"""Dispatch for the MeZO-MinExpLSTM fused optimizer kernel.

Fused memory pipeline for the MeZO-MinExpLSTM optimizer (PRD
v2.0 sections 2.2-2.5 and 6). The step consumes the scalar
SPSA estimate and the perturbation seeds (never a materialized
gradient vector), replays the FIR and MeZO perturbations from
the seeds, runs the shared MinExpLSTM cell, and applies the
FiLM-modulated MeZO update with coordinate-wise clipping and
decoupled weight decay.
"""

import torch

from .dispatch import Backend, _select_backend


def dispatch_mezo_minexplstm_step(
    fir: torch.Tensor,
    mezo: torch.Tensor,
    weight: torch.Tensor,
    h_int8: torch.Tensor,
    scale_h: torch.Tensor,
    # Cell parameters
    phi_w: torch.Tensor,
    phi_b: torch.Tensor,
    iota_w: torch.Tensor,
    iota_b: torch.Tensor,
    h_w: torch.Tensor,
    h_b: torch.Tensor,
    out_w: torch.Tensor,
    out_b: torch.Tensor,
    film_w: torch.Tensor,
    film_b: torch.Tensor,
    # Hyperparameters
    lr: float,
    wd: float,
    clamp_val: float,
    clip_bound: float,
    backend: Backend = "cpu",
) -> None:
    """
    Dispatch the MeZO-MinExpLSTM fused optimizer step to the
    selected backend.

    The fused kernel performs the complete optimizer step in a
    single pass:
    1. Dequantize the INT8 hidden state to FP32
    2. Layer-wise RMS normalization of fir/mezo/h
    3. Cell forward (log-domain gates, 4D latent FiLM)
    4. FiLM-modulated MeZO update with coordinate clipping
    5. Decoupled weight decay
    6. Re-quantize the hidden state to INT8 (symmetric max-abs)

    The FIR signal and the raw MeZO vector are already the
    seed-replayed [N] float vectors (the Python reference
    ``_replay`` replays them from the PRNG seeds host-side,
    so no kernel needs the seeds or the FIR kernel weights).

    Args:
        fir: FIR momentum signal [N] (already seed-replayed)
        mezo: Raw MeZO directional vector [N] (mu * z)
        weight: Weight tensor [N]
        h_int8: Hidden state (INT8) [N]
        scale_h: Hidden state scale (scalar)
        phi_w, phi_b: Cell phi projection weights/bias
        iota_w, iota_b: Cell iota projection weights/bias
        h_w, h_b: Cell h projection weights/bias
        out_w, out_b: Cell output projection weights/bias
        film_w, film_b: FiLM projection weights/bias
        lr: Learning rate
        wd: Weight decay
        clamp_val: Clamp value for beta
        clip_bound: Coordinate-wise clip bound (a_t * delta_q)
        backend: Backend to use ("naive", "cpu", "gpu")

    Note: All tensors are modified in-place. The function returns None.
    """
    # Validate inputs
    if backend not in ("naive", "cpu", "gpu"):
        raise ValueError(f"Unknown backend: {backend}. Must be one of: naive, cpu, gpu")

    def _naive():
        _mezo_minexplstm_step_naive(
            fir,
            mezo,
            weight,
            h_int8,
            scale_h,
            phi_w,
            phi_b,
            iota_w,
            iota_b,
            h_w,
            h_b,
            out_w,
            out_b,
            film_w,
            film_b,
            lr,
            wd,
            clamp_val,
            clip_bound,
        )

    def _ensure_cpu_op() -> None:
        """Build/import the C++ extension once."""
        from nanochat.ops.kernels.cpu_loader import load_cpu_mezo_minexplstm_extension

        load_cpu_mezo_minexplstm_extension()

    def _cpu():
        # CPU fused kernel via torch.ops
        _ensure_cpu_op()
        torch.ops.nanochat.mezo_minexplstm_step(
            fir,
            mezo,
            weight,
            h_int8,
            scale_h,
            phi_w,
            phi_b,
            iota_w,
            iota_b,
            h_w,
            h_b,
            out_w,
            out_b,
            film_w,
            film_b,
            lr,
            wd,
            clamp_val,
            clip_bound,
        )

    def _gpu():
        # GPU (Taichi) fused kernel. The GPU path receives the per-tensor scale as a float (Taichi kernels take scalars, not torch tensors) and returns the updated scale; write it back so the caller's scale tensor reflects this step's quantization, matching the in-place mutation of the naive and CPU paths.
        from nanochat.ops.kernels.gpu_loader import run_mezo_minexplstm_step

        scale_h_new = run_mezo_minexplstm_step(
            fir,
            mezo,
            weight,
            h_int8,
            scale_h.item(),
            phi_w,
            phi_b,
            iota_w,
            iota_b,
            h_w,
            h_b,
            out_w,
            out_b,
            film_w,
            film_b,
            lr,
            wd,
            clamp_val,
            clip_bound,
        )
        scale_h.copy_(torch.tensor(scale_h_new))

    _select_backend(backend, naive=_naive, cpu=_cpu, gpu=_gpu)


def _mezo_minexplstm_step_naive(
    fir: torch.Tensor,
    mezo: torch.Tensor,
    weight: torch.Tensor,
    h_int8: torch.Tensor,
    scale_h: torch.Tensor,
    # Cell parameters
    phi_w: torch.Tensor,
    phi_b: torch.Tensor,
    iota_w: torch.Tensor,
    iota_b: torch.Tensor,
    h_w: torch.Tensor,
    h_b: torch.Tensor,
    out_w: torch.Tensor,
    out_b: torch.Tensor,
    film_w: torch.Tensor,
    film_b: torch.Tensor,
    # Hyperparameters
    lr: float,
    wd: float,
    clamp_val: float,
    clip_bound: float,
) -> None:
    """Naive PyTorch implementation for testing/debugging.

    Single-tensor transcription of
    ``MeZo_MinExpLstm_Optimizer._step_shape_group``
    (nanochat/training/mezo_minexplstm.py): dequantize the
    INT8 hidden state, layer-wise RMS-normalize the FIR and
    MeZO signals, run the log-domain cell with the 4D latent
    FiLM head, apply the clipped FiLM-modulated MeZO update
    with decoupled weight decay, and symmetric max-abs
    requantize the hidden state. All state tensors mutate in
    place, matching the C++/Taichi kernels. Inputs may carry
    the parameter's original shape; the flattened form is only
    the internal working layout.
    """
    eps = 1e-8

    # The fused kernels are not autograd-tracked; neither is the naive reference (the optimizer runs its step under no_grad).
    with torch.no_grad():
        fir_f = fir.float().reshape(-1)
        mezo_f = mezo.float().reshape(-1)
        weight_f = weight.float().reshape(-1)

        # 1. Dequantize the INT8 hidden state with its scale.
        h_prev = h_int8.to(torch.float32).reshape(-1) * scale_h

        # 2. Layer-wise RMS normalization (the cell's scale-invariant input contract).
        rms_fir = torch.sqrt(fir_f.square().mean() + eps)
        rms_mezo = torch.sqrt(mezo_f.square().mean() + eps)
        rms_h = torch.sqrt(h_prev.square().mean() + eps)
        fir_norm = fir_f / (rms_fir + eps)
        mezo_norm = mezo_f / (rms_mezo + eps)
        h_norm = h_prev / (rms_h + eps)

        # 3. Cell forward: log-domain gates, 4D latent FiLM head.
        x = torch.stack([fir_norm, mezo_norm, h_norm], dim=-1)
        phi_val = x @ phi_w.float().reshape(-1) + phi_b
        iota_val = x @ iota_w.float().reshape(-1) + iota_b
        hat_phi = torch.sigmoid(phi_val - iota_val)
        tilde_h = x @ h_w.float().reshape(-1) + h_b
        h_next = hat_phi * h_prev + (1.0 - hat_phi) * tilde_h
        out_w_f = out_w.float().reshape(-1)
        z = torch.relu(h_next.unsqueeze(-1) * out_w_f + out_b)
        film_out = z @ film_w.float().reshape(2, 4).t() + film_b
        gamma = film_out[..., 0]
        beta = film_out[..., 1]

        # 4. FiLM clamp bounds: gain keeps (1 + gamma) in [0.1, 3.0]; beta is clamped by clamp_val.
        gamma_c = torch.clamp(gamma, -0.9, 2.0)
        beta_c = torch.clamp(beta, -clamp_val, clamp_val)

        # 5. FiLM-modulated MeZO update with coordinate-wise clipping (PRD 2.4/2.5) and decoupled weight decay. PRD 2.5: w_{t+1} = (1 - lambda*eta) * w_t - eta * clipped. The decay is applied multiplicatively to the weight (AdamW-style decoupled), not folded into the update term -- folding it in yields lr^2 * wd, which the fused kernels do not do.
        raw_update = gamma_c * mezo_f + beta_c
        clipped = torch.clamp(raw_update, -clip_bound, clip_bound)
        weight_next = weight_f * (1.0 - lr * wd) - lr * clipped

        # 6. Symmetric max-abs requantization of the new hidden state.
        scale_h_new = h_next.abs().max() / 127.0 + eps
        h_int8_new = torch.clamp(torch.round(h_next / scale_h_new), -128, 127).to(
            torch.int8
        )

        # In-place write-back (copy_ preserves arbitrary shapes/strides).
        weight.copy_(weight_next.reshape(weight.shape).to(weight.dtype))
        h_int8.copy_(h_int8_new.reshape(h_int8.shape))
        scale_h.copy_(scale_h_new)


__all__ = ["dispatch_mezo_minexplstm_step"]

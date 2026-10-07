"""
Dispatch for the SGD-MinExpLSTM fused optimizer kernel.

Fused memory pipeline for the SGD-MinExpLSTM optimizer (PRD v3.2 Section 6).
"""

import torch

from .dispatch import Backend, _select_backend


def dispatch_sgd_minexplstm_step(
    grad: torch.Tensor,
    weight: torch.Tensor,
    ema_fast_int8: torch.Tensor,
    ema_slow_int8: torch.Tensor,
    h_int8: torch.Tensor,
    scale_fast: torch.Tensor,
    scale_slow: torch.Tensor,
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
    w_beta: torch.Tensor,
    b_beta: torch.Tensor,
    # Hyperparameters
    lr: float,
    wd: float,
    lam: float,
    clamp_val: float,
    backend: Backend = "cpu",
) -> None:
    """
    Dispatch the SGD-MinExpLSTM fused optimizer step to the selected backend.

    The fused kernel performs the complete optimizer step in a single pass:
    1. Dequantize INT8 states to FP32
    2. Layer-wise RMS normalization
    3. Lambda-coupled beta prediction
    4. Dynamic EMA blending
    5. EMA normalization
    6. Cell forward (log-domain gates, 4D latent FiLM)
    7. FiLM-modulated SGD step with weight decay
    8. Re-quantize to INT8 (symmetric max-abs)

    Args:
        grad: Gradient tensor [N]
        weight: Weight tensor [N]
        ema_fast_int8: Fast EMA state (INT8) [N]
        ema_slow_int8: Slow EMA state (INT8) [N]
        h_int8: Hidden state (INT8) [N]
        scale_fast: Fast EMA scale (scalar)
        scale_slow: Slow EMA scale (scalar)
        scale_h: Hidden state scale (scalar)
        phi_w, phi_b: Cell phi projection weights/bias
        iota_w, iota_b: Cell iota projection weights/bias
        h_w, h_b: Cell h projection weights/bias
        out_w, out_b: Cell output projection weights/bias
        film_w, film_b: FiLM projection weights/bias
        w_beta, b_beta: Beta predictor parameters
        lr: Learning rate
        wd: Weight decay
        lam: Lambda (beta_slow - beta_fast)
        clamp_val: Clamp value for delta
        backend: Backend to use ("naive", "cpu", "gpu")

    Note: All tensors are modified in-place. The function returns None.
    """
    # Validate inputs
    if backend not in ("naive", "cpu", "gpu"):
        raise ValueError(f"Unknown backend: {backend}. Must be one of: naive, cpu, gpu")

    def _naive():
        # Naive PyTorch implementation for testing/debugging
        _sgd_minexplstm_step_naive(
            grad,
            weight,
            ema_fast_int8,
            ema_slow_int8,
            h_int8,
            scale_fast,
            scale_slow,
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
            w_beta,
            b_beta,
            lr,
            wd,
            lam,
            clamp_val,
        )

    def _ensure_cpu_op() -> None:
        """Build/import the C++ extension once."""
        from nanochat.ops.kernels.cpu_loader import load_cpu_sgd_minexplstm_extension

        load_cpu_sgd_minexplstm_extension()

    def _cpu():
        # CPU fused kernel via torch.ops
        _ensure_cpu_op()
        torch.ops.nanochat.sgd_minexplstm_step(
            grad,
            weight,
            ema_fast_int8,
            ema_slow_int8,
            h_int8,
            scale_fast,
            scale_slow,
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
            w_beta,
            b_beta,
            lr,
            wd,
            lam,
            clamp_val,
        )

    def _gpu():
        # GPU (Taichi) fused kernel. The GPU path receives the
        # per-tensor scales as floats (Taichi kernels take
        # scalars, not torch tensors) and returns the updated
        # scales; write them back so the caller's scale
        # tensors reflect this step's quantization, matching
        # the in-place mutation of the naive and CPU paths.
        from nanochat.ops.kernels.gpu_loader import run_sgd_minexplstm_step

        scale_fast_new, scale_slow_new, scale_h_new = run_sgd_minexplstm_step(
            grad,
            weight,
            ema_fast_int8,
            ema_slow_int8,
            h_int8,
            scale_fast.item(),
            scale_slow.item(),
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
            w_beta,
            b_beta,
            lr,
            wd,
            lam,
            clamp_val,
        )
        scale_fast.copy_(torch.tensor(scale_fast_new))
        scale_slow.copy_(torch.tensor(scale_slow_new))
        scale_h.copy_(torch.tensor(scale_h_new))

    _select_backend(backend, naive=_naive, cpu=_cpu, gpu=_gpu)


def _sgd_minexplstm_step_naive(
    grad: torch.Tensor,
    weight: torch.Tensor,
    ema_fast_int8: torch.Tensor,
    ema_slow_int8: torch.Tensor,
    h_int8: torch.Tensor,
    scale_fast: torch.Tensor,
    scale_slow: torch.Tensor,
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
    w_beta: torch.Tensor,
    b_beta: torch.Tensor,
    # Hyperparameters
    lr: float,
    wd: float,
    lam: float,
    clamp_val: float,
) -> None:
    """Naive PyTorch implementation for testing/debugging.

    Single-tensor transcription of
    ``SGD_MinExpLstm_Optimizer._step_shape_group`` (nanochat/training/
    minexp_lstm.py) and ``batched_cell_forward`` (nanochat/training/
    batched_step.py): dequantize, layer-wise RMS normalization,
    lambda-coupled beta prediction, dual-EMA blending, EMA
    normalization, log-domain cell forward with the 4D latent FiLM
    head, FiLM-modulated SGD update, and symmetric max-abs
    requantization. All seven state tensors mutate in place, matching
    the C++/Taichi kernels. Inputs may carry the parameter's original
    shape; the flattened form is only the internal working layout.
    """
    eps = 1e-8

    # The fused kernels are not autograd-tracked; neither is the
    # naive reference (the optimizer runs its step under no_grad).
    with torch.no_grad():
        grad_f = grad.float().reshape(-1)
        weight_f = weight.float().reshape(-1)

        # 1. Dequantize INT8 states with their per-tensor scales.
        ema_fast = ema_fast_int8.to(torch.float32).reshape(-1) * scale_fast
        ema_slow = ema_slow_int8.to(torch.float32).reshape(-1) * scale_slow
        h_prev = h_int8.to(torch.float32).reshape(-1) * scale_h

        # 2. Layer-wise RMS normalization of the gradient.
        rms_g = torch.sqrt(grad_f.square().mean() + eps)
        g_norm = grad_f / (rms_g + eps)

        # 3. Lambda-coupled beta prediction (hard sigmoid).
        u = g_norm * w_beta + b_beta
        sigma = torch.clamp(0.5 + 0.125 * u, 0.0, 1.0)
        beta_slow = 2.0 * lam + (0.999 - 2.0 * lam) * sigma
        beta_fast = beta_slow - lam

        # 4. Dynamic EMA blending.
        ema_fast_new = beta_fast * ema_fast + (1.0 - beta_fast) * grad_f
        ema_slow_new = beta_slow * ema_slow + (1.0 - beta_slow) * grad_f

        # 5. Normalize EMAs for scale-invariant cell input.
        rms_fast = torch.sqrt(ema_fast_new.square().mean() + eps)
        rms_slow = torch.sqrt(ema_slow_new.square().mean() + eps)
        fast_norm = ema_fast_new / (rms_fast + eps)
        slow_norm = ema_slow_new / (rms_slow + eps)

        # 6. Cell forward: log-domain gates, 4D latent FiLM head.
        x = torch.stack([g_norm, fast_norm, slow_norm], dim=-1)
        phi_val = x @ phi_w.float().reshape(-1) + phi_b
        iota_val = x @ iota_w.float().reshape(-1) + iota_b
        hat_phi = torch.sigmoid(phi_val - iota_val)
        tilde_h = x @ h_w.float().reshape(-1) + h_b
        h_next = hat_phi * h_prev + (1.0 - hat_phi) * tilde_h
        out_w_f = out_w.float().reshape(-1)
        z = torch.relu(h_next.unsqueeze(-1) * out_w_f + out_b)
        film_out = z @ film_w.float().reshape(2, 4).t() + film_b
        gamma = film_out[..., 0]
        delta = film_out[..., 1]

        # 7. FiLM clamp bounds: gain keeps (1 + gamma) in [0.1, 3.0].
        gamma_c = torch.clamp(gamma, -0.9, 2.0)
        delta_c = torch.clamp(delta, -clamp_val, clamp_val)

        # 8. FiLM-modulated SGD step with weight decay.
        update = (1.0 + gamma_c) * (-lr * grad_f) + delta_c - lr * wd * weight_f

        # 9. Symmetric max-abs requantization of the new states.
        def _quantize(fp32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            scale = fp32.abs().max() / 127.0 + eps
            int8 = torch.clamp(torch.round(fp32 / scale), -128, 127).to(torch.int8)
            return int8, scale

        ef_int8, scale_fast_new = _quantize(ema_fast_new)
        es_int8, scale_slow_new = _quantize(ema_slow_new)
        h_int8_new, scale_h_new = _quantize(h_next)

        # In-place write-back (copy_ preserves arbitrary shapes/strides).
        weight.copy_((weight_f + update).reshape(weight.shape).to(weight.dtype))
        ema_fast_int8.copy_(ef_int8.reshape(ema_fast_int8.shape))
        ema_slow_int8.copy_(es_int8.reshape(ema_slow_int8.shape))
        h_int8.copy_(h_int8_new.reshape(h_int8.shape))
        scale_fast.copy_(scale_fast_new)
        scale_slow.copy_(scale_slow_new)
        scale_h.copy_(scale_h_new)


__all__ = ["dispatch_sgd_minexplstm_step"]

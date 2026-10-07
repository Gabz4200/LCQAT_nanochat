"""Batch helpers for the SGD-MinExpLSTM optimizer.

The cell is elementwise and permutation-invariant, so same-shape
parameter tensors are concatenated into a single batch and processed
in one forward pass instead of looping over 200+ tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rms_batch(t: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(t.float().square().mean(dim=1) + eps)


def dequantize_batch(int8_batch: Tensor, scale: Tensor) -> Tensor:
    return int8_batch.to(torch.float32) * scale


def quantize_batch(fp32: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric max-abs INT8 quantization of ``fp32`` batch.

    Returns ``(int8, scale)`` with ``scale = max|fp32| / 127 + eps``
    (PRD section 4.1). Scales stay 0-dim tensors: a float ``.item()``
    would force a CPU sync per tensor per step.
    """
    scale = fp32.detach().abs().max() / 127.0 + 1e-8
    int8 = torch.clamp(torch.round(fp32 / scale), -128, 127).to(torch.int8)
    return int8, scale


def batched_cell_forward(
    cell,
    grad_batch: Tensor,
    ema_fast_batch: Tensor,
    ema_slow_batch: Tensor,
    h_batch: Tensor,
    lam: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Batched dual-EMA cell forward over one shape group (PRD 3.2-3.5).

    Shared between ``step`` and the closure-based ``meta_step`` so the
    two paths stay bit-identical. Returns
    ``(gamma, delta, next_h, ema_fast_new, ema_slow_new)``.

    ``grad_batch`` / ``ema_fast_batch`` / ``ema_slow_batch`` /
    ``h_batch`` are ``[B, *shape]`` where B is the number of tensors
    in the group. They are flattened to ``[B, N]`` for the cell
    forward, then reshaped back by the caller.
    """
    B = grad_batch.shape[0]
    N = grad_batch[0].numel()
    flat_grad = grad_batch.reshape(B, N)
    flat_fast = ema_fast_batch.reshape(B, N)
    flat_slow = ema_slow_batch.reshape(B, N)
    flat_h = h_batch.reshape(B, N, 1)

    # Layer-wise RMS normalization (PRD 3.1): one RMS per tensor.
    rms_g = rms_batch(flat_grad)
    g_norm = flat_grad / (rms_g.unsqueeze(1) + 1e-8)

    # Lambda-coupled per-coordinate betas from the normalized
    # gradient (PRD 3.2). The predictor broadcasts: g_norm [B, N].
    beta_fast, beta_slow = cell.predict_betas(g_norm, lam=lam)

    # Dynamic EMA IIR blending (PRD 3.3).
    ema_fast_new = beta_fast * flat_fast + (1.0 - beta_fast) * flat_grad
    ema_slow_new = beta_slow * flat_slow + (1.0 - beta_slow) * flat_grad

    # Normalize EMAs for scale-invariant cell input (PRD 3.3).
    rms_fast = rms_batch(ema_fast_new)
    rms_slow = rms_batch(ema_slow_new)
    fast_norm = ema_fast_new / (rms_fast.unsqueeze(1) + 1e-8)
    slow_norm = ema_slow_new / (rms_slow.unsqueeze(1) + 1e-8)

    x_in = torch.stack([g_norm, fast_norm, slow_norm], dim=-1)
    gamma, delta, next_h = cell(x_in, flat_h)
    return gamma, delta, next_h, ema_fast_new, ema_slow_new

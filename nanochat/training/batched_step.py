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


def differentiable_cell_step(
    cell,
    grad: Tensor,
    ema_fast: Tensor,
    ema_slow: Tensor,
    h_fp32: Tensor,
    lam: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The differentiable MinExpLSTM cell step (PRD 3.1-3.5).

    One formula, two callers: ``batched_cell_forward`` (the fused
    step, which flattens same-shape tensors into a ``[B, N]``
    batch first) and each optimizer's ``_meta_step_closure`` (which
    rebuilds one parameter's updated value from its gradient, EMA
    state, and hidden state as a flat ``[N]`` tensor). Extracting
    the math here means the step and the closure cannot drift --
    the exact failure the multi-step parity test guards against.

    Inputs are flat per-parameter tensors. A ``[B, N]`` batch
    (the fused step) is RMS-normalized per row; a ``[N]`` tensor
    (the closure) is RMS-normalized by a scalar -- matching the
    two layouts the callers hold. Returns
    ``(gamma, delta, next_h, ema_fast_new, ema_slow_new)`` in the
    same layout. The caller reshapes and applies the FiLM update
    (:func:`film_sgd_update`).
    """
    # Layer-wise RMS normalization (PRD 3.1): one RMS per
    # tensor. A batched [B, N] input normalizes per row; a
    # flat [N] input normalizes by a scalar (the closure's
    # per-parameter layout).
    batched = grad.dim() == 2
    rms_g = rms_batch(grad) if batched else scalar_rms(grad)
    g_norm = grad / (rms_g.unsqueeze(-1) if batched else rms_g + 1e-8)

    # Lambda-coupled per-coordinate betas from the normalized
    # gradient (PRD 3.2). The predictor broadcasts: g_norm.
    beta_fast, beta_slow = cell.predict_betas(g_norm, lam=lam)

    # Dynamic EMA IIR blending (PRD 3.3).
    ema_fast_new = beta_fast * ema_fast + (1.0 - beta_fast) * grad
    ema_slow_new = beta_slow * ema_slow + (1.0 - beta_slow) * grad

    # Normalize EMAs for scale-invariant cell input (PRD 3.3).
    rms_fast = rms_batch(ema_fast_new) if batched else scalar_rms(ema_fast_new)
    rms_slow = rms_batch(ema_slow_new) if batched else scalar_rms(ema_slow_new)
    fast_norm = ema_fast_new / (rms_fast.unsqueeze(-1) if batched else rms_fast + 1e-8)
    slow_norm = ema_slow_new / (rms_slow.unsqueeze(-1) if batched else rms_slow + 1e-8)

    x_in = torch.stack([g_norm, fast_norm, slow_norm], dim=-1)
    gamma, delta, next_h = cell(x_in, h_fp32)
    return gamma, delta, next_h, ema_fast_new, ema_slow_new


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

    return differentiable_cell_step(cell, flat_grad, flat_fast, flat_slow, flat_h, lam)


def film_sgd_update(
    gamma: Tensor,
    delta: Tensor,
    grad: Tensor,
    weight: Tensor,
    lr: float | Tensor,
    wd: float,
    clamp_val: float,
) -> Tensor:
    """FiLM-modulated SGD update (PRD section 3.6), shared by every
    SGD-MinExpLSTM path.

    One formula, four callers: the fused-kernel step, the pure-PyTorch
    batched step, the closure-based meta-step rebuild, and the naive
    reference oracle. Extracting it means a math change (for example
    the decoupled-weight-decay form) is made once instead of drifting
    between copies -- the exact failure the multi-step parity test
    guards against.

    ``gamma``/``delta`` are the raw FiLM pair (unclamped); ``grad``
    and ``weight`` are the parameter's gradient and current value
    (same shape). ``lr`` is the (data-dependent) learning rate: a
    float for the fixed-LR paths, a differentiable 0-dim tensor
    for the learnable-LR path (broadcast over the update). Returns
    the parameter *delta* (not the new value):
    ``theta_new = theta + film_sgd_update(...)``.
    """
    gamma_c = torch.clamp(gamma, -0.9, 2.0)
    delta_c = torch.clamp(delta, -clamp_val, clamp_val)
    return (1.0 + gamma_c) * (-lr * grad) + delta_c - lr * wd * weight


def film_mezo_update(
    gamma: Tensor,
    delta: Tensor,
    mezo: Tensor,
    clamp_val: float,
    clip_bound: float,
) -> Tensor:
    """FiLM-modulated MeZO update (PRD sections 2.4/2.5), shared by
    every MeZO-MinExpLSTM path.

    The MeZO twin of :func:`film_sgd_update`: one formula for the
    fused-kernel step, the pure-PyTorch step, the closure-based
    meta-step rebuild, and the naive reference oracle, so a math
    change is made once instead of drifting between copies.

    ``gamma``/``delta`` are the raw FiLM pair (unclamped);
    ``mezo`` is the raw MeZO directional vector ``mu * z``.
    Returns the clipped update (not the new value): the caller
    applies ``theta_new = theta * (1 - lr * wd) - lr * update``.
    """
    gamma_c = torch.clamp(gamma, -0.9, 2.0)
    delta_c = torch.clamp(delta, -clamp_val, clamp_val)
    raw_update = gamma_c * mezo + delta_c
    return torch.clamp(raw_update, -clip_bound, clip_bound)


def differentiable_mezo_cell_step(
    cell,
    fir: Tensor,
    mezo: Tensor,
    h_fp32: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """The differentiable MeZO cell step (PRD sections 2.3-2.5).

    The MeZO twin of :func:`differentiable_cell_step`: one formula
    for the closure-based meta-step rebuild and the naive reference
    oracle. The MeZO cell takes three RMS-normalized inputs -- the
    FIR signal, the MeZO directional vector, and the hidden state
    -- rather than the SGD cell's gradient/EMA inputs.

    Inputs are per-parameter tensors in their natural shape
    (the closure holds ``[..., *shape]``; the fused step
    flattens to ``[N]``). Each is RMS-normalized by a scalar
    over its own tensor, stacked on the last axis into
    ``[..., *shape, 3]``, then flattened to ``[-1, 3]`` for
    the cell (which takes ``x [..., 3]`` and ``h_prev [..., 1]``).
    Returns ``(gamma, delta, next_h)`` in the ``[-1]`` layout.
    The caller applies the FiLM update (:func:`film_mezo_update`).
    """
    # Scale-invariant cell inputs (PRD 2.3): each input is
    # RMS-normalized over its own tensor (a scalar RMS, not
    # the SGD cell's per-row RMS), so the cell sees
    # direction, not magnitude.
    fir_norm = fir / (scalar_rms(fir) + eps)
    mezo_norm = mezo / (scalar_rms(mezo) + eps)
    h_norm = h_fp32 / (scalar_rms(h_fp32) + eps)
    x_in = torch.stack([fir_norm, mezo_norm, h_norm], dim=-1).reshape(-1, 3)
    gamma, delta, next_h = cell(x_in, h_fp32.reshape(-1, 1))
    return gamma, delta, next_h


def scalar_rms(t: Tensor, eps: float = 1e-8) -> Tensor:
    """Layer-wise RMS of ``t``: ``sqrt(mean(t^2) + eps)``, scalar."""
    return torch.sqrt(t.float().square().mean() + eps)


def meta_step_closure(
    optimizer: torch.optim.Optimizer,
    meta_optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    closure,
    rebuild,
) -> Tensor:
    """The shared skeleton of both MinExpLSTM optimizers' ``meta_step(model, closure)``.

    ``rebuild`` returns the updated parameter value (a tensor,
    possibly a view of ``p.data``) or ``None`` to skip the parameter.
    """
    if closure is None:
        raise TypeError(
            "meta_step(model) needs a closure(params) -> loss; pass a "
            "loss tensor directly only when it already runs through "
            "the cell"
        )
    meta_optimizer.zero_grad(set_to_none=True)

    params = dict(model.named_parameters())
    name_of = {id(p): n for n, p in model.named_parameters()}
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}

    rebuilt = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in owned:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                continue
            updated = rebuild(p, state, group)
            if updated is None:
                continue
            params[name_of[id(p)]] = updated
            rebuilt += 1

    # A zero rebuild means the caller zeroed grads before
    # meta_step (the historical bug) or owns no trained params.
    if rebuilt == 0:
        raise RuntimeError(
            "meta_step(model, closure) rebuilt 0 parameters: every "
            "optimizer-owned parameter had a None gradient or empty "
            "state. Call meta_step BEFORE zero_grad (the closure "
            "reads p.grad to rebuild each updated parameter), or "
            "pass a differentiable loss tensor directly via "
            "meta_step(loss)."
        )

    loss = closure(params)
    if not isinstance(loss, Tensor) or not loss.requires_grad:
        raise TypeError(
            "meta_step closure must return a differentiable loss "
            "tensor built from the params it was handed; the cell "
            "has no gradient path otherwise"
        )
    loss.backward()
    meta_optimizer.step()
    return loss

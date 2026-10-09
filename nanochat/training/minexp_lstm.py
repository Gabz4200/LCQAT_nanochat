"""SGD-MinExpLSTM joint online learned optimizer (PRD v3.2).

A drop-in AdamW replacement: a shared elementwise
``SGDMinExpLSTMCellFiLM4D`` modulates a base SGD step through FiLM
gain/shift parameters, and the three persistent optimizer states
(fast EMA, slow EMA, hidden state) live in INT8 -- 3 bytes/param
instead of AdamW's 8 (PRD section 4.2, a 62.5% RAM traffic cut).

The math anchors:

* MinLSTM (Feng et al. 2024, arXiv:2410.01201): gates depend only on
  the current input, so the recurrence is a linear scan.
* Log-domain exponential gating (Horuz et al. 2025,
  arXiv:2508.03614): ``sigmoid(phi - iota)`` with no ``exp()``.
* Input-dependent decay (Gu & Dao 2023, arXiv:2312.00752): static
  momentum hyperparameters become data-dependent dynamic decays.
* 8-bit Optimizers (Dettmers et al. 2021, arXiv:2110.02861):
  dynamic symmetric max-abs quantization of optimizer states --
  here per-tensor (one scale per state tensor) rather than
  block-wise, trading a little precision for zero blocking
  overhead on small CPU tensors.
* Coordinate-wise learned optimizer (Andrychowicz et al. 2016,
  arXiv:1606.04474) with scale-invariant RMS inputs (Metz et al.
  2022, arXiv:2203.00845) and FiLM modulation (Perez et al. 2017,
  arXiv:1709.07871), anchored on SGD so step 0 is exactly SGD
  (He et al. 2015, arXiv:1512.03385).

Integration contract: this optimizer trains any ``nn.Module`` the
way ``torch.optim.SGD`` does -- DiffusionBlocks engines (train_step
/ denoise_step), LC-QAT codebook parameters (the straight-through
estimator routes gradients to the raw deltas), SparseProp masked
linear layers (pruned positions hold exact-zero gradients and stay
frozen while the head is at its zero init), and DARD's ``lm_head``
under the EDM objective alike.

Deliberate deviations from the PRD text, each for a stated reason:

* No separate C++/Taichi fused kernel ships yet. The step is one
  vectorized pass per shape group (a single RMS reduction, one
  batched cell GEMM, one in-place update; INT8 traffic only), which
  is the memory behavior PRD section 6 asks for. A native fused
  kernel following the section 6.1 pipeline is future work.
* The online meta-pass rebuilds the loss differentiably through the
  cell via a closure (``torch.func.functional_call``), as before.
  The PRD's ``meta_step(loss)`` shorthand (step, then
  backward an already-computed loss) cannot train the cell: ``step``
  runs under ``no_grad`` with in-place ``.data`` updates and a
  non-differentiable round-to-INT8, so that graph is severed. Both
  spellings exist -- the closure form for real co-training, the
  loss-tensor form for graphs that genuinely run through the cell.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from nanochat.training.batched_step import (
    batched_cell_forward,
    differentiable_cell_step,
    film_sgd_update,
    meta_step_closure,
    quantize_batch,
)
from nanochat.training.cell import MinExpLSTMCell, SGDMinExpLSTMCellFiLM4D


def _inv_sigmoid(target: float) -> float:
    """The logit whose sigmoid is ``target``: logit = log(t/(1-t)).

    Inverse of the LR parametrization: a predictor bias set to
    this value makes ``sigmoid(bias) == target`` for any target
    in (0, 1), so a zero-weight LR predictor starts at the
    requested scalar ``lr``. ``target=0`` (the degenerate
    "freeze updates" construction) maps to a large negative
    logit: the LR is strictly positive (never exactly 0) but
    numerically negligible, so a zero-LR optimizer still steps
    ~nothing while the parametrization stays smooth.
    """
    if target <= 0.0:
        return -12.0
    if target >= 1.0:
        return 12.0
    return float(torch.log(torch.tensor(target / (1.0 - target))))


class SGD_MinExpLstm_Optimizer(torch.optim.Optimizer):
    """Joint online learned optimizer, an AdamW replacement.

    Per coordinate: RMS-normalize the gradient and both EMAs, predict
    lambda-coupled decays, blend the dual EMAs, run the shared FiLM
    cell, and apply (PRD section 3.6)

        theta <- theta + (1 + clamp(gamma)) * (-lr * g)
                         + clamp(delta) - lr * wd * theta

    with ``gamma`` in ``[-0.9, 2.0]`` and ``delta`` in
    ``[-clamp_val, clamp_val]``. The persistent states (fast EMA,
    slow EMA, hidden) are INT8 with per-tensor scales (3
    bytes/param). With a zero-initialized FiLM head the trajectory
    matches native SGD exactly (the PRD step-0 equivalence check).

    Tensors are grouped by shape so the cell runs once per shape
    group instead of once per parameter.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        lam: float = 0.10,
        clamp_val: float = 1.0,
        meta_lr: float = 1e-3,
        # Fused-kernel backend ("naive", "cpu", "gpu"): the same selection surface the MeZO optimizer exposes. "naive" is the pure-PyTorch reference path; "cpu" and "gpu" run the fused kernels (nanochat/ops).
        backend: Literal["naive", "cpu", "gpu"] = "naive",
        # Deprecated pre-PRD-v3.2 spellings, mapped below so existing scripts keep running. `alpha_residual=0` freezes the FiLM head (pure SGD forever); `learnable_beta=False` freezes the beta predictor at the fixed `beta`.
        alpha_residual: float | None = None,
        beta: float = 0.9,
        learnable_beta: bool = True,
        beta_min: float = 0.5,
        beta_max: float = 0.999,
        # Quantization-Aware Training on the 4-bit main-model layout (nanochat/models/quant/fourbit.py). When True the optimizer (1) quantizes gradients to INT8 (8-bit gradients, the PRD's SGD allowance) and (2) refreshes each FourBitLinear's 4-bit storage mirror after the step, so the persistent weight form stays at 4 bits between steps. Off by default: the shipped optimizers keep their fp32/INT8-state path byte-for-byte.
        qat_4bit: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 < lam <= 0.4:
            raise ValueError(f"Invalid lam: {lam}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            lam=lam,
            clamp_val=clamp_val,
        )
        super().__init__(params, defaults)
        self.cell = SGDMinExpLSTMCellFiLM4D(input_size=3, hidden_size=1, latent_dim=4)
        if alpha_residual is not None and alpha_residual == 0.0:
            # The old "disable the residual" spelling: pin the FiLM
            # head at its zero init so the update stays pure SGD.
            for q in self.cell.film_proj.parameters():
                q.requires_grad_(False)
        if not learnable_beta:
            # The old fixed-decay spelling: freeze the predictor with
            # b_beta solved so beta_slow(g_norm=0) == beta.
            with torch.no_grad():
                sigma = (beta - 2.0 * lam) / (0.999 - 2.0 * lam)
                sigma = min(max(sigma, 0.0), 1.0)
                self.cell.b_beta.fill_((sigma - 0.5) / 0.125)
            for q in (self.cell.w_beta, self.cell.b_beta):
                q.requires_grad_(False)
        if beta_min != 0.5 or beta_max != 0.999:
            warnings.warn(
                "beta_min/beta_max have no equivalent in the lambda-coupled "
                "design (bounds are structural now) and are ignored",
                stacklevel=2,
            )
        # The optimizer's optimizer (PRD principle 5): AdamW over the cell's own < 50 params plus the LR predictor. One instance is reused across meta-steps so its moments persist. The meta LR (meta_lr) is a plain hyperparameter -- Lightning's LR finder tunes it before training (warmup); it is NOT learned online.
        self.meta_optimizer = torch.optim.AdamW(
            list(self.cell.parameters()), lr=meta_lr
        )
        # Data-dependent, learnable learning rate. The LR is a function of the current step's data (the RMS-normalized gradient magnitude), not a stored per-coordinate vector: the model is elementwise and parallel across the batch dimension, so one LR value per batch element suffices. A tiny linear predictor maps the normalized gradient magnitude to a logit, and a sigmoid parametrizes the LR to (0, 1) -- strictly positive (the optimizer always moves) and bounded by 1 (the step is bounded). The predictor is zero-initialized so a fresh optimizer's LR is the constant `lr` (the inverse-sigmoid of the requested scalar), matching native SGD at step 0; the meta-optimizer then co-trains it so the LR adapts to the loss landscape.
        self.lr_predictor = nn.Linear(1, 1)
        with torch.no_grad():
            # Zero weight, bias = inv_sigmoid(lr): the initial LR is the constant `lr` regardless of the input.
            self.lr_predictor.weight.zero_()
            self.lr_predictor.bias.fill_(float(_inv_sigmoid(lr)))
        self.meta_optimizer.add_param_group(
            {"params": list(self.lr_predictor.parameters())}
        )
        # Backend for the fused step (the same "naive"/"cpu"/"gpu" contract the MeZO-MinExpLSTM dispatcher uses. Default "naive" keeps the pure-PyTorch path the historical behavior; "cpu"/"gpu" route through dispatch_sgd_minexplstm_step.
        self.backend: Literal["naive", "cpu", "gpu"] = backend
        # QAT on the 4-bit main-model layout (see __init__ docstring). The step quantizes grads to INT8 and refreshes the FourBitLinear 4-bit mirrors.
        self.qat_4bit = qat_4bit

    def _all_params(self) -> list[Tensor]:
        """Every parameter in every group, in traversal order."""
        out: list[Tensor] = []
        for group in self.param_groups:
            out.extend(group["params"])
        return out

    def predict_lr(self, grad_rms: Tensor) -> Tensor:
        """The data-dependent LR for a shape group, in (0, 1).

        ``grad_rms`` is the layer-wise RMS of the group's
        gradients (the same scale-invariant input the cell and
        the beta predictor consume). The predictor maps it to a
        logit and a sigmoid parametrizes the LR to (0, 1). The
        result is differentiable in the predictor parameters, so
        the meta-optimizer's closure rebuild routes the loss
        gradient back to the LR predictor.

        ``grad_rms`` is a scalar (0-dim) tensor; it is reshaped
        to a 1-element batch so ``nn.Linear`` accepts it, and
        the output is squeezed back to a scalar.
        """
        logit = self.lr_predictor(grad_rms.reshape(1))
        return torch.sigmoid(logit).squeeze(0)

    @staticmethod
    def _rms(t: Tensor, eps: float = 1e-8) -> Tensor:
        """Layer-wise RMS of ``t``: sqrt(mean(t^2) + eps), scalar."""
        return torch.sqrt(torch.mean(t.float().square()) + eps)

    @staticmethod
    def _dequantize(int8: Tensor, scale: Tensor) -> Tensor:
        """Dequantize ``int8`` to FP32: ``int8 * scale``."""
        return int8.to(torch.float32) * scale

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Apply one SGD + FiLM-modulated step to every parameter.

        Tensors are grouped by shape to avoid the per-tensor Python
        dispatch overhead: same-shape tensors are concatenated into a
        single batched forward.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            shape_groups: dict[tuple[int, ...], list] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                key = tuple(p.shape)
                if key not in shape_groups:
                    shape_groups[key] = []
                shape_groups[key].append(p)

            for shape, params in shape_groups.items():
                self._step_shape_group(params, shape, group)

        # QAT on the 4-bit main-model layout: after every parameter has been stepped, refresh each FourBitLinear's 4-bit storage mirror (packed nibbles + scale) from the fp32 surrogate, so the persistent weight form stays at 4 bits between steps. Parameters that are not FourBitLinear weights (codebook deltas, the LR predictor, biases) are left untouched.
        if self.qat_4bit:
            for group in self.param_groups:
                for p in group["params"]:
                    module = getattr(p, "_fourbit_linear", None)
                    if module is not None:
                        module.quantize_weight_()

        return loss

    def _step_shape_group(
        self,
        params: list[Tensor],
        shape: tuple[int, ...],
        group_cfg: dict,
    ) -> None:
        """Batched optimizer step over one group of same-shape params."""
        wd = group_cfg["weight_decay"]
        lam = group_cfg["lam"]
        clamp_val = group_cfg["clamp_val"]
        grad_batch = torch.stack([p.grad for p in params if p.grad is not None])
        # Data-dependent learning rate (PRD: the LR is a function of the current step's data). The predictor consumes the group's RMS-normalized gradient magnitude -- the same scale-invariant signal the cell and the beta predictor read, computed after the normalization pre-filter -- and a sigmoid parametrizes the LR to (0, 1). The result is a differentiable tensor, so the meta-rebuild routes the loss gradient back to the LR predictor.
        lr = self.predict_lr(self._rms(grad_batch))
        state_list: list[dict] = []

        fast_list: list[Tensor] = []
        slow_list: list[Tensor] = []
        h_list: list[Tensor] = []
        for p in params:
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["ema_fast_int8"] = torch.zeros_like(p, dtype=torch.int8)
                state["ema_slow_int8"] = torch.zeros_like(p, dtype=torch.int8)
                state["h_int8"] = torch.zeros_like(p, dtype=torch.int8)
                state["scale_fast"] = torch.ones(())
                state["scale_slow"] = torch.ones(())
                state["scale_h"] = torch.ones(())
            state["step"] += 1
            state_list.append(state)
            fast_list.append(
                self._dequantize(state["ema_fast_int8"], state["scale_fast"])
            )
            slow_list.append(
                self._dequantize(state["ema_slow_int8"], state["scale_slow"])
            )
            h_list.append(self._dequantize(state["h_int8"], state["scale_h"]))

        ema_fast = torch.stack(fast_list)
        ema_slow = torch.stack(slow_list)
        h_fp32 = torch.stack(h_list)

        # QAT (PRD 8-bit-gradient allowance for the SGD path): each gradient is quantized to INT8 (symmetric max-abs, one scale per tensor) and the INT8 buffer + scale live in the state, so the grad footprint is 1 byte/param instead of 4. The cell and the FiLM update still consume the fp32 gradient; the INT8 buffer is the stored 8-bit-gradient form.
        if self.qat_4bit:
            for p in params:
                if p.grad is None:
                    continue
                gi, gs = quantize_batch(p.grad.reshape(1, -1))
                self.state[p]["grad_int8"] = gi.reshape(p.grad.shape)
                self.state[p]["scale_grad"] = gs

        if self.backend == "naive":
            # Pure-PyTorch path (the historical behavior and the naive reference): one batched cell forward, one in-place update, INT8 requantization.
            gamma, delta, next_h, ema_fast_new, ema_slow_new = batched_cell_forward(
                self.cell, grad_batch, ema_fast, ema_slow, h_fp32, lam
            )
            B = gamma.shape[0]
            N = grad_batch[0].numel()
            # FiLM application (PRD 3.6): the shared formula clamps gamma to [-0.9, 2.0] (so (1 + gamma) stays in [0.1, 3.0] and never flips the SGD direction) and delta to [-clamp_val, clamp_val]. It runs on the flat [B, N] layout the cell forward returns, matching batched_cell_forward's internal flatten.
            weight_flat = torch.stack([p.data.reshape(N) for p in params])
            grad_flat = grad_batch.reshape(B, N)
            update = film_sgd_update(
                gamma,
                delta,
                grad_flat,
                weight_flat,
                lr,
                wd,
                clamp_val,
            )
            ema_fast_r = ema_fast_new.reshape(B, *shape)
            ema_slow_r = ema_slow_new.reshape(B, *shape)
            next_h_r = next_h.reshape(B, *shape)
            update_r = update.reshape(B, *shape)

            for i, p in enumerate(params):
                p.data.add_(update_r[i])

            for i, s in enumerate(state_list):
                ema_fast_int8, scale_fast = quantize_batch(ema_fast_r[i])
                ema_slow_int8, scale_slow = quantize_batch(ema_slow_r[i])
                h_int8, scale_h = quantize_batch(next_h_r[i])
                s["ema_fast_int8"] = ema_fast_int8
                s["scale_fast"] = scale_fast
                s["ema_slow_int8"] = ema_slow_int8
                s["scale_slow"] = scale_slow
                s["h_int8"] = h_int8
                s["scale_h"] = scale_h
            return

        # Fused-kernel path: one dispatch_sgd_minexplstm_step
        # per parameter, matching the MeZO optimizer's
        # per-parameter fused step. The fused kernel runs the
        # identical pipeline the naive path runs (dequantize
        # -> RMS -> beta prediction -> EMA blending -> cell
        # forward -> FiLM update -> requantize), so the two
        # backends agree by construction (pinned by the
        # parity tests). The states were initialized in the
        # dequantize loop above, so each entry is populated.
        from nanochat.ops.sgd_minexplstm import dispatch_sgd_minexplstm_step

        cell = self.cell
        phi_w = cell.phi_proj.weight.reshape(-1)
        phi_b = cell.phi_proj.bias
        iota_w = cell.iota_proj.weight.reshape(-1)
        iota_b = cell.iota_proj.bias
        h_w = cell.h_proj.weight.reshape(-1)
        h_b = cell.h_proj.bias
        out_w = cell.out_proj.weight.reshape(-1)
        out_b = cell.out_proj.bias
        film_w = cell.film_proj.weight.reshape(-1)
        film_b = cell.film_proj.bias
        w_beta = cell.w_beta
        b_beta = cell.b_beta

        for i, p in enumerate(params):
            grad = p.grad
            if grad is None:
                # No gradient for this parameter (e.g.
                # a frozen submodule): skip the fused
                # step, matching the naive path which
                # only steps parameters with gradients.
                continue
            fused_state: dict = state_list[i]
            dispatch_sgd_minexplstm_step(
                grad.reshape(-1),
                p.data.reshape(-1),
                fused_state["ema_fast_int8"].reshape(-1),
                fused_state["ema_slow_int8"].reshape(-1),
                fused_state["h_int8"].reshape(-1),
                fused_state["scale_fast"],
                fused_state["scale_slow"],
                fused_state["scale_h"],
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
                backend=self.backend,
            )

    def meta_step(
        self,
        model_or_loss,
        closure=None,
        meta_lr: float | None = None,
    ) -> Tensor:
        """Joint online co-training step for the cell.

        Two spellings. ``meta_step(meta_loss)`` (PRD section 5.2)
        backprops a loss tensor that already carries a gradient path
        to the cell and steps the AdamW meta-optimizer.
        ``meta_step(model, closure)`` rebuilds the loss from the
        *updated* parameters via ``closure(params: dict) -> loss``
        (canonically ``torch.func.functional_call``), so the graph
        runs cell -> (gamma, delta) -> theta_new -> loss -- this is
        the form the training loop uses, because the primary
        ``step()`` runs under ``no_grad`` and severs the cell graph.
        """
        if meta_lr is not None:
            for group in self.meta_optimizer.param_groups:
                group["lr"] = meta_lr
        if isinstance(model_or_loss, Tensor):
            return self._meta_step_tensor(model_or_loss)
        return self._meta_step_closure(model_or_loss, closure)

    def _meta_step_tensor(self, meta_loss: Tensor) -> Tensor:
        """AdamW meta-step on a loss already wired to the cell."""
        if not isinstance(meta_loss, Tensor) or not meta_loss.requires_grad:
            raise TypeError(
                "meta_step(loss) needs a differentiable loss tensor built "
                "through the cell; the cell has no gradient path otherwise"
            )
        self.meta_optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss

    def _meta_step_closure(self, model: nn.Module, closure) -> Tensor:
        """Co-training step rebuilding the loss from updated params.

        ``closure`` receives the model's parameters with every
        optimizer-owned parameter replaced by its *updated* value and
        evaluates the loss on them.
        """
        cell = self.cell

        def rebuild(p: Tensor, state: dict, group: dict) -> Tensor | None:
            if p.grad is None:
                return None
            grad = p.grad.data
            ema_fast = self._dequantize(state["ema_fast_int8"], state["scale_fast"])
            ema_slow = self._dequantize(state["ema_slow_int8"], state["scale_slow"])
            h_fp32 = self._dequantize(state["h_int8"], state["scale_h"])
            # Same math step() runs, so the rebuilt theta_new matches.
            gamma, delta, _, _, _ = differentiable_cell_step(
                cell,
                grad,
                ema_fast,
                ema_slow,
                h_fp32.unsqueeze(-1),
                group["lam"],
            )
            # Same formula step() applies to the base weights. The LR is the data-dependent learnable LR: computed from the rebuilt gradient's RMS-normalized magnitude (the same signal step() consumes), so the LR predictor sits on the gradient path and the meta-optimizer co-trains it alongside the cell.
            lr = self.predict_lr(self._rms(grad.unsqueeze(0)))
            return p.data + film_sgd_update(
                gamma,
                delta,
                grad,
                p.data,
                lr,
                group["weight_decay"],
                group["clamp_val"],
            )

        return meta_step_closure(self, self.meta_optimizer, model, closure, rebuild)


__all__ = [
    "MinExpLSTMCell",
    "SGDMinExpLSTMCellFiLM4D",
    "SGD_MinExpLstm_Optimizer",
]

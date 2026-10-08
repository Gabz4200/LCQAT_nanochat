"""MeZO-MinExpLSTM zeroth-order learned optimizer (PRD v2.0).

MeZO (Malladi et al. 2023, arXiv:2305.17333) estimates the gradient
with two forward passes and a Gaussian perturbation vector replayed
from a PRNG seed, so the memory footprint is that of inference alone.
This module extends the estimator with the shared, coordinate-wise
``SGDMinExpLSTMCellFiLM4D`` from the SGD stack: the SPSA directional
estimate is smoothed by a K-tap FIR filter over the seed-replayed
perturbations, passed through a division-free log-domain minExpLSTM
recurrence, and projected to a FiLM (gamma, beta) pair that modulates
the MeZO update (PRD sections 2.2-2.5).

The persistent optimizer state is a single INT8 hidden state per
parameter (1 byte/param) with a per-tensor symmetric max-abs scale --
the same quantization contract the SGD-MinExpLSTM optimizer uses
(nanochat/training/batched_step.py: ``quantize_batch``).

Design deviations from the PRD text, each for a stated reason:

* The PRD writes the FiLM output head as a separate 1->4 ``film_head``
  and reads ``gamma = w1*h + b1`` / ``beta = w2*h + b2``. The SGD
  stack already ships a 4D-latent FiLM head (``out_proj`` -> ReLU ->
  ``film_proj``); this module reuses that head so both optimizers
  share one cell implementation and one kernel pipeline. The math is
  equivalent up to the ReLU-bottleneck parameterization, which is what
  keeps the fused C++/Taichi kernels a single shared code path.
* The PRD keeps the main model in 4-bit storage. This repo's training
  stack keeps FP32/bfloat16 master weights (LC-QAT quantizes inside
  ``Linear`` modules with a straight-through estimator). The optimizer
  therefore steps FP32 weights directly; the INT8 hidden state and
  the seed-only history (K scalars, not K vectors) preserve the
  memory and bus-traffic properties the PRD targets.
* The clip bound is ``a_t * delta_q`` with ``a_t`` LR-scheduled, as
  specified; ``delta_q`` defaults to the 4-bit grid spacing 1/16.
* Perturbations are replayed per parameter from seeds derived from the
  step seed and the parameter's canonical index (splitmix64 finalizer).
  MeZO replays one global generator across parameters; deriving a
  per-parameter seed gives the same partition of the perturbation
  space while keeping every kernel launch self-contained and the
  replay independent of parameter iteration order.

Integration contract: the optimizer steps any ``nn.Module`` the way
``torch.optim.SGD`` does -- the same engines, LC-QAT codebooks,
SparseProp masked layers and DARD ``lm_head`` the SGD stack trains.
The gradient is never materialized: ``step`` consumes the scalar SPSA
estimate ``(loss_pos - loss_neg) / (2 * eps)`` staged by
``stage_mezo_estimate`` and the perturbation seed, and reconstructs
the perturbation vectors in place via seed replay.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from nanochat.ops.mezo_minexplstm import dispatch_mezo_minexplstm_step
from nanochat.training.batched_step import (
    differentiable_mezo_cell_step,
    film_mezo_update,
    meta_step_closure,
)
from nanochat.training.cell import SGDMinExpLSTMCellFiLM4D


def _meta_step_loss_tensor(
    meta_optimizer: torch.optim.Optimizer, meta_loss: Tensor
) -> Tensor:
    """AdamW meta-step on a loss already wired to the cell.

    Shared verbatim by both MinExpLSTM optimizers'
    ``meta_step(loss)`` spellings: validate the tensor is
    differentiable, backprop, step the cell's AdamW. One
    implementation means the two optimizers' meta-steps
    cannot drift on this spelling.
    """
    if not isinstance(meta_loss, Tensor) or not meta_loss.requires_grad:
        raise TypeError(
            "meta_step(loss) needs a differentiable loss tensor built "
            "through the cell; the cell has no gradient path otherwise"
        )
    meta_optimizer.zero_grad(set_to_none=True)
    meta_loss.backward()
    meta_optimizer.step()
    return meta_loss


#: FIR history length default (PRD 2.2: K-tap Conv1d pre-filter).
DEFAULT_K = 4

_EPS = 1e-8


def _derive_seed(global_seed: int, index: int) -> int:
    """Derive a per-parameter seed from the step seed and canonical index.

    splitmix64 finalizer kept in uint32 range (``torch.Generator``
    accepts the full int64, but staying in uint32 keeps the derived
    seeds of different parameters well-separated and reproducible).
    """
    x = (global_seed + 0x9E3779B9 * (index + 1)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return int(x)


class MeZoMinExpLSTMCellFiLM4D(SGDMinExpLSTMCellFiLM4D):
    """MeZO variant of the shared MinExpLSTM cell.

    Identical recurrence to the SGD cell (log-domain gates, 4D latent
    FiLM head) plus the K-scalar FIR kernel ``conv_w`` of the PRD's
    Conv1d pre-filter. The FIR weights live on the cell so the
    meta-optimizer (AdamW over ``cell.parameters()``) trains them
    together with the gates; the c-scalars ``c_j = conv_w[j] * mu``
    are computed host-side in O(K) and the FIR signal itself is
    rebuilt by seed replay, never stored (PRD 2.2).
    """

    def __init__(
        self,
        K: int = DEFAULT_K,
        input_size: int = 3,
        hidden_size: int = 1,
        latent_dim: int = 4,
    ) -> None:
        super().__init__(input_size, hidden_size, latent_dim)
        if K < 1:
            raise ValueError(f"Invalid K: {K}")
        self.K = K
        # PRD 2.2: the K-tap Conv1d kernel, initialized to a uniform
        # average so the FIR signal starts as the mean of the last K
        # perturbations.
        self.conv_w = nn.Parameter(torch.ones(K) / float(K))


class MeZo_MinExpLstm_Optimizer(torch.optim.Optimizer):
    """Zeroth-order learned optimizer: MeZO + MinExpLSTM + FiLM.

    The step consumes the scalar SPSA estimate and the perturbation
    seed instead of a materialized gradient vector (PRD 2.1). The
    FIR pre-filter (PRD 2.2) blends the last ``K`` seed-replayed
    perturbations with the learned ``conv_w`` kernels and the scalar
    ``mu`` history; the result feeds the shared MinExpLSTM cell,
    whose FiLM pair modulates the MeZO update:

        update = clamp(gamma * mu * z + beta, -clip_bound, clip_bound)
        theta <- theta * (1 - lr * wd) - lr * update

    Persistent state: one INT8 hidden state per parameter (1 byte/param)
    plus the K-entry seed/mu history (scalars, O(1) memory). The
    perturbation vectors are never stored -- they are regenerated from
    the seeds, which is MeZO's memory-efficiency mechanism.

    The cell is the SGD stack's cell, so the fused C++ and Taichi
    kernels are shared between the two optimizers (the MeZO step is
    the SGD step minus the dual-EMA blending, with the FIR signal and
    the MeZO vector as cell inputs).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        lam: float = 0.10,
        clamp_val: float = 1.0,
        meta_lr: float = 1e-3,
        # MeZO-specific hyperparameters (PRD 2.1/2.2/2.5).
        epsilon: float = 1e-3,
        K: int = DEFAULT_K,
        delta_q: float = 0.0625,
        a_base: float = 1.5,
        a_min: float = 0.1,
        lr_max: float | None = None,
        backend: Literal["naive", "cpu", "gpu"] = "cpu",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 < lam <= 0.4:
            raise ValueError(f"Invalid lam: {lam}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if delta_q <= 0.0:
            raise ValueError(f"Invalid delta_q: {delta_q}")
        if not 0.0 < a_min <= a_base:
            raise ValueError(f"Invalid a_min/a_base: {a_min}/{a_base}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            lam=lam,
            clamp_val=clamp_val,
        )
        super().__init__(params, defaults)
        self.epsilon = epsilon
        self.K = K
        self.delta_q = delta_q
        self.a_base = a_base
        self.a_min = a_min
        # lr_max anchors the clip-bound schedule; default to the init LR.
        self.lr_max = lr if lr_max is None else lr_max
        # The MeZO cell: the shared MinExpLSTM cell plus the FIR kernel.
        self.cell = MeZoMinExpLSTMCellFiLM4D(K=K)
        # The optimizer's optimizer (PRD principle 5): AdamW over the
        # cell's own parameters is < 1KB of state. One instance is
        # reused across meta-steps so its moments persist.
        self.meta_optimizer = torch.optim.AdamW(self.cell.parameters(), lr=meta_lr)
        # Rolling seed / scalar-estimate history (PRD 2.2). Seeds and
        # mu are scalars, so the history is O(K) memory regardless of
        # the parameter count -- MeZO's whole point.
        self.seed_history: list[int] = []
        self.mu_history: list[float] = []
        # The staged (seed, mu) pair between the dual forward passes
        # and ``step``. Staged on the optimizer, not per param group:
        # the SPSA estimate is a global scalar (PRD 2.1).
        self._staged: tuple[int, float] | None = None
        # Backend selection for the fused step (the same
        # "naive"/"cpu"/"gpu" contract the SGD-MinExpLSTM
        # dispatcher uses).
        self.backend: Literal["naive", "cpu", "gpu"] = backend

    # MeZO probe (PRD 2.1)

    def _replay(self, seed: int, shape, device: torch.device) -> Tensor:
        """Regenerate a perturbation vector from its seed.

        Deterministic Gaussian replay -- the same seed always yields the
        same vector, which is what lets the FIR filter reconstruct
        historical perturbations without storing them (PRD 2.2). The
        generator is always CPU-side so the replay is bit-identical
        across the naive/CPU/GPU backends (CUDA and CPU generators
        produce different sequences for the same seed).
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(
            torch.Size(shape), generator=gen, device="cpu", dtype=torch.float32
        ).to(device)

    @torch.no_grad()
    def perturb(self, seed: int, scaling: float = 1.0) -> None:
        """Apply the seed-replayed perturbation in place (MeZO Algorithm 1).

        ``scaling=+1`` adds ``eps * z``; ``scaling=-2`` moves from the
        positive point to the negative point; ``scaling=+1`` again
        restores the baseline. Parameters are visited in canonical
        (group, index) order and each gets the perturbation derived
        from the step seed and its index, so the replay is independent
        of how many parameters share a shape.
        """
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                z = self._replay(_derive_seed(seed, idx), p.shape, p.device)
                p.data.add_(z, alpha=scaling * self.epsilon)
                idx += 1

    def stage_mezo_estimate(
        self, seed: int, loss_pos: Tensor, loss_neg: Tensor
    ) -> float:
        """Stage the scalar SPSA estimate for the next ``step``.

        Call between the two forward passes: ``perturb(seed, +1)`` ->
        ``loss_pos``, ``perturb(seed, -2)`` -> ``loss_neg``. Returns
        the scalar estimate ``(loss_pos - loss_neg) / (2 * epsilon)``
        so the loop can log it without a second reduction.
        """
        if not isinstance(loss_pos, Tensor) or not isinstance(loss_neg, Tensor):
            raise TypeError("loss_pos/loss_neg must be scalar tensors")
        mu = (loss_pos.detach() - loss_neg.detach()) / (2.0 * self.epsilon)
        self._staged = (seed, float(mu.item()))
        return float(mu.item())

    # The MeZO step (PRD 2.2-2.5)

    def _clip_bound(self, lr: float) -> float:
        """Coordinate-wise clip bound: ``a_t * delta_q`` (PRD 2.5)."""
        a_t = max(self.a_min, self.a_base * (lr / max(self.lr_max, 1e-12)))
        return a_t * self.delta_q

    @staticmethod
    def _quantize(fp32: Tensor) -> tuple[Tensor, Tensor]:
        """Symmetric max-abs INT8 quantization (PRD 4.1)."""
        scale = fp32.detach().abs().max() / 127.0 + _EPS
        int8 = torch.clamp(torch.round(fp32 / scale), -128, 127).to(torch.int8)
        return int8, scale

    @staticmethod
    def _rms(t: Tensor) -> Tensor:
        """Layer-wise RMS of ``t``: ``sqrt(mean(t^2) + eps)``, scalar."""
        return torch.sqrt(t.float().square().mean() + _EPS)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Apply one MeZO + FiLM-modulated step to every parameter.

        Expects ``stage_mezo_estimate`` to have been called between the
        two forward passes. Same-shape parameters are grouped so the
        cell runs once per shape group, matching the SGD optimizer's
        batching (nanochat/training/minexp_lstm.py).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._staged is None:
            raise RuntimeError(
                "MeZo_MinExpLstm_Optimizer.step() needs a staged MeZO "
                "estimate: call stage_mezo_estimate(seed, loss_pos, "
                "loss_neg) between the two forward passes"
            )
        seed, mu = self._staged
        self._staged = None
        self.seed_history.append(seed)
        self.mu_history.append(mu)
        if len(self.seed_history) > self.K:
            self.seed_history.pop(0)
            self.mu_history.pop(0)

        # Canonical parameter indices: both ``perturb`` and this step
        # derive per-parameter seeds from (seed, index), so the two
        # traversals must agree on the ordering.
        idx_of: dict[int, int] = {}
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                idx_of[id(p)] = idx
                idx += 1

        for group in self.param_groups:
            shape_groups: dict[tuple[int, ...], list] = {}
            for p in group["params"]:
                shape_groups.setdefault(tuple(p.shape), []).append(p)
            for shape, params in shape_groups.items():
                self._step_shape_group(params, shape, group, idx_of)

        return loss

    def _step_shape_group(
        self,
        params: list[Tensor],
        shape: tuple[int, ...],
        group_cfg: dict,
        idx_of: dict[int, int],
    ) -> None:
        """Batched MeZO + MinExpLSTM step over one shape group.

        Mirrors ``SGD_MinExpLstm_Optimizer._step_shape_group``
        minus the dual-EMA blending: the FIR signal and the MeZO
        vector replace ``g_norm``/``ema_norm`` as cell inputs,
        and the FiLM pair modulates the MeZO directional update
        instead of the SGD step.

        The FIR/MeZO vectors are replayed host-side from the
        seed history (PRNG seed replay, PRD 2.2), then each
        parameter's fused step runs through
        ``dispatch_mezo_minexplstm_step`` so the naive, CPU
        and GPU backends execute the identical math. The
        naive reference is ``_mezo_minexplstm_step_naive``
        (nanochat/ops/mezo_minexplstm.py), which both fused
        kernels are tested against.
        """
        lr = group_cfg["lr"]
        wd = group_cfg["weight_decay"]
        clamp_val = group_cfg["clamp_val"]
        clip_bound = self._clip_bound(lr)
        device = params[0].device

        # Cell parameters as flat tensors for the fused op.
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

        # FIR pre-filter (PRD 2.2) and MeZO vector, per
        # parameter: replay the seed history and weight the
        # perturbations with the learned conv_w kernels and
        # the scalar mu history. K_curr < K during warmup --
        # the oldest seeds are simply absent, matching the
        # PRD's rolling-window semantics.
        K_curr = len(self.seed_history)
        conv_w = cell.conv_w
        for p in params:
            p_idx = idx_of[id(p)]
            fir = torch.zeros(shape, dtype=torch.float32, device=device)
            for j in range(K_curr):
                c_j = conv_w[j] * self.mu_history[-(j + 1)]
                z_j = self._replay(
                    _derive_seed(self.seed_history[-(j + 1)], p_idx),
                    shape,
                    device,
                )
                fir = fir.add(z_j, alpha=float(c_j))
            z_curr = self._replay(
                _derive_seed(self.seed_history[-1], p_idx), shape, device
            )
            mezo = z_curr * self.mu_history[-1]

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["h_int8"] = torch.zeros_like(p, dtype=torch.int8)
                state["scale_h"] = torch.ones(())
            state["step"] += 1

            # Fused step: naive, CPU or GPU backend. The op
            # mutates p, h_int8 and scale_h in place.
            dispatch_mezo_minexplstm_step(
                fir.reshape(-1),
                mezo.reshape(-1),
                p.reshape(-1),
                state["h_int8"].reshape(-1),
                state["scale_h"],
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
                backend=self.backend,
            )

    # Online meta-optimization (REINFORCE feedback, PRD section 5)

    def meta_step(
        self,
        model_or_loss,
        closure=None,
        meta_lr: float | None = None,
    ) -> Tensor:
        """Online co-training step for the MeZO cell.

        Two spellings, matching the SGD optimizer. ``meta_step(meta_loss)``
        backprops a loss tensor already wired to the cell.
        ``meta_step(model, closure)`` rebuilds the loss from the
        *updated* parameters via ``closure(params: dict) -> loss``
        (canonically ``torch.func.functional_call``), so the graph runs
        cell -> (gamma, beta) -> theta_new -> loss -- the form the
        training loop uses, because the primary ``step()`` runs under
        ``no_grad`` with in-place ``.data`` updates and a
        non-differentiable round-to-INT8, which severs the cell graph.

        If ``self.cell`` was swapped after construction (tests and the
        training loop often inject a configured cell), the
        meta-optimizer is rebuilt over the new cell's parameters so
        its AdamW moments track the live cell.
        """
        if meta_lr is not None:
            for group in self.meta_optimizer.param_groups:
                group["lr"] = meta_lr
        self._sync_meta_optimizer()
        if isinstance(model_or_loss, Tensor):
            return self._meta_step_tensor(model_or_loss)
        return self._meta_step_closure(model_or_loss, closure)

    def _sync_meta_optimizer(self) -> None:
        """Rebuild the meta-optimizer when ``self.cell`` was replaced.

        The meta-optimizer's parameter list is bound at construction;
        swapping ``self.cell`` afterwards would leave its moments on
        the dead cell. Rebuilding keeps the AdamW state on the live
        cell's parameters.
        """
        live = set(id(q) for q in self.cell.parameters())
        bound = {
            id(q) for group in self.meta_optimizer.param_groups for q in group["params"]
        }
        if live != bound:
            self.meta_optimizer = torch.optim.AdamW(
                self.cell.parameters(),
                lr=self.meta_optimizer.param_groups[0]["lr"],
            )

    def _meta_step_tensor(self, meta_loss: Tensor) -> Tensor:
        """AdamW meta-step on a loss already wired to the cell."""
        return _meta_step_loss_tensor(self.meta_optimizer, meta_loss)

    def _meta_step_closure(self, model: nn.Module, closure) -> Tensor:
        """Co-training step rebuilding the loss from updated params.

        ``closure`` receives the model's parameters with every
        optimizer-owned parameter replaced by its *updated* value and
        evaluates the loss on them. The rebuilt update mirrors
        ``_step_shape_group`` differentiably: the FIR coefficients
        ``c_j`` stay tensors so the AdamW gradient reaches
        ``conv_w``, and the replayed perturbations are constants.
        """
        # Canonical index of every optimizer-owned parameter,
        # for the per-parameter seed derivation.
        idx_of: dict[int, int] = {}
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                idx_of[id(p)] = idx
                idx += 1
        opt = self

        def rebuild(p: Tensor, state: dict, group: dict) -> Tensor | None:
            shape = tuple(p.shape)
            device = p.device
            p_idx = idx_of[id(p)]
            # Rebuild the FIR signal and MeZO vector
            # differentiably through the same replay arithmetic
            # the step uses.
            K_curr = len(opt.seed_history)
            conv_w = opt.cell.conv_w
            m_fir = torch.zeros(shape, dtype=torch.float32, device=device)
            for j in range(K_curr):
                c_j = conv_w[j] * opt.mu_history[-(j + 1)]
                z_j = opt._replay(
                    _derive_seed(opt.seed_history[-(j + 1)], p_idx),
                    shape,
                    device,
                )
                # c_j is a 0-dim tensor (conv_w[j] is a Parameter),
                # so the FIR accumulates via tensor multiply-add.
                m_fir = m_fir + z_j * c_j
            z_curr = opt._replay(
                _derive_seed(opt.seed_history[-1], p_idx), shape, device
            )
            raw_mezo = opt.mu_history[-1] * z_curr
            h_fp = state["h_int8"].to(torch.float32) * state["scale_h"]

            # Same math step() runs, so the rebuilt theta_new matches.
            gamma, beta, _ = differentiable_mezo_cell_step(
                opt.cell, m_fir, raw_mezo, h_fp, _EPS
            )
            clipped = film_mezo_update(
                gamma,
                beta,
                raw_mezo.reshape(-1),
                group["clamp_val"],
                opt._clip_bound(group["lr"]),
            )
            return p.data.mul(1.0 - group["lr"] * group["weight_decay"]).sub(
                group["lr"] * clipped.reshape(shape)
            )

        return meta_step_closure(self, self.meta_optimizer, model, closure, rebuild)


__all__ = ["MeZoMinExpLSTMCellFiLM4D", "MeZo_MinExpLstm_Optimizer"]

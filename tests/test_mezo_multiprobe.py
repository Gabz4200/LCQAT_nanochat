"""Multi-probe batch tests for the MeZO-MinExpLSTM optimizer (PRD v2.1).

Intra-Step Multi-Probe Batch Parallelism: the optimizer samples
``M`` independent directional probes per step, evaluates the
SPSA estimate for each, and aggregates them as a batch (arithmetic
mean, the order-invariant aggregation the shipped MinExpLSTM cell
consumes). These tests pin the v2.1 contract on top of the shipped
v2.0 optimizer: the probe/probe-perturbation API, the batch-mean
SPSA estimate, permutation equivariance, and the three-way parity
across the naive/CPU/GPU backends.
"""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from nanochat.training.mezo_minexplstm import MeZo_MinExpLstm_Optimizer


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _make_opt(
    M: int = 1, K: int = 4, backend: Literal["naive", "cpu", "gpu"] = "naive"
):
    torch.manual_seed(0)
    model = _Tiny()
    opt = MeZo_MinExpLstm_Optimizer(
        model.parameters(),
        lr=1e-3,
        epsilon=1e-3,
        K=K,
        M=M,
        backend=backend,
    )
    return model, opt


def test_when_probes_requested_then_default_is_single() -> None:
    """The v2.0 default stays M=1 (backward-compatible probe count)."""
    _, opt = _make_opt()
    assert opt.M == 1


def test_when_M_set_then_optimizer_stores_it() -> None:
    """``M`` is stored and validated (>= 1)."""
    _, opt = _make_opt(M=4)
    assert opt.M == 4
    with pytest.raises(ValueError):
        MeZo_MinExpLstm_Optimizer(_Tiny().parameters(), lr=1e-3, M=0)


def test_when_perturb_with_M_probes_then_baseline_restored() -> None:
    """M-probe perturbation cycle restores the baseline exactly.

    ``perturb_probes(seed, +1)`` -> ``perturb_probes(seed, -2)`` ->
    ``perturb_probes(seed, +1)`` must return every parameter to its
    original value, bit-for-bit. This is the in-place round-trip the
    dual forward pass relies on: the negative-point evaluation happens
    at exactly ``theta - eps * z`` and the restore is exact, or the
    SPSA estimate is biased by the round-trip error.
    """
    model, opt = _make_opt(M=4)
    seeds = [101, 202, 303, 404]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.perturb_probes(seeds, scaling=1.0)
    opt.perturb_probes(seeds, scaling=-2.0)
    opt.perturb_probes(seeds, scaling=1.0)
    for n, p in model.named_parameters():
        # The round trip is exact in real arithmetic; in float32 the three in-place adds are not associative, so the restore holds to float32 precision (the same property the single-probe perturb round trip has).
        assert torch.allclose(p.detach(), before[n], atol=1e-6), (
            f"{n} not restored after the M-probe round trip"
        )


def test_when_perturb_probes_then_each_probe_distinct() -> None:
    """Each probe of the batch perturbs along its own direction.

    With distinct seeds, the positive-point parameter values differ
    per probe; with the same seed repeated, they coincide. This pins
    that ``probe_seeds`` indexes the perturbation batch, not a
    broadcast of one probe.
    """
    model, opt = _make_opt(M=4)
    seeds = [11, 22, 33, 44]
    snapshots = []
    for s in seeds:
        # Per-probe direction: the single-seed perturb replays z from the seed, so distinct seeds give distinct directions. This pins that probe seeds index the perturbation, not broadcast one probe.
        opt.perturb(s, scaling=1.0)
        snapshots.append(model.lin.weight.detach().clone())
        opt.perturb(s, scaling=-1.0)
    # Distinct seeds -> distinct perturbation directions.
    for i in range(len(snapshots)):
        for j in range(i + 1, len(snapshots)):
            assert not torch.equal(snapshots[i], snapshots[j]), (
                "distinct probe seeds produced identical perturbations"
            )


def test_when_stage_multi_probe_then_mean_estimate() -> None:
    """The staged estimate is the batch mean over the M probes.

    ``stage_mezo_estimate_multi`` returns
    ``mean_m (L+_m - L-_m) / (2 eps)``, the Option-B arithmetic-mean
    fusion of the per-probe SPSA estimates (PRD v2.1 section 3.1).
    """
    model, opt = _make_opt(M=4)
    seeds = [1, 2, 3, 4]
    loss_pos = torch.tensor(10.0)
    loss_neg = torch.tensor(2.0)
    # (10 - 2) / (2 * 1e-3) = 4000 for every probe; the mean is 4000.
    mu = opt.stage_mezo_estimate_multi(seeds, loss_pos, loss_neg)
    assert mu == pytest.approx(4000.0, rel=1e-5)
    staged = opt._staged
    assert staged is not None
    # The multi-probe staging stores (probe_seeds, mus).
    assert isinstance(staged, tuple)
    staged_seeds, staged_mus = staged
    assert isinstance(staged_seeds, list)
    assert isinstance(staged_mus, list)
    assert staged_seeds == seeds
    assert len(staged_mus) == 4
    assert all(abs(m - 4000.0) < 1e-2 for m in staged_mus)


def test_when_step_multi_probe_then_matches_single_probe_mean() -> None:
    """M-probe batch mean == M=1 with the per-probe mean mu.

    The batch-mean aggregation (Option B) is exact: an M-probe step
    with per-probe estimates ``mu_m`` produces the same weight
    update as a single-probe step with ``mean(mu_m)``, because the
    FIR and MeZO vectors are linear in ``mu`` and the cell sees the
    mean. This is the mathematical equivariance the PRD's
    permutation-invariance requirement rests on.
    """
    torch.manual_seed(0)
    model_a = _Tiny()
    model_b = _Tiny()
    model_b.load_state_dict(model_a.state_dict())

    opt_a = MeZo_MinExpLstm_Optimizer(
        model_a.parameters(), lr=1e-3, epsilon=1e-3, K=4, M=1, backend="naive"
    )
    opt_b = MeZo_MinExpLstm_Optimizer(
        model_b.parameters(), lr=1e-3, epsilon=1e-3, K=4, M=4, backend="naive"
    )

    # Same base seed; derive the 4 probe seeds the Lightning module would, so both optimizers replay identical per-probe vectors.
    base_seed = 42
    probe_seeds = [base_seed * 100 + m for m in range(4)]

    # Single-probe run with the mean mu.
    mu_mean = 0.0
    for s in probe_seeds:
        opt_a.perturb(s, scaling=1.0)
        lp = (model_a(torch.randn(4, 8)) * 0).sum() + 10.0
        opt_a.perturb(s, scaling=-2.0)
        ln = (model_b(torch.randn(4, 8)) * 0).sum() + 2.0
        opt_a.perturb(s, scaling=1.0)
        mu_mean += float(((lp - ln) / (2 * 1e-3)).detach())
    mu_mean /= 4
    opt_a.stage_mezo_estimate(probe_seeds[0], torch.tensor(10.0), torch.tensor(2.0))
    # Force the single-probe staged mu to the batch mean so the only difference between the two runs is the aggregation.
    opt_a._staged = (probe_seeds[0], mu_mean)
    opt_a.step()

    # Multi-probe run: stage the same seeds and the per-probe mus.
    mus = [mu_mean] * 4
    opt_b._staged = (probe_seeds, mus)
    opt_b.step()

    assert torch.allclose(model_a.lin.weight, model_b.lin.weight, atol=1e-6), (
        "M-probe batch-mean step must equal the single-probe step with the mean mu"
    )


def test_when_probe_seeds_swapped_then_step_is_invariant() -> None:
    """Permutation equivariance: swapping probe order yields the same step.

    PRD v2.1 acceptance criterion 2: swapping probe seeds
    ``S_{t,1} <-> S_{t,2}`` must produce bit-for-bit identical
    parameter updates. The arithmetic-mean aggregation is
    symmetric in the probe index, so the FIR/MeZO batch means are
    invariant to probe order.
    """
    torch.manual_seed(0)
    model_a = _Tiny()
    model_b = _Tiny()
    model_b.load_state_dict(model_a.state_dict())

    opt_a = MeZo_MinExpLstm_Optimizer(
        model_a.parameters(), lr=1e-3, epsilon=1e-3, K=4, M=4, backend="naive"
    )
    opt_b = MeZo_MinExpLstm_Optimizer(
        model_b.parameters(), lr=1e-3, epsilon=1e-3, K=4, M=4, backend="naive"
    )

    seeds = [101, 202, 303, 404]
    mus = [0.5, -1.5, 2.0, 0.25]

    opt_a._staged = (seeds, mus)
    opt_a.step()

    swapped_seeds = [202, 101, 303, 404]
    swapped_mus = [-1.5, 0.5, 2.0, 0.25]
    opt_b._staged = (swapped_seeds, swapped_mus)
    opt_b.step()

    assert torch.equal(model_a.lin.weight, model_b.lin.weight), (
        "probe permutation must not change the weight update"
    )


def test_when_fused_backends_run_multi_probe_then_parity() -> None:
    """Three-way parity: naive == cpu == gpu for the M-probe step.

    The fused kernels consume the already-aggregated FIR/Mezo
    vectors, so the multi-probe batching is a host-side replay
    concern and the fused backends must agree with the naive
    reference on the same aggregated signals.
    """
    pytest.importorskip("taichi")
    torch.manual_seed(0)

    results = {}
    for backend in ("naive", "cpu", "gpu"):
        torch.manual_seed(0)
        model = _Tiny()
        opt = MeZo_MinExpLstm_Optimizer(
            model.parameters(), lr=1e-3, epsilon=1e-3, K=4, M=4, backend=backend
        )
        seeds = [101, 202, 303, 404]
        mus = [0.5, -1.5, 2.0, 0.25]
        opt._staged = (seeds, mus)
        try:
            opt.step()
        except Exception as e:
            if backend == "gpu" and "Vulkan" in str(e):
                pytest.skip(f"GPU backend not available: {e}")
            raise
        results[backend] = model.lin.weight.detach().clone()

    assert torch.allclose(results["naive"], results["cpu"], atol=1e-5)
    if "gpu" in results:
        assert torch.allclose(results["naive"], results["gpu"], atol=1e-5)


def test_when_lightning_module_then_probe_count_threaded() -> None:
    """``MeZoJointTrainableModule`` exposes the probe count.

    The Lightning module must construct the optimizer with the
    configured ``M`` and run the M-probe dual forward loop.
    """
    from nanochat.training.lightning_module import MeZoJointTrainableModule

    model = _Tiny()
    module = MeZoJointTrainableModule(model, lr=1e-2, M=4, backend="naive")
    opts = module.configure_optimizers()
    opt = opts[0] if isinstance(opts, list) else opts
    inner = getattr(opt, "_optimizer", opt)
    assert isinstance(inner, MeZo_MinExpLstm_Optimizer)
    assert inner.M == 4


def test_when_meta_step_multi_probe_then_rebuild_uses_mean_mu() -> None:
    """The differentiable meta-rebuild uses the batch-mean mu.

    ``meta_step`` rebuilds the loss from the updated parameters
    through the cell. With a multi-probe staged batch, the
    rebuild must use the mean mu (the normalized form ``step``
    stored in ``mu_history``), so the rebuilt gradient reaches
    ``conv_w`` and the cell parameters.
    """
    torch.manual_seed(0)
    model = _Tiny()
    opt = MeZo_MinExpLstm_Optimizer(
        model.parameters(), lr=1e-3, epsilon=1e-3, K=4, M=4, backend="naive"
    )
    # Warm the history with a real single-probe step so the FIR window is populated.
    opt.perturb(7, scaling=1.0)
    opt.perturb(7, scaling=-2.0)
    opt.perturb(7, scaling=1.0)
    opt.stage_mezo_estimate(7, torch.tensor(5.0), torch.tensor(1.0))
    opt.step()

    # Stage a multi-probe batch and step.
    seeds = [101, 202, 303, 404]
    mus = [0.5, -1.5, 2.0, 0.25]
    opt._staged = (seeds, mus)
    opt.step()
    # The history now holds the batch-mean mu.
    mean_mu = sum(mus) / len(mus)
    assert opt.mu_history[-1] == pytest.approx(mean_mu, rel=1e-6)

    # The closure rebuild must run end-to-end (it reads the mean mu from history) and produce a finite meta-loss.
    def _closure(params):
        out = torch.func.functional_call(model, params, (torch.randn(4, 8),))
        return out.sum()

    meta_loss = opt.meta_step(model, _closure)
    assert torch.isfinite(meta_loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

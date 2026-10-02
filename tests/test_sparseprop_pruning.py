"""SparseProp pruning criterion, scope, and gradual schedule.

The three properties that matter and are not obvious from reading the code:

* the criterion is *magnitude*, not random -- a random mask spends the same
  budget on a large weight it spends on a negligible one;
* global scope is *not* uniform scope -- the paper measures them as different,
  and at equal average sparsity they distribute very differently;
* gradual pruning is *monotone* -- nothing is ever un-pruned, which is what
  makes a resumed ramp safe and what GMP is defined to guarantee.
"""

import pytest
import torch
import torch.nn as nn

from nanochat.models.quant.pruning import GradualPruningSchedule, collect
from nanochat.models.quant.sparseprop import (
    PRUNE_SCOPES,
    SCOPE_GLOBAL,
    SCOPE_LAYER,
    SparsePropLinear,
    apply_global_pruning,
    apply_static_sparsity_mask,
    magnitude_mask,
)


def _keys(mask: torch.Tensor) -> set[tuple[int, int]]:
    rows, cols = torch.nonzero(mask, as_tuple=True)
    return {(int(r), int(c)) for r, c in zip(rows, cols)}


def _layer(out_f: int = 8, in_f: int = 16, scale: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    linear = nn.Linear(in_f, out_f)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_f, in_f) * scale)
    return SparsePropLinear.from_linear(linear, sparsity=0.0)


class TestMagnitudeCriterion:
    def test_when_pruning_then_the_smallest_weights_are_dropped_per_row(self):
        """Per-row magnitude: each row keeps its own largest entries.

        The comparison is per row, not global -- that is what the per-row nnz
        guarantee implies, and a global comparison would fail on any matrix
        whose rows have different scales.
        """
        layer = _layer()
        original = layer.weight.detach().abs().clone()
        layer.sparsity_mask.copy_(magnitude_mask(layer.weight, 0.5))
        layer._apply_mask()
        for row in range(layer.weight.shape[0]):
            kept = original[row][layer.sparsity_mask[row]]
            dropped = original[row][~layer.sparsity_mask[row]]
            assert kept.numel() > 0 and dropped.numel() > 0
            assert kept.min() >= dropped.max(), (
                f"row {row} kept a smaller weight ({kept.min()}) than one it "
                f"dropped ({dropped.max()})"
            )

    def test_when_pruning_then_result_is_reproducible(self):
        """Same weights in, same mask out: no RNG anywhere in the criterion."""
        a, b = _layer(seed=3), _layer(seed=3)
        with torch.no_grad():
            a.weight.copy_(b.weight)
        a.sparsity_mask.copy_(magnitude_mask(a.weight, 0.6))
        b.sparsity_mask.copy_(magnitude_mask(b.weight, 0.6))
        assert torch.equal(a.sparsity_mask, b.sparsity_mask)

    @pytest.mark.parametrize("sparsity", [0.0, 0.25, 0.75, 0.99])
    def test_when_pruning_then_every_row_keeps_at_least_one_entry(self, sparsity):
        """>= 1 nnz per row is a hard requirement, not a best effort."""
        mask = magnitude_mask(torch.randn(32, 40), sparsity)
        assert bool(mask.any(dim=1).all()), f"a row was emptied at sparsity {sparsity}"

    def test_when_sparsity_is_zero_then_the_mask_is_dense_and_forward_matches(self):
        """sparsity=0.0 means 'no pruning', which must be a usable layer.

        The mask buffer is all-zeros when constructed; treating that as 'no
        pruning' yields an empty CSR structure whose forward raises.
        """
        layer = SparsePropLinear(4, 3, sparsity=0.0)
        assert bool(layer.sparsity_mask.all())
        x = torch.randn(5, 4)
        torch.testing.assert_close(
            layer(x),
            torch.nn.functional.linear(x, layer.weight, layer.bias),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_when_a_row_is_all_zero_then_it_still_reaches_the_target_sparsity(self):
        """A `>=` threshold keeps an all-zero row entirely, so the layer escapes pruning.

        nanochat zero-initializes attn.c_proj / mlp.c_proj, so every fresh model
        has all-zero rows. Pruning them by magnitude alone would silently leave
        those layers at 0% sparsity and report the model as pruned.
        """
        weights = torch.zeros(4, 10)
        mask = magnitude_mask(weights, 0.6)
        expected = round(10 * 0.4)
        assert mask.sum(dim=1).tolist() == [expected] * 4, mask.sum(dim=1).tolist()

    def test_when_mixed_with_normal_rows_then_both_hit_the_target(self):
        """The degenerate-row fix must not disturb ordinary rows."""
        weights = torch.randn(6, 20)
        weights[2] = 0.0
        weights[5] = 0.0
        mask = magnitude_mask(weights, 0.75)
        assert mask.sum(dim=1).tolist() == [5] * 6, mask.sum(dim=1).tolist()

    def test_when_pruned_then_weights_hold_exact_zero(self):
        """Structural zeros, not small values: this is what the codebook anchors on."""
        layer = _layer()
        layer.sparsity_mask.copy_(magnitude_mask(layer.weight, 0.75))
        layer._apply_mask()
        assert bool((layer.weight.detach()[~layer.sparsity_mask] == 0.0).all())


class TestPruningScope:
    def test_when_scope_is_layer_then_every_layer_gets_the_same_fraction(self):
        big = _layer(seed=1, scale=1.0)
        small = _layer(seed=2, scale=0.001)
        for module in (big, small):
            module.sparsity_mask.copy_(magnitude_mask(module.weight, 0.5))
        assert int(big.sparsity_mask.sum()) == big.weight.numel() // 2
        assert int(small.sparsity_mask.sum()) == small.weight.numel() // 2

    def test_when_scope_is_global_then_low_magnitude_layers_absorb_the_sparsity(self):
        """Global scope lets a uniformly-small layer take almost all the budget.

        This is the whole point of the flag: at equal *average* sparsity, global
        ranking keeps the heavy-tailed layer dense and prunes the small one
        hard. Uniform scope cannot express that, and SparseProp Fig. 6 measures
        the two as genuinely different.
        """
        big = _layer(seed=1, scale=1.0)
        small = _layer(seed=2, scale=0.001)
        achieved = apply_global_pruning([big, small], 0.5)
        big_s = 1 - int(big.sparsity_mask.sum()) / big.weight.numel()
        small_s = 1 - int(small.sparsity_mask.sum()) / small.weight.numel()
        assert 0.35 < achieved < 0.65, achieved
        assert big_s < small_s, (
            f"global scope did not shift the budget: big={big_s:.3f} small={small_s:.3f}"
        )

    def test_when_scope_is_global_then_every_row_stays_nonempty(self):
        """The global threshold can empty a row; the per-row guarantee repairs it."""
        tiny = _layer(out_f=4, in_f=4, seed=5, scale=1e-6)
        huge = _layer(out_f=64, in_f=64, seed=6, scale=1.0)
        apply_global_pruning([tiny, huge], 0.99)
        for module in (tiny, huge):
            assert bool(module.sparsity_mask.any(dim=1).all())

    def test_when_unknown_scope_then_apply_static_raises(self):
        layer = _layer()
        with pytest.raises(ValueError, match="scope"):
            apply_static_sparsity_mask(nn.Sequential(layer), 0.5, scope="dense")

    def test_when_scope_names_are_listed_then_both_paper_variants_are_available(self):
        assert PRUNE_SCOPES == (SCOPE_LAYER, SCOPE_GLOBAL)


class TestGradualSchedule:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_sparsity": 1.5},
            {"target_sparsity": 0.5, "start_frac": -0.1},
            {"target_sparsity": 0.5, "ramp_steps": 0},
            {"target_sparsity": 0.5, "every": -1},
            {"target_sparsity": 0.5, "scope": "uniform"},
        ],
    )
    def test_when_config_is_invalid_then_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            GradualPruningSchedule(**kwargs)

    def test_when_disabled_then_apply_is_a_no_op(self):
        layer = _layer()
        schedule = GradualPruningSchedule(target_sparsity=0.9, every=0)
        assert not schedule.enabled
        before = _keys(layer.sparsity_mask)
        assert schedule.apply(layer, 0) is None
        assert _keys(layer.sparsity_mask) == before

    def test_when_start_frac_is_nonzero_then_step_zero_prunes_partway(self):
        """A ramp's first prune is a partial prune, not the full target."""
        layer = _layer()
        schedule = GradualPruningSchedule(target_sparsity=0.75, start_frac=0.5, every=1)
        assert schedule.initial_sparsity == pytest.approx(0.375)
        achieved = schedule.apply(layer, 0)
        assert achieved is not None
        assert achieved == pytest.approx(0.375, abs=0.02), achieved

    def test_when_ramp_is_configured_then_targets_rise_to_the_target(self):
        schedule = GradualPruningSchedule(
            target_sparsity=0.9, start_frac=0.5, ramp_steps=4, every=10
        )
        targets = [schedule.target_at(s) for s in (0, 10, 20, 30, 40, 100)]
        assert targets[0] == pytest.approx(0.45)
        assert targets == sorted(targets), "ramp is not monotonic"
        assert targets[-1] == pytest.approx(0.9)
        assert all(t <= 0.9 for t in targets), "ramp overshoots the target"

    def test_when_every_is_set_then_only_multiples_prune(self):
        schedule = GradualPruningSchedule(target_sparsity=0.8, every=5)
        assert [schedule.should_prune(s) for s in range(12)] == [
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
        ]

    @pytest.mark.parametrize("scope", [SCOPE_LAYER, SCOPE_GLOBAL])
    def test_when_ramp_runs_then_no_weight_is_ever_un_pruned(self, scope):
        """GMP is defined to be monotone; a resumed ramp depends on it.

        Re-selecting each row independently does NOT give a nested mask -- equal
        magnitudes in one row swap in and out as the threshold drops -- so this
        is the property that catches a non-intersecting implementation.
        """
        torch.manual_seed(7)
        layer = _layer(out_f=12, in_f=24, seed=7)
        schedule = GradualPruningSchedule(
            target_sparsity=0.8,
            start_frac=0.0,
            ramp_steps=5,
            every=2,
            scope=scope,
        )
        for step in range(20):
            before = _keys(layer.sparsity_mask)
            achieved = schedule.apply(layer, step)
            after = _keys(layer.sparsity_mask)
            if achieved is None:
                assert after == before, f"non-event step {step} mutated the mask"
                continue
            assert after <= before, (
                f"{scope} step {step} revived {len(after - before)} pruned entries"
            )

    @pytest.mark.parametrize("scope", [SCOPE_LAYER, SCOPE_GLOBAL])
    def test_when_ramp_completes_then_target_sparsity_is_reached(self, scope):
        torch.manual_seed(11)
        layer = _layer(out_f=8, in_f=16, seed=11)
        schedule = GradualPruningSchedule(
            target_sparsity=0.75,
            start_frac=0.0,
            ramp_steps=3,
            every=1,
            scope=scope,
        )
        for step in range(5):
            schedule.apply(layer, step)
        achieved = 1 - int(layer.sparsity_mask.sum()) / layer.weight.numel()
        assert achieved >= 0.7, achieved
        assert bool(layer.sparsity_mask.any(dim=1).all())

    def test_when_dense_threshold_is_set_then_only_certain_layers_are_listed(self):
        """SparseProp Sec. 4.1: below ~80% sparsity the sparse kernel is a loss."""
        shallow = _layer(out_f=4, in_f=16, seed=13)
        with torch.no_grad():
            shallow.weight.copy_(torch.randn(4, 16))
        deep = _layer(out_f=8, in_f=32, seed=14)
        shallow.sparsity_mask.copy_(magnitude_mask(shallow.weight, 0.5))
        shallow._apply_mask()
        deep.sparsity_mask.copy_(magnitude_mask(deep.weight, 0.95))
        deep._apply_mask()
        root = nn.ModuleList([shallow, deep])
        schedule = GradualPruningSchedule(target_sparsity=0.9, dense_threshold=0.8)
        assert schedule.layers_above_threshold(root) == [deep]


class TestEngineTreeTraversal:
    """The schedule's `root` is a DiffusionBlockEngine, not an nn.Module."""

    @staticmethod
    def _sparse_engine(sparsity: float = 0.0):
        """A real engine with SparseProp injected, so the module tree is real."""
        from tests.test_dbcpu_engine import make_engine

        engine = make_engine(num_blocks=2, n_layer=4, active=False)
        engine.apply_sparseprop(sparsity=sparsity, with_lcqat=False)
        return engine

    def test_when_root_is_the_engine_then_its_layers_are_found(self):
        """Without a `modules()` view on the engine the schedule raises on call.

        The schedule walks the engine tree at setup and every training step, so
        a missing view is invisible until a run is started.
        """
        engine = self._sparse_engine()
        assert collect(engine), "no SparsePropLinear reachable from the engine"

    def test_when_schedule_runs_against_the_engine_then_it_prunes(self):
        engine = self._sparse_engine()
        # start_frac=1.0: a default schedule's target at step 0 is 0.0, so
        # step 0 prunes nothing. One-shot-at-target needs the ramp to start
        # at its end, which is what start_frac=1.0 expresses.
        schedule = GradualPruningSchedule(target_sparsity=0.5, start_frac=1.0, every=1)
        achieved = schedule.apply(engine, 0)
        assert achieved == pytest.approx(0.5, abs=0.02), achieved
        assert all(bool(m.sparsity_mask.any(dim=1).all()) for m in collect(engine))

    def test_when_schedule_runs_then_adapters_and_heads_are_pruned_too(self):
        """Not just the transformer: the engine's own layers carry the budget.

        The engine owns three subtrees. Reaching only `model` would leave the
        per-block adapters and denoise heads dense, which is invisible in the
        achieved-sparsity number unless the per-subtree counts are checked.
        """
        engine = self._sparse_engine()
        model_layers = {
            id(m) for m in engine.model.modules() if isinstance(m, SparsePropLinear)
        }
        all_layers = collect(engine)
        assert len(all_layers) > len(model_layers), (
            "engine traversal found only the transformer's sparse layers; "
            "adapters and denoise heads are being skipped"
        )
        GradualPruningSchedule(target_sparsity=0.5, start_frac=1.0, every=1).apply(
            engine, 0
        )
        for module in all_layers:
            sparsity = 1 - int(module.sparsity_mask.sum()) / module.weight.numel()
            assert sparsity == pytest.approx(0.5, abs=0.05), sparsity


class TestScheduleFromArgs:
    @pytest.mark.parametrize(
        ("argv", "attribute"),
        [
            (["--sparseprop-sparsity", "0.9"], "target_sparsity"),
            (["--sparseprop-scope", "global"], "scope"),
            (["--sparseprop-every", "17"], "every"),
            (["--sparseprop-start-frac", "0.25"], "start_frac"),
            (["--sparseprop-ramp-steps", "3"], "ramp_steps"),
            (["--sparseprop-dense-threshold", "0.7"], "dense_threshold"),
        ],
    )
    def test_when_flag_is_passed_then_it_reaches_the_schedule(self, argv, attribute):
        """The three entry points share one flag surface; a dropped flag is silent."""
        import argparse

        from nanochat.models.quant.pruning import (
            add_sparseprop_pruning_args,
            schedule_from_args,
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--sparseprop-sparsity", type=float, default=0.75)
        add_sparseprop_pruning_args(parser)
        args = parser.parse_args(argv)
        schedule = schedule_from_args(args)
        expected = args.__dict__[argv[0].lstrip("-").replace("-", "_")]
        assert getattr(schedule, attribute) == expected, (
            f"{argv[0]} did not reach schedule.{attribute}"
        )

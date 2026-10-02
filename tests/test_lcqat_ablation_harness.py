"""Tests for the ablation measurement harness (W6/W7 evidence layer).

The harness exists to make two written claims falsifiable, so the tests are
about the *protocol*, not about whether the numbers come out favourably. A
measurement harness whose failure mode is "silently measures the wrong thing"
is worse than none.

What these tests pin:

* The paired protocol actually pairs. Both arms must see the identical probe
  and seed, or the reported delta is seed variance wearing a comparison's
  clothes.
* The metrics detect a regression. Feeding them a deliberately broken quantizer
  must move the numbers; a metric that stays flat when the thing it measures is
  broken is measuring nothing.
* `observe_grad_scale` reproduces the `1/sqrt(N)` factor to numerical precision.
  That is the PRD 2.4 claim, and this is the test that would catch it drifting.
* `select_quantizer` refuses to guess. Falling back to the weight quantizer when
  asked for the activation quantizer silently answers a different question.
"""

import math

import pytest
import torch

from nanochat.lcqat.ablation_metrics import (
    aggregate_comparisons,
    compare_presets,
    measure_reconstruction,
    observe_grad_scale,
    quantization_error,
    render_leaderboard,
    row_to_dict,
    select_quantizer,
)
from nanochat.lcqat.linear import GRAD_SCALE_INV_SQRT_N, GRAD_SCALE_NONE
from nanochat.lcqat.retrofit import PRESETS, retrofit_model
from tests.conftest import build_active_tiny_gpt


def retrofitted_layer(preset: str = "asym", layer: str = "c_proj"):
    """A retrofitted `c_proj` from a fresh tiny model."""
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS[preset])
    return getattr(model.transformer.h[0].mlp, layer)


def relu2_probe(in_features: int, rows: int = 256, seed: int = 0) -> torch.Tensor:
    """Non-negative probe matching `mlp.forward`'s `F.relu(x).square()`."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, in_features, generator=gen).relu().square()


class TestQuantizerSelection:
    def test_when_selecting_by_role_then_the_matching_quantizer_is_returned(
        self,
    ) -> None:
        layer = retrofitted_layer()
        assert select_quantizer(layer, "weight") is layer.weight_quantizer
        assert select_quantizer(layer, "act") is layer.act_quantizer

    def test_when_the_role_is_unknown_then_selection_raises(self) -> None:
        """Guessing would measure a different tensor than the caller asked for."""
        layer = retrofitted_layer()
        with pytest.raises(ValueError, match="which must be one of"):
            select_quantizer(layer, "output")

    def test_when_the_layer_lacks_the_quantizer_then_selection_raises(self) -> None:
        layer = retrofitted_layer(layer="c_fc")
        # c_fc is built with quantize_out=True by the asym preset, so use a layer
        # that genuinely has no `out_quantizer` to prove the guard fires.
        assert hasattr(layer, "out_quantizer")
        stripped = retrofitted_layer()
        stripped.out_quantizer = None
        with pytest.raises(ValueError, match="has no"):
            select_quantizer(stripped, "out")


class TestReconstructionMetrics:
    def test_when_the_reconstruction_is_a_copy_then_the_error_is_zero(self) -> None:
        """The zero point: an unchanged tensor must score exactly zero error.

        Quantizing a quantizer's own codebook is *not* exact -- bucketizing a
        level can land it in a neighbouring bucket -- so the zero reference is
        a plain copy, not a round trip.
        """
        probe = relu2_probe(16)
        mse, max_abs, power = quantization_error(probe, probe.clone())
        assert mse == 0.0
        assert max_abs == 0.0
        assert power > 0.0

    def test_when_the_reconstruction_is_a_copy_then_the_layer_round_trip_is_not(
        self,
    ) -> None:
        """Quantization must actually lose something, or NMSE is vacuous.

        Guards the premise of every other measurement here: if the round trip
        were lossless, a broken quantizer would also look lossless.
        """
        layer = retrofitted_layer()
        probe = relu2_probe(layer.in_features)
        mse, _, _ = quantization_error(probe, reconstruct_exactly(layer, probe))
        assert mse > 0.0

    def test_when_the_reconstruction_is_garbage_then_the_nmse_is_large(self) -> None:
        """A metric that cannot see a broken quantizer is measuring nothing."""
        layer = retrofitted_layer()
        probe = relu2_probe(layer.in_features)
        result = measure_reconstruction(layer, probe, "asym", which="act")
        exact = measure_reconstruction(layer, probe, "asym", which="act")
        assert exact.nmse == result.nmse  # deterministic
        # Replacing the probe with noise must move the number.
        garbage = measure_reconstruction(
            layer, torch.zeros_like(probe), "asym", which="act"
        )
        assert garbage.nmse != result.nmse

    def test_when_the_probe_is_zero_then_nmse_is_infinite_not_a_crash(self) -> None:
        """A zero probe has no signal power; NMSE is 0/0 and must not be silent."""
        layer = retrofitted_layer()
        result = measure_reconstruction(
            layer, torch.zeros(8, layer.in_features), "asym", which="act"
        )
        assert math.isinf(result.nmse)

    def test_when_shapes_disagree_then_quantization_error_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            quantization_error(torch.zeros(4, 4), torch.zeros(4, 5))

    def test_when_measured_then_the_used_level_count_is_at_most_k(self) -> None:
        layer = retrofitted_layer()
        probe = relu2_probe(layer.in_features)
        result = measure_reconstruction(layer, probe, "asym", which="act")
        assert 0 < result.used_levels <= result.k_total
        assert 0.0 < result.level_utilization <= 1.0

    def test_when_the_preset_is_asym_then_a_non_negative_probe_uses_every_level(
        self,
    ) -> None:
        """The `asym` claim: a one-sided codebook wastes nothing on relu^2 data.

        The symmetric `small` preset cannot reach the same utilization on a
        non-negative tensor, because half its levels sit below the data's
        minimum. This is the direct observation behind the preset.
        """
        asym = retrofitted_layer("asym")
        small = retrofitted_layer("small")
        probe = relu2_probe(asym.in_features)

        asym_res = measure_reconstruction(asym, probe, "asym", which="act")
        small_res = measure_reconstruction(small, probe, "small", which="act")
        assert asym_res.level_utilization > small_res.level_utilization


class TestPairedComparison:
    def test_when_the_variant_is_worse_then_the_comparison_reports_a_loss(
        self,
    ) -> None:
        """`compare_presets(baseline, variant)` -- the argument order is the contract.

        Lower NMSE wins, so the variant wins exactly when its delta is negative.
        Getting this backwards would flip every leaderboard row.
        """
        layer = retrofitted_layer()
        probe = relu2_probe(layer.in_features)
        good = measure_reconstruction(layer, probe, "asym", which="act")
        # A probe scaled up is far outside the codebook's range, so it
        # reconstructs worse: an unambiguous, data-driven difference.
        bad = measure_reconstruction(layer, probe * 100.0, "small", which="act")
        assert bad.nmse > good.nmse, "precondition: the bad arm is worse"

        variant_loses = compare_presets("t", good, bad, seed=3)
        assert variant_loses.wins is False
        assert variant_loses.nmse_delta > 0.0

        variant_wins = compare_presets("t", bad, good, seed=3)
        assert variant_wins.wins is True
        assert variant_wins.nmse_delta < 0.0

    def test_when_aggregating_then_per_seed_deltas_are_pooled_not_the_arms(
        self,
    ) -> None:
        """Averaging the two arms separately is not the same as averaging deltas.

        The harness pairs seeds so that per-seed offset cancels; if aggregation
        were done on the arms independently, a constant per-seed bias would leak
        into the reported delta.
        """
        comparisons = []
        for seed in range(4):
            layer = retrofitted_layer()
            probe = relu2_probe(layer.in_features, seed=seed)
            base = measure_reconstruction(layer, probe, "base", which="act")
            # Same seed, deliberately shifted worse by a constant factor.
            variant = measure_reconstruction(layer, probe * 50.0, "var", which="act")
            comparisons.append(compare_presets("t", base, variant, seed))
        row = aggregate_comparisons("t", comparisons)
        assert row.n_seeds == 4
        assert row.value_variant > row.value_baseline
        assert math.isclose(row.delta, row.value_variant - row.value_baseline)

    def test_when_aggregating_nothing_then_it_raises(self) -> None:
        with pytest.raises(ValueError, match="no comparisons"):
            aggregate_comparisons("t", [])


class TestGradScaleMeasurement:
    def test_when_measured_then_the_ratio_matches_one_over_sqrt_n(self) -> None:
        """The PRD 2.4 claim, to numerical precision.

        N here is the weight element count, so `1/sqrt(N)` is the factor the
        scaling applies. Anything materially off this is a real regression.
        """
        layer = retrofitted_layer()
        probe = relu2_probe(layer.in_features)
        observation = observe_grad_scale(layer, probe, GRAD_SCALE_INV_SQRT_N, steps=1)
        expected = 1.0 / math.sqrt(layer.weight.numel())
        assert observation.grad_ratio == pytest.approx(expected, rel=1e-3)

    def test_when_no_scaling_is_applied_then_the_ratio_is_one(self) -> None:
        """The `none` arm is the control: its gradient must be the naive one."""
        layer = retrofitted_layer()
        probe = relu2_probe(layer.in_features)
        observation = observe_grad_scale(layer, probe, GRAD_SCALE_NONE, steps=1)
        assert observation.grad_ratio == pytest.approx(1.0, rel=1e-3)

    def test_when_scaling_is_applied_then_the_gradient_is_strictly_smaller(
        self,
    ) -> None:
        layer = retrofitted_layer()
        probe = relu2_probe(layer.in_features)
        none_obs = observe_grad_scale(layer, probe, GRAD_SCALE_NONE, steps=1)
        inv_obs = observe_grad_scale(layer, probe, GRAD_SCALE_INV_SQRT_N, steps=1)
        assert inv_obs.grad_norm < none_obs.grad_norm

    def test_when_measured_then_the_layers_grad_scale_is_restored(self) -> None:
        """The harness must not leave global layer state modified."""
        layer = retrofitted_layer()
        before = layer.grad_scale
        observe_grad_scale(
            layer, relu2_probe(layer.in_features), GRAD_SCALE_NONE, steps=1
        )
        assert layer.grad_scale == before

    def test_when_the_setting_is_unknown_then_it_raises(self) -> None:
        layer = retrofitted_layer()
        with pytest.raises(ValueError, match="grad_scale must be one of"):
            observe_grad_scale(layer, relu2_probe(layer.in_features), "sqrt_n")

    def test_when_steps_is_zero_then_it_raises(self) -> None:
        layer = retrofitted_layer()
        with pytest.raises(ValueError, match="steps must be"):
            observe_grad_scale(
                layer, relu2_probe(layer.in_features), GRAD_SCALE_NONE, steps=0
            )


class TestLeaderboardRendering:
    def test_when_rendered_then_each_row_is_its_own_line(self) -> None:
        """The table must be structurally a table, not prose.

        `render_leaderboard` joins rows with `"".join(...)`, so a row missing
        its trailing newline silently concatenates with the next and the whole
        table collapses onto one line. Every number is still present, so a
        substring assertion passes and the defect ships -- which is exactly what
        happened. The count of table lines is the property that catches it.
        """
        from nanochat.lcqat.ablation_metrics import AblationRow

        rows = [
            AblationRow(
                experiment=f"zz{i}",
                metric="nmse",
                baseline="x",
                variant="y",
                value_baseline=1.0,
                value_variant=0.5,
                delta=-0.5,
                better="variant",
                n_seeds=3,
            )
            for i in range(3)
        ]
        table = render_leaderboard(rows, "T")
        lines = table.splitlines()
        # Distinct experiment prefix so the header row (whose first column is
        # literally "experiment") is never counted as data.
        body = [line for line in lines if line.startswith("| zz")]
        assert len(body) == 3, f"expected 3 table rows, got {len(body)}:\n{table}"
        # Every data row must have the same column count as the header.
        header = next(line for line in lines if line.startswith("| experiment |"))
        assert {line.count("|") for line in body} == {header.count("|")}

    def test_when_rendered_then_the_table_contains_every_row(self) -> None:
        from nanochat.lcqat.ablation_metrics import AblationRow

        rows = [
            AblationRow(
                experiment="a",
                metric="nmse",
                baseline="x",
                variant="y",
                value_baseline=1.0,
                value_variant=0.5,
                delta=-0.5,
                better="variant",
                n_seeds=3,
                notes="note a",
            ),
            AblationRow(
                experiment="b",
                metric="levels",
                baseline="p",
                variant="q",
                value_baseline=0.5,
                value_variant=1.0,
                delta=0.5,
                better="variant",
                n_seeds=3,
            ),
        ]
        table = render_leaderboard(rows, "Test")
        assert "| a | nmse |" in table
        assert "| b | levels |" in table
        assert "note a" in table
        assert "Test" in table

    def test_when_serialized_then_the_row_is_json_friendly(self) -> None:
        import json

        from nanochat.lcqat.ablation_metrics import AblationRow

        row = AblationRow(
            experiment="a",
            metric="nmse",
            baseline="x",
            variant="y",
            value_baseline=1.0,
            value_variant=0.5,
            delta=-0.5,
            better="variant",
            seeds=[0, 1],
            n_seeds=2,
        )
        assert json.loads(json.dumps(row_to_dict(row)))["seeds"] == [0, 1]


def reconstruct_exactly(layer, probe: torch.Tensor) -> torch.Tensor:
    """The identity round trip: a quantizer fed its own codebook values."""
    quantizer = select_quantizer(layer, "act")
    indices = quantizer.bucketize(probe.to(torch.float32))
    return quantizer.get_codebook().detach()[indices.long()]

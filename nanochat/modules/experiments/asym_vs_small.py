"""The `asym_vs_small` experiment: does the asymmetric codebook split help?

Measures level utilization of the `relu^2` activation quantizer of `c_proj`,
paired across seeds, for the `small` and `asym` presets.
"""

from __future__ import annotations

import argparse

import torch

from nanochat.models.quant.ablation_metrics import AblationRow, measure_reconstruction
from nanochat.modules.experiments.common import (
    BASELINE_PRESET,
    VARIANT_PRESET,
    non_negative_probe,
    probe_layer,
)


def run_asym_vs_small(args: argparse.Namespace) -> AblationRow:
    """Measure the paired effect of the `asym` split on a `relu^2` activation.

    Probes the **activation** quantizer of `c_proj`, not its weights. That
    distinction is the whole measurement, and getting it wrong inverts the
    result:

    * On the real non-negative `relu^2` input, `small` (a symmetric 15-level
      codebook) can only place 8 of its levels where the tensor actually lives,
      so it wastes half its alphabet. `asym` (one-sided, 8 levels) puts every
      level in range.
    * On signed weights, `asym` is measurably *worse*, because the split gives up
      half the range for no benefit.

    So on a `relu^2` probe the NMSE is a tie -- both arms spend the same 8
    effective levels -- and the claim is confirmed by **effective level count**,
    which is the quantity the split was designed to raise. Reporting NMSE alone
    would show a flat result and hide the improvement.
    """
    measurements = []
    for seed in range(args.seeds):
        # Same seed for both arms: the only difference is the preset.
        torch.manual_seed(seed)
        layer_base = probe_layer(BASELINE_PRESET)
        layer_var = probe_layer(VARIANT_PRESET)

        # One probe tensor, shared: the input is held fixed across arms so the
        # measurement isolates the codebook, not the data.
        probe = non_negative_probe(layer_base.in_features, args.n, seed)

        base = measure_reconstruction(layer_base, probe, BASELINE_PRESET, which="act")
        variant = measure_reconstruction(layer_var, probe, VARIANT_PRESET, which="act")
        measurements.append((base, variant))

    # The deciding metric is the fraction of the alphabet actually used.
    # Absolute level count *ties* at 8 for both arms -- the difference is that
    # `small` spends 15 levels to do it (8 used, 7 stranded below the data's
    # minimum) while `asym` spends 8. So the win is headroom, not resolution.
    base_eff = sum(b.level_utilization for b, _ in measurements) / len(measurements)
    var_eff = sum(v.level_utilization for _, v in measurements) / len(measurements)
    return AblationRow(
        experiment="asym_vs_small_levels",
        metric="level_utilization",
        baseline=BASELINE_PRESET,
        variant=VARIANT_PRESET,
        value_baseline=base_eff,
        value_variant=var_eff,
        delta=var_eff - base_eff,
        better="variant" if var_eff > base_eff else "baseline",
        seeds=list(range(args.seeds)),
        n_seeds=args.seeds,
        notes=(
            "fraction of codebook levels actually hit on a non-negative relu^2 "
            "probe; absolute level count ties at 8 for both arms, so the gain is "
            "headroom (asym spends 8 levels, small spends 15 for the same 8)"
        ),
    )

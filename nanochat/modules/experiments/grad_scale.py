"""The `grad_scale` experiment: does the codebook gradient scale as 1/sqrt(N)?

Compares `observe_grad_scale` over the unscaled and `inv_sqrt_n` codebook
gradient scales, and returns the per-seed observations alongside the row.
"""

from __future__ import annotations

import argparse

import torch

from nanochat.models.quant.ablation_metrics import (
    AblationRow,
    GradScaleObservation,
    observe_grad_scale,
)
from nanochat.models.quant.linear import GRAD_SCALE_INV_SQRT_N, GRAD_SCALE_NONE
from nanochat.modules.experiments.common import (
    VARIANT_PRESET,
    expected_1_over_sqrt_n,
    probe_layer,
)


def run_grad_scale(
    args: argparse.Namespace,
) -> tuple[AblationRow, list[GradScaleObservation]]:
    """Measure the `1/sqrt(N)` codebook gradient scale directly."""
    rows = []
    observations: list[GradScaleObservation] = []
    n_elements = 0
    base_norm = 0.0
    inv_norm = 0.0
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        layer = probe_layer(VARIANT_PRESET)
        gen = torch.Generator().manual_seed(seed)
        probe = torch.randn(args.n, layer.in_features, generator=gen)

        none_obs = observe_grad_scale(
            layer,
            probe,
            GRAD_SCALE_NONE,
            steps=args.grad_scale_steps,
            lr=args.grad_scale_lr,
        )
        inv_obs = observe_grad_scale(
            layer,
            probe,
            GRAD_SCALE_INV_SQRT_N,
            steps=args.grad_scale_steps,
            lr=args.grad_scale_lr,
        )
        observations.extend([none_obs, inv_obs])
        base_norm += none_obs.grad_norm
        inv_norm += inv_obs.grad_norm
        n_elements = layer.weight.numel()

    base_avg = base_norm / args.seeds
    inv_avg = inv_norm / args.seeds
    ratio = inv_avg / base_avg if base_avg else float("nan")
    rows.append(
        AblationRow(
            experiment="grad_scale",
            metric="codebook_grad_ratio",
            baseline=GRAD_SCALE_NONE,
            variant=GRAD_SCALE_INV_SQRT_N,
            value_baseline=base_avg,
            value_variant=inv_avg,
            delta=inv_avg - base_avg,
            # Lower codebook gradient is the point of the scaling, so the
            # variant "wins" when it is strictly smaller.
            better="variant" if inv_avg < base_avg else "baseline",
            seeds=list(range(args.seeds)),
            n_seeds=args.seeds,
            notes=(
                f"observed ratio {ratio:.6g} vs predicted 1/sqrt(N) = "
                f"{expected_1_over_sqrt_n(n_elements):.6g} for N={n_elements}"
            ),
        )
    )
    return rows[0], observations

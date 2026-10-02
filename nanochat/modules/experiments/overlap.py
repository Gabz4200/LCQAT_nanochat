"""The `overlap` experiment: how much of each sigma draw leaves its band.

Sweeps the partitioner's log-sigma overlap and reports, per setting, the
fraction of `(sigma, block)` draws that land outside the *nominal* band.
"""

from __future__ import annotations

import argparse

import torch

from nanochat.models.quant.ablation_metrics import AblationRow
from nanochat.training.diffusion_blocks import EquiProbabilityPartitioner

#: Overlap settings the `overlap` experiment sweeps. The disjoint partition
#: (0.0) is the baseline; the rest are the values DiffusionBlocks App. C quotes
#: for text (0.1) rounded up, plus `--db-overlap` so the operator's choice is
#: swept too. Deduped and sorted by `sorted_overlap_sweep`.
OVERLAP_FLOORS = (0.0, 0.125)


def sorted_overlap_sweep(args: argparse.Namespace) -> list[float]:
    """The overlap settings to sweep, ascending, with the 0.0 baseline first."""
    return sorted({*OVERLAP_FLOORS, float(args.db_overlap)})


def measure_overlap_out_of_band(
    args: argparse.Namespace, overlap: float, seed: int
) -> float:
    """Fraction of sampled `(sigma, block)` pairs outside the *nominal* band.

    The band is the partitioner's own `boundaries()[b], boundaries()[b + 1]` --
    the disjoint equi-probability partition. `sample_sigma(b, overlap=g)` draws
    from `[lo/alpha, hi*alpha]` with `alpha = (hi/lo) ** g`, so this fraction is
    expected to *rise* with `g`: overlap deliberately widens the sampling
    interval past the nominal band, which is the mechanism by which it absorbs
    the mass that would otherwise be misrouted (§12.1).

    That is exactly why the handoff's "strictly decreases" framing is not the
    assertion made here. Measuring against the *widened* interval instead would
    be circular -- `overlap` defines that interval, so the measured fraction
    would fall by construction and test nothing (§10.3).
    """
    partitioner = EquiProbabilityPartitioner(num_blocks=args.ablation_blocks)
    bounds = partitioner.boundaries()
    generator = torch.Generator().manual_seed(seed)
    out_of_band = 0
    total = 0
    for block in range(args.ablation_blocks):
        lo = float(bounds[block].item())
        hi = float(bounds[block + 1].item())
        for _ in range(args.overlap_samples):
            sigma = float(
                partitioner.sample_sigma(
                    block, generator=generator, overlap=overlap
                ).item()
            )
            total += 1
            if not lo <= sigma <= hi:
                out_of_band += 1
    return out_of_band / total if total else 0.0


def run_overlap(args: argparse.Namespace) -> list[AblationRow]:
    """`--db-overlap` sweep: how much of each draw leaves its nominal band.

    One row per swept setting, so the trend is readable down the leaderboard
    rather than collapsed into a single delta. The disjoint partition is the
    baseline every setting is compared against.
    """
    settings = sorted_overlap_sweep(args)
    out_of_band: dict[float, list[float]] = {g: [] for g in settings}
    for g in settings:
        for seed in range(args.seeds):
            out_of_band[g].append(measure_overlap_out_of_band(args, g, seed))
    rows: list[AblationRow] = []
    baseline = settings[0]
    base_val = sum(out_of_band[baseline]) / len(out_of_band[baseline])
    for g in settings:
        vals = out_of_band[g]
        mean = sum(vals) / len(vals)
        rows.append(
            AblationRow(
                experiment=f"overlap_g{g:g}",
                metric="sigma_out_of_nominal_band",
                baseline=f"g={baseline:g}",
                variant=f"g={g:g}",
                value_baseline=base_val,
                value_variant=mean,
                delta=mean - base_val,
                better="variant" if mean < base_val else "baseline",
                seeds=list(range(args.seeds)),
                n_seeds=args.seeds,
                notes=(
                    f"overlap g={g:g} over {args.overlap_samples * args.ablation_blocks} "
                    "draws per seed; the band is the partitioner's nominal "
                    "equi-probability range, not the widened draw interval. CONTRADICTS "
                    "the 'strictly decreases with overlap' framing: this fraction "
                    "RISES with g, because sample_sigma draws from [lo/alpha, "
                    "hi*alpha] and alpha widens the interval by construction -- that "
                    "widening IS the mechanism by which overlap absorbs out-of-range "
                    "mass (handoff 12.1). Measuring against the widened interval "
                    "would be circular (10.3). No model runs for this row."
                ),
            )
        )
    return rows

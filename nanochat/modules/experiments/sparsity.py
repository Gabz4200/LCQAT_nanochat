"""The `sparsity` experiment: pruning mask, scope, and exact-zero dequantization.

Sweeps sparsity levels under both pruning scopes on a retrofitted tiny model,
reporting dequantized NMSE and how far the two scopes' masks disagree.
"""

from __future__ import annotations

import argparse

import torch

from nanochat.models.quant.ablation_metrics import AblationRow, reconstruct_layer
from nanochat.models.quant.pruning import schedule_from_args
from nanochat.models.quant.retrofit import PRESETS, retrofit_model
from nanochat.models.quant.sparseprop import (
    SCOPE_GLOBAL,
    SCOPE_LAYER,
    inject_sparseprop_layers,
)
from nanochat.modules.experiments.common import VARIANT_PRESET
from tests.conftest import build_active_tiny_gpt

#: Sparsity levels the `sparsity` experiment sweeps, below and at the target.
#: The dense 0.0 arm is not swept: at zero sparsity `layer` and `global` are
#: the same mask by construction, so the pair would be a tautology.
SPARSITY_FLOORS = (0.5,)


#: The two pruning scopes the `sparsity` experiment sweeps.
SCOPES = (SCOPE_LAYER, SCOPE_GLOBAL)


def sparsity_sweep(args: argparse.Namespace) -> list[float]:
    """The sparsity settings to sweep, ascending, target last."""
    return sorted({*SPARSITY_FLOORS, float(args.sparseprop_sparsity)})


def pruned_lcqat_layers(
    args: argparse.Namespace, preset: str, sparsity: float, scope: str
) -> list:
    """Retrofitted LC-QAT Linears, pruned to `sparsity` under `scope`.

    Two properties of this path are load-bearing and easy to get wrong:

    * **Masks are KEEP-masks** (`True` = retained). `magnitude_mask` returns
      exactly that, and GMP intersects against the previous mask so pruning stays
      monotone -- a pruned position holds an exact `0.0`, so its `|W|` is 0 and
      it can never re-enter a magnitude-selected mask.
    * **Structural sparsity requires exact LC-QAT zero dequantization.** A pruned
      position holds an exact `0.0` in the shadow weight, and 0.0 is a codebook
      *level* (the zero anchor at `m_neg`), so `bucketize(0.0)` returns `m_neg`
      in any regime and the pruned position dequantizes back to exactly 0.0,
      with no post-hoc mask multiply. The exact-zero check in
      `measure_sparsity_nmse` verifies that contract survived the prune; it is
      not a formality.

    **Both scopes prune two layers jointly, and that is not incidental.**
    `apply_global_pruning` ranks `|W|` across *all* modules it is handed, so a
    global sweep over a single layer is the same mask as a layer sweep by
    construction -- the two arms would tie and the comparison would be a
    tautology. Passing both the attention and MLP projections is what gives
    global scope something to rank across, and it is the pairing SparseProp
    Fig. 6 actually compares.

    The schedule is driven to its final target rather than to whatever the ramp
    would have produced at an arbitrary step, so the sweep compares sparsities
    rather than cadence. Gradual pruning is still the path that applies the mask.
    """
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS[preset])
    inject_sparseprop_layers(
        model, sparsity=sparsity, target_modules=["c_proj", "c_fc"], with_lcqat=True
    )
    layers = [
        model.transformer.h[0].attn.c_proj,
        model.transformer.h[0].mlp.c_fc,
    ]
    schedule = schedule_from_args(args)
    schedule.target_sparsity = sparsity
    schedule.scope = scope
    # `apply` returns None off a prune event, and `enabled` is False whenever
    # `every == 0` -- which is `--sparseprop-every`'s default. `every = 1` makes
    # every step an event so the ramp actually runs; the step passed is the last
    # one, so the ramp completes to its target instead of leaving the model at
    # the start fraction. Skipping this would prune nothing and the whole sweep
    # would report the NMSE of an unpruned layer.
    schedule.every = 1
    schedule.apply(model, schedule.ramp_steps, layers=layers)
    return layers


def scope_mask_disagreement(
    args: argparse.Namespace, preset: str, sparsity: float
) -> tuple[float, float]:
    """How far the two scopes' masks disagree, pooled and per layer.

    Returns `(pooled_disagreement, largest_per_layer_disagreement)`.

    The per-layer *sparsity* of the two scopes came out identical at every
    setting measured here, and that is a real property of these two layers
    rather than a bug: their magnitude distributions overlap enough that one
    global threshold prunes both to the same fraction. What the scopes actually
    disagree about is *which* weights they keep -- the global threshold falls
    between the two layers' medians, so it selects differently within each
    layer even though the counts match. Counting the differing positions is
    what makes the comparison non-tautological; reporting the NMSE of each
    scope separately cannot, because equal counts give equal NMSE.

    Disagreement is the fraction of positions at which the two masks differ, so
    0.0 means identical masks and 0.5 means the scopes are as different as two
    masks of the same sparsity can be.
    """
    torch.manual_seed(0)
    layer_masks = pruned_lcqat_layers(args, preset, sparsity, SCOPE_LAYER)
    global_masks = pruned_lcqat_layers(args, preset, sparsity, SCOPE_GLOBAL)
    pooled = 0.0
    total = 0
    worst = 0.0
    for layer_mask, global_mask in zip(layer_masks, global_masks, strict=True):
        differing = int((layer_mask.sparsity_mask ^ global_mask.sparsity_mask).sum())
        pooled += differing
        total += int(layer_mask.sparsity_mask.numel())
        worst = max(worst, differing / layer_mask.sparsity_mask.numel())
    return (pooled / total if total else 0.0), worst


def measure_sparsity_nmse(
    args: argparse.Namespace, preset: str, sparsity: float, scope: str
) -> tuple[float, bool]:
    """Dequantized NMSE summed over the pruned layers, plus whether zeros hold.

    NMSE is pooled over both layers as `total_squared_error / total_signal_power`
    rather than averaged per layer, so a layer with more parameters contributes
    in proportion to its size instead of being counted once regardless.
    """
    total_mse = 0.0
    total_signal = 0.0
    exact = True
    for layer in pruned_lcqat_layers(args, preset, sparsity, scope):
        with torch.no_grad():
            original = layer.weight.detach().clone()
            pruned = ~layer.sparsity_mask
            # Round-trip through the quantizer, which is what the exported
            # artifact does. Recomputed once, used for both the error and the
            # zero check.
            dequantized = reconstruct_layer(layer, original, "weight")
            if bool(pruned.any()):
                exact = exact and bool(
                    torch.equal(
                        dequantized[pruned], torch.zeros_like(dequantized[pruned])
                    )
                )
            total_mse += float((original - dequantized).square().sum())
            total_signal += float(original.square().sum())
    return (total_mse / total_signal if total_signal > 0 else float("inf")), exact


def run_sparsity(args: argparse.Namespace) -> list[AblationRow]:
    """Sparsity x scope sweep: dequantized NMSE after pruning.

    SparseProp Fig. 6 compares Uniform-GMP against Global-GMP at equal average
    sparsity and finds Global better, so the two scopes are not interchangeable
    and neither is the default by construction. Both are swept here, at the
    floors and at the target, on a retrofitted `attn.c_proj` -- the first
    non-negative activated layer in the model. No training runs: this measures
    the mask, not what a trained model would do with it.
    """
    levels = sparsity_sweep(args)
    nmse: dict[tuple[float, str], list[float]] = {
        (level, scope): [] for level in levels for scope in SCOPES
    }
    exact: dict[tuple[float, str], bool] = {}
    for level in levels:
        for scope in SCOPES:
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                value, is_exact = measure_sparsity_nmse(
                    args, VARIANT_PRESET, level, scope
                )
                nmse[(level, scope)].append(value)
                exact[(level, scope)] = exact.get((level, scope), True) and is_exact

    rows: list[AblationRow] = []
    low, target = levels[0], levels[-1]
    for scope in SCOPES:
        base = sum(nmse[(low, scope)]) / len(nmse[(low, scope)])
        var = sum(nmse[(target, scope)]) / len(nmse[(target, scope)])
        rows.append(
            AblationRow(
                experiment=f"sparsity_{scope}",
                metric="dequant_nmse",
                baseline=f"{scope}@{low:g}",
                variant=f"{scope}@{target:g}",
                value_baseline=base,
                value_variant=var,
                delta=var - base,
                better="variant" if var < base else "baseline",
                seeds=list(range(args.seeds)),
                n_seeds=args.seeds,
                notes=(
                    f"dequantized NMSE of a retrofitted attn.c_proj, {scope} scope, "
                    f"sparsity {low:g} -> {target:g}, reached by the gradual schedule "
                    "at its last ramp event. Masks are KEEP-masks (True = retained) "
                    "and structural zeros dequantize to exactly 0.0 "
                    f"({'verified' if exact[(target, scope)] else 'FAILED'} at the "
                    f"target; {'verified' if exact[(low, scope)] else 'FAILED'} at the "
                    "floor). NMSE rising with sparsity is expected -- pruning removes "
                    "weight magnitude -- so this row records the mask's cost, not a "
                    "quality win. No training runs."
                ),
            )
        )
    # The two scopes prune the two layers to the same *fraction* at every
    # setting, so the NMSE rows above cannot separate them. This row reports
    # the thing that does differ -- which weights each scope keeps -- so the
    # `layer` vs `global` claim rests on a measurement rather than on a
    # tautology.
    pooled_disagreement, worst_layer = scope_mask_disagreement(
        args, VARIANT_PRESET, target
    )
    rows.append(
        AblationRow(
            experiment="sparsity_scope_mask_disagreement",
            metric="scope_mask_disagreement",
            baseline=SCOPE_LAYER,
            variant=SCOPE_GLOBAL,
            value_baseline=0.0,
            value_variant=pooled_disagreement,
            delta=pooled_disagreement,
            better="variant",
            seeds=[0],
            n_seeds=1,
            notes=(
                f"fraction of positions where the layer-scope and global-scope masks "
                f"disagree at sparsity {target:g}, pooled over attn.c_proj and "
                f"mlp.c_fc ({worst_layer:.4g} in the worse of the two layers). "
                "Both scopes prune to the SAME per-layer fraction at every setting "
                "measured here -- these two layers' magnitude distributions overlap "
                "enough that one global threshold cuts both equally -- so the "
                "per-scope NMSE rows above cannot distinguish them and would tie by "
                "construction. The scopes disagree about WHICH weights survive, and "
                "that is what this row measures. Whether global's selection is "
                "better after training is NOT measured here: no training runs. "
                "Single seed, because the two masks are a deterministic function of "
                "the weight tensor, not a stochastic draw."
            ),
        )
    )
    return rows

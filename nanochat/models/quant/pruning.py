"""
Gradual magnitude pruning schedule for SparseProp layers.

SparseProp Sec. 4.1 ("prune gradually every 10 epochs until epoch 80, at which
point we fine-tune") plus the Zhu & Gupta criterion it cites: hold the model
dense for a warmup, then raise the pruning target in equal increments until the
final sparsity, then hold. The pruning *pattern* is static between re-prunes
(AGENTS.md: "Sparsity is static. SparseProp masks are materialised once"), so
this module owns the schedule, not the kernel selection.

Linear ramp from `start_frac` to the target over `prune_every` steps of the
chosen cadence, then a constant hold. A linear ramp in the *sparsity* target is
the standard GMP choice; the paper's per-epoch increments differ only in the
cadence unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nanochat.models.quant.sparseprop import (
    DEFAULT_DENSE_THRESHOLD,
    SCOPE_GLOBAL,
    SCOPE_LAYER,
    SparsePropLinear,
    _give_back_empty_rows,
    apply_global_pruning,
    magnitude_mask,
)


@dataclass
class GradualPruningSchedule:
    """Stateful target-sparsity schedule over training steps.

    Args:
        target_sparsity: the final sparsity, reached at `ramp_steps`.
        start_frac: fraction of `target_sparsity` to start from. 0.0 prunes the
            full target immediately (one-shot, the paper's transfer setting);
            0.5 ramps in half the sparsity first.
        ramp_steps: number of prune events in the ramp. The schedule reaches
            `target_sparsity` on the last one.
        every: prune every N training steps. 0 disables gradual pruning (the
            model is built once at its target and never re-pruned).
        scope: `layer` (uniform per layer) or `global` (joint magnitude ranking).
        dense_threshold: sparsity at which a layer is considered to have crossed
            into the sparse-kernel regime (SparseProp's 80% rule).
    """

    target_sparsity: float
    start_frac: float = 0.0
    ramp_steps: int = 8
    every: int = 0
    scope: str = SCOPE_LAYER
    dense_threshold: float = 0.8

    def __post_init__(self) -> None:
        for name, value in (
            ("target_sparsity", self.target_sparsity),
            ("start_frac", self.start_frac),
            ("dense_threshold", self.dense_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
        if self.ramp_steps < 1:
            raise ValueError(f"ramp_steps must be >= 1, got {self.ramp_steps}")
        if self.every < 0:
            raise ValueError(f"every must be >= 0, got {self.every}")
        if self.scope not in (SCOPE_LAYER, SCOPE_GLOBAL):
            raise ValueError(f"scope must be layer or global, got {self.scope!r}")

    @property
    def enabled(self) -> bool:
        """True when the schedule re-prunes at all."""
        return self.every > 0 and self.target_sparsity > 0.0

    @property
    def initial_sparsity(self) -> float:
        """Sparsity the model is built with, before any ramp step."""
        return self.target_sparsity * self.start_frac

    def should_prune(self, step: int) -> bool:
        """True when `step` is a prune event.

        Step 0 is always an event when the schedule is enabled, so the model is
        never left at the ramp's start fraction if the caller wires it in late.
        """
        if not self.enabled:
            return False
        return step % self.every == 0

    def target_at(self, step: int) -> float:
        """Sparsity target in force at `step`.

        `ramp_steps` equal increments of `(target - initial) / ramp_steps`, then
        the target. Ramp position is counted in prune events, not raw steps, so
        `every` controls the cadence without changing the trajectory.
        """
        if not self.enabled:
            return self.initial_sparsity
        events = step // self.every
        initial = self.initial_sparsity
        delta = (self.target_sparsity - initial) / self.ramp_steps
        return min(self.target_sparsity, initial + events * delta)

    def apply(
        self, root: object, step: int, layers: list[SparsePropLinear] | None = None
    ) -> float | None:
        """Prune `root`'s sparse layers to the `step` target. Returns achieved sparsity.

        Returns None when `step` is not a prune event, so the caller can log
        unconditionally. Re-pruning is monotone: a weight that has already been
        pruned holds exactly 0.0, so its |W| is 0 and it cannot re-enter a
        magnitude-selected mask. Nothing is ever un-pruned.
        """
        if not self.should_prune(step):
            return None
        target = self.target_at(step)
        layers = collect(root) if layers is None else layers
        if not layers:
            return None
        if self.scope == SCOPE_GLOBAL:
            achieved = apply_global_pruning(layers, target, current_sparsity=0.0)
        else:
            with torch.no_grad():
                for layer in layers:
                    # Only layers below the target need re-pruning, and a layer
                    # already past the target is left alone: raising the target
                    # is monotone, so this is the only direction that can occur.
                    current = (
                        1.0 - int(layer.sparsity_mask.sum()) / layer.weight.numel()
                    )
                    if current >= target:
                        continue
                    # `mask & magnitude_mask(W, target)`, not a fresh
                    # magnitude_mask: re-selecting per row independently does
                    # NOT yield a nested mask. Two weights of equal magnitude in
                    # one row swap in and out as the threshold drops, so a plain
                    # re-mask can hand a slot back to an entry a previous prune
                    # already removed. Gradual Magnitude Pruning is defined to be
                    # monotone, and intersecting with the previous mask is both
                    # correct and free.
                    layer.sparsity_mask.logical_and_(
                        magnitude_mask(layer.weight, target)
                    )
                    # The AND can empty a row whose survivors were all pruned
                    # by the intersection; give the row back its largest
                    # remaining entry so no row is ever annihilated.
                    _give_back_empty_rows(layer, layer.sparsity_mask)
                    layer._apply_mask()
            total = sum(m.weight.numel() for m in layers)
            nnz = sum(int(m.sparsity_mask.sum()) for m in layers)
            achieved = 1.0 - nnz / total if total else 0.0
        return achieved

    def layers_above_threshold(self, root: object) -> list[SparsePropLinear]:
        """Sparse layers that have crossed into the sparse-kernel regime."""
        return [
            m
            for m in collect(root)
            if 1.0 - int(m.sparsity_mask.sum()) / m.weight.numel()
            >= self.dense_threshold
        ]


def collect(root: object) -> list[SparsePropLinear]:
    """Every SparsePropLinear under `root`, in module-tree order.

    `root` may be an `nn.Module` or a `DiffusionBlockEngine`; the engine is not
    an `nn.Module` (it owns three of them) but exposes the same `modules()`
    view, so both walk identically here.
    """
    return [m for m in root.modules() if isinstance(m, SparsePropLinear)]


def add_sparseprop_pruning_args(parser) -> None:
    """Register the SparseProp pruning flags on an argparse parser.

    Shared by base_train / chat_sft / chat_rl so the three entry points cannot
    drift apart on the same hardware knob (AGENTS.md: "Hardware knobs exist for
    a reason ... Don't hide them behind config files").
    """
    parser.add_argument(
        "--sparseprop-scope",
        choices=(SCOPE_LAYER, SCOPE_GLOBAL),
        default=SCOPE_LAYER,
        help=(
            "how the sparsity target is distributed: 'layer' prunes every "
            "layer to the same fraction (Uniform-GMP), 'global' ranks |W| "
            "across all layers jointly (Global-GMP, better at equal average "
            "sparsity in SparseProp Fig. 6)"
        ),
    )
    parser.add_argument(
        "--sparseprop-start-frac",
        type=float,
        default=0.0,
        help=(
            "starting sparsity as a fraction of --sparseprop-sparsity for "
            "gradual pruning; 0.0 = one-shot pruning at the full target"
        ),
    )
    parser.add_argument(
        "--sparseprop-every",
        type=int,
        default=0,
        help=(
            "re-prune every N optimizer steps (gradual magnitude pruning, "
            "Zhu & Gupta 2017); 0 = prune once at setup and never again"
        ),
    )
    parser.add_argument(
        "--sparseprop-ramp-steps",
        type=int,
        default=8,
        help="number of prune events over which sparsity ramps to the target",
    )
    parser.add_argument(
        "--sparseprop-dense-threshold",
        type=float,
        default=DEFAULT_DENSE_THRESHOLD,
        help=(
            "sparsity at which a layer is reported as having crossed into the "
            "sparse-kernel regime (SparseProp keeps modules dense below 80%%)"
        ),
    )


def schedule_from_args(args) -> GradualPruningSchedule:
    """Build a `GradualPruningSchedule` from parsed CLI flags."""
    return GradualPruningSchedule(
        target_sparsity=args.sparseprop_sparsity,
        start_frac=args.sparseprop_start_frac,
        ramp_steps=args.sparseprop_ramp_steps,
        every=args.sparseprop_every,
        scope=args.sparseprop_scope,
        dense_threshold=args.sparseprop_dense_threshold,
    )


__all__ = [
    "GradualPruningSchedule",
    "add_sparseprop_pruning_args",
    "schedule_from_args",
]

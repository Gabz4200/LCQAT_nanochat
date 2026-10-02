"""
EfQAT selective layer freezing (PRD section 3.2).

To prevent VRAM exhaustion from Adam momentum states over codebook delta
parameters across multi-billion parameter models, LC-QAT implements
EfQAT-style selective parameter freezing:

* Middle transformer layers exhibiting low activation variance and stable
  gradient trajectories have their codebook updates (`raw_pos_deltas`,
  `raw_neg_deltas`) and weight gradients frozen after an initial warm-up.
* Active backward optimization is restricted to critical outlier layers
  (input embedding projections, attention keys/queries, final output layers).

The freezer is a stateful callable driven by the training step count. It
mutates `requires_grad` in place; because AdamW skips params whose `.grad is
None`, frozen params stop updating without touching the optimizer's momentum
buffers (those buffers remain but receive no fresh updates).

`BlockLatchFreezer` adds the *per-block permanent* half, which the
middle-band freezer above cannot express. Under DiffusionBlocks each block
trains on a disjoint equi-probability sigma range, so a block's quantization
parameters (LC-QAT codebook deltas, weights) finish specializing early. Once
that happens the block must stop moving: later optimizer steps that happen to
sample this block again must not revive it. The latch is therefore one-way --
`latch_block` has no inverse -- and it is consulted by the engine's
`_requires_grad_for` *before* block activation, so activation can never
re-enable a latched parameter.
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn

# Module-name substrings that are always kept trainable (PRD 3.2 "critical
# outlier layers"). Everything not matching these patterns is a candidate for
# freezing once warmup elapses.
CRITICAL_PATTERNS = (
    "transformer.wte",  # input embedding projection
    "value_embeds",  # value embeddings
    "attn.c_q",  # attention query projection
    "attn.c_k",  # attention key projection
    "lm_head",  # final output (logit) projection
    "backout",  # backout residual scalar
    "smear",  # smear gate / scalar
    "resid_lambdas",
    "x0_lambdas",
)


class SelectiveFreezer:
    """Freeze middle-layer codebook + weight grads after a warm-up window.

    Single source of truth for `requires_grad` under DiffusionBlocks: the engine
    asks `is_trainable(name)` before enabling a block, instead of enabling
    everything and letting the freezer lose the race. Previously `_activate_block`
    unconditionally re-enabled every `transformer.h.*` parameter each micro-step,
    which undid any freeze applied on the previous step, so EfQAT never took
    effect at all.

    Args:
        model: the (retrofitted) model tree.
        warmup_steps: training steps to keep everything trainable before
            freezing middle layers.
        freeze_middle_frac: fraction of middle transformer layers to freeze.
            0.0 = freeze none (EfQAT disabled), 1.0 = freeze everything that is
            not critical. The frozen layers are the contiguous middle band
            (bottom + top kept unfrozen, matching the PRD's "critical outlier
            layers at the boundaries").
    """

    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int = 1000,
        freeze_middle_frac: float = 0.5,
    ):
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if not 0.0 <= freeze_middle_frac <= 1.0:
            raise ValueError(
                f"freeze_middle_frac must be in [0, 1], got {freeze_middle_frac}"
            )
        self.model = model
        self.warmup_steps = int(warmup_steps)
        self.freeze_middle_frac = float(freeze_middle_frac)
        self._frozen = False
        self._frozen_params: set[int] = set()

    @staticmethod
    def is_critical(name: str) -> bool:
        return any(pat in name for pat in CRITICAL_PATTERNS)

    def is_trainable(self, name: str) -> bool:
        """Whether `name` may receive gradients right now.

        The engine's block activation consults this instead of setting
        `requires_grad_` directly, so a frozen parameter stays frozen across
        steps. Unfrozen is the default, so this is safe to call before `freeze()`.
        """
        if not self._frozen:
            return True
        if self.is_critical(name):
            return True
        return id(self._lookup(name)) not in self._frozen_params

    def _lookup(self, name: str) -> nn.Parameter | None:
        params = dict(self.model.named_parameters())
        return params.get(name)

    def _layer_params(self) -> list[tuple[int, str, nn.Parameter]]:
        """Return (layer_index, param_name, param) for every transformer.h param."""
        params = []
        prefix = "transformer.h."
        for name, p in self.model.named_parameters():
            if not name.startswith(prefix):
                continue
            layer_token = name[len(prefix) :].split(".", 1)[0]
            if not layer_token.isdigit():
                continue
            params.append((int(layer_token), name, p))
        return params

    def layer_bounds(self) -> tuple[int, int, int]:
        """Return (n_layer, frozen_start, frozen_end) layer index bounds (exclusive end).

        The frozen band is the middle `freeze_middle_frac` of layers. The top
        and bottom unfrozen bands keep gradients live (critical outlier layers).
        """
        h = getattr(self.model, "transformer", None)
        if h is None:
            return 0, 0, 0
        n_layer = len(h.h)
        if n_layer == 0 or self.freeze_middle_frac <= 0.0:
            return n_layer, 0, 0
        n_freeze = round(n_layer * self.freeze_middle_frac)
        # Clamp: never freeze the very first or last layer (keep boundaries alive).
        n_freeze = max(0, min(n_freeze, n_layer - 2)) if n_layer >= 2 else 0
        # Center the frozen band.
        margin = (n_layer - n_freeze) // 2
        start = margin
        end = start + n_freeze
        return n_layer, start, end

    def freeze(self) -> int:
        """Freeze middle-layer codebook deltas and weight grads. Idempotent.

        Returns the number of parameters frozen.
        """
        if self._frozen:
            return len(self._frozen_params)
        n_layer, start, end = self.layer_bounds()
        if end <= start:
            self._frozen = True
            return 0
        frozen = 0
        for layer_idx, name, p in self._layer_params():
            if start <= layer_idx < end and not self.is_critical(name):
                p.requires_grad_(False)
                self._frozen_params.add(id(p))
                frozen += 1
        self._frozen = True
        return frozen

    def update(self, step: int) -> bool:
        """Call once per training step. Freezes when `step >= warmup_steps`.

        Returns True if this call crossed the freeze threshold.
        """
        if self._frozen or step < self.warmup_steps:
            return False
        self.freeze()
        return True

    def is_frozen(self) -> bool:
        return self._frozen

    def unfreeze(self) -> int:
        """Reverse the freeze (for testing / resume). Returns count unfrozen."""
        if not self._frozen:
            return 0
        count = 0
        for _, name, p in self._layer_params():
            if id(p) in self._frozen_params:
                p.requires_grad_(True)
                count += 1
        self._frozen_params.clear()
        self._frozen = False
        return count


class BlockLatchFreezer:
    """One-way per-block freeze of DiffusionBlocks quantization parameters.

    A DiffusionBlocks block trains on one disjoint equi-probability sigma range,
    so its quantization parameters converge to that range early. Once they have,
    later optimizer steps must not move them again -- otherwise a step that
    happens to resample a converged block silently un-converges it. The latch
    has no inverse by design: `latch_block` is the only transition, and
    `is_trainable` keeps returning False for a latched block's names forever.

    The engine consults `is_trainable(name)` from `_requires_grad_for` *before*
    setting `requires_grad`, which is what makes the latch survive block
    activation. Setting `requires_grad` first and vetoing afterwards would let
    the next `_activate_block` revive the block.

    Names follow `DiffusionBlockEngine.named_parameters()` conventions: the
    block index is the ModuleList position after the prefix, e.g.
    `db_denoise_heads.2.weight` or `transformer.h.5.attn.c_q.weight` (block 2
    owning layer 5 under a 4-layer / 2-block split).

    Args:
        engine: the `DiffusionBlockEngine` whose parameters are gated.
        block_layers: `engine.block_layers()`, mapping block index to the
            transformer layer indices it owns. Needed to attribute shared-name
            transformer parameters to a block.
    """

    def __init__(self, engine, block_layers: list[list[int]]) -> None:
        self.engine = engine
        self.block_layers = [list(group) for group in block_layers]
        self._latched: set[int] = set()

    # -- name -> block resolution -------------------------------------------

    def block_of(self, name: str) -> int | None:
        """Return the block index owning `name`, or None if it is shared.

        Shared parameters (embeddings, `lm_head`, per-model scalars) belong to
        every block, so no single block's freeze can claim them; returning None
        keeps them trainable, which is what block isolation already does.
        """
        if name.startswith("transformer.h."):
            layer_token = name.split(".")[2]
            if not layer_token.isdigit():
                return None
            layer = int(layer_token)
            for block_idx, group in enumerate(self.block_layers):
                if layer in group:
                    return block_idx
            return None
        for prefix in ("db_adapters.", "db_denoise_heads."):
            if name.startswith(prefix):
                parts = name.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    return int(parts[1])
        return None

    def block_parameter_names(self, block_idx: int) -> list[str]:
        """Every engine parameter name attributed to `block_idx`."""
        return [
            name
            for name, _ in self.engine.named_parameters()
            if self.block_of(name) == block_idx
        ]

    # -- the latch -----------------------------------------------------------

    def is_latched(self, block_idx: int) -> bool:
        """Whether `block_idx` has been permanently frozen."""
        return int(block_idx) in self._latched

    def latched_blocks(self) -> list[int]:
        """Sorted list of latched block indices (for logging / checkpointing)."""
        return sorted(self._latched)

    def latch_block(self, block_idx: int) -> int:
        """Permanently freeze `block_idx`'s parameters. Idempotent.

        Returns the number of parameters newly frozen (0 if already latched).
        """
        b = int(block_idx)
        if b in self._latched:
            return 0
        params = dict(self.engine.named_parameters())
        frozen = 0
        for name, p in params.items():
            if self.block_of(name) != b:
                continue
            if p.requires_grad:
                p.requires_grad_(False)
                p.grad = None
                frozen += 1
        self._latched.add(b)
        return frozen

    def latch_blocks(self, block_indices: Iterable[int]) -> int:
        """Latch several blocks; returns the total number of params frozen."""
        return sum(self.latch_block(b) for b in block_indices)

    def is_trainable(self, name: str) -> bool:
        """Whether `name` may receive gradients right now.

        A latched block's names are False forever; everything else is True, so
        this is safe to call before any latch is taken.
        """
        if not self._latched:
            return True
        block_idx = self.block_of(name)
        return block_idx is None or block_idx not in self._latched

    def metadata(self) -> dict:
        """JSON-serializable latch state for checkpoint metadata.

        Codebook/weight tensors are already covered by the engine state_dict, so
        this records only the (non-recoverable) freeze decision: a resumed run
        must not train a block the previous run had converged.
        """
        return {
            "latched_blocks": self.latched_blocks(),
            "n_blocks": len(self.block_layers),
        }

    def load_metadata(self, meta: dict | None) -> None:
        """Restore latch state from checkpoint metadata. No-op when absent."""
        if not meta:
            return
        self.latch_blocks(meta.get("latched_blocks", []))

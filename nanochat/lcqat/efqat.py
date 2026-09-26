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
"""

from __future__ import annotations

import torch.nn as nn


# Module-name substrings that are always kept trainable (PRD 3.2 "critical
# outlier layers"). Everything not matching these patterns is a candidate for
# freezing once warmup elapses.
CRITICAL_PATTERNS = (
    "transformer.wte",     # input embedding projection
    "value_embeds",        # value embeddings
    "attn.c_q",            # attention query projection
    "attn.c_k",            # attention key projection
    "lm_head",             # final output (logit) projection
    "backout",             # backout residual scalar
    "smear",               # smear gate / scalar
    "resid_lambdas",
    "x0_lambdas",
)


class SelectiveFreezer:
    """Freeze middle-layer codebook + weight grads after a warm-up window.

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

    def _layer_params(self) -> list[tuple[int, str, nn.Parameter]]:
        """Return (layer_index, param_name, param) for every transformer.h param."""
        params = []
        for name, p in self.model.named_parameters():
            # Only consider params owned by transformer block layers.
            if name.startswith("transformer.h.") and name[len("transformer.h.") :].isdigit() is False:
                # extract layer index token
                pass
            prefix = "transformer.h."
            if name.startswith(prefix):
                rest = name[len(prefix) :]
                layer_idx = int(rest.split(".", 1)[0]) if rest.split(".", 1)[0].isdigit() else -1
                if layer_idx >= 0:
                    params.append((layer_idx, name, p))
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

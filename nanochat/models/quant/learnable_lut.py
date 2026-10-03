"""
Learnable index-to-index activation LUT for the LC-QAT fused MLP chain.

`activation_lut` is already a `K_in -> K_out` integer table: the runtime
replaces `relu^2` + `bucketize` with a single gather
(`linear.py:quantized_mlp_chain`). Today it is *baked* at export time by
`compile_activation_lut`, so its content is a consequence of the two codebooks
rather than anything the task loss can shape.

This module makes that table a trained parameter. The initialisation is
`compile_activation_lut(act_fn, C_in, C_out)` -- **bit-identical to the current
bake** -- so switching this in is a strict superset of today's behaviour: at
step 0 the model computes exactly what it computes now, and from step 1 the
table is free to move.

Training uses a softmax relaxation over output-codebook logits plus a
straight-through round, so the forward value stays a discrete index and the
inference path is unchanged. The relaxed value is a convex combination of the
output codebook levels, which is what makes the gradient meaningful: a bare
`argmax` has no gradient at all.

The companion `LearnableActivationLut.from_callable` (`ablation.py`) learns a
*continuous* 1-D function seeded from the closed form instead. Honest framing:
with K knots it is piecewise-linear, exact at the knots and approximate between
them -- not "exactly relu^2". It is trainable, so within a step budget it can
beat the closed form, but it starts as an approximation.

Both are per-layer, and therefore per-block for free: `c_fc` belongs to exactly
one diffusion block, so each block's activation table is trained on that block's
own noise-range activation distribution.

Parity requirement (`AGENTS.md`: behavior over bytes): the training path and the
fused `quantized_mlp_chain` must read the *same* table. `resolved_table()` is
the single accessor both use.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanochat.models.quant.codebook import codebook_midpoints
from nanochat.models.quant.lut import compile_activation_lut, get_activation

#: Relaxation temperature for the softmax over output-level logits. Small
#: enough that the relaxed value is dominated by the argmax at init, so the
#: trained path and the hard path start numerically indistinguishable.
DEFAULT_TEMPERATURE = 0.1

#: How strongly the initial logits favour the baked index, in units of the
#: softmax temperature. Multiplied by `temperature` on purpose: a gap that is
#: large in absolute logits is *saturated*, and `exp(-gap / T)` then underflows
#: to exactly 0 in FP32, which makes the softmax gradient identically zero and
#: freezes the table forever. At 4 temperature units the initial softmax puts
#: ~0.93 on the baked index -- peaked enough that the relaxed value starts near
#: the hard one, and still far from the underflow cliff.
LOGIT_SEED_STRENGTH = 4.0

#: Selector relaxations for the trained table.
#:
#: `logits` is the original free `(K_in, K_out)` logit matrix with a
#: straight-through round. `proximity` selects the output level by inverse-square
#: distance to a learnable knot grid instead (the `ablation/simple.py` form,
#: measured in HANDOFF §10.2.1: 30 parameters rather than 225 at K=15, and a
#: ~6x larger gradient at init). The distance selector has no saturation cliff,
#: which is the load-bearing difference -- a frozen table would silently void the
#: PRD 2.4 gradient scaling.
RELAXATION_LOGITS = "logits"
RELAXATION_PROXIMITY = "proximity"
RELAXATIONS = (RELAXATION_LOGITS, RELAXATION_PROXIMITY)

#: Floor on the inverse-square distance, matching `ablation/simple.py`. Without
#: it, an input sitting exactly on a knot divides by zero.
PROXIMITY_EPS = 1e-6

#: Knot storage dtype. fp8_e4m3fn round-trips the K=15 knot grid over [-1, 1]
#: with the resolved index table bit-identical to fp32 knots (measured, HANDOFF
#: §10.2.2), so knots cost 1 byte each instead of 4 at zero behavioural cost.
KNOT_DTYPE = torch.float8_e4m3fn

#: Fraction of the local codebook gap by which each knot is nudged off the input
#: position it serves. Non-zero is load-bearing, not cosmetic: at offset 0 the
#: nearest knot is at `diff == 0`, the inverse-square weight is `1/eps == 1e6`,
#: and every other weight underflows to exactly 0 in FP32. The softmax becomes a
#: hard argmax whose knot gradient is identically zero -- the same permanent
#: freeze the proximity relaxation was adopted to escape, one level down. At 1/4
#: of the local gap the nearest knot still wins decisively (so the table stays
#: the bake) while the off-diagonal weights stay in FP32 range.
KNOT_INIT_OFFSET_FRACTION = 0.25

#: How the inverse-square selector is scaled, as a multiple of the local knot
#: spacing squared. Expressed as a multiple of the grid rather than as an
#: absolute constant so it tracks K: at K=15 the gap is ~0.14, so an unscaled
#: weight is `1/0.0013 ~ 770` at the nearest knot and `1/0.021 ~ 48` at the
#: second -- a gap of ~700 in the softmax argument, which underflows to exactly
#: 0 in FP32 and turns the selector back into a hard argmax with a zero knot
#: gradient. That is the same saturation cliff `LOGIT_SEED_STRENGTH` exists to
#: stay clear of, one level down, and it was measured here rather than assumed:
#: `test_when_backpropagated_then_the_knot_gradient_is_nonzero` fails at 1.0.
#:
#: At 1.0 the nearest knot scores ~16 and its neighbour ~2, giving `exp(14)` --
#: comfortably representable, so every knot keeps a live gradient. Measured as a
#: parameter-reduction *and* gradient-quality win, not a cosmetic constant.
PROXIMITY_SCALE_GRID_UNITS = 1.0


class LearnableIndexLut(nn.Module):
    """A trained `K_in -> K_out` index table.

    The forward path is exactly the baked table (an integer gather). Training
    happens through a softmax relaxation over per-level logits:

        out = soft + hard - detach(soft)

    so gradients reach `logits` while the value that reaches the matmul is the
    discrete index, matching `quantized_mlp_chain` to the bit.

    Args:
        input_codebook: `(K_in,)` FP32 codebook of the pre-activation tensor.
        output_codebook: `(K_out,)` FP32 codebook of the post-activation
            quantizer. Kept as a non-persistent buffer: it is already owned by
            `out_quantizer` in the checkpoint, and duplicating it persistently
            would let the two copies drift apart silently.
        act_name: key into `lut._ACTIVATIONS`, used only for the initialisation
            bake. Changing it after construction has no effect on the table.
        temperature: softmax temperature over the output-level logits.
        freeze: when True the table is a buffer rather than a parameter -- the
            exact current behaviour, kept as an ablation baseline.

    Shape:
        `forward` accepts any integer tensor and returns the same shape with
        values in `[0, K_out)`.
    """

    def __init__(
        self,
        input_codebook: torch.Tensor,
        output_codebook: torch.Tensor,
        act_name: str = "relu2",
        temperature: float = DEFAULT_TEMPERATURE,
        freeze: bool = False,
        relaxation: str = RELAXATION_LOGITS,
        smooth_body=None,
    ) -> None:
        super().__init__()
        if input_codebook.ndim != 1 or output_codebook.ndim != 1:
            raise ValueError(
                "codebooks must be 1-D, got "
                f"input {tuple(input_codebook.shape)}, output {tuple(output_codebook.shape)}"
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if relaxation not in RELAXATIONS:
            raise ValueError(
                f"relaxation must be one of {RELAXATIONS}, got {relaxation!r}"
            )

        self.k_in = int(input_codebook.numel())
        self.k_out = int(output_codebook.numel())
        self.temperature = float(temperature)
        self.relaxation = relaxation

        with torch.no_grad():
            initial = compile_activation_lut(
                get_activation(act_name),
                input_codebook.to(torch.float32),
                output_codebook.to(torch.float32),
            ).to(torch.int64)
            if smooth_body is not None:
                # The fitted body is a *better starting table* than the
                # per-codebook bake: it approximates the activation over the
                # whole span rather than at K_in sampled points, so its nearest
                # codebook level is a better choice at every index. The table
                # shape and dtype are unchanged -- this only picks which output
                # level each input index starts on.
                body_values = smooth_body(input_codebook.to(torch.float32)).reshape(-1)
                midpoints = codebook_midpoints(output_codebook.to(torch.float32))
                initial = torch.bucketize(body_values, midpoints).to(torch.int64)
            # Seed the logits so softmax(logits / T) peaks on the baked index.
            # A uniform init would make the relaxed value the mean of the
            # output codebook at step 0, which is a large, arbitrary gradient
            # pushing the table somewhere before it has learned anything.
            gap = LOGIT_SEED_STRENGTH * self.temperature
            logits = torch.full((self.k_in, self.k_out), -gap, dtype=torch.float32)
            rows = torch.arange(self.k_in)
            logits[rows, initial] = gap

        # The input codebook is read by the proximity selector (it evaluates the
        # knot distances *at* the codebook entries, since the table is indexed
        # by input codebook position). Non-persistent for the same reason
        # `output_codebook` is: it is already owned by the quantizer in the
        # checkpoint and a second persistent copy could drift.
        self.register_buffer(
            "input_codebook",
            input_codebook.detach().to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "output_codebook",
            output_codebook.detach().to(torch.float32),
            persistent=False,
        )
        self.register_buffer("initial_table", initial, persistent=True)

        if relaxation == RELAXATION_PROXIMITY:
            self._init_proximity(initial)
        elif freeze:
            self.register_buffer("logits", logits, persistent=True)
        else:
            self.logits = nn.Parameter(logits)

    def _init_proximity(self, initial: torch.Tensor) -> None:
        """Install the knot/level parameterization (proximity relaxation).

        Two vectors replace the `(K_in, K_out)` logit matrix, which is what the
        measured parameter reduction in HANDOFF §10.2.1 counts: `knots (K_in,)`
        are positions in the input codebook's domain, `levels (K_in,)` are the
        post-activation values at those positions. The output codebook is not
        re-parameterized -- a level is a level of it.

        Init is bit-identical to the bake by construction, not by seeding:

        * `levels[i] = output_codebook[initial[i]]`, i.e. exactly what the baked
          table emits at input index `i`. Since `output_codebook[initial[i]]` is
          a codebook entry, `bucketize` against the codebook midpoints returns
          `initial[i]` for it, so `resolved_table()` is the baked table to the
          bit on the first step.
        * `knots = input_codebook + KNOT_INIT_OFFSET_FRACTION * local_gap`, so
          the nearest knot to input index `i` is still knot `i` and the relaxed
          value equals the hard value -- but the selector is not *exactly*
          one-hot. Initializing the knots at the exact input positions makes
          `diff == 0` for the nearest knot, so `1/(diff^2 + eps) == 1/eps == 1e6`
          and every other weight underflows to exactly 0 in FP32. The softmax is
          then a hard argmax with identically zero gradient: precisely the
          saturation cliff the free-logit relaxation was replaced to avoid,
          reproduced one level down. `resolved_table()` reads `levels` only, so
          the nudge leaves the baked table bit-identical -- verified by
          `test_when_knots_are_exported_to_fp8_then_the_table_is_unchanged` and
          the bit-identity test.

        The zero anchor is pinned, not hoped for. LC-QAT's input codebook always
        has exactly one level at FP32 0.0 (index `m_neg`), and the bake maps it
        to the output codebook's own 0.0 anchor -- so that entry's level is
        *already* exactly 0.0 and pinning it keeps it there as the level
        optimizes. An unpinned RBF-style convex combination returns -4.47e-08 at
        input 0.0 (measured, HANDOFF §10.3), which would break the SparseProp
        structural-zero contract at the junction of the two methods.
        """
        codebook = self.output_codebook
        levels = codebook.to(torch.float32)[initial]
        anchor = self.zero_input_index(initial)

        self.knots = nn.Parameter(self._initial_knots())
        self.levels = nn.Parameter(levels)
        # `pin_mask` marks the knot whose level must stay exactly 0.0. Held as a
        # buffer rather than folded into `levels` so the level itself stays a
        # free parameter everywhere else, and so the pin is visible in the
        # state_dict instead of being an invisible clamp.
        pin = torch.zeros(self.k_in, dtype=torch.bool)
        pin[anchor] = True
        self.register_buffer("pin_mask", pin, persistent=True)
        # Pin value is exactly 0.0 (FP32), matching the codebook anchor.
        self.register_buffer(
            "pin_value", torch.zeros((), dtype=torch.float32), persistent=True
        )

    def zero_input_index(self, initial: torch.Tensor | None = None) -> int:
        """Index of the input-codebook level that is exactly FP32 0.0.

        Also asserts that the *baked* value at that level is exactly 0.0 -- by
        value, not by index. The LC-QAT zero anchor is a structural property of
        the input codebook (index `m_neg`), but which output index it lands on
        depends on the output codebook's spacing. Given a codebook whose zero
        level is not at the mid-position, `bucketize(0.0)` can return an index
        whose codebook value is not 0.0. A pin that then forces
        `levels[zero] = 0.0` would install a value the bake never had: the
        resolved *table* could stay bit-exact while the stored level no longer
        describes what the frozen LUT gathers.

        So the contract is checked on the emitted value, which is what SparseProp
        actually depends on, and a codebook pair whose bake maps the zero anchor
        to a nonzero level is refused rather than silently repaired.
        """
        if initial is None:
            initial = self.initial_table
        exact = torch.nonzero(self.input_codebook == 0.0, as_tuple=True)[0]
        if exact.numel() != 1:
            raise ValueError(
                "the proximity activation LUT needs exactly one input-codebook "
                f"level at FP32 0.0 (the LC-QAT zero anchor), found "
                f"{exact.numel()} in a K_in={self.k_in} codebook"
            )
        idx = int(exact[0])
        baked = self.output_codebook.to(torch.float32)[initial[idx]]
        if baked != 0.0:
            raise ValueError(
                "the activation does not preserve zero: the bake maps input "
                f"level 0.0 (index {idx}) to output codebook value "
                f"{float(baked):.6g}, not 0.0. A pinned zero level would "
                "disagree with the bake it is supposed to match, so this is "
                "refused rather than applied silently."
            )
        return idx

    def pinned_levels(self) -> torch.Tensor:
        """`levels` with the zero anchor forced to exactly 0.0."""
        return torch.where(
            self.pin_mask.to(self.levels.device),
            self.pin_value.to(self.levels.dtype),
            self.levels,
        )

    def _initial_knots(self) -> torch.Tensor:
        """Knot init: input positions nudged off by a quarter of the local gap.

        The nudge direction alternates so the knots interleave rather than all
        shifting one way: knot `i` sits a quarter-gap *after* input `i` when the
        gap is positive. Ordering is preserved (the nudge is a quarter of the gap,
        far less than the gap itself), so the nearest knot to input `i` is still
        knot `i` and the resolved table is the bake.
        """
        codebook = self.input_codebook.to(torch.float32)
        if self.k_in < 2:
            # A single knot has no gap to offset into, and one knot already
            # selects itself with a finite weight (diff is the codebook value,
            # not 0, unless the codebook is a lone 0.0).
            return codebook.clone()
        gaps = codebook[1:] - codebook[:-1]
        offset = KNOT_INIT_OFFSET_FRACTION * torch.cat([gaps[:1], gaps])
        return codebook + offset

    def proximity_weights(self) -> torch.Tensor:
        """Soft assignment of each input index to a knot, shape `(K_in, K_in)`.

        `softmax_k 1 / ((x_i - knot_k)^2 + eps)`. The selector is a distance,
        not a logit gap, which is the point of the relaxation: it has no
        saturation cliff, so the gradient cannot collapse to exactly zero in
        FP32 and freeze the table (the failure `LOGIT_SEED_STRENGTH` exists to
        stay clear of).
        """
        diff = self.input_codebook.unsqueeze(-1) - self.knots.unsqueeze(0)
        weights = 1.0 / (diff.pow(2) + PROXIMITY_EPS)
        # Divide the *weight*, not the exponent: the softmax is invariant to a
        # global factor, so scaling by `spacing^2` is exactly equivalent to
        # subtracting `2*log(spacing)` from every exponent, and it keeps the
        # argument in a range where neighbouring knots stay representable. See
        # PROXIMITY_SCALE_GRID_UNITS for why the unscaled form saturates.
        spacing = self._knot_spacing()
        if spacing > 0.0:
            weights = weights * (PROXIMITY_SCALE_GRID_UNITS * spacing**2)
        return torch.softmax(weights, dim=-1)

    def _knot_spacing(self) -> float:
        """Median local gap between knots, used to scale the selector.

        Read under `no_grad` and returned as a Python float. The knots
        themselves must stay differentiable -- that is the entire point of the
        relaxation -- but this *normalization* is derived from those same knots,
        and backpropagating through a median makes the knot gradient a one-hot
        scatter that is exactly zero for every knot but one. Treating it as the
        constant it is keeps all `K_in` knots trainable.

        The median rather than the mean: a codebook whose levels are unevenly
        spaced (an asymmetric codebook is, by construction) would have its mean
        gap dragged around by the wide side, and the scale is supposed to
        represent a typical neighbour distance.
        """
        if self.knots.numel() < 2:
            return 0.0
        with torch.no_grad():
            gaps = (self.knots[1:] - self.knots[:-1]).abs()
            return float(gaps.median())

    def fp8_knots(self) -> torch.Tensor:
        """Knots round-tripped through fp8_e4m3fn, back in FP32.

        The trained parameter is FP32 -- an fp8 `nn.Parameter` has no
        meaningful `.grad` and would make the knot gradient the one this
        relaxation was adopted for unusable. fp8 is the *export* saving, and
        `test_when_knots_are_exported_to_fp8_then_the_table_is_unchanged` pins
        that the saving is free: `resolved_table()` does not read `knots` at
        all (levels alone determine it), so the table is bit-identical for any
        knot dtype.
        """
        return self.knots.to(KNOT_DTYPE).to(torch.float32)

    def relaxed(self) -> torch.Tensor:
        """Differentiable `(K_in,)` value in output-codebook units.

        A softmax-weighted combination of the output codebook levels, i.e. the
        post-activation value implied by the (soft) index distribution.
        """
        if self.relaxation == RELAXATION_PROXIMITY:
            return self.proximity_weights() @ self.pinned_levels()
        weights = torch.softmax(self.logits / self.temperature, dim=-1)
        return (weights * self.output_codebook.to(weights.dtype)).sum(dim=-1)

    def resolved_table(self) -> torch.Tensor:
        """The hard `(K_in,)` integer table the inference path gathers with.

        The single accessor both the training path and `quantized_mlp_chain`
        must use, so the two cannot read different tables. This is an integer
        gather target: `output_codebook[table[input_index]]` is the value the
        matmul consumes.

        Under the proximity relaxation this is
        `bucketize(level, midpoints(output_codebook))` -- each knot's learned
        level snapped back onto the output codebook. Because a level *is* a
        codebook entry at init, the snap is the identity there and the table is
        the baked one to the bit.
        """
        if self.relaxation == RELAXATION_PROXIMITY:
            midpoints = codebook_midpoints(self.output_codebook)
            return torch.bucketize(self.pinned_levels(), midpoints).to(torch.int64)
        return torch.argmax(self.logits, dim=-1)

    def hard_values(self, indices: torch.Tensor) -> torch.Tensor:
        """The exact discrete value the inference path produces, in codebook units.

        `output_codebook[resolved_table()[indices]]`, with no straight-through
        term -- bit-identical to the baked table's output.
        """
        table = self.resolved_table().to(indices.device)
        return self.output_codebook.to(table.device)[table[indices.long()]]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Map input indices to output values, differentiably.

        Returns FP32 values in output-codebook units whose *forward value* is
        exactly `hard_values(indices)` (straight-through), so the training path
        and `quantized_mlp_chain` agree to the bit while the gradient still
        reaches the trained parameters.

        The return is FP32, not an index, and that is not a convenience: an
        integer tensor carries no gradient, so a forward that returned indices
        would silently train nothing. Use `resolved_table()` when an integer
        gather is what the caller actually wants.
        """
        table = self.resolved_table().to(indices.device)
        codebook = self.output_codebook.to(table.device)
        hard = codebook[table[indices.long()]]
        if not torch.is_grad_enabled():
            return hard
        # `_trained_parameters` is the source of truth for "does this
        # relaxation have trainable parameters at all"; it reports non-None
        # under proximity (knots + levels), so the guard below is exactly the
        # frozen-logits case and the two branches share one relaxation.
        if self.relaxation != RELAXATION_PROXIMITY and (
            self._trained_parameters() is None
        ):
            return hard
        soft = self.relaxed().to(table.device)[indices.long()]
        return soft + hard - soft.detach()

    def _trained_parameters(self) -> list[nn.Parameter] | None:
        """The parameters this relaxation trains, or None when frozen."""
        if self.relaxation == RELAXATION_PROXIMITY:
            return [self.knots, self.levels]
        return [self.logits] if isinstance(self.logits, nn.Parameter) else None

    def extra_repr(self) -> str:
        return (
            f"k_in={self.k_in}, k_out={self.k_out}, temperature={self.temperature}, "
            f"relaxation={self.relaxation}"
        )


def bake_learnable_table(
    input_codebook: torch.Tensor,
    output_codebook: torch.Tensor,
    act_name: str = "relu2",
) -> torch.Tensor:
    """The frozen `(K_in,)` table `LearnableIndexLut` starts from.

    Exposed so a caller that wants the baked behaviour with no module in the
    way (or wants to check a trained table against its starting point) does not
    have to reach into `lut.py`.
    """
    return compile_activation_lut(
        get_activation(act_name),
        input_codebook.to(torch.float32),
        output_codebook.to(torch.float32),
    ).to(torch.int64)


__all__ = [
    "DEFAULT_TEMPERATURE",
    "LOGIT_SEED_STRENGTH",
    "LearnableIndexLut",
    "bake_learnable_table",
]

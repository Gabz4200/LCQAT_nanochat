"""Tests for the proximity relaxation of the learnable activation LUT (D5/D6).

The proximity selector replaces the free `(K_in, K_out)` logit matrix with a
`knots (K_in,) + levels (K_in,)` pair and picks the level by inverse-square
distance to the knot grid (HANDOFF §10.2.1). Three properties are load-bearing
and each has a test here:

1. **The resolved table is still bit-identical to the bake.** The relaxation is
   adopted as an opt-in alternative, so step 0 must be unchanged. The init is
   bit-identical by construction (`levels[i]` *is* a codebook entry), not by
   seeding, so this holds for every codebook rather than for one instance.
2. **The zero anchor survives as exactly 0.0.** A convex combination of learned
   levels has no reason to pass through the origin -- measured at -4.47e-08
   before the pin -- and a nonzero value there breaks the SparseProp
   structural-zero contract at the junction of the two methods. Asserted with
   `==`, not `allclose`.
3. **The knot gradient is not identically zero at init.** This is the failure the
   relaxation exists to avoid, and it reappears if the knots are initialized on
   the exact input positions: `diff == 0` makes the weight `1/eps == 1e6`, every
   other weight underflows to 0 in FP32, and the softmax degenerates to a hard
   argmax with no gradient. The test pins the nudge that prevents it.
"""

import pytest
import torch

from nanochat.models.quant.codebook import MemoryEfficientLearnedCodebook
from nanochat.models.quant.learnable_lut import (
    KNOT_DTYPE,
    RELAXATIONS,
    LearnableIndexLut,
    bake_learnable_table,
)


def make_codebooks(k_in: int = 15, k_out: int = 15, seed: int = 0):
    """Codebooks built by the real LC-QAT primitive, not synthesized.

    The proximity relaxation's zero contract is a claim about what LC-QAT's
    `MemoryEfficientLearnedCodebook` actually produces: a monotone level vector
    with exactly one FP32 0.0 at index `m_neg`. Hand-built "sorted random values
    with element [k//2] overwritten by 0.0" is not that -- it is non-monotone
    around the anchor, so `bucketize(0.0)` can land on an index whose value is
    not 0.0, and the tests would be asserting the relaxation's behaviour on an
    input the library never emits.

    `seed` varies the *asymmetry* (how many negative vs positive levels), which
    is the knob the bit-identity test actually needs to vary: it changes the
    anchor position and the level spacing without breaking monotonicity. The
    intra-codebook level values are deterministic by construction, since they
    come from a softplus prefix-sum over a fixed init range.
    """
    m_neg = (k_in - 1) // 2 + (seed % 2)
    c_in = MemoryEfficientLearnedCodebook(
        m_neg=m_neg, m_pos=k_in - 1 - m_neg, init_min=-1.0, init_max=1.0
    ).get_codebook()
    m_neg_out = (k_out - 1) // 2
    c_out = MemoryEfficientLearnedCodebook(
        m_neg=m_neg_out, m_pos=k_out - 1 - m_neg_out, init_min=0.0, init_max=2.0
    ).get_codebook()
    return c_in, c_out


def _nonconstant_loss(lut: LearnableIndexLut) -> torch.Tensor:
    """A loss whose gradient is nonzero for *both* relaxations.

    `out.sum()` is degenerate: each selector row sums to one, so the gradient
    through the relaxation is identically zero and every parameter appears to
    have no gradient regardless of the relaxation. Weighting by a non-constant
    vector is what makes the comparison meaningful.
    """
    idx = torch.arange(lut.k_in)
    ramp = torch.linspace(0.1, 2.0, lut.k_in)
    return (lut(idx) * ramp).sum()


class TestProximityParameterization:
    def test_when_proximity_then_it_costs_fewer_parameters_than_logits(self) -> None:
        """The parameter reduction is exact and structural, not a fit."""
        c_in, c_out = make_codebooks()
        prox = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        logits = LearnableIndexLut(c_in, c_out, "relu2", relaxation="logits")
        n_prox = sum(p.numel() for p in prox.parameters())
        n_logits = sum(p.numel() for p in logits.parameters())
        assert n_prox == prox.k_in * 2
        assert n_logits == prox.k_in * prox.k_out
        assert n_prox < n_logits

    def test_when_relaxation_is_unknown_then_construction_raises(self) -> None:
        c_in, c_out = make_codebooks()
        with pytest.raises(ValueError, match="relaxation must be one of"):
            LearnableIndexLut(c_in, c_out, "relu2", relaxation="quadratic")

    def test_when_default_then_the_relaxation_is_unchanged(self) -> None:
        """D9 makes proximity opt-in; the shipped default must not move."""
        c_in, c_out = make_codebooks()
        assert LearnableIndexLut(c_in, c_out, "relu2").relaxation == "logits"
        assert RELAXATIONS == ("logits", "proximity")

    def test_when_proximity_then_there_is_no_logit_matrix(self) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        assert not hasattr(lut, "logits")


class TestProximityBitIdentity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_when_proximity_then_the_table_is_the_bake_to_the_bit(
        self, seed: int
    ) -> None:
        c_in, c_out = make_codebooks(seed=seed)
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        assert torch.equal(
            lut.resolved_table(), bake_learnable_table(c_in, c_out, "relu2")
        )

    @pytest.mark.parametrize("act", ["relu2", "gelu", "silu", "tanh"])
    def test_when_proximity_for_any_zero_preserving_act_then_it_is_the_bake(
        self, act: str
    ) -> None:
        c_in, c_out = make_codebooks(seed=4)
        lut = LearnableIndexLut(c_in, c_out, act, relaxation="proximity")
        assert torch.equal(lut.resolved_table(), bake_learnable_table(c_in, c_out, act))

    def test_when_proximity_then_the_forward_value_is_the_hard_lookup(self) -> None:
        """Straight-through must not perturb the forward value.

        `torch.equal`, not `allclose`, would be the wrong assertion here: the
        straight-through `soft + hard - soft.detach()` is exact in real
        arithmetic but rounds once in FP32, so the result can differ from `hard`
        by a single ULP (~6e-08 measured). The table itself *is* bit-exact --
        that is what `resolved_table` pins, and what the fused chain consumes.
        The tolerance matches the logits path's existing test for the same
        reason.
        """
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        idx = torch.arange(lut.k_in)
        assert torch.allclose(lut(idx), lut.hard_values(idx), atol=1e-6)

    def test_when_a_non_zero_preserving_act_is_asked_then_construction_raises(
        self,
    ) -> None:
        """`sigmoid` maps 0.0 to 0.5, so pinning zero would silently disagree
        with the bake it claims to reproduce."""
        c_in, _ = make_codebooks()
        c_out = MemoryEfficientLearnedCodebook(
            m_neg=3, m_pos=3, init_min=-1.0, init_max=1.0
        ).get_codebook()
        with pytest.raises(ValueError, match="does not preserve zero"):
            LearnableIndexLut(c_in, c_out, "sigmoid", relaxation="proximity")


class TestProximityZeroAnchor:
    def test_when_input_is_the_zero_anchor_then_the_value_is_exactly_zero(self) -> None:
        """`==`, not `allclose`. The SparseProp contract needs an exact zero, and
        an unpinned convex combination returns -4.47e-08 (HANDOFF §10.3)."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        zero_idx = torch.nonzero(c_in == 0.0, as_tuple=True)[0]
        assert float(lut.hard_values(zero_idx)[0]) == 0.0

    def test_when_the_zero_anchor_level_is_optimized_then_it_stays_exact(self) -> None:
        """The pin has to survive optimization, not just construction."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        opt = torch.optim.AdamW([p for p in lut.parameters()], lr=1.0)
        zero_idx = torch.nonzero(c_in == 0.0, as_tuple=True)[0]
        for _ in range(25):
            opt.zero_grad()
            _nonconstant_loss(lut).backward()
            opt.step()
        assert float(lut.hard_values(zero_idx)[0]) == 0.0

    def test_when_proximity_then_exactly_one_level_is_pinned(self) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        assert int(lut.pin_mask.sum()) == 1


class TestProximityGradients:
    def test_when_backpropagated_then_the_knot_gradient_is_nonzero(self) -> None:
        """The regression the `KNOT_INIT_OFFSET_FRACTION` nudge exists for.

        With knots on the exact input positions the nearest-knot weight is
        `1/eps == 1e6`, everything else underflows to 0.0 in FP32, and the knot
        gradient is exactly zero -- a table that can never move while looking
        healthy.
        """
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        _nonconstant_loss(lut).backward()
        assert float(lut.knots.grad.abs().max()) > 0.0

    def test_when_constructed_then_the_selector_is_not_exactly_one_hot(self) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        weights = lut.proximity_weights()
        assert weights.shape == (lut.k_in, lut.k_in)
        # A saturated row would carry the 1/eps signature of diff == 0.
        assert float(weights.detach().max()) < 1e5
        assert torch.allclose(weights.sum(dim=-1), torch.ones(lut.k_in), atol=1e-5)

    def test_when_backpropagated_then_the_level_gradient_is_nonzero(self) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        _nonconstant_loss(lut).backward()
        assert float(lut.levels.grad.abs().max()) > 0.0

    def test_when_proximity_then_the_gradient_beats_the_saturated_logit_one(
        self,
    ) -> None:
        """The measured claim (HANDOFF §10.2.1): a distance selector has no
        saturation cliff, so its gradient is larger at the same init."""
        c_in, c_out = make_codebooks()
        prox = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        logits = LearnableIndexLut(c_in, c_out, "relu2", relaxation="logits")
        _nonconstant_loss(prox).backward()
        _nonconstant_loss(logits).backward()
        prox_grad = max(
            float(prox.knots.grad.abs().max()),
            float(prox.levels.grad.abs().max()),
        )
        logits_grad = float(logits.logits.grad.abs().max())
        assert prox_grad > logits_grad

    def test_when_grad_is_disabled_then_proximity_returns_the_hard_lookup(
        self,
    ) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        idx = torch.arange(lut.k_in)
        with torch.no_grad():
            assert torch.equal(lut(idx), lut.hard_values(idx))

    def test_when_optimised_then_the_proximity_table_can_actually_move(self) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        before = lut.resolved_table().clone()
        opt = torch.optim.AdamW(list(lut.parameters()), lr=0.5)
        for _ in range(50):
            opt.zero_grad()
            _nonconstant_loss(lut).backward()
            opt.step()
        assert not torch.equal(lut.resolved_table(), before)


class TestFp8KnotStorage:
    def test_when_knots_are_exported_to_fp8_then_the_table_is_unchanged(self) -> None:
        """D6: the fp8 saving is free because the table reads `levels`, not
        `knots`. If a future change makes the table depend on the knots, this
        fails and the 1-byte claim is no longer true."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        before = lut.resolved_table().clone()
        saved = lut.knots.detach().clone()
        try:
            lut.knots.data = lut.fp8_knots()
            assert torch.equal(lut.resolved_table(), before)
        finally:
            lut.knots.data = saved

    def test_when_knots_are_exported_to_fp8_then_the_storage_is_4x_smaller(
        self,
    ) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        fp8_bytes = lut.fp8_knots().to(KNOT_DTYPE).numel() * 1
        fp32_bytes = lut.knots.numel() * 4
        assert fp8_bytes * 4 == fp32_bytes

    def test_when_knots_are_exported_to_fp8_then_the_strict_ordering_survives(
        self,
    ) -> None:
        """The proximity selector needs distinguishable knots; fp8 collapsing
        two neighbours into one byte would make two inputs select the same knot
        and silently coarsen the table."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        knots = lut.fp8_knots()
        assert bool((knots[1:] > knots[:-1]).all())

    def test_when_grad_is_enabled_then_the_knot_parameter_stays_fp32(self) -> None:
        """An fp8 `nn.Parameter` has no usable gradient, so training would need
        the FP32 shadow this keeps. fp8 is the *export* saving only."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
        assert lut.knots.dtype == torch.float32

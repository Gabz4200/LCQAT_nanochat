"""Tests for the learnable activation LUT (W4).

Two properties matter more than anything else here:

1. The table is *bit-identical* to the frozen `compile_activation_lut` bake at
   initialisation. Without this, adopting the learnable table would be a
   behaviour change at step 0 rather than a strict superset.
2. The gradient actually reaches the logits. A straight-through relaxation
   whose seed logits are saturated in FP32 has an identically-zero gradient and
   trains nothing while looking healthy -- that is the failure this file pins.
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.lcqat.ablation import LearnableActivationLut
from nanochat.lcqat.learnable_lut import (
    DEFAULT_TEMPERATURE,
    LOGIT_SEED_STRENGTH,
    LearnableIndexLut,
    bake_learnable_table,
)
from nanochat.lcqat.lut import compile_activation_lut, get_activation


def make_codebooks(k_in: int = 15, k_out: int = 15, seed: int = 0):
    """Codebooks with the mandatory exact-zero anchor at a known index."""
    gen = torch.Generator().manual_seed(seed)
    c_in = torch.sort(torch.randn(k_in, generator=gen)).values
    c_in[k_in // 2] = 0.0
    c_out = torch.sort(torch.rand(k_out, generator=gen) * 2.0).values
    c_out[k_out // 2] = 0.0
    return c_in, c_out


class TestLearnableIndexLutInit:
    def test_when_constructed_then_the_table_equals_the_frozen_bake(self) -> None:
        """Adopting the learnable table must not change what step 0 computes."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        assert torch.equal(
            lut.resolved_table(), bake_learnable_table(c_in, c_out, "relu2")
        )

    def test_when_constructed_then_the_table_matches_compile_activation_lut(
        self,
    ) -> None:
        """The bake is `compile_activation_lut`, not a reimplementation of it."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        reference = compile_activation_lut(get_activation("relu2"), c_in, c_out).long()
        assert torch.equal(lut.resolved_table(), reference)

    @pytest.mark.parametrize("act", ["relu2", "gelu", "silu", "tanh"])
    def test_when_constructed_for_any_activation_then_it_equals_that_bake(
        self, act: str
    ) -> None:
        c_in, c_out = make_codebooks(seed=3)
        lut = LearnableIndexLut(c_in, c_out, act)
        expected = compile_activation_lut(get_activation(act), c_in, c_out).long()
        assert torch.equal(lut.resolved_table(), expected)

    def test_when_frozen_then_logits_are_a_buffer_not_a_parameter(self) -> None:
        """`freeze=True` reproduces the current frozen behaviour exactly."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2", freeze=True)
        names = {name for name, _ in lut.named_parameters()}
        assert names == set()
        assert "logits" in dict(lut.named_buffers())

    def test_when_codebooks_are_not_1d_then_construction_raises(self) -> None:
        c_in, c_out = make_codebooks()
        with pytest.raises(ValueError, match="1-D"):
            LearnableIndexLut(c_in.unsqueeze(0), c_out, "relu2")

    def test_when_temperature_is_not_positive_then_construction_raises(self) -> None:
        c_in, c_out = make_codebooks()
        with pytest.raises(ValueError, match="temperature must be positive"):
            LearnableIndexLut(c_in, c_out, "relu2", temperature=0.0)


class TestLearnableIndexLutValues:
    def test_when_forwarded_then_the_value_is_the_hard_lookup(self) -> None:
        """Straight-through must not perturb the forward value."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        idx = torch.randint(0, c_in.numel(), (3, 7))
        assert torch.allclose(lut(idx), lut.hard_values(idx), atol=1e-6)

    def test_when_forwarded_then_the_value_equals_codebook_bake_of_the_index(
        self,
    ) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        idx = torch.randint(0, c_in.numel(), (3, 7))
        table = bake_learnable_table(c_in, c_out, "relu2")
        assert torch.allclose(lut(idx), c_out[table[idx]], atol=1e-6)

    def test_when_forwarded_then_shape_is_preserved(self) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        for shape in [(5,), (2, 3), (2, 3, 4)]:
            idx = torch.randint(0, c_in.numel(), shape)
            assert lut(idx).shape == idx.shape

    def test_when_grad_is_disabled_then_the_value_is_the_hard_lookup(self) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        idx = torch.randint(0, c_in.numel(), (3, 7))
        with torch.no_grad():
            out = lut(idx)
        assert torch.allclose(out, c_out[lut.resolved_table()[idx]], atol=1e-6)


class TestLearnableIndexLutGradients:
    """The gradient must be non-zero. A saturated softmax trains nothing."""

    def test_when_backpropagated_then_the_logit_gradient_is_nonzero(self) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        idx = torch.randint(0, c_in.numel(), (3, 7))
        weights = torch.randn(3, 7)
        (lut(idx) * weights).sum().backward()
        assert lut.logits.grad is not None
        assert float(lut.logits.grad.abs().sum()) > 0.0

    def test_when_constructed_then_the_seed_softmax_is_not_saturated(self) -> None:
        """Guards the FP32 underflow cliff.

        A seed gap large in absolute logits makes `exp(-gap / T)` underflow to
        exactly 0, so the softmax gradient is identically 0 and the table can
        never train -- while every value-level check still passes.
        """
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        table = lut.resolved_table()
        weights = torch.softmax(lut.logits.detach() / lut.temperature, dim=-1)
        mass = weights.gather(1, table.unsqueeze(1)).mean()
        # Peaked enough to start near the hard value, loose enough to have slope.
        assert 0.9 < float(mass) < 0.999
        assert LOGIT_SEED_STRENGTH * lut.temperature < 20.0

    def test_when_backpropagated_then_the_table_does_not_change_on_the_same_step(
        self,
    ) -> None:
        """Straight-through: the gradient exists but the forward value is hard.

        A table that changed on the step would mean the relaxation leaked into
        the forward pass, which would silently break parity with the frozen
        `quantized_mlp_chain` gather.
        """
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        before = lut.resolved_table().clone()
        idx = torch.randint(0, c_in.numel(), (3, 7))
        (lut(idx) * torch.randn(3, 7)).sum().backward()
        assert torch.equal(lut.resolved_table().detach(), before)

    def test_when_optimised_then_the_table_can_actually_move(self) -> None:
        """End-to-end: a trained table must be able to leave its initialisation."""
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        before = lut.resolved_table().clone()
        idx = torch.randint(0, c_in.numel(), (4, 8))
        weights = torch.randn(4, 8)
        opt = torch.optim.SGD(lut.parameters(), lr=5.0)
        for _ in range(30):
            opt.zero_grad()
            (lut(idx) * weights).sum().backward()
            opt.step()
        assert not torch.equal(lut.resolved_table().detach(), before)

    def test_when_the_default_temperature_is_used_then_it_is_the_documented_one(
        self,
    ) -> None:
        c_in, c_out = make_codebooks()
        lut = LearnableIndexLut(c_in, c_out, "relu2")
        assert lut.temperature == DEFAULT_TEMPERATURE


class TestActivationLutFromCallable:
    def test_when_fitted_exactly_then_it_reproduces_the_callable_at_every_knot(
        self,
    ) -> None:
        """`fit="exact"` is exact *at the knots* -- that is its whole claim."""
        relu2 = lambda t: F.relu(t).square()
        lut = LearnableActivationLut.from_callable(
            relu2, low=-2.0, high=4.0, n_points=9, fit="exact"
        )
        knots = lut.codebook_x().detach()
        assert torch.allclose(lut(knots), relu2(knots), atol=1e-6)

    def test_when_fitted_least_squares_then_between_knot_error_beats_exact_fit(
        self,
    ) -> None:
        """The point of `lsq`: minimise error *between* knots, not only at them.

        With a coarse grid, preserving knot values exactly is not the best use
        of the degrees of freedom.
        """
        relu2 = lambda t: F.relu(t).square()
        samples = torch.linspace(-2.0, 4.0, 257)[2:-2]
        target = relu2(samples)
        exact = LearnableActivationLut.from_callable(
            relu2, low=-2.0, high=4.0, n_points=9, fit="exact"
        )
        lsq = LearnableActivationLut.from_callable(
            relu2, low=-2.0, high=4.0, n_points=9, fit="lsq"
        )
        with torch.no_grad():
            exact_err = float((exact(samples) - target).abs().max())
            lsq_err = float((lsq(samples) - target).abs().max())
        assert lsq_err < exact_err

    def test_when_fitted_then_it_is_between_the_baked_and_the_target(self) -> None:
        """It approximates the closed form; it does not equal it everywhere."""
        relu2 = lambda t: F.relu(t).square()
        lut = LearnableActivationLut.from_callable(
            relu2, low=-2.0, high=4.0, n_points=9, fit="exact"
        )
        samples = torch.linspace(-2.0, 4.0, 257)[2:-2]
        with torch.no_grad():
            err = float((lut(samples) - relu2(samples)).abs().max())
        assert err > 0.0

    def test_when_fitted_then_gradients_reach_the_y_knots(self) -> None:
        relu2 = lambda t: F.relu(t).square()
        lut = LearnableActivationLut.from_callable(
            relu2, low=-2.0, high=4.0, n_points=9
        )
        lut(torch.linspace(-2.0, 4.0, 33)).sum().backward()
        assert lut.codebook_y.grad is not None
        assert float(lut.codebook_y.grad.abs().sum()) > 0.0

    @pytest.mark.parametrize("bad", [1.0, 2.0])
    def test_when_the_domain_is_empty_then_construction_raises(
        self, bad: float
    ) -> None:
        with pytest.raises(ValueError, match="require low < high"):
            LearnableActivationLut.from_callable(
                torch.relu, low=bad, high=bad - 1.0, n_points=4
            )

    def test_when_n_points_is_too_small_then_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="n_points"):
            LearnableActivationLut.from_callable(
                torch.relu, low=0.0, high=1.0, n_points=1
            )

    def test_when_fit_is_unknown_then_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="fit must be"):
            LearnableActivationLut.from_callable(
                torch.relu,
                low=0.0,
                high=1.0,
                n_points=4,
                fit="cubic",  # type: ignore[arg-type]
            )

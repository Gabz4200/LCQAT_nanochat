"""Tests for the classic-QAT front stage in the ablation reference (D14r).

The change under test is an *insertion*, not a replacement:
`QuantizedLinear.forward` became

    weight -> ClassicUniformFakeQuant -> LearnedLookup -> F.linear

where it used to be `weight -> LearnedLookup -> F.linear`. The reason is a
train/inference mismatch: the lookup saw continuous weights in training and a
quantized weight distribution at inference, so its correction was fitted to an
input it would never be given.

What has to hold, and why:

1. **`LearnedLookup` is still there.** This was an insertion; if a "cleanup"
   later drops the lookup because the front stage quantizes, the correction --
   the entire point -- is gone.
2. **The default is unchanged.** `lut_front="codebook"` reproduces the original
   path, so no existing result moves.
3. **The gradient reaches the shadow weight through both stages.** One STE plus
   one smooth map, not two STEs in series.
4. **Exact zero survives the front stage for free.** `round(0) == 0` and
   `0 * scale == 0.0`, so the SparseProp structural-zero contract holds here
   without a pin -- the one place in this stack where it is automatic.
"""

import pytest
import torch

from nanochat.lcqat.reference.simple import (
    LUT_FRONTS,
    ClassicUniformFakeQuant,
    LearnedLookup,
    QuantizedLinear,
)


class TestClassicUniformFakeQuant:
    def test_when_the_input_is_zero_then_the_output_is_exactly_zero(self) -> None:
        """`==`, not `allclose`. The SparseProp contract needs an exact zero, and
        here it is automatic: `round(0) == 0` and `0 * scale == 0.0` for any K."""
        fq = ClassicUniformFakeQuant(entries=255)
        assert float(fq(torch.zeros(8))[0]) == 0.0

    def test_when_the_tensor_is_all_zero_then_it_does_not_divide_by_zero(self) -> None:
        """nanochat zero-initializes `attn.c_proj` / `mlp.c_proj`, so an all-zero
        weight is a real input, not a hypothetical one."""
        fq = ClassicUniformFakeQuant(entries=255)
        assert float(fq(torch.zeros(8)).abs().max()) == 0.0

    def test_when_the_grid_is_symmetric_then_it_spans_the_tensor(self) -> None:
        fq = ClassicUniformFakeQuant(entries=255)
        x = torch.randn(1000)
        scale = float(fq.scale_for(x))
        assert scale > 0.0
        # The scale is chosen so the extreme value lands on the top level.
        assert float(x.abs().max() / scale) == pytest.approx(float(fq.levels), abs=1e-3)

    def test_when_backpropagated_then_the_gradient_reaches_the_input(self) -> None:
        """STE identity: a step function of the input still passes the gradient
        through unattenuated, which is what lets the shadow weight train."""
        fq = ClassicUniformFakeQuant(entries=255)
        x = torch.randn(64, requires_grad=True)
        fq(x).sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_when_the_grid_is_too_small_then_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="entries must be >= 3"):
            ClassicUniformFakeQuant(entries=2)

    def test_when_constructed_then_it_holds_no_parameters(self) -> None:
        """The scale is read off the tensor, not learned. A learnable scale would
        be a second grid to tune and would confound the comparison this stage
        exists to make."""
        assert list(ClassicUniformFakeQuant(entries=255).parameters()) == []

    def test_when_the_inputs_are_already_on_the_grid_then_it_is_a_no_op(self) -> None:
        """Sanity on the round trip: a value already on a grid point comes back
        unchanged, so the stage is not adding error of its own."""
        fq = ClassicUniformFakeQuant(entries=255)
        x = torch.randn(4096)
        scale = fq.scale_for(x)
        on_grid = torch.round(x / scale).clamp(-fq.levels, fq.levels) * scale
        assert torch.allclose(fq(on_grid), on_grid, atol=1e-6)


class TestQuantizedLinearFrontStage:
    def test_when_classic_then_the_lookup_is_still_in_the_chain(self) -> None:
        """The insertion must not have replaced the lookup -- it is the
        correction, and losing it would make the front stage pointless."""
        layer = QuantizedLinear(8, 4, lut_front="classic")
        assert isinstance(layer.quantize_weight, LearnedLookup)
        assert isinstance(layer.fake_quant, ClassicUniformFakeQuant)

    def test_when_codebook_then_no_front_stage_is_installed(self) -> None:
        layer = QuantizedLinear(8, 4, lut_front="codebook")
        assert layer.fake_quant is None
        assert isinstance(layer.quantize_weight, LearnedLookup)

    def test_when_default_then_the_front_is_absent(self) -> None:
        """The default has to reproduce the original behaviour, or every existing
        measurement moves."""
        assert QuantizedLinear(8, 4).lut_front == "codebook"
        assert LUT_FRONTS == ("codebook", "classic")

    def test_when_the_front_is_unknown_then_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="lut_front must be one of"):
            QuantizedLinear(8, 4, lut_front="lsq")

    def test_when_the_two_modes_are_compared_then_they_differ(self) -> None:
        """The A/B has to be a real difference; identical outputs would mean the
        front stage is inert."""
        x = torch.randn(2, 8)
        torch.manual_seed(0)
        codebook = QuantizedLinear(8, 4, lut_front="codebook")
        torch.manual_seed(0)
        classic = QuantizedLinear(8, 4, lut_front="classic")
        assert not torch.equal(codebook(x), classic(x))

    def test_when_both_modes_backpropagate_then_the_weight_gradient_is_live(
        self,
    ) -> None:
        x = torch.randn(2, 8)
        for front in LUT_FRONTS:
            torch.manual_seed(0)
            layer = QuantizedLinear(8, 4, lut_front=front)
            layer(x).sum().backward()
            assert layer.weight.grad is not None, front
            assert float(layer.weight.grad.abs().max()) > 0.0, front

    def test_when_both_modes_backpropagate_then_the_lookup_still_trains(self) -> None:
        """The STE must not cut the lookup off: the correction is a smooth map
        *after* the STE, so it receives a real gradient."""
        x = torch.randn(2, 8)
        for front in LUT_FRONTS:
            torch.manual_seed(0)
            layer = QuantizedLinear(8, 4, lut_front=front)
            layer(x).sum().backward()
            assert layer.quantize_weight.outputs.grad is not None, front
            assert float(layer.quantize_weight.outputs.grad.abs().max()) > 0.0, front

    def test_when_bias_quantization_is_off_then_bias_is_consumed_raw(self) -> None:
        """Bias quantization is the one genuine capability gap here (nothing in
        `LCQATLinear` quantizes bias), so it stays opt-in rather than changing
        the default layer's behaviour."""
        x = torch.randn(2, 8)
        torch.manual_seed(0)
        off = QuantizedLinear(8, 4, quantize_bias=False)
        torch.manual_seed(0)
        on = QuantizedLinear(8, 4, quantize_bias=True)
        assert not torch.equal(off(x), on(x))

    def test_when_bias_quantization_is_on_then_bias_trains(self) -> None:
        torch.manual_seed(0)
        layer = QuantizedLinear(8, 4, lut_front="classic", quantize_bias=True)
        layer(torch.randn(2, 8)).sum().backward()
        assert layer.quantize_bias.outputs.grad is not None

    def test_when_there_is_no_bias_then_no_lookup_is_built(self) -> None:
        layer = QuantizedLinear(8, 4, bias=False)
        assert layer.bias is None
        assert layer.quantize_bias is None


class TestReferenceLookupStorage:
    def test_when_constructed_then_the_knot_parameter_is_fp32(self) -> None:
        """An fp8 `nn.Parameter` has no usable gradient, so training needs the
        FP32 shadow and fp8 is the export saving only. Constructing directly in
        fp8 also fails outright: `torch.linspace` has no fp8 CPU kernel."""
        lookup = LearnedLookup(entries=15)
        assert lookup.index.dtype == torch.float32

    def test_when_exported_then_the_knots_take_the_fp8_dtype(self) -> None:
        lookup = LearnedLookup(entries=15, index_dtype=torch.float8_e4m3fn)
        assert lookup.index_fp8().dtype == torch.float8_e4m3fn

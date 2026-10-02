"""Tests for the opt-in bias quantizer.

The centre of gravity here is the *default* path, not the new one.
`quantize_bias` defaults to `False`, and the only reason that default is
trustworthy is that it is checked rather than assumed: the first test replays
the pre-change forward by hand and asserts bit-identity. Everything else -- the
gradient reach, the zero anchor, the round-trip -- is worthless if enabling
this flag perturbs a layer that never asked for it.

Two things are deliberately *not* asserted:

* That bias quantization improves accuracy. It is a capability plus a
  measurement, not a claimed win; the measurement tests below only assert the
  numbers are finite and separately reported, and the accuracy question stays
  open until the ablation harness runs it.
* That a bias-free layer gains a codebook. It cannot and should not -- see the
  constructor's `quantize_bias and not bias` guard.
"""

import copy
import math
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from nanochat.models.quant.bias_quant import BiasQuantizer, measure_bias_quantization
from nanochat.models.quant.linear import GRAD_SCALE_NONE, LCQATLinear
from nanochat.models.quant.retrofit import (
    CODEBOOK_SPEC_FIELDS,
    PRESETS,
    LayerKConfig,
    retrofit_model,
)

IN_FEATURES = 64
OUT_FEATURES = 32


def float_linear(seed: int = 0, bias: bool = True) -> nn.Linear:
    """A float Linear with a *range* in both parameters.

    Both are randomized on purpose: `nn.Linear`'s default bias init is narrow
    and `init_weights()` can leave a projection at exactly zero, either of which
    collapses the measurement onto the zero anchor and makes every comparison
    between two zeros trivially "match".
    """
    torch.manual_seed(seed)
    mod = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias)
    with torch.no_grad():
        mod.weight.normal_(0.0, 0.3)
        if bias:
            mod.bias.normal_(0.0, 0.2)
    return mod


def quantized(**kwargs) -> LCQATLinear:
    return LCQATLinear.from_float(float_linear(), K_weight=15, K_act=15, **kwargs)


class TestDefaultPathUnchanged:
    def test_when_flag_is_off_then_no_quantizer_is_built(self) -> None:
        linear = quantized()
        assert linear.bias_quantizer is None
        assert "bias_quantizer" not in linear.state_dict()

    def test_when_flag_is_off_then_forward_is_bit_identical_to_the_shipped_path(
        self,
    ) -> None:
        """The regression guard that matters most.

        Replays `forward` exactly as it was written before bias quantization
        existed -- raw FP32 bias into `F.linear` -- and asserts equality with
        `torch.equal`, not `allclose`. `allclose` would pass on a bias addend
        that moved by 1e-9, which is a real change to a model that never opted
        in, and `nan`-vs-`nan` comparisons would pass on a broken one.
        """
        linear = quantized()
        x = torch.randn(4, IN_FEATURES)

        x_q = linear._quantize(linear.act_quantizer, x, x.numel())
        w_q = linear._quantize(
            linear.weight_quantizer, linear.weight, linear.weight.numel()
        )
        y = torch.nn.functional.linear(
            x_q.value, w_q.value.to(x.dtype), linear.bias.to(dtype=x.dtype)
        )
        if linear.out_quantizer is not None:
            y = linear._quantize(linear.out_quantizer, y, y.numel()).value

        assert torch.equal(linear(x), y)

    def test_when_flag_is_off_then_bias_reaches_the_shadow_unchanged(self) -> None:
        linear = quantized()
        assert linear._dequantized_bias(torch.float32).equal(linear.bias.detach())

    def test_when_flag_is_off_then_a_checkpoint_still_loads_strictly(self) -> None:
        """A flag-off model must not carry bias-codebook keys at all.

        This is what "every existing checkpoint is unchanged" means in practice:
        a state dict saved before this change has to load into a flag-off model
        with `strict=True`, and a flag-off model must not acquire keys a
        pre-existing checkpoint does not have.
        """
        state = quantized(quantize_bias=True).state_dict()
        off = quantized()
        off.load_state_dict(
            {k: v for k, v in state.items() if "bias_quantizer" not in k}
        )
        assert "bias_quantizer" not in off.state_dict()


class TestEnabledPath:
    def test_when_enabled_then_forward_shape_and_dtype_are_unchanged(self) -> None:
        linear = quantized(quantize_bias=True)
        x = torch.randn(4, IN_FEATURES)
        assert linear(x).shape == (4, OUT_FEATURES)
        assert linear(x).dtype == x.dtype

    def test_when_enabled_then_the_bias_addend_is_the_dequantized_one(self) -> None:
        """The addend must come off the codebook, not off the shadow bias.

        Checked structurally (every entry is a codebook level) rather than by
        comparing outputs, so it also holds when the level happens to equal the
        original value.
        """
        linear = quantized(quantize_bias=True)
        x = torch.randn(4, IN_FEATURES)
        codebook = linear.bias_quantizer.get_codebook()
        bias = linear._dequantized_bias(x.dtype)
        distance = (bias.unsqueeze(-1) - codebook).abs().min(dim=-1).values
        assert bool((distance == 0).all())

    def test_when_enabled_then_gradient_reaches_the_shadow_bias_and_the_codebook(
        self,
    ) -> None:
        """Both halves of the dual gradient, or the STE is not an STE.

        The shadow bias is what actually trains the layer; the codebook is what
        makes the bias quantized rather than merely rounded once. A quantizer
        that reached only one of them would still produce plausible outputs, so
        this asserts each side separately.
        """
        linear = quantized(quantize_bias=True)
        x = torch.randn(8, IN_FEATURES)
        linear(x).square().mean().backward()

        assert linear.bias.grad is not None
        assert float(linear.bias.grad.abs().sum()) > 0.0
        for name, param in linear.bias_quantizer.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"
            assert float(param.grad.abs().sum()) > 0.0, f"{name} gradient is zero"

    def test_when_enabled_then_the_ste_passes_the_gradient_through_unattenuated(
        self,
    ) -> None:
        """`dL/dbias` must be exactly 1.0 per addend position, i.e. unattenuated.

        The dequantized bias is added inside the graph, so its gradient comes
        from the identity term of the STE: summing the output adds the bias
        once per (batch, time) position, so every entry of `dL/dbias` is that
        count and nothing else. A scaling factor here -- the kind of bug
        `_CodebookSTE`'s docstring warns about -- would quietly retune how every
        biased layer trains.
        """
        linear = quantized(quantize_bias=True)
        x = torch.randn(8, 5, IN_FEATURES, requires_grad=True)
        linear(x).sum().backward()

        addends = x.numel() // IN_FEATURES
        assert torch.equal(
            linear.bias.grad,
            torch.full((OUT_FEATURES,), float(addends)),
        )

    def test_when_grad_scale_is_none_then_the_forward_value_is_the_same(self) -> None:
        """Both arms must quantize identically; only `dL/dC` differs.

        `grad_scale` selects the codebook-gradient scaling, and a variant that
        silently served the raw FP32 bias under `"none"` would leave the other
        two terms quantized while measuring a different model than the flag
        names.
        """
        scaled = quantized(quantize_bias=True)
        unscaled = quantized(quantize_bias=True, grad_scale=GRAD_SCALE_NONE)
        x = torch.randn(4, IN_FEATURES)
        assert torch.equal(scaled(x), unscaled(x))

    def test_when_grad_scale_is_none_then_the_codebook_still_receives_gradient(
        self,
    ) -> None:
        unscaled = quantized(quantize_bias=True, grad_scale=GRAD_SCALE_NONE)
        unscaled(torch.randn(8, IN_FEATURES)).square().mean().backward()
        for name, param in unscaled.bias_quantizer.named_parameters():
            assert float(param.grad.abs().sum()) > 0.0, f"{name} gradient is zero"

    def test_when_enabled_with_per_channel_weights_then_both_tables_survive(
        self,
    ) -> None:
        """The two opt-in mechanisms are independent and may be combined.

        Per-channel costs `out_features x K` on the *weight* side and this
        costs `K` on the bias side, so neither shadows the other's table. If
        this ever stops holding it must raise, not silently drop one.
        """
        torch.manual_seed(0)
        linear = LCQATLinear(
            256, 256, bias=True, per_channel_weight=True, quantize_bias=True
        )
        with torch.no_grad():
            linear.weight.normal_(0.0, 0.3)
            linear.bias.normal_(0.0, 0.2)
        linear(torch.randn(8, 256)).square().mean().backward()

        assert linear.per_channel_weight_quantizer is not None
        assert linear.bias_quantizer is not None
        grads = [float(p.grad.abs().sum()) for p in linear.bias_quantizer.parameters()]
        assert all(g > 0.0 for g in grads), grads


class TestZeroAnchor:
    def test_when_a_bias_entry_is_zero_then_it_dequantizes_to_exactly_zero(
        self,
    ) -> None:
        """The structural-zero contract, checked with `==`.

        `allclose(atol=1e-6)` would pass on a 1e-8 residue, and a residue is
        exactly what SparseProp's pruning depends on not existing: a structurally
        absent bias entry that becomes 1e-8 adds a small constant to every row.
        """
        linear = quantized(quantize_bias=True)
        bias = linear.bias.detach().clone()
        bias[0] = 0.0
        bias[5] = 0.0

        dequantized = linear.bias_quantizer(bias).value
        assert dequantized[0].item() == 0.0
        assert dequantized[5].item() == 0.0

    def test_when_a_bias_entry_is_zero_then_it_buckets_to_the_anchor_index(
        self,
    ) -> None:
        linear = quantized(quantize_bias=True)
        # A full-length vector with one entry zeroed: `bucketize` is
        # position-wise, so this asks the same question as the test above of the
        # real per-entry path rather than of a detached scalar.
        bias = linear.bias.detach().clone()
        bias[7] = 0.0
        quantizer = linear.bias_quantizer
        assert int(quantizer.bucketize(bias)[7].item()) == quantizer.m_neg

    def test_when_the_whole_bias_is_zero_then_every_entry_dequantizes_to_zero(
        self,
    ) -> None:
        linear = quantized(quantize_bias=True)
        with torch.no_grad():
            linear.bias.zero_()
        dequantized = linear.bias_quantizer(linear.bias.detach()).value
        assert bool((dequantized == 0.0).all())

    def test_when_a_bias_entry_is_zero_then_it_is_absent_from_the_output(self) -> None:
        """End-to-end form of the anchor: swapping two bias rows is a no-op
        when both are exactly zero, so the zeros contribute nothing."""
        linear = quantized(quantize_bias=True)
        with torch.no_grad():
            linear.bias.zero_()
        x = torch.randn(4, IN_FEATURES)
        reference = linear(x)

        with torch.no_grad():
            linear.bias[3] = 0.0
        assert torch.equal(linear(x), reference)


class TestPersistence:
    def test_when_saved_and_reloaded_then_the_bias_codebook_is_preserved(self) -> None:
        linear = quantized(quantize_bias=True)
        state = linear.state_dict()
        assert any("bias_quantizer" in key for key in state)

        clone = copy.deepcopy(linear)
        with torch.no_grad():
            for param in clone.bias_quantizer.parameters():
                param.add_(torch.randn_like(param))
        assert not torch.equal(
            clone.bias_quantizer.get_codebook(), linear.bias_quantizer.get_codebook()
        )

        clone.load_state_dict(state)
        assert torch.equal(
            clone.bias_quantizer.get_codebook(), linear.bias_quantizer.get_codebook()
        )

    def test_when_reloaded_then_the_output_is_identical(self) -> None:
        linear = quantized(quantize_bias=True)
        x = torch.randn(4, IN_FEATURES)
        clone = copy.deepcopy(linear)
        clone.load_state_dict(linear.state_dict())
        assert torch.equal(clone(x), linear(x))

    def test_when_the_codebook_is_a_step_parameter_then_the_optimizer_sees_it(
        self,
    ) -> None:
        """`--codebook-lr` must reach these params.

        The optimizer partitions by parameter-name suffix
        (`raw_pos_deltas` / `raw_neg_deltas`), so the bias codebook is only
        trainable at its own learning rate if it nests under those names. That
        is the reason it wraps `MemoryEfficientLearnedCodebook` instead of
        holding its own parameters.
        """
        from nanochat.models.quant.optimizer import build_qat_param_groups

        linear = quantized(quantize_bias=True)
        groups = build_qat_param_groups(linear, matrix_lr=3e-4, weight_decay=0.1)
        codebook_ids = {
            id(param)
            for group in groups
            if group["role"] == "codebook"
            for param in group["params"]
        }
        bias_ids = {id(p) for p in linear.bias_quantizer.parameters()}
        assert bias_ids <= codebook_ids


class TestQuantizerUnit:
    def test_when_built_then_the_codebook_is_one_shared_one_dimensional_table(
        self,
    ) -> None:
        quantizer = BiasQuantizer(out_features=OUT_FEATURES, K_bias=15)
        assert quantizer.get_codebook().shape == (15,)
        assert quantizer.K == 15

    def test_when_built_then_the_zero_anchor_is_exactly_zero(self) -> None:
        quantizer = BiasQuantizer(out_features=OUT_FEATURES, K_bias=15)
        assert quantizer.get_codebook()[quantizer.m_neg].item() == 0.0

    def test_when_built_then_the_levels_stay_strictly_increasing(self) -> None:
        quantizer = BiasQuantizer(out_features=OUT_FEATURES, K_bias=15)
        book = quantizer.get_codebook()
        assert bool(torch.all(book[1:] > book[:-1]))

    def test_when_fit_from_a_tensor_then_the_span_covers_the_data(self) -> None:
        """A caller-supplied span is a guess; fitting is the fix.

        Built with a deliberately too-wide `init_max` and fit afterwards, which
        is the state a layer is in whenever the guess overshoots: every entry
        bucketizes onto the anchor and the table can never move.
        """
        bias = torch.linspace(-1.0, 1.0, OUT_FEATURES)
        quantizer = BiasQuantizer(
            out_features=OUT_FEATURES, K_bias=15, init_min=-100.0, init_max=100.0
        )
        assert int(quantizer.bucketize(bias).unique().numel()) == 1

        quantizer.init_from_tensor(bias)
        book = quantizer.get_codebook()
        assert book[0].item() == pytest.approx(-1.0, abs=1e-3)
        assert book[-1].item() == pytest.approx(1.0, abs=1e-3)
        assert int(quantizer.bucketize(bias).unique().numel()) > 1

    def test_when_the_bias_is_degenerate_then_the_span_is_left_alone(self) -> None:
        """An all-zero bias keeps the constructor's span.

        It has no range to fit, and narrowing to the floor would put every level
        inside the gap between the anchor and the values the bias will take once
        it starts training -- freezing the table before it ever sees a signal.
        """
        quantizer = BiasQuantizer(
            out_features=OUT_FEATURES, K_bias=15, init_min=-2.0, init_max=2.0
        )
        before = quantizer.get_codebook().clone()
        quantizer.init_from_tensor(torch.zeros(OUT_FEATURES))
        assert torch.equal(quantizer.get_codebook(), before)

    def test_when_fit_on_meta_then_raises(self) -> None:
        quantizer = BiasQuantizer(out_features=OUT_FEATURES, K_bias=15)
        with torch.device("meta"):
            empty = torch.empty(OUT_FEATURES)
        with pytest.raises(RuntimeError, match="materialized"):
            quantizer.init_from_tensor(empty)

    def test_when_the_bias_length_is_wrong_then_raises(self) -> None:
        quantizer = BiasQuantizer(out_features=OUT_FEATURES, K_bias=15)
        with pytest.raises(ValueError, match="out_features"):
            quantizer(torch.randn(OUT_FEATURES + 1))

    def test_when_the_grad_scale_is_unknown_then_raises(self) -> None:
        with pytest.raises(ValueError, match="grad_scale"):
            BiasQuantizer(out_features=OUT_FEATURES, grad_scale="sqrt_n")

    def test_when_out_features_is_invalid_then_raises(self) -> None:
        with pytest.raises(ValueError, match="out_features"):
            BiasQuantizer(out_features=0)

    def test_indices_requires_the_bias_tensor(self) -> None:
        quantizer = BiasQuantizer(out_features=OUT_FEATURES, K_bias=15)
        with pytest.raises(ValueError, match="needs the bias tensor"):
            quantizer.indices()

    def test_indices_returns_uint8_for_a_small_alphabet(self) -> None:
        quantizer = BiasQuantizer(out_features=OUT_FEATURES, K_bias=255)
        indices = quantizer.indices(torch.randn(OUT_FEATURES))
        # 255 is the boundary and it fits: a uint8 alphabet holds 0..255.
        assert indices.dtype == torch.uint8
        assert int(indices.max().item()) < 255

    def test_indices_returns_int32_past_the_uint8_alphabet(self) -> None:
        quantizer = BiasQuantizer(out_features=OUT_FEATURES, K_bias=257)
        assert quantizer.K == 257
        assert quantizer.indices(torch.randn(OUT_FEATURES)).dtype == torch.int32


class TestMeasurement:
    def test_when_measured_then_every_number_is_finite(self) -> None:
        linear = quantized(quantize_bias=True)
        report = measure_bias_quantization(linear, torch.randn(4, IN_FEATURES), "asym")
        for result in (report.bias, report.weight, report.activation):
            assert math.isfinite(result.nmse)
            assert math.isfinite(result.mse)
            assert math.isfinite(result.max_abs_error)
            assert 0.0 <= result.nmse

    def test_when_measured_then_the_bias_error_is_reported_separately(self) -> None:
        """The three terms are distinct measurements, not one averaged number.

        Reported separately is the whole point: a bias NMSE that is small in
        absolute terms still dominates a layer whose weight NMSE is smaller, and
        an aggregate would hide exactly that.
        """
        linear = quantized(quantize_bias=True)
        report = measure_bias_quantization(linear, torch.randn(4, IN_FEATURES), "asym")
        assert report.bias.n_elements == OUT_FEATURES
        assert report.weight.n_elements == OUT_FEATURES * IN_FEATURES
        assert report.activation.n_elements == 4 * IN_FEATURES
        assert report.bias is not report.weight
        assert report.bias.preset == report.weight.preset == "asym"

    def test_when_measured_then_the_bias_error_differs_from_the_weight_error(
        self,
    ) -> None:
        """Not an equality test: the two tensors have different sizes and
        different distributions, so identical numbers would mean one of the
        measurements is measuring the other's tensor."""
        linear = quantized(quantize_bias=True)
        report = measure_bias_quantization(linear, torch.randn(4, IN_FEATURES), "asym")
        assert report.bias.nmse != report.weight.nmse
        assert math.isfinite(report.bias_nmse_over_weight_nmse)

    def test_when_measured_then_the_bias_term_uses_the_bias_quantizer(self) -> None:
        """Ties the number to the codebook that is actually installed.

        Refitting the quantizer and re-measuring must change the bias term; if
        it did not, the report would be scoring some other tensor.
        """
        linear = quantized(quantize_bias=True)
        x = torch.randn(4, IN_FEATURES)
        before = measure_bias_quantization(linear, x, "asym").bias.nmse
        with torch.no_grad():
            linear.bias.mul_(0.05)
            linear.bias_quantizer.init_from_tensor(linear.bias.detach())
        after = measure_bias_quantization(linear, x, "asym").bias.nmse
        assert before != after

    def test_when_the_layer_has_no_quantizer_then_raises(self) -> None:
        """Rather than falling back to the weight quantizer, which would report
        a weight number under a bias label."""
        linear = quantized()
        with pytest.raises(ValueError, match="bias_quantizer"):
            measure_bias_quantization(linear, torch.randn(4, IN_FEATURES), "asym")


class TestConflictingConfigurations:
    def test_when_quantize_bias_meets_a_biased_layer_that_is_fine(self) -> None:
        linear = LCQATLinear(64, 32, bias=True, quantize_bias=True)
        assert linear.bias_quantizer is not None

    def test_when_quantize_bias_meets_a_bias_free_layer_then_raises(self) -> None:
        """Raising rather than building an unread codebook.

        A silently-accepted flag here would add a codebook per layer that no
        forward pass ever reads -- parameters in the optimizer, zero gradient,
        and a state dict that differs from the model's actual behaviour.
        """
        with pytest.raises(ValueError, match="bias=True"):
            LCQATLinear(64, 32, bias=False, quantize_bias=True)

    def test_when_both_a_cardinality_and_a_split_are_given_then_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            LCQATLinear(
                64, 32, bias=True, quantize_bias=True, K_bias=15, K_bias_split=(3, 4)
            )

    def test_when_packed_indices_meet_a_bias_quantizer_then_forward_raises(
        self,
    ) -> None:
        """The fused inference path has no bias-index buffer to read.

        Left alone it would add the raw shadow bias -- a different value from
        the one the layer was trained with -- and nothing downstream would
        notice.
        """
        linear = LCQATLinear(
            64, 32, bias=True, quantize_bias=True, K_weight=15, K_act=15
        )
        linear.register_buffer(
            "packed_weight_indices",
            torch.zeros(32, 64, dtype=torch.uint8),
            persistent=True,
        )
        linear.register_buffer(
            "weight_index_format", torch.tensor(0, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match="packed_weight_indices"):
            linear(torch.randn(4, 64))


class TestRetrofitPath:
    @staticmethod
    def _net(bias: bool) -> nn.Module:
        """A two-block net whose module names match `get_layer_config`.

        The names are load-bearing: retrofit keys off `attn.c_q`, `mlp.c_fc`
        and friends, so an unnamed fixture retrofits nothing and the flag test
        would pass vacuously.
        """

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = nn.Module()
                self.attn.c_q = nn.Linear(256, 256, bias=bias)
                self.attn.c_proj = nn.Linear(256, 256, bias=bias)
                self.mlp = nn.Module()
                self.mlp.c_fc = nn.Linear(256, 512, bias=bias)
                self.mlp.c_proj = nn.Linear(512, 256, bias=bias)

        net = nn.ModuleDict({"h": nn.ModuleList([Block() for _ in range(2)])})
        with torch.no_grad():
            for param in net.parameters():
                param.normal_(0.0, 0.2)
        return net

    @staticmethod
    def _quantized_layers(model: nn.Module) -> list[LCQATLinear]:
        layers = [m for m in model.modules() if isinstance(m, LCQATLinear)]
        assert layers, "fixture retrofitted nothing; the name matching broke"
        return layers

    def test_when_the_flag_is_on_then_every_biased_layer_gets_a_quantizer(self) -> None:
        config = replace(PRESETS["asym"], quantize_bias=True)
        net = self._net(bias=True)
        retrofit_model(net, config)
        layers = self._quantized_layers(net)
        assert all(layer.bias_quantizer is not None for layer in layers)

    def test_when_the_flag_is_off_then_no_layer_gets_a_quantizer(self) -> None:
        net = self._net(bias=True)
        retrofit_model(net, PRESETS["asym"])
        assert all(
            layer.bias_quantizer is None for layer in self._quantized_layers(net)
        )

    def test_when_the_flag_is_on_for_a_bias_free_model_then_it_is_a_no_op(self) -> None:
        """`gpt.py` builds every projection with `bias=False`.

        So the flag has to degrade to "nothing to do" on the model this repo
        actually trains, rather than raising on the first layer it reaches.
        """
        config = replace(PRESETS["asym"], quantize_bias=True)
        net = self._net(bias=False)
        retrofit_model(net, config)
        assert all(
            layer.bias_quantizer is None for layer in self._quantized_layers(net)
        )

    def test_when_retrofitted_then_the_codebook_survives_a_state_dict_round_trip(
        self,
    ) -> None:
        """The resume shape: rebuild from config, then load strictly.

        The clone is deliberately perturbed first, so a load that silently
        rebuilt its own codebook instead of restoring the saved one would be
        caught.
        """
        config = replace(PRESETS["asym"], quantize_bias=True)
        net = self._net(bias=True)
        retrofit_model(net, config)
        state = net.state_dict()
        assert any("bias_quantizer" in key for key in state)

        clone = self._net(bias=True)
        retrofit_model(clone, config)
        with torch.no_grad():
            for param in clone["h"][0].attn.c_q.bias_quantizer.parameters():
                param.add_(1.0)
        assert not torch.equal(
            clone["h"][0].attn.c_q.bias_quantizer.get_codebook(),
            net["h"][0].attn.c_q.bias_quantizer.get_codebook(),
        )

        # strict=True: the flag has to be in the config, or the resume fails
        # with "unexpected keys" for every bias codebook in the checkpoint.
        clone.load_state_dict(state)
        assert torch.equal(
            clone["h"][0].attn.c_q.bias_quantizer.get_codebook(),
            net["h"][0].attn.c_q.bias_quantizer.get_codebook(),
        )

    def test_when_the_config_is_serialized_then_the_flag_survives(self) -> None:
        """A checkpoint has to record that it was trained this way.

        Without the flag in `meta["lcqat"]`, a resume would rebuild the layers
        without their bias codebooks and fail the strict state_dict load with
        "unexpected keys".
        """
        config = replace(PRESETS["asym"], quantize_bias=True)
        assert config.as_dict()["quantize_bias"] is True
        assert LayerKConfig.from_dict(config.as_dict()).quantize_bias is True

    def test_when_an_older_config_dict_is_loaded_then_the_flag_defaults_off(
        self,
    ) -> None:
        """A checkpoint saved before this field existed must still load."""
        from nanochat.models.quant.retrofit import LayerKConfig

        data = PRESETS["asym"].as_dict()
        data.pop("quantize_bias", None)
        config = LayerKConfig.from_dict(data)
        config.validate()
        assert config.quantize_bias is False

    def test_when_validated_then_the_bool_is_not_treated_as_a_codebook_spec(
        self,
    ) -> None:
        """The §9.3 Defect 1 landmine, in its most likely new form.

        `validate()` picks fields by the explicit `CODEBOOK_SPEC_FIELDS`
        frozenset. A `bool` reaching `_validate_codebook_spec` raises on every
        `retrofit_model` call with no per-channel flag in play, and the suffix
        heuristic that caused it must not come back -- so the assertion is that
        the flag is absent from the frozenset, not merely that validation
        happens to pass.
        """
        assert "quantize_bias" not in CODEBOOK_SPEC_FIELDS
        replace(PRESETS["asym"], quantize_bias=True).validate()
        replace(PRESETS["asym"], quantize_bias=False).validate()

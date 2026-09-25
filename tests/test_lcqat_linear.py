"""
Tests for LCQATLinear dual quantization (PRD sections 3.2 and 6).

python -m pytest tests/test_lcqat_linear.py -v
"""

import pytest
import torch
import torch.nn as nn

from nanochat.lcqat import LCQATLinear


def from_float(
    in_features: int = 64,
    out_features: int = 32,
    bias: bool = True,
    K_weight: int = 3,
    K_act: int = 15,
    quantize_out: bool = False,
    dtype: torch.dtype = torch.float32,
    zeros: bool = False,
) -> LCQATLinear:
    torch.manual_seed(0)
    mod = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    if zeros:
        nn.init.zeros_(mod.weight)
    return LCQATLinear.from_float(
        mod, K_weight=K_weight, K_act=K_act, quantize_out=quantize_out
    )


def test_when_forward_then_shapes_and_finiteness_hold() -> None:
    linear = from_float()
    x = torch.randn(4, 16, 64)
    y = linear(x)
    assert y.shape == (4, 16, 32)
    assert torch.isfinite(y).all()


def test_when_input_is_bf16_then_output_is_bf16() -> None:
    linear = from_float()
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    y = linear(x)
    assert y.dtype == torch.bfloat16


def test_when_weights_are_zero_then_output_is_exact_zero() -> None:
    linear = from_float(bias=False, zeros=True)
    y = linear(torch.randn(8, 64))
    assert (y == 0).all()


def test_when_from_float_then_weight_and_bias_are_copied() -> None:
    torch.manual_seed(1)
    mod = nn.Linear(64, 32, bias=True)
    linear = LCQATLinear.from_float(mod, K_weight=3, K_act=15)
    assert torch.equal(linear.weight, mod.weight)
    assert torch.equal(linear.bias, mod.bias)
    assert linear.weight.device == mod.weight.device
    assert linear.weight.dtype == mod.weight.dtype


def test_when_from_float_on_meta_device_then_raises() -> None:
    with torch.device("meta"):
        mod = nn.Linear(64, 32)
    with pytest.raises(RuntimeError, match="materialized"):
        LCQATLinear.from_float(mod)


def test_when_from_float_on_lcqat_then_raises() -> None:
    linear = from_float()
    with pytest.raises(ValueError, match="plain float"):
        LCQATLinear.from_float(linear)


def test_when_quantize_out_then_output_values_come_from_the_output_codebook() -> None:
    linear = from_float(quantize_out=True)
    assert linear.out_quantizer is not None
    y = linear(torch.randn(8, 64))
    out_cb = linear.out_quantizer.get_codebook()
    distance = (y.unsqueeze(-1) - out_cb).abs().min(dim=-1).values
    assert (distance == 0).all()


def test_when_no_quantize_out_then_output_quantizer_is_absent() -> None:
    assert from_float(quantize_out=False).out_quantizer is None


def test_when_backward_then_gradients_reach_weights_inputs_and_all_codebooks() -> None:
    linear = from_float(quantize_out=True)
    x = torch.randn(8, 64, requires_grad=True)
    linear(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert linear.weight.grad is not None and linear.weight.grad.abs().sum() > 0
    assert linear.act_quantizer.raw_pos_deltas.grad.abs().sum() > 0
    assert linear.weight_quantizer.raw_pos_deltas.grad.abs().sum() > 0
    assert linear.out_quantizer.raw_pos_deltas.grad.abs().sum() > 0


def test_when_bias_present_then_bias_receives_gradient() -> None:
    linear = from_float(bias=True)
    linear(torch.randn(8, 64)).sum().backward()
    assert linear.bias.grad is not None
    assert linear.bias.grad.shape == linear.bias.shape


def test_when_weight_quantized_then_values_are_codebook_levels() -> None:
    linear = from_float(K_weight=3)
    w_q = linear.weight_quantizer(linear.weight)
    assert set(w_q.value.flatten().tolist()) <= set(w_q.codebook.tolist())
    assert int(w_q.indices.max()) < 3

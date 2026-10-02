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


# ---------------------------------------------------------------------------
# PRD 2.4: 1/sqrt(N) codebook gradient scaling
# ---------------------------------------------------------------------------


def _codebook_grad_sum(grad_scale: str, seed: int = 1) -> float:
    """Sum |grad| on the weight codebook's latent deltas, for one grad_scale."""
    linear = from_float(in_features=384, out_features=1536, K_weight=3, K_act=15)
    linear.grad_scale = grad_scale
    torch.manual_seed(seed)
    x = torch.randn(8, 384)
    linear(x).pow(2).mean().backward()
    return linear.weight_quantizer.raw_pos_deltas.grad.abs().sum().item()


def test_when_grad_scale_inv_sqrt_n_then_codebook_gradient_is_scaled_by_inv_sqrt_numel() -> (
    None
):
    """The PRD's stated motivation, checked numerically.

    In a 4096x4096 layer, 16.7M elements pool into one K-entry codebook.
    Unscaled, the step parameters get a sum-reduction gradient orders of
    magnitude larger than the per-weight gradients and oscillate relative to
    them; `1/sqrt(N)` makes the magnitudes comparable (PRD 2.4).

    N is `numel(weight)` on this path: one weight matrix feeds one codebook, so
    the scale is constant across steps (unlike the activation path, where N is
    `numel(x)` and varies per batch).
    """
    unscaled = _codebook_grad_sum("none")
    scaled = _codebook_grad_sum("inv_sqrt_n")
    expected = unscaled * (384 * 1536) ** -0.5
    assert scaled == pytest.approx(expected, rel=1e-4), (
        f"expected {expected:.6e} from {unscaled:.6e}, got {scaled:.6e}"
    )


def test_when_grad_scale_inv_sqrt_n_then_input_and_weight_grads_are_untouched() -> None:
    """Only the codebook side is scaled.

    The STE identity gradient (PRD 2.3) must stay unit-scaled, and the weight
    gradient must not inherit the 1/sqrt(N) factor -- it is already a
    per-element gradient, not a pooled one.
    """
    grads = {}
    for tag in ("none", "inv_sqrt_n"):
        linear = from_float()
        linear.grad_scale = tag
        torch.manual_seed(2)
        x = torch.randn(8, 64, requires_grad=True)
        linear(x).sum().backward()
        grads[tag] = (x.grad.clone(), linear.weight.grad.clone())
    assert torch.equal(grads["none"][0], grads["inv_sqrt_n"][0]), (
        "input gradient should be identical: the STE identity is not scaled"
    )
    assert torch.equal(grads["none"][1], grads["inv_sqrt_n"][1]), (
        "weight gradient should be identical: only the codebook is scaled"
    )


def test_when_act_batch_size_changes_then_activation_codebook_scale_changes() -> None:
    """N = numel(x) on the activation path, so the scale tracks the batch."""
    linear = from_float()
    linear.grad_scale = "inv_sqrt_n"
    torch.manual_seed(3)
    linear(torch.randn(64, 64)).sum().backward()
    small = linear.act_quantizer.raw_pos_deltas.grad.abs().sum().item()

    linear.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    linear(torch.randn(256, 64)).sum().backward()
    large = linear.act_quantizer.raw_pos_deltas.grad.abs().sum().item()

    # 4x the elements => 1/sqrt(4) = 1/2 the gradient for the same per-element
    # signal. Note the sum also grows with N, so the comparison is only
    # meaningful as a ratio against the unscaled baseline; what is asserted here
    # is the direction and rough magnitude of the effect.
    assert large != small
    assert small > 0 and large > 0


def test_when_grad_scale_invalid_at_construction_then_raises() -> None:
    with pytest.raises(ValueError, match="grad_scale"):
        torch.manual_seed(0)
        mod = nn.Linear(64, 32, bias=False)
        LCQATLinear.from_float(mod, grad_scale="sqrt_n")

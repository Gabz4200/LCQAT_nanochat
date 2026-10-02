"""PRD 2.4 codebook gradient scaling (W2.5), reachable end to end.

The PRD's motivation is specific: in a 4096x4096 layer, 16.7M elements pool into
one K-entry codebook, and unscaled sum-reduction makes the step parameters
oscillate relative to the weights. `1/sqrt(N)` makes the magnitudes comparable.

This was the A1 gap in disguise: `LayerKConfig.grad_scale` existed and
`retrofit_model` threaded it, but no training script set it, so the ablation axis
existed in the library and nowhere else.
"""

import pytest
import torch

from nanochat.lcqat import PRESETS, LCQATLinear, retrofit_model
from nanochat.lcqat.linear import GRAD_SCALE_INV_SQRT_N, GRAD_SCALE_NONE
from nanochat.lcqat.retrofit import DEFAULT_PRESET, lcqat_config_from_args
from tests.conftest import build_active_tiny_gpt


def _codebook_grad_sum(grad_scale, in_features=384, out_features=1536, seed=1):
    torch.manual_seed(0)
    linear = LCQATLinear(in_features, out_features, bias=False, K_weight=3, K_act=15)
    linear.grad_scale = grad_scale
    torch.manual_seed(seed)
    linear(torch.randn(8, in_features)).pow(2).mean().backward()
    return linear.weight_quantizer.raw_pos_deltas.grad.abs().sum().item()


def test_when_inv_sqrt_n_then_gradient_is_scaled_by_inverse_sqrt_numel():
    """N is `numel(weight)` on this path: one weight matrix, one codebook, so
    the scale is constant across steps (unlike the activation path)."""
    unscaled = _codebook_grad_sum(GRAD_SCALE_NONE)
    scaled = _codebook_grad_sum(GRAD_SCALE_INV_SQRT_N)
    expected = unscaled * (384 * 1536) ** -0.5
    assert scaled == pytest.approx(expected, rel=1e-4), (
        f"expected {expected:.6e} from {unscaled:.6e}, got {scaled:.6e}"
    )


def test_when_none_then_no_scaling_is_applied():
    assert _codebook_grad_sum(GRAD_SCALE_NONE) > 0


def test_when_grad_scale_set_then_input_and_weight_grads_are_untouched():
    """Only the codebook side is pooled, so only it should be scaled."""
    grads = {}
    for tag in (GRAD_SCALE_NONE, GRAD_SCALE_INV_SQRT_N):
        torch.manual_seed(0)
        linear = LCQATLinear(64, 32, bias=False, K_weight=3, K_act=15)
        linear.grad_scale = tag
        torch.manual_seed(2)
        x = torch.randn(8, 64, requires_grad=True)
        linear(x).sum().backward()
        grads[tag] = (x.grad.clone(), linear.weight.grad.clone())
    assert torch.equal(grads[GRAD_SCALE_NONE][0], grads[GRAD_SCALE_INV_SQRT_N][0])
    assert torch.equal(grads[GRAD_SCALE_NONE][1], grads[GRAD_SCALE_INV_SQRT_N][1])


def test_when_activation_batch_grows_then_act_codebook_gradient_changes():
    """N = numel(x) on the activation path, so the scale tracks the batch."""
    torch.manual_seed(0)
    linear = LCQATLinear(64, 32, bias=False, K_weight=3, K_act=15)
    linear.grad_scale = GRAD_SCALE_INV_SQRT_N
    torch.manual_seed(3)
    linear(torch.randn(64, 64)).sum().backward()
    small = linear.act_quantizer.raw_pos_deltas.grad.abs().sum().item()
    linear.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    linear(torch.randn(256, 64)).sum().backward()
    large = linear.act_quantizer.raw_pos_deltas.grad.abs().sum().item()
    assert small > 0 and large > 0 and small != large


def test_when_config_from_args_then_grad_scale_is_selectable_and_validated():
    """The CLI surface the plan called for and that did not exist."""
    args = type(
        "A",
        (),
        {
            "lcqat_preset": "asym",
            "lcqat_k_map": "",
            "codebook_grad_scale": "none",
        },
    )()
    assert lcqat_config_from_args(args).grad_scale == "none"

    default = type(
        "A",
        (),
        {
            "lcqat_preset": "asym",
            "lcqat_k_map": "",
            "codebook_grad_scale": "inv_sqrt_n",
        },
    )()
    assert lcqat_config_from_args(default).grad_scale == "inv_sqrt_n"

    # A caller that does not provide the flag at all still works (optional getattr).
    legacy = type("A", (), {"lcqat_preset": "asym", "lcqat_k_map": ""})()
    assert lcqat_config_from_args(legacy).grad_scale == "inv_sqrt_n"

    bad = type(
        "A",
        (),
        {"lcqat_preset": "asym", "lcqat_k_map": "", "codebook_grad_scale": "sqrt_n"},
    )()
    with pytest.raises(ValueError, match="codebook-grad-scale"):
        lcqat_config_from_args(bad)


def test_when_grad_scale_none_in_config_then_retrofitted_modules_use_it():
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = replace(PRESETS[DEFAULT_PRESET], grad_scale=GRAD_SCALE_NONE)
    model = retrofit_model(build_active_tiny_gpt(), cfg)
    for module in model.modules():
        if isinstance(module, LCQATLinear):
            assert module.grad_scale == GRAD_SCALE_NONE


def test_when_grad_scale_persists_in_config_then_resume_cannot_change_it_silently():
    """It is a `LayerKConfig` field, so `as_dict()` carries it into meta."""
    from dataclasses import replace

    cfg = replace(PRESETS[DEFAULT_PRESET], grad_scale=GRAD_SCALE_NONE)
    assert cfg.as_dict()["grad_scale"] == GRAD_SCALE_NONE
    from nanochat.lcqat.retrofit import LayerKConfig

    assert LayerKConfig.from_dict(cfg.as_dict()).grad_scale == GRAD_SCALE_NONE


def test_when_grad_scale_invalid_at_construction_then_raises():
    torch.manual_seed(0)

    with pytest.raises(ValueError, match="grad_scale"):
        torch.manual_seed(0)
        mod = torch.nn.Linear(64, 32, bias=False)
        LCQATLinear.from_float(mod, grad_scale="sqrt_n")

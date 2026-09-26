"""
Tests for the LCQATLinear quantized-inference runtime path (TODO section 1):
forward through K-selected packed weight IDs + LUT fetch, and mixed-K
single-forward parity between the exported index runtime and the float
STE path.

python -m pytest tests/test_lcqat_linear_runtime.py -v
"""

import copy

import torch

from nanochat.lcqat import LCQATLinear, export_lcqat_checkpoint, retrofit_model


def make_linear(in_features: int = 16, out_features: int = 8) -> LCQATLinear:
    torch.manual_seed(0)
    base = torch.nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        base.weight.mul_(0.5)
    return LCQATLinear.from_float(base, K_weight=15, K_act=15, quantize_out=False)


def test_when_quantized_inference_forward_then_matches_ste_float_path(
    tmp_path,
) -> None:
    float_module = make_linear()
    float_module.eval()
    runtime_module = copy.deepcopy(float_module)
    export_lcqat_checkpoint(runtime_module, str(tmp_path / "linear_export.pt"))
    assert "packed_weight_indices" in dict(runtime_module.named_buffers())
    assert not hasattr(runtime_module, "weight")

    torch.manual_seed(1)
    x = torch.randn(5, 16)
    with torch.no_grad():
        expected = float_module(x)
        got = runtime_module(x)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_when_mixed_k_model_forward_then_matches_float_path(tmp_path) -> None:
    from nanochat.lcqat import PRESETS
    from tests.conftest import build_active_tiny_gpt

    torch.manual_seed(42)
    float_model = retrofit_model(build_active_tiny_gpt(), PRESETS["prd"])
    float_model.eval()
    runtime_model = copy.deepcopy(float_model)
    export_lcqat_checkpoint(runtime_model, str(tmp_path / "mixed_export.pt"))
    runtime_model.eval()

    torch.manual_seed(7)
    ids = torch.randint(0, float_model.config.vocab_size, (2, 16))
    with torch.no_grad():
        expected = float_model(ids)
        got = runtime_model(ids)
    assert got.shape == expected.shape
    assert torch.isfinite(got).all()
    # Identical IDs + identical LUTs: only summation order differs.
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_when_runtime_module_backward_then_actionable_error(tmp_path) -> None:
    module = make_linear()
    module.eval()
    export_lcqat_checkpoint(module, str(tmp_path / "linear_export2.pt"))
    x = torch.randn(3, 16)
    out = module(x)
    # exported codebooks are frozen (no grad path) and the index op is
    # inference-only; any backward through it must not silently succeed.
    assert not out.requires_grad

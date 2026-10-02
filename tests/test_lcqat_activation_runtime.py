"""
Tests for the fused activation-LUT runtime consumer (TODO section 1,
PRD section 6): export wires relu^2 tables between c_fc and c_proj, the
MLP forward consumes them as index->index fetches, and the fused chain
agrees exactly with the float elementwise path.

python -m pytest tests/test_lcqat_activation_runtime.py -v
"""

import copy

import torch
import torch.nn.functional as F

from nanochat.models.quant import LCQATLinear, export_lcqat_checkpoint, retrofit_model


def build_pair(preset) -> tuple:
    """(float_model, runtime_model) both retrofitted identically."""
    from tests.conftest import build_active_tiny_gpt

    torch.manual_seed(42)
    float_model = retrofit_model(build_active_tiny_gpt(), preset)
    float_model.eval()
    runtime_model = copy.deepcopy(float_model)
    return float_model, runtime_model


def test_when_exporting_then_activation_lut_wired_on_every_mlp(
    tiny_gpt_factory, tmp_path
) -> None:
    from nanochat.models.quant import PRESETS

    model = retrofit_model(tiny_gpt_factory(), PRESETS["prd"])
    state = export_lcqat_checkpoint(model, str(tmp_path / "act_export.pt"))
    lut_keys = [k for k in state if k.endswith("mlp.c_fc.activation_lut")]
    assert len(lut_keys) == model.config.n_layer
    for key in lut_keys:
        assert state[key].dtype == torch.uint8
        assert state[key].shape == (15,)  # fc K_act = 15 table entries


def test_when_exporting_then_lut_matches_value_path_quantizer_ids(
    tiny_gpt_factory, tmp_path
) -> None:
    # Independent oracle: run the REAL out-quantizer -> relu^2 -> act-quantizer
    # value path and require the compiled table to reproduce those exact IDs.
    from nanochat.models.quant import PRESETS

    model = retrofit_model(tiny_gpt_factory(), PRESETS["prd"])
    model.eval()
    runtime = copy.deepcopy(model)
    export_lcqat_checkpoint(runtime, str(tmp_path / "act_export2.pt"))

    for name, fc in runtime.named_modules():
        if not name.endswith("mlp.c_fc") or not isinstance(fc, LCQATLinear):
            continue
        parent = runtime.get_submodule(name[: -len(".c_fc")])
        proj = parent.c_proj
        cb_in = fc.out_quantizer.get_codebook()
        table = dict(runtime.named_buffers())[f"{name}.activation_lut"]
        y_value = F.relu(cb_in).square()
        expected = proj.act_quantizer(y_value.unsqueeze(0)).indices.squeeze(0)
        assert torch.equal(table.long(), expected.long()), name


def test_when_quantized_mlp_forward_then_matches_float_path(
    tiny_gpt_factory, tmp_path
) -> None:
    from nanochat.models.quant import PRESETS

    float_model, runtime_model = build_pair(PRESETS["prd"])
    export_lcqat_checkpoint(runtime_model, str(tmp_path / "mlp_export.pt"))
    runtime_model.eval()

    torch.manual_seed(3)
    x = torch.randn(4, float_model.config.n_embd)
    float_mlp = float_model.transformer.h[0].mlp
    runtime_mlp = runtime_model.transformer.h[0].mlp
    # the runtime branch is active: packed weights + fused table present
    assert "activation_lut" in dict(runtime_mlp.c_fc.named_buffers())
    with torch.no_grad():
        expected = float_mlp(x)
        got = runtime_mlp(x)
    assert got.shape == expected.shape
    assert torch.isfinite(got).all()
    # Identical IDs on both matmuls: only summation order differs.
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)

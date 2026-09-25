"""
Tests for the LC-QAT export pipeline (PRD sections 8 and 6).

python -m pytest tests/test_lcqat_export.py -v
"""

import pytest
import torch

from nanochat.lcqat import (
    LCQATLinear,
    export_lcqat_checkpoint,
    is_exported_lcqat_state,
    is_lcqat_state,
    prepare_lcqat_before_load,
)


def test_when_exporting_then_artifact_has_indices_luts_and_no_shadow_weights(
    tiny_gpt_lcqat, tmp_path
) -> None:
    model = tiny_gpt_lcqat
    expected_codebooks = {
        name: module.weight_quantizer.get_codebook().clone()
        for name, module in model.named_modules()
        if isinstance(module, LCQATLinear)
    }
    export_path = str(tmp_path / "lcqat_export.pt")
    state = export_lcqat_checkpoint(model, export_path)

    assert is_exported_lcqat_state(state)
    assert not is_lcqat_state(state)
    for key, value in state.items():
        assert not key.endswith("raw_pos_deltas")
        assert not key.endswith("raw_neg_deltas")

    # every retrofitted module contributes a uint8 index matrix and keeps its LUTs
    index_keys = [k for k in state if k.endswith("packed_weight_indices")]
    assert len(index_keys) == len(expected_codebooks)
    for key in index_keys:
        assert state[key].dtype == torch.uint8
    for name, codebook in expected_codebooks.items():
        key = f"{name}.weight_quantizer.compiled_codebook"
        assert key in state, f"missing compiled codebook: {key}"
        assert torch.equal(state[key], codebook)

    # the un-quantized head stays in floating point
    assert "lm_head.weight" in state
    # retrofitted weight matrices are gone
    assert "transformer.h.0.attn.c_q.weight" not in state

    reloaded = torch.load(export_path, weights_only=True)
    assert set(reloaded) == set(state)


def test_when_loading_exported_artifact_then_actionable_error(
    tiny_gpt_lcqat, tmp_path, tiny_gpt_factory
) -> None:
    export_path = str(tmp_path / "lcqat_export.pt")
    state = export_lcqat_checkpoint(tiny_gpt_lcqat, export_path)
    fresh = tiny_gpt_factory()
    with pytest.raises(RuntimeError, match="exported LC-QAT artifact"):
        prepare_lcqat_before_load(fresh, state, None, None)

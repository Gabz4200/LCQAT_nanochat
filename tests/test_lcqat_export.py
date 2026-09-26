"""
Tests for the LC-QAT export pipeline (PRD sections 8 and 6).

python -m pytest tests/test_lcqat_export.py -v
"""

import copy
import math

import pytest
import torch

from nanochat.lcqat import (
    LCQATLinear,
    export_lcqat_checkpoint,
    is_exported_lcqat_state,
    is_lcqat_state,
    prepare_lcqat_before_load,
)
from nanochat.lcqat.packing import (
    FORMAT_NIBBLES,
    FORMAT_TRITS,
    FORMAT_UINT8,
    index_format_for_k,
    unpack_weight_indices,
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


def test_when_loading_exported_artifact_then_loads_and_runs(
    tiny_gpt_factory, tmp_path
) -> None:
    from dataclasses import asdict

    from nanochat.lcqat import PRESETS, retrofit_model

    torch.manual_seed(1)
    source = retrofit_model(tiny_gpt_factory(), PRESETS["prd"])
    float_baseline = copy.deepcopy(source)
    source.eval()
    float_baseline.eval()
    state = export_lcqat_checkpoint(source, str(tmp_path / "lcqat_export.pt"))

    fresh = tiny_gpt_factory()
    cfg = prepare_lcqat_before_load(fresh, state, asdict(PRESETS["prd"]), None)
    assert cfg is not None
    fresh.load_state_dict(state, strict=True, assign=True)
    fresh.eval()

    torch.manual_seed(9)
    ids = torch.randint(0, fresh.config.vocab_size, (2, 16))
    with torch.no_grad():
        expected = float_baseline(ids)
        got = fresh(ids)
    assert got.shape == expected.shape
    assert torch.isfinite(got).all()
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_when_build_model_with_exported_artifact_then_eval_runs_and_train_rejects(
    tiny_gpt_factory, tmp_path, monkeypatch
) -> None:
    from dataclasses import asdict

    from nanochat import checkpoint_manager as cm
    from nanochat.lcqat import PRESETS, retrofit_model

    torch.manual_seed(2)
    source = retrofit_model(tiny_gpt_factory(), PRESETS["prd"])
    float_baseline = copy.deepcopy(source)
    source.eval()
    float_baseline.eval()
    state = export_lcqat_checkpoint(source, str(tmp_path / "e2.pt"))
    meta = {
        "model_config": asdict(source.config),
        "lcqat": asdict(PRESETS["prd"]),
    }
    cm.save_checkpoint(str(tmp_path / "ck"), 1, state, None, meta)

    class FakeTokenizer:
        def get_vocab_size(self):
            return source.config.vocab_size

    monkeypatch.setattr(cm, "get_tokenizer", lambda: FakeTokenizer())
    model, _, loaded_meta = cm.build_model(
        str(tmp_path / "ck"), 1, torch.device("cpu"), "eval"
    )
    assert loaded_meta["lcqat"]["down_weight"] == 255
    torch.manual_seed(9)
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    with torch.no_grad():
        assert torch.allclose(model(ids), float_baseline(ids), atol=1e-4, rtol=1e-4)

    with pytest.raises(RuntimeError, match="inference-only"):
        cm.build_model(str(tmp_path / "ck"), 1, torch.device("cpu"), "train")


def test_when_exporting_then_indices_are_packed_by_k_format(
    tiny_gpt_lcqat, tmp_path
) -> None:
    model = tiny_gpt_lcqat
    original = {
        name: module.weight_quantizer(module.weight).indices.clone()
        for name, module in model.named_modules()
        if isinstance(module, LCQATLinear)
    }
    state = export_lcqat_checkpoint(model, str(tmp_path / "packed_export.pt"))

    for name, idx in original.items():
        k = next(
            m.K_weight
            for n, m in model.named_modules()
            if isinstance(m, LCQATLinear) and n == name
        )
        fmt = index_format_for_k(k)
        packed = state[f"{name}.packed_weight_indices"]
        tag = int(state[f"{name}.weight_index_format"])
        assert tag == fmt, name
        n = idx.shape[1]
        if fmt == FORMAT_TRITS:
            assert packed.shape == (idx.shape[0], math.ceil(n / 5))
        elif fmt == FORMAT_NIBBLES:
            assert packed.shape == (idx.shape[0], math.ceil(n / 2))
        else:
            assert packed.shape == idx.shape
        assert torch.equal(unpack_weight_indices(packed, n, k), idx)


def test_when_exporting_preset_prd_then_uint8_and_trit_formats_coexist(
    tiny_gpt_factory, tmp_path
) -> None:
    from nanochat.lcqat import PRESETS, retrofit_model

    model = retrofit_model(tiny_gpt_factory(), PRESETS["prd"])
    state = export_lcqat_checkpoint(model, str(tmp_path / "prd_export.pt"))
    tags = {int(v) for k, v in state.items() if k.endswith("weight_index_format")}
    # prd preset: q/k K=3 (trits), v/o/fc K=15 (nibbles), down K=255 (uint8)
    assert tags == {FORMAT_TRITS, FORMAT_NIBBLES, FORMAT_UINT8}


def test_when_exporting_k_above_255_then_int32_indices_not_rejected(
    tiny_gpt_factory, tmp_path
) -> None:
    from nanochat.lcqat import LayerKConfig, retrofit_model

    cfg = LayerKConfig(down_weight=257, down_act=15)
    model = retrofit_model(tiny_gpt_factory(), cfg)
    export_path = str(tmp_path / "k257_export.pt")
    state = export_lcqat_checkpoint(model, export_path)
    down_keys = [k for k in state if "mlp.c_proj.packed_weight_indices" in k]
    assert down_keys
    for key in down_keys:
        assert state[key].dtype == torch.int32
    format_keys = [k for k in state if k.endswith("weight_index_format")]
    assert any(int(state[k]) == 3 for k in format_keys)  # FORMAT_INT32 present

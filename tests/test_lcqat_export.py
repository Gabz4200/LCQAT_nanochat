"""
Tests for the LC-QAT export pipeline (PRD sections 8 and 6).

python -m pytest tests/test_lcqat_export.py -v
"""

import copy
import math

import pytest
import torch

from nanochat.models.quant import (
    PRESETS,
    LCQATLinear,
    export_lcqat_checkpoint,
    is_exported_lcqat_state,
    is_lcqat_state,
    prepare_lcqat_before_load,
    retrofit_model,
)
from nanochat.models.quant.packing import (
    FORMAT_NIBBLES,
    FORMAT_TRITS,
    FORMAT_UINT8,
    index_format_for_k,
    unpack_weight_indices,
)
from nanochat.models.quant.retrofit import DEFAULT_PRESET
from nanochat.modules.experiments.tiny_models import build_active_tiny_gpt
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
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


def test_when_loading_exported_db_artifact_then_engine_owns_its_quantized_layers(
    tmp_path,
) -> None:
    """The exported half of the engine contract: a stripped artifact reloads whole.

    The forward direction (`build_model` retrofitting engine-owned layers) is
    covered in `test_w0_ckpt_guard.py`. This is the reverse: export writes
    `db_*` packed indices and compiled codebooks, and reloading must put the
    engine's adapters and denoise heads back into the exported runtime shape.

    It did not. `prepare_lcqat_before_load` reshapes only the bare GPT -- it is
    handed `model`, and `_prepare_exported_buffers` walks `model.named_modules()`.
    The engine is built afterwards from plain float `nn.Linear`s, so every
    `db_*` index and codebook key landed in a module with no buffer to receive
    it and was dropped by the `strict=False` load. Export succeeded, the file
    looked right, and inference then served float adapters.
    """
    from dataclasses import asdict

    from nanochat.data.tokenizer import get_tokenizer
    from nanochat.models.backbone import GPT, GPTConfig
    from nanochat.modules import checkpoint_manager as cm

    vocab = get_tokenizer().get_vocab_size()
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=vocab,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        window_pattern="L",
    )
    with torch.device("meta"):
        source = GPT(config)
    source.to_empty(device="cpu")
    source.init_weights()
    quantized = retrofit_model(source, PRESETS[DEFAULT_PRESET])
    engine = DiffusionBlockEngine(quantized, EquiProbabilityPartitioner(num_blocks=2))
    engine.apply_lcqat(PRESETS[DEFAULT_PRESET])

    ckpt = tmp_path / "ck"
    state = export_lcqat_checkpoint(engine, str(tmp_path / "db_e2e.pt"))
    exported_db_keys = {k for k in state if k.startswith("db_")}
    assert exported_db_keys, "export wrote no db_* keys to begin with"
    cm.save_checkpoint(
        str(ckpt),
        1,
        state,
        None,
        {
            "model_config": asdict(config),
            "lcqat": asdict(PRESETS[DEFAULT_PRESET]),
            "db": {"num_blocks": 2},
        },
    )

    loaded, _tokenizer, _meta = cm.build_model(
        str(ckpt), 1, torch.device("cpu"), "eval"
    )

    assert isinstance(loaded, DiffusionBlockEngine)
    assert not (exported_db_keys - set(loaded.state_dict())), (
        "exported db_* state dropped on reload: "
        f"{sorted(exported_db_keys - set(loaded.state_dict()))[:5]}"
    )
    adapter = loaded.adapters[0].mlp[0]
    assert getattr(adapter, "weight_quantizer", None) is not None, (
        f"adapter reloaded as plain {type(adapter).__name__}: export/load are asymmetric"
    )
    assert not hasattr(adapter, "weight"), "shadow weight survived the roundtrip"
    assert "db_adapters.0.mlp.0.packed_weight_indices" in loaded.state_dict()


def test_when_loading_exported_sparseprop_artifact_then_it_reloads(
    tmp_path, monkeypatch
) -> None:
    """An exported artifact from a SparseProp-trained model must be loadable.

    Pre-existing and independent of DiffusionBlocks: this raised on a plain
    retrofitted GPT too, so *no* exported artifact from a default-config model
    could be read back -- and SparseProp is on by default.

    `inject_sparseprop_layers` runs *after* `prepare_lcqat_before_load`, which
    for an exported artifact has already run `_prepare_exported_buffers` and
    deleted every `weight`. The injection then builds a
    `SparsePropLinearLCQAT` from that layer and reads `lcqat_linear.weight.device`
    -- AttributeError on a stripped layer.

    The artifact needs no injection: export already wrote the CSR structure
    (`sparse_row_ptr` and friends) and `_prepare_exported_buffers` re-registers
    those buffers on the reshaped layers.

    Known gap, deliberately not asserted here: the reloaded layer is a plain
    `LCQATLinear`, so its CSR buffers are registered but nothing reads them --
    `dispatch_index_linear` takes the dense `packed_weight_indices`, and no
    forward path routes to `sparse_index_linear_cpu`. Exporting the CSR form
    ahead of a runtime that consumes it is harmless (the dense buffer is written
    alongside and is numerically identical); asserting the sparse path works
    would be asserting a feature that does not exist yet.
    """
    from dataclasses import asdict

    from nanochat.models.quant.sparseprop import inject_sparseprop_layers
    from nanochat.modules import checkpoint_manager as cm

    source = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    inject_sparseprop_layers(source, sparsity=0.75, with_lcqat=True)
    state = export_lcqat_checkpoint(source, str(tmp_path / "sp.pt"), sparse=True)
    assert any(k.endswith("sparse_row_ptr") for k in state)

    ckpt = tmp_path / "sp_ck"
    cm.save_checkpoint(
        str(ckpt),
        1,
        state,
        None,
        {
            "model_config": asdict(source.config),
            "lcqat": asdict(PRESETS[DEFAULT_PRESET]),
            "sparseprop": {"enabled": True, "sparsity": 0.75},
        },
    )

    class FakeTokenizer:
        def get_vocab_size(self):
            return source.config.vocab_size

    monkeypatch.setattr(cm, "get_tokenizer", lambda: FakeTokenizer())
    loaded, _tokenizer, _meta = cm.build_model(
        str(ckpt), 1, torch.device("cpu"), "eval"
    )

    layer = loaded.transformer.h[0].attn.c_q
    # Strips weights, gains the index buffer, and keeps the artifact's CSR
    # structure registered rather than silently discarded.
    assert not hasattr(layer, "weight"), "exported layer kept its shadow weight"
    assert "transformer.h.0.attn.c_q.packed_weight_indices" in loaded.state_dict()
    assert "transformer.h.0.attn.c_q.sparse_row_ptr" in loaded.state_dict()

    ids = torch.randint(0, loaded.config.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.isfinite(loaded(ids)).all()


def test_when_loading_exported_artifact_then_learnable_lut_params_are_dropped(
    tiny_gpt_factory, tmp_path, monkeypatch
) -> None:
    """An exported artifact must not carry training-time-only parameters.

    Pre-existing, and it blocked the reload of *every* exported artifact from a
    default-config model, because `--lcqat-lut-relaxation` attaches a
    `LearnableIndexLut` to each c_fc -> c_proj pair and export leaves its
    `logits` / `initial_table` parameters in the state. On load,
    `_prepare_exported_buffers` registers the baked `activation_lut` buffer but
    never attaches the learnable module, so the strict load rejected the keys as
    unexpected.

    They are redundant in an artifact, not lost: the function they parameterize
    is the baked `activation_lut` table sitting beside them, and the artifact is
    inference-only by contract (`build_model` rejects `phase="train"` outright).
    So they are stripped on load rather than reconstructed.
    """
    from dataclasses import asdict

    from nanochat.models.quant.retrofit import _attach_luts
    from nanochat.modules import checkpoint_manager as cm

    torch.manual_seed(3)
    source = retrofit_model(tiny_gpt_factory(), PRESETS[DEFAULT_PRESET])
    _attach_luts(source, PRESETS[DEFAULT_PRESET])
    state = export_lcqat_checkpoint(source, str(tmp_path / "lut.pt"))
    learnable = [k for k in state if "learnable_activation_lut" in k]
    assert learnable, "fixture did not attach a learnable table"
    # The baked table is what actually runs at inference and must survive.
    assert any(k.endswith("activation_lut") for k in state)

    ckpt = tmp_path / "lut_ck"
    cm.save_checkpoint(
        str(ckpt),
        1,
        state,
        None,
        {
            "model_config": asdict(source.config),
            "lcqat": asdict(PRESETS[DEFAULT_PRESET]),
        },
    )

    class FakeTokenizer:
        def get_vocab_size(self):
            return source.config.vocab_size

    monkeypatch.setattr(cm, "get_tokenizer", lambda: FakeTokenizer())
    loaded, _tokenizer, _meta = cm.build_model(
        str(ckpt), 1, torch.device("cpu"), "eval"
    )

    assert not any("learnable_activation_lut" in k for k in loaded.state_dict())
    assert loaded.transformer.h[0].mlp.c_fc.activation_lut is not None
    ids = torch.randint(0, loaded.config.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.isfinite(loaded(ids)).all()


def test_when_loading_exported_artifact_then_loads_and_runs(
    tiny_gpt_factory, tmp_path
) -> None:
    from dataclasses import asdict

    from nanochat.models.quant import PRESETS, retrofit_model

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

    from nanochat.models.quant import PRESETS, retrofit_model
    from nanochat.modules import checkpoint_manager as cm

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


def _lcqat_engine(num_blocks: int = 2) -> DiffusionBlockEngine:
    """A DiffusionBlocks engine with LC-QAT on the model *and* its own layers.

    This is the shape `scripts/base_train.py` builds by default: `--lcqat` plus
    `--db-blocks=4` calls `retrofit_model` on the GPT and then
    `engine.apply_lcqat` on the adapters and denoise heads, so all three owned
    subtrees carry quantized Linears. `load_model` returns the engine for any
    checkpoint declaring `meta["db"]`, which is every default run.
    """
    from nanochat.models.quant import PRESETS, retrofit_model

    torch.manual_seed(0)
    cfg = PRESETS[DEFAULT_PRESET]
    model = retrofit_model(build_active_tiny_gpt(), cfg)
    engine = DiffusionBlockEngine(
        model, EquiProbabilityPartitioner(num_blocks=num_blocks), dtype=torch.float32
    )
    engine.apply_lcqat(cfg)
    return engine


def test_when_exporting_a_diffusion_blocks_engine_then_every_owned_subtree_is_stripped(
    tmp_path,
) -> None:
    """Regression: export died on the default configuration.

    `export_lcqat_checkpoint` walked `model.named_modules()`, which a
    `DiffusionBlockEngine` does not have, so every default run -- the ones
    declaring `meta["db"]` -- raised `AttributeError` at export. The engine also
    has to expose `get_submodule` for the activation-LUT sibling walk, which
    resolves `transformer.h.<n>.mlp.c_proj` by dotted path.

    Asserting on the artifact, not on the absence of an exception: an engine
    that merely walked `self.model` would pass a "does not raise" test and then
    ship FP32 adapters with no index buffers.
    """
    engine = _lcqat_engine()

    state = export_lcqat_checkpoint(engine, str(tmp_path / "db_export.pt"))

    assert is_exported_lcqat_state(state)
    assert not is_lcqat_state(state)
    # Every quantized layer in every owned subtree loses its FP32 shadow weight
    # and gains packed indices -- base GPT, adapters, and denoise heads alike.
    quantized = [
        name
        for name, module in engine.named_modules()
        if getattr(module, "weight_quantizer", None) is not None
        and hasattr(module, "K_weight")
    ]
    assert quantized, "fixture did not retrofit anything"
    prefixes = {"transformer": 0, "db_adapters": 0, "db_denoise_heads": 0}
    for name in quantized:
        subtree = name.split(".")[0]
        assert subtree in prefixes, f"unexpected subtree in {name}"
        prefixes[subtree] += 1
        assert not any(k == f"{name}.weight" for k in state), (
            f"shadow weight kept: {name}"
        )
        assert f"{name}.packed_weight_indices" in state, f"no indices for {name}"
    # A pass that only covered the bare GPT would leave these at zero.
    assert prefixes["db_adapters"] > 0, "adapters were not exported"
    assert prefixes["db_denoise_heads"] > 0, "denoise heads were not exported"
    assert prefixes["transformer"] > 0, "base transformer was not exported"


def test_when_exporting_a_diffusion_blocks_engine_then_activation_luts_are_wired(
    tmp_path,
) -> None:
    """The `get_submodule` half of the contract.

    `_activation_pairs` resolves a dotted path to find the quantized sibling that
    consumes an out-quantized layer's output. On an engine that resolution runs
    through `get_submodule`, so a stub returning None would skip every table
    silently and emit an artifact with no activation LUTs at all.
    """
    engine = _lcqat_engine()

    state = export_lcqat_checkpoint(engine, str(tmp_path / "db_lut_export.pt"))

    activation_lut_keys = [
        k for k, v in state.items() if k.endswith("activation_lut") and v.numel() > 1
    ]
    assert activation_lut_keys, "no fused activation tables were wired"


def test_when_exporting_preset_prd_then_uint8_and_trit_formats_coexist(
    tiny_gpt_factory, tmp_path
) -> None:
    from nanochat.models.quant import PRESETS, retrofit_model

    model = retrofit_model(tiny_gpt_factory(), PRESETS["prd"])
    state = export_lcqat_checkpoint(model, str(tmp_path / "prd_export.pt"))
    tags = {int(v) for k, v in state.items() if k.endswith("weight_index_format")}
    # prd preset: q/k K=3 (trits), v/o/fc K=15 (nibbles), down K=255 (uint8)
    assert tags == {FORMAT_TRITS, FORMAT_NIBBLES, FORMAT_UINT8}


def test_when_exporting_k_above_255_then_int32_indices_not_rejected(
    tiny_gpt_factory, tmp_path
) -> None:
    from nanochat.models.quant import LayerKConfig, retrofit_model

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

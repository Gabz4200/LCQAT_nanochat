"""The checkpoint_manager guard for missing diffusion-engine state.

A checkpoint that declares `meta["db"]` but carries no `db_*` keys is always a
writer bug, not a legitimate state. It used to load silently: the loader skipped
the absent sub-dicts and the engine's zero-initialized adapters and denoise
heads were left in place, so the model evaluated as garbage while looking
healthy.
"""

import pytest
import torch

from nanochat.models.quant import PRESETS, retrofit_model
from nanochat.models.quant.retrofit import DEFAULT_PRESET
from nanochat.modules.checkpoint_manager import build_model, save_checkpoint
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
)
from tests.conftest import build_active_tiny_gpt


def _config_kwargs(model):
    c = model.config
    return {
        "sequence_len": c.sequence_len,
        "vocab_size": c.vocab_size,
        "n_layer": c.n_layer,
        "n_head": c.n_head,
        "n_kv_head": c.n_kv_head,
        "n_embd": c.n_embd,
        "window_pattern": c.window_pattern,
    }


def _engine_matching_tokenizer():
    """An engine whose vocab_size matches the on-disk tokenizer.

    `build_model` asserts that match, so the tiny fixture (vocab 128) cannot be
    loaded through it. n_embd is kept small so the vocab-sized embedding stays
    cheap in a test.
    """
    from nanochat.data.tokenizer import get_tokenizer
    from nanochat.models.backbone import GPT, GPTConfig

    vocab = get_tokenizer().get_vocab_size()
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=vocab,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    model = retrofit_model(model, PRESETS[DEFAULT_PRESET])
    engine = DiffusionBlockEngine(model, EquiProbabilityPartitioner(num_blocks=2))
    return engine, _config_kwargs(model)


def test_when_db_meta_present_but_db_keys_missing_then_raises(tmp_path):
    """The W0.2b regression."""
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    # Deliberately the bug: meta declares a diffusion engine, state has no db_*.
    save_checkpoint(
        str(tmp_path),
        1,
        model.state_dict(),
        {},
        {"model_config": _config_kwargs(model), "db": {"num_blocks": 2}},
    )
    with pytest.raises(RuntimeError, match="db_adapters"):
        build_model(str(tmp_path), 1, torch.device("cpu"), phase="eval")


def test_when_db_meta_present_with_db_keys_then_builds(tmp_path):
    """The guard must not fire on a well-formed diffusion checkpoint."""
    engine, config = _engine_matching_tokenizer()
    save_checkpoint(
        str(tmp_path),
        1,
        engine.state_dict(),
        {},
        {"model_config": config, "db": {"num_blocks": 2}},
    )
    built, _tokenizer, meta = build_model(
        str(tmp_path), 1, torch.device("cpu"), phase="eval"
    )
    assert isinstance(built, DiffusionBlockEngine)
    assert len(built.denoise_heads) == meta["db"]["num_blocks"] == 2


def _quantized_engine_matching_tokenizer():
    """An engine whose OWN layers carry LC-QAT + SparseProp, as `base_train` does.

    `_engine_matching_tokenizer` builds a float engine, which is why the test
    above cannot see the loader bug below: it has no `db_*` codebook keys to
    drop. `scripts/_train/build.py` calls `engine.apply_lcqat` and
    `engine.apply_sparseprop` after constructing the engine, so a checkpoint
    written by the default entry point carries `db_adapters.*` /
    `db_denoise_heads.*` quantizer and sparsity keys.
    """
    from nanochat.models.quant.sparseprop import SparsePropLinearLCQAT

    engine, config = _engine_matching_tokenizer()
    engine.apply_lcqat(PRESETS[DEFAULT_PRESET])
    engine.apply_sparseprop(sparsity=0.75, with_lcqat=True)
    assert isinstance(engine.denoise_heads[0], SparsePropLinearLCQAT)
    return engine, config


def test_when_db_engine_layers_are_quantized_then_build_model_retrofits_them(
    tmp_path,
) -> None:
    """`build_model` retrofitted only the bare GPT, not the engine-owned layers.

    The engine is constructed *after* `prepare_lcqat_before_load` /
    `inject_sparseprop_layers` have already run on the GPT, and nothing
    re-applied them to the adapters and denoise heads. `engine.load_state_dict`
    then runs with `strict=False`, so every `db_*` key the fresh layers did not
    expect -- all the codebook params, and `sparsity_mask` / `w_ptr` / `w_col` /
    `w_ptr_csc` -- was discarded without a word.

    The engine came back structurally valid and numerically wrong: adapters and
    heads as plain float `Linear`. Nothing warned. The weights survived, because
    a plain `Linear` does have a `weight`, which is what makes this so easy to
    miss -- the model looks loaded while silently serving unquantized,
    unmasked adapters.
    """
    engine, config = _quantized_engine_matching_tokenizer()
    save_checkpoint(
        str(tmp_path),
        1,
        engine.state_dict(),
        {},
        {"model_config": config, "db": {"num_blocks": 2}},
    )
    saved = engine.state_dict()

    built, _tokenizer, _meta = build_model(
        str(tmp_path), 1, torch.device("cpu"), phase="eval"
    )

    adapter = built.adapters[0].mlp[0]
    head = built.denoise_heads[0]
    for name, module in (("adapter", adapter), ("denoise head", head)):
        assert getattr(module, "weight_quantizer", None) is not None, (
            f"{name} came back unquantized: {type(module).__name__}"
        )
        assert module.sparsity_mask is not None, f"{name} came back without sparsity"
        # Not just "is quantized" -- the trained state has to be the state that
        # loads. A freshly initialized codebook would pass the check above.
        key = "db_denoise_heads.0.weight_quantizer.raw_pos_deltas"
        if name == "adapter":
            key = "db_adapters.0.mlp.0.weight_quantizer.raw_pos_deltas"
        assert torch.equal(module.weight_quantizer.raw_pos_deltas, saved[key]), (
            f"{name} loaded a fresh codebook, not the trained one"
        )
        mask_key = (
            "db_adapters.0.mlp.0.sparsity_mask"
            if name == "adapter"
            else "db_denoise_heads.0.sparsity_mask"
        )
        assert torch.equal(module.sparsity_mask, saved[mask_key])


def test_when_db_engine_layers_are_quantized_then_no_db_key_is_silently_dropped(
    tmp_path,
) -> None:
    """No `db_*` key may vanish, which is what `strict=False` was hiding.

    Naming every key rather than spot-checking two: the failure mode is silent
    truncation of a prefix, so the count is the assertion that matters.
    """
    engine, config = _quantized_engine_matching_tokenizer()
    save_checkpoint(
        str(tmp_path),
        1,
        engine.state_dict(),
        {},
        {"model_config": config, "db": {"num_blocks": 2}},
    )

    built, _tokenizer, _meta = build_model(
        str(tmp_path), 1, torch.device("cpu"), phase="eval"
    )

    missing = set(engine.state_dict()) - set(built.state_dict())
    assert not missing, f"db_* state dropped on load: {sorted(missing)[:6]}"


def test_when_legacy_single_denoise_head_and_multiple_blocks_then_raises(tmp_path):
    """A legacy checkpoint had one head; it cannot be split across B blocks."""
    torch.manual_seed(0)
    base = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    engine = DiffusionBlockEngine(base, EquiProbabilityPartitioner(num_blocks=2))
    state = dict(engine.state_dict())
    # Rewrite the per-block heads back into the legacy single-head layout.
    for b in range(2):
        for suffix in ("weight", "bias"):
            state[f"db_denoise_head.{suffix}"] = state.pop(
                f"db_denoise_heads.{b}.{suffix}"
            )
    save_checkpoint(
        str(tmp_path),
        1,
        state,
        {},
        {"model_config": _config_kwargs(base), "db": {"num_blocks": 2}},
    )
    with pytest.raises(RuntimeError, match="legacy single"):
        build_model(str(tmp_path), 1, torch.device("cpu"), phase="eval")

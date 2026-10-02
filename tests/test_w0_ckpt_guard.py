"""The checkpoint_manager guard for missing diffusion-engine state.

A checkpoint that declares `meta["db"]` but carries no `db_*` keys is always a
writer bug, not a legitimate state. It used to load silently: the loader skipped
the absent sub-dicts and the engine's zero-initialized adapters and denoise
heads were left in place, so the model evaluated as garbage while looking
healthy.
"""

import pytest
import torch

from nanochat.checkpoint_manager import build_model, save_checkpoint
from nanochat.diffusion_blocks import DiffusionBlockEngine, EquiProbabilityPartitioner
from nanochat.lcqat import PRESETS, retrofit_model
from nanochat.lcqat.retrofit import DEFAULT_PRESET
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
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer

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

"""Shared fixtures for LC-QAT tests: a tiny CPU GPT, float and retrofitted."""

import pytest
import torch

from nanochat.models.backbone import GPT, GPTConfig
from nanochat.models.quant import PRESETS, retrofit_model
from nanochat.models.quant.retrofit import DEFAULT_PRESET


def build_tiny_gpt() -> GPT:
    """Materialize a small CPU GPT through the standard meta -> to_empty -> init flow."""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def build_active_tiny_gpt() -> GPT:
    """Tiny GPT whose zero-initialized projections are randomized.

    nanochat zero-inits attn.c_proj / mlp.c_proj, so in an untrained model
    attention and MLP outputs are exactly zero and block outputs equal the
    embedding stream - KV-cache or runtime parity tests would pass vacuously.
    Randomize those projections BEFORE retrofitting so quantizer init spans
    see the real weight ranges.
    """
    model = build_tiny_gpt()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("c_proj.weight"):
                param.normal_(std=0.02)
    return model


@pytest.fixture
def tiny_gpt() -> GPT:
    return build_tiny_gpt()


@pytest.fixture
def tiny_gpt_factory():
    """Callable returning a fresh tiny float GPT (for roundtrip tests)."""
    return build_tiny_gpt


@pytest.fixture
def tiny_gpt_lcqat() -> GPT:
    # Independent instance: retrofitting mutates the model in place, so it must
    # not share state with the `tiny_gpt` fixture.
    #
    # Built with the DEFAULT preset so that a state_dict roundtrip through
    # `prepare_lcqat_before_load(fresh, state, None, None)` -- which falls back
    # to the default when no meta is supplied -- reconstructs the same config.
    return retrofit_model(build_tiny_gpt(), PRESETS[DEFAULT_PRESET])

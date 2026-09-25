"""Shared fixtures for LC-QAT tests: a tiny CPU GPT, float and retrofitted."""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.lcqat import PRESETS, retrofit_model


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
    return retrofit_model(build_tiny_gpt(), PRESETS["small"])

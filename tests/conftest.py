"""Shared fixtures for LC-QAT tests: a tiny CPU GPT, float and retrofitted."""

import pytest

from nanochat.models.backbone import GPT
from nanochat.models.quant import PRESETS, retrofit_model
from nanochat.models.quant.retrofit import DEFAULT_PRESET

# The builders themselves live in the package, not here: the shipped ablation
# harness imports them, and a package that imports `tests/` is only importable
# when the repo root happens to be on sys.path.
from nanochat.modules.experiments.tiny_models import (  # noqa: F401  (re-export)
    build_active_tiny_gpt,
    build_tiny_gpt,
)


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

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
    # Active (randomized) model: quantizer levels must be fitted to real ranges,
    # not the zero-init span of c_proj (see build_active_tiny_gpt gotcha).
    return retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])


# ---------------------------------------------------------------------------
# `base_train` subprocess harness
# ---------------------------------------------------------------------------
# Shared by test_w0_smoke.py and test_lm_mode.py, which both run the real entry
# point rather than calling into it. The interpreter resolution and the smoke
# argument list were duplicated in both; the argument lists had already drifted
# (one carried `--eval-tokens 128` and the other did not), so the two suites were
# validating different configurations while both claimed to test the smoke run.

import os as _os
import subprocess as _subprocess
import sys as _sys


def subprocess_python() -> str:
    """The interpreter that can import this project's dependencies.

    `sys.executable` is only correct when pytest itself runs from the project
    venv. Under a plain `python -m pytest` against the system interpreter it
    points at /usr/bin/python, and every subprocess test then dies at
    `import wandb` before reaching any of the code under test -- which is how
    tests/test_w0_smoke.py came to have subprocess tests that never ran.

    Resolve from the *package* location instead of `sys.executable`: the
    project's own dependencies are importable from the interpreter whose
    site-packages contains `nanochat`, and that interpreter is the venv one
    whether or not pytest itself is running inside it. Falls back to
    `sys.executable` when the venv layout is not there, which is the normal
    in-venv case and needs no special handling.
    """
    venv_python = _os.path.join(_sys.prefix, "bin", "python")
    # A repo checkout inside a venv: the project root is two levels above the
    # tests directory, and the venv interpreter sits beside bin/activate.
    candidate = _os.path.join(
        _os.path.dirname(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        ),
        ".venv",
        "bin",
        "python",
    )
    if _os.path.exists(candidate):
        return candidate
    if _os.path.exists(venv_python):
        return venv_python
    return _sys.executable


#: d6 / seq 64 / batch 2 keeps a run inside the 7.6 GB host budget. Every knob
#: that would make the run long or chatty is disabled; the loop still has to
#: execute `--num-iterations 2` steps.
SMOKE_ARGS = [
    "--depth",
    "6",
    "--num-iterations",
    "2",
    "--max-seq-len",
    "64",
    "--device-batch-size",
    "2",
    "--total-batch-size",
    "256",
    "--run",
    "dummy",
    "--eval-tokens",
    "128",
    "--eval-every",
    "-1",
    "--core-metric-every",
    "-1",
    "--save-every",
    "-1",
    "--sample-every",
    "-1",
]


def run_base_train(extra=(), model_tag=None, timeout=900):
    """Run `python -m scripts.base_train` with `SMOKE_ARGS` plus `extra`.

    Returns the CompletedProcess; callers assert on the return code and on
    stdout, because the failure mode these tests exist for is a *clean-looking*
    run that did the wrong thing.

    `model_tag` gives a run its own checkpoint directory. Tests that exercise
    different flag combinations in sequence need it, or they write into one
    shared directory and a later test resumes an earlier test's checkpoint --
    which passes for the wrong reason.
    """
    argv = [subprocess_python(), "-m", "scripts.base_train", *SMOKE_ARGS]
    if model_tag is not None:
        argv += ["--model-tag", model_tag]
    argv += list(extra)
    return _subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

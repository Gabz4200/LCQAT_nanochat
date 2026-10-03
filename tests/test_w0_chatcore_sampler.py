"""The ChatCORE samplers must be handed the bare GPT, not the engine.

`scripts/chat_sft.py` aliased `orig_model = model` *before* unwrapping a
`DiffusionBlockEngine`. `load_model` returns the engine whenever the checkpoint
declares `meta["db"]` -- which is what every `base_train` run in this repo
writes, since DiffusionBlocks is on by default -- so `orig_model` was the
engine.

`DiffusionBlockEngine` has no `forward`. `Engine(orig_model, tokenizer)` and
`run_chat_eval(..., orig_model)` therefore died with

    AttributeError: 'DiffusionBlockEngine' object has no attribute 'forward'

at the first ChatCORE evaluation, i.e. at step 0 or at the first
`--chatcore-every` boundary of every SFT run started from this repo's own
pretrained checkpoint. `scripts/base_eval.py` had already worked around the same
trap with `getattr(model, "model", model)`; `chat_sft` had not.

These tests pin the aliasing order and the sampler arguments. The behavioural
half is here too: it asserts the unwrap yields something the Engine can run.
"""

import ast
import pathlib

import pytest
import torch

from nanochat.training.diffusion_blocks import DiffusionBlockEngine

CHAT_SFT = "scripts/chat_sft.py"


def _tree() -> ast.Module:
    return ast.parse(pathlib.Path(CHAT_SFT).read_text())


def _assignments(tree, name):
    """Every `(lineno, value)` bound to `name`.

    Walks the whole module: `base_model` is bound inside the engine-unwrap
    `if/else`, so a top-level-only scan would miss it.
    """
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    out.append((node.lineno, ast.unparse(node.value)))
    return out


def _engine_constructions(tree):
    """Every `Engine(...)` construction with its first argument spelled out."""
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Engine"
        ):
            arg = ast.unparse(node.args[0]) if node.args else None
            out.append((node.lineno, arg))
    return out


def _unwrap_line(tree) -> int:
    """The line where `chat_sft` unwraps the engine, i.e. `base_model = ...`."""
    found = _assignments(tree, "base_model")
    if not found:
        raise AssertionError("chat_sft never binds base_model")
    return min(line for line, _ in found)


class TestNoPrematureOrigModelAlias:
    def test_orig_model_is_not_aliased_before_the_engine_unwrap(self):
        tree = _tree()
        aliases = _assignments(tree, "orig_model")
        assert not aliases, (
            f"chat_sft binds orig_model at {aliases}. Any alias taken before the "
            f"line {_unwrap_line(tree)} unwrap can capture a DiffusionBlockEngine, "
            "which has no forward."
        )

    def test_there_is_no_orig_model_binding_at_all(self):
        """Simpler statement of the same rule; keep one owner of the bare GPT."""
        assert not _assignments(_tree(), "orig_model")

    def test_the_unwrap_binds_base_model_to_the_inner_gpt(self):
        tree = _tree()
        base = _assignments(tree, "base_model")
        assert base, "chat_sft must bind base_model"
        line, value = base[0]
        # Either `engine.model` (already an engine) or the bare model.
        assert value in ("engine.model", "model"), value
        assert line > 0


class TestSamplersReceiveTheBareGpt:
    def test_the_ar_engine_is_built_from_base_model(self):
        constructions = _engine_constructions(_tree())
        assert constructions, "chat_sft builds no Engine; the test is vacuous"
        for line, arg in constructions:
            assert arg == "base_model", (
                f"line {line}: Engine({arg}) -- must be base_model, the bare GPT"
            )

    def test_run_chat_eval_is_given_base_model(self):
        """The categorical and generative eval paths both read it."""
        found = []
        for node in ast.walk(_tree()):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_chat_eval"
            ):
                found.append([ast.unparse(a) for a in node.args])
        assert found, "chat_sft never calls run_chat_eval"
        for args in found:
            assert "base_model" in args, args
            assert "orig_model" not in args, args


class TestTheTrapIsReal:
    """The premise: an engine handed to `Engine` cannot run."""

    def test_a_diffusionblockengine_has_no_forward(self):
        assert not hasattr(DiffusionBlockEngine, "forward")

    def test_the_unwrap_is_what_makes_it_runnable(self):
        """`getattr(model, "model", model)` -- the base_eval workaround."""
        engine = DiffusionBlockEngine.__new__(DiffusionBlockEngine)
        engine.model = torch.nn.Linear(2, 2)
        base_model = getattr(engine, "model", engine)
        assert base_model is engine.model
        assert callable(base_model.forward)


class TestChatRlHasTheSameShape:
    """`chat_rl` already named the sampler `ar_engine` and passed `base_model`."""

    def test_chat_rl_passes_base_model_to_its_sampler(self):
        tree = ast.parse(pathlib.Path("scripts/chat_rl.py").read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Engine"
            ):
                arg = ast.unparse(node.args[0]) if node.args else None
                assert arg == "base_model", f"Engine({arg})"


@pytest.mark.parametrize("script", ["scripts/chat_sft.py", "scripts/chat_rl.py"])
def test_neither_chat_script_aliases_the_loaded_object(script):
    """Both load through `load_model`, which may return the engine."""
    tree = ast.parse(pathlib.Path(script).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "orig_model":
                    raise AssertionError(
                        f"{script}:{node.lineno} aliases orig_model = "
                        f"{ast.unparse(node.value)}; load_model can return a "
                        "DiffusionBlockEngine, which has no forward"
                    )

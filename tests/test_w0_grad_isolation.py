"""The optimizer step must clear every gradient the optimizer owns.

`scripts/_train/loop.py` zeroed `ctx.model` after `optimizer.step()`, but the
optimizer owns `ctx.trainable_root` -- which in DiffusionBlocks mode is the
*engine*, three `nn.Module`s it holds rather than one. The engine's
`db_adapters.*` and `db_denoise_heads.*` are unreachable from `ctx.model`, so
their gradients survived the step.

`_apply_requires_grad` does not save them: it clears `p.grad` only for
parameters the *active* block does not own. So if a block is drawn twice inside
one optimizer step, the second draw's adapter and denoise-head gradients
accumulate onto the first and the update lands at exactly double the intended
magnitude -- silently, and with no role check able to see it, because no role is
missing.

The two end-to-end tests below measure exactly that ratio, and it is 1.0000 with
the fix and 2.0000 without it.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from nanochat.modules.experiments.tiny_models import build_active_tiny_gpt, make_engine

#: `denoise_step` draws some of its randomness from the global RNG, so both
#: draws below pin it. Without this the two draws differ by ~1.3x on their own
#: and the accumulation is invisible in the noise.
_GLOBAL_SEED = 7


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _engine_grads(engine, prefix="db_"):
    """Clone the engine-owned gradients (`db_adapters.*`, `db_denoise_heads.*`)."""
    return {
        n: p.grad.detach().clone()
        for n, p in engine.named_parameters()
        if p.grad is not None and n.startswith(prefix)
    }


def _draw(engine, idx, block_idx, seed=0):
    """One EDM step plus its backward, with both RNGs pinned.

    `denoise_step` returns the loss; the training loop calls `backward()` itself,
    so a test that forgets that sees no gradients at all.
    """
    torch.manual_seed(_GLOBAL_SEED)
    loss, _sigma = engine.denoise_step(idx, block_idx=block_idx, generator=_gen(seed))
    loss.backward()
    return _engine_grads(engine)


def _ids(params):
    return {id(p) for p in params}


@pytest.fixture
def engine():
    return make_engine(num_blocks=2, n_layer=4)


@pytest.fixture
def draw_env():
    """(engine, token ids, block index) ready for two comparable draws."""
    eng = make_engine(num_blocks=2, n_layer=4)
    eng.train()
    idx = torch.randint(0, 128, (1, 8))
    block = eng.sample_block(_gen(0))
    return eng, idx, block


class TestEngineOwnsMoreThanItsModel:
    def test_the_engine_owns_parameters_the_model_cannot_reach(self, engine):
        """The premise. If this stops holding, the wrong-tree zero_grad is moot."""
        extra = _ids(engine.parameters()) - _ids(engine.model.parameters())
        assert extra, (
            "DiffusionBlockEngine currently owns nothing its model does not, so "
            "the wrong-tree zero_grad is unreachable -- delete this test and the "
            "loop fix together"
        )

    def test_the_unreachable_parameters_are_the_engine_owned_ones(self, engine):
        model_ids = _ids(engine.model.parameters())
        extra = [n for n, p in engine.named_parameters() if id(p) not in model_ids]
        assert extra
        assert all(
            n.startswith("db_adapters.") or n.startswith("db_denoise_heads.")
            for n in extra
        ), extra


class TestZeroingTheRightTree:
    def test_when_zeroing_the_model_then_engine_grads_survive(self, engine):
        """The bug, asserted so it cannot be reintroduced silently."""
        for p in engine.parameters():
            p.grad = torch.ones_like(p)

        engine.model.zero_grad(set_to_none=True)

        survivors = [n for n, p in engine.named_parameters() if p.grad is not None]
        assert survivors, "expected a model-only zero_grad to miss engine params"
        assert all(
            n.startswith("db_adapters.") or n.startswith("db_denoise_heads.")
            for n in survivors
        ), survivors

    def test_when_zeroing_the_trainable_root_then_every_grad_is_cleared(self, engine):
        """The fix: `trainable_root` is the engine."""
        for p in engine.parameters():
            p.grad = torch.ones_like(p)

        engine.zero_grad(set_to_none=True)

        assert [n for n, p in engine.named_parameters() if p.grad is not None] == []

    def test_in_lm_mode_the_two_roots_are_the_same_object(self):
        """`--db-blocks 0` sets `trainable_root = model`, so the fix is a no-op.

        Checked through `build_optimizer` rather than by comparing a parameter
        set to itself, which would assert nothing.
        """
        from scripts._train.build import build_optimizer

        model = build_active_tiny_gpt()
        args = SimpleNamespace(
            matrix_lr=1e-3,
            embedding_lr=0.3,
            unembedding_lr=0.008,
            scalar_lr=0.5,
            codebook_lr=1e-3,
            dmodel_lr_scale=1.0,
        )
        run_config = SimpleNamespace(
            batch_lr_scale=1.0,
            weight_decay_scaled=0.1,
            dmodel_lr_scale=1.0,
        )
        _optimizer, trainable_root = build_optimizer(
            args, model, None, False, run_config, "cpu", False, None
        )
        assert trainable_root is model


class TestOptimizerAndZeroingAgree:
    def test_every_optimizer_parameter_lives_under_the_trainable_root(self, engine):
        """The invariant the loop depends on, asserted directly.

        Built the way `build_optimizer` does it: `trainable_root` is the engine
        when DiffusionBlocks is on, and the optimizer is built over that root.
        """
        from nanochat.models.quant.optimizer import build_qat_param_groups

        param_groups = build_qat_param_groups(engine, matrix_lr=1e-3, weight_decay=0.1)
        owned = {id(p) for g in param_groups for p in g["params"]}

        assert owned == _ids(engine.parameters()), (
            "the optimizer and trainable_root disagree about which parameters "
            "exist, so zeroing the root would miss or over-reach"
        )
        assert owned - _ids(engine.model.parameters())


class TestGradientAccumulation:
    def test_with_the_right_zero_then_a_second_draw_does_not_accumulate(self, draw_env):
        """The fix, measured end to end: the ratio is exactly 1."""
        engine, idx, block = draw_env

        first = _draw(engine, idx, block)
        assert first, "no engine-owned gradient arrived; the test would be vacuous"

        engine.zero_grad(set_to_none=True)
        second = _draw(engine, idx, block)

        assert set(second) == set(first), "the set of live engine grads changed"
        for name, before in first.items():
            (
                torch.testing.assert_close(second[name], before, rtol=1e-5, atol=1e-6),
                f"{name} accumulated across the zero_grad",
            )

    def test_with_the_wrong_zero_then_a_second_draw_doubles(self, draw_env):
        """The bug, measured end to end: the ratio is exactly 2.

        This is the anti-vacuity guard for the test above -- it asserts the
        accumulation is real and large, not a rounding artefact.
        """
        engine, idx, block = draw_env

        first = _draw(engine, idx, block)
        assert first

        engine.model.zero_grad(set_to_none=True)  # the bug
        second = _draw(engine, idx, block)

        assert set(second) == set(first)
        for name, before in first.items():
            (
                torch.testing.assert_close(
                    second[name], before * 2, rtol=1e-4, atol=1e-6
                ),
                f"{name} did not double, so the premise of this file is wrong",
            )

    def test_the_two_draws_agree_when_nothing_accumulates(self, draw_env):
        """Determinism check: with a full zero_grad the two draws are identical.

        Without this, the 1x and 2x assertions above could both be passing
        because the draws differ in some third way.
        """
        engine, idx, block = draw_env
        first = _draw(engine, idx, block)
        engine.zero_grad(set_to_none=True)
        second = _draw(engine, idx, block)
        for name, before in first.items():
            torch.testing.assert_close(second[name], before, rtol=0, atol=0)


class TestLoopUsesTrainableRoot:
    def test_the_loop_zeroes_the_trainable_root(self):
        """Source-level guard: the loop must not regress to `ctx.model`.

        A behavioural test needs a full `base_train` subprocess; this pins the
        one token that is the whole fix, and the tests above pin why it has to
        be that token.
        """
        src = pathlib.Path("scripts/_train/loop.py").read_text()
        zeroed = [
            ast.unparse(node.func.value)
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "zero_grad"
        ]
        assert "ctx.trainable_root" in zeroed, zeroed
        assert "ctx.model" not in zeroed, (
            "ctx.model.zero_grad is back; it cannot reach db_adapters.* / "
            "db_denoise_heads.*"
        )

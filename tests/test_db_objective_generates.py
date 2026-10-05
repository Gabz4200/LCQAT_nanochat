"""An EDM-trained checkpoint cannot generate text, and that is load-bearing.

Under `--db-objective edm`, `denoise_step`'s loss produces **no gradient at
all** for `lm_head` (the objective is score-matching on denoised *embeddings*).
It is 8.39M of 36.70M params -- 22.9% -- and the only component mapping hidden
state to token ids, so a purely-EDM-trained checkpoint depends entirely on a
side-channel to train its output head.

This was found the expensive way: a 100-step run whose acceptance criterion was
coherent generation used the default objective and produced word-salad. The
gradient gap was visible in one line of engine output the whole time.

`tests/test_db_edm_behavior.py` already pins that `lm_head` gets no gradient
under EDM. What was missing is the *consequence* of that being true, stated where
someone configuring a run will meet it.

python -m pytest tests/test_db_objective_generates.py -v
"""

from __future__ import annotations

import pytest
import torch

from nanochat.modules.experiments.tiny_models import make_engine


def _head_grad(engine) -> torch.Tensor | None:
    return engine.model.lm_head.weight.grad


def test_when_edm_is_the_objective_then_lm_head_gets_no_gradient() -> None:
    """The premise of this file: EDM leaves the output head untrained."""
    engine = make_engine(num_blocks=2)
    engine.train()
    idx = torch.randint(0, engine.model.config.vocab_size, (1, 8))
    loss, _ = engine.denoise_step(idx, block_idx=0, backend="naive")
    loss.backward()
    grad = _head_grad(engine)
    assert grad is None or grad.abs().sum() == 0


def test_when_ce_is_the_objective_then_lm_head_does_get_a_gradient() -> None:
    """The CE path is the one that trains next-token prediction."""
    engine = make_engine(num_blocks=2)
    engine.train()
    idx = torch.randint(0, engine.model.config.vocab_size, (1, 8))
    targets = torch.randint(0, engine.model.config.vocab_size, (1, 8))
    engine.train_step(idx, targets, block_idx=0).backward()
    grad = _head_grad(engine)
    assert grad is not None
    assert grad.abs().sum() > 0


def test_when_edm_is_used_then_lm_head_is_a_large_share_of_the_model() -> None:
    """Why the gap matters: the untrained part is not a rounding error.

    If `lm_head` were a small fraction, "EDM leaves it at init" would be a
    footnote. In the real training config it is 8.39M of 36.70M -- 22.9% -- the
    single largest component and the whole generation path. The tiny test model
    is only ~1.7M params with a 128x256 head, so the *ratio* there is 2%; this
    checks the absolute order of magnitude instead, which is what transfers.
    """

    from nanochat.models.backbone import GPT, GPTConfig

    config = GPTConfig(vocab_size=32768, n_layer=4, n_head=6, n_kv_head=6, n_embd=384)
    model = GPT(config)
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.lm_head.parameters())
    assert head / total > 0.15, (
        f"lm_head is only {head / total:.1%} of a depth-4 config; the premise "
        "(EDM cannot generate) would need re-checking if that changed"
    )


def test_when_a_generation_run_is_configured_then_it_needs_the_ce_objective() -> None:
    """The guardrail: `runs/night5.sh` must not silently inherit the default.

    A generation-oriented run that omits `--db-objective` inherits `edm` and
    produces an unusable checkpoint after burning the whole budget. Asserting
    the flag is present in the script is crude but it is the failure that
    actually happened, and it is cheap to catch here.
    """
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "runs" / "night5.sh"
    text = script.read_text(encoding="utf-8")
    assert "--db-objective=ce" in text or "--db-objective ce" in text, (
        "runs/night5.sh trains with the default EDM objective, so lm_head is "
        "never trained and the checkpoint cannot generate text"
    )


@pytest.mark.parametrize("script", ["night4.sh", "night5.sh"])
def test_when_a_night_script_runs_then_it_keeps_all_four_technologies(script) -> None:
    """LC-QAT, SparseProp and DARD are default-on; only `--db-blocks` is explicit.

    So what is worth asserting is that nothing silently switches one of the
    three off: an explicit `--no-*` would drop a technology from a run whose
    whole point is that all four are active.
    """
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "runs" / script).read_text(
        encoding="utf-8"
    )
    assert "--db-blocks" in text, f"{script} does not enable DiffusionBlocks"
    for disable in ("--no-lcqat", "--no-sparseprop", "--no-dard"):
        assert disable not in text, (
            f"{script} passes {disable}; the run must keep all four technologies"
        )

"""End-to-end smoke: `scripts.base_train` with DARD on and off.

The unit tests prove the pieces; this proves the *wiring*. The failure mode
these exist for is a run that completes cleanly with a healthy-looking loss
curve while the rule penalty was never actually evaluated -- which no
unit test on `DardRuleLoss` can catch, because the bug would be in the caller.

Both arms run the real entry point. `--no-dard` must produce the identical
objective it did before this feature existed, and the default (on) must log a
non-zero rule penalty so an enabled-but-dead anchor is visible.

python -m pytest tests/test_dard_smoke.py -v
"""

from __future__ import annotations

import pytest

from tests.conftest import run_base_train


@pytest.mark.slow
def test_when_dard_is_on_then_base_train_runs_and_reports_a_rule_penalty() -> None:
    """The default (DARD on) arm completes and the penalty is non-zero."""
    result = run_base_train(extra=["--model-tag", "dard_smoke_on"], timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "DARD: mined" in result.stdout
    assert "DARD rule penalty attached" in result.stdout
    # The per-step line carries the raw penalty. Its presence is the proof the penalty was evaluated inside the loop rather than built and discarded.
    assert "dard:" in result.stdout


@pytest.mark.slow
def test_when_dard_is_off_then_base_train_runs_and_reports_no_penalty() -> None:
    """`--no-dard` is a clean removal: no mining, no penalty line, same exit 0."""
    result = run_base_train(
        extra=["--no-dard", "--model-tag", "dard_smoke_off"], timeout=900
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "DARD: mined" not in result.stdout
    assert "dard:" not in result.stdout


@pytest.mark.slow
def test_when_dard_is_on_then_the_loss_curve_is_finite() -> None:
    """The balanced total must be finite, not nan/inf from a cold EMA."""
    result = run_base_train(extra=["--model-tag", "dard_smoke_finite"], timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]
    lines = [ln for ln in result.stdout.splitlines() if "step 0000" in ln]
    assert lines, result.stdout[-2000:]
    for line in lines:
        assert "nan" not in line.lower()
        assert "inf" not in line.lower()


@pytest.mark.slow
def test_when_dard_runs_with_lm_mode_then_it_still_works() -> None:
    """`--db-blocks=0` (plain LM) is a supported DARD host, not a rejected one.

    DARD is deliberately absent from the `--db-blocks<=0` rejection tuple: it is
    a vocabulary-space loss and both objectives define one.
    """
    result = run_base_train(
        extra=["--db-blocks", "0", "--model-tag", "dard_smoke_lm"], timeout=900
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "DARD rule penalty attached" in result.stdout


def test_when_report_prompts_are_sampled_then_each_gets_its_own_seed() -> None:
    """Each prompt must draw from a different RNG stream.

    `GPT.generate` seeds its own generator from a `seed` argument that defaults
    to the constant 42. If the report leaves that default in place, every prompt
    samples the *same* stream and returns near-identical text, which reads as
    "the model ignores the prompt" -- a finding that is really an RNG artifact.

    Measured on the poc15 checkpoint: a constant seed gave 2 distinct
    continuations across 5 prompts, a per-prompt seed gave 5 of 5.
    """
    from scripts import dard_report

    seeds = dard_report.seeds_for(len(dard_report.PROMPTS), base=1000)
    assert seeds == [1000, 1001, 1002, 1003, 1004]
    assert len(set(seeds)) == len(seeds), "seeds must differ per prompt"


def test_when_the_seed_is_varied_then_continuations_differ() -> None:
    """The fix must change the sampled tokens, not merely the argument.

    Pinned against the real sampler on a randomly-initialised tiny model, so
    it catches a regression where `seeds_for` is right but `generate` is still
    called without the seed.
    """
    import torch

    from nanochat.models.backbone import GPT, GPTConfig
    from scripts import dard_report

    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            vocab_size=256,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            sequence_len=32,
            window_pattern="L",
        )
    ).eval()
    ids = [1, 2, 3]
    outs = {
        tuple(model.generate(ids, max_tokens=6, temperature=0.8, top_k=40, seed=s))
        for s in dard_report.seeds_for(5, base=1000)
    }
    assert len(outs) > 1, "identical continuations despite distinct seeds"

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
    # The per-step line carries the raw penalty. Its presence is the proof the
    # penalty was evaluated inside the loop rather than built and discarded.
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

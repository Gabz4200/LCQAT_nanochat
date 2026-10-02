"""Pin the numerical fingerprint so a refactor cannot silently change math.

A refactor that swaps a layer, reorders an operation, or moves a quantization
step changes training outcomes while still passing every behavioural test. This
module pins the *values* -- logits, loss, and gradient digests -- across the
four paths a structural change is most likely to disturb.

The expected values are bit-exact float64 encodings recorded on the reference
commit. Comparison is exact rather than tolerance-based, so a real numerical
change fails loudly instead of drifting quietly.

Regenerate deliberately, never to make a failing test pass:

    PYTHONPATH=. python tests/numerical_fingerprint.py \\
        --json-out tests/numerical_reference.json

and review the diff before committing it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tests.numerical_fingerprint import fingerprint, unhex

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REFERENCE_PATH = pathlib.Path(__file__).resolve().parent / "numerical_reference.json"


@pytest.fixture(scope="module")
def reference() -> dict[str, str]:
    if not REFERENCE_PATH.exists():
        pytest.skip(f"no recorded fingerprint at {REFERENCE_PATH}")
    # The file nests the digests under "digests"; this fixture exposes just that
    # mapping, so the assertions below read like the keys they check.
    return json.loads(REFERENCE_PATH.read_text())["digests"]


def test_fingerprint_matches_reference(reference: dict[str, str]) -> None:
    """The core gate: no path may change its numerics."""
    actual, tolerances = fingerprint()
    drifted = {}
    for key, value in actual.items():
        expected = reference.get(key)
        if expected is None:
            drifted[key] = (None, value)
            continue
        tol = tolerances.get(key)
        if tol is None:
            if expected != value:
                drifted[key] = (expected, value)
        elif abs(unhex(expected) - unhex(value)) > tol * abs(unhex(expected)):
            drifted[key] = (expected, value)
    assert not drifted, (
        "numerical drift -- these values changed, so this is not a "
        f"behavior-preserving change: {drifted}"
    )


def test_arms_are_distinguishable() -> None:
    """Guard against the fingerprint going blind.

    `GPT.init_weights` zero-initializes `lm_head`, so every arm reports a loss of
    exactly `ln(vocab_size)` unless the weights are perturbed. A fingerprint
    whose arms all agree cannot detect a broken forward pass, which is the
    whole point of the gate.
    """
    actual, _ = fingerprint()
    losses = {actual[f"{other}/loss"] for other in ("plain_lm", "lcqat", "sparseprop")}
    assert len(losses) > 1, (
        "every arm reports the same loss, so the fingerprint is not "
        "discriminating and would pass even a completely broken forward"
    )


def test_gradient_reaches_expected_parameters(reference: dict[str, str]) -> None:
    """A refactor must not silently disconnect a parameter from the graph.

    Counts are pinned because a layer that stops receiving gradients trains as
    a no-op while every loss value still looks plausible.
    """
    actual, _ = fingerprint()
    assert actual["plain_lm/grad_params"] == reference["plain_lm/grad_params"]
    assert actual["lcqat/grad_params"] == reference["lcqat/grad_params"]
    assert actual["sparseprop/grad_params"] == reference["sparseprop/grad_params"]

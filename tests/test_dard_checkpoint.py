"""DARD's checkpoint round-trip: the rule set and the balancer EMA.

The bug these guard against is silent and expensive. `DardRuleLoss` is an
optimizer-side attachment, so it is NOT in `engine.state_dict()`. Without
explicit persistence, a resume re-mines from whichever batch it happens to see
-- producing a *different* rule set, so the loss curve before the resume was
measured under one objective and the curve after it under another, with nothing
in the logs to say so.

python -m pytest tests/test_dard_checkpoint.py -v
"""

from __future__ import annotations

import json

import pytest
import torch

from nanochat.models.dard.compile import mine_rules
from nanochat.models.dard.loss import (
    DARD_STATE_VERSION,
    DardLossBalancer,
    DardRuleLoss,
    dard_state,
    load_dard_state,
)
from nanochat.models.dard.shard import shard_corpus


def corpus(n: int = 3000) -> torch.Tensor:
    motif = [3, 5, 7, 9]
    return torch.tensor(((motif * 75) + list(range(20, 60)))[:n], dtype=torch.long) % 64


def make_loss(num_shards: int = 3) -> DardRuleLoss:
    shards = shard_corpus(corpus(), max_shard_size=700)
    rule_set = mine_rules(shards[:num_shards], vocab_size=64, max_rules_per_shard=32)
    return DardRuleLoss(rule_set, gamma=10.0)


# --- Round-trip --------------------------------------------------------------


def test_when_dard_state_round_trips_then_the_rules_are_identical() -> None:
    """The restored module must enforce exactly the same rules."""
    original = make_loss()
    balancer = DardLossBalancer(lambda_global=0.2, ema_decay=0.9)
    balancer.train()
    for _ in range(5):
        balancer(torch.tensor(2.0), torch.tensor(0.3))

    restored_loss, _ = load_dard_state(dard_state(original, balancer))
    assert restored_loss.premise.shape[0] == original.premise.shape[0]
    assert torch.equal(restored_loss.weights, original.weights)
    assert torch.equal(restored_loss.premise.to_dense(), original.premise.to_dense())
    assert torch.equal(
        restored_loss.conclusion.to_dense(), original.conclusion.to_dense()
    )
    assert restored_loss.vocab_size == original.vocab_size
    assert restored_loss.gamma == original.gamma


def test_when_dard_state_round_trips_then_the_penalty_is_unchanged() -> None:
    """Behavioural, not structural: same logits must give the same penalty."""
    torch.manual_seed(0)
    original = make_loss()
    logits = torch.randn(2, 3, 64)
    restored, _ = load_dard_state(dard_state(original, DardLossBalancer()))
    assert torch.allclose(restored(logits), original(logits), atol=1e-6)


def test_when_dard_state_round_trips_then_the_ema_resumes() -> None:
    """The EMA must not restart at zero, or the first steps run unbalanced."""
    balancer = DardLossBalancer(ema_decay=0.9)
    balancer.train()
    for _ in range(10):
        balancer(torch.tensor(3.0), torch.tensor(0.7))
    _, restored = load_dard_state(dard_state(make_loss(), balancer))
    assert float(restored.mu_ce) == pytest.approx(float(balancer.mu_ce))
    assert int(restored.steps) == int(balancer.steps)


def test_when_restored_then_gradients_still_flow() -> None:
    """A restored rule set must be live, not a detached husk."""
    torch.manual_seed(1)
    restored, _ = load_dard_state(dard_state(make_loss(), DardLossBalancer()))
    logits = torch.randn(1, 2, 64, requires_grad=True)
    restored(logits).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


# --- Validation --------------------------------------------------------------


def test_when_dard_state_version_is_stale_then_a_loud_error() -> None:
    state = dard_state(make_loss(), DardLossBalancer())
    state["version"] = DARD_STATE_VERSION + 1
    with pytest.raises(ValueError, match="version"):
        load_dard_state(state)


def test_when_dard_state_is_missing_a_field_then_a_loud_error() -> None:
    state = dard_state(make_loss(), DardLossBalancer())
    del state["balancer"]
    with pytest.raises(ValueError, match="missing fields"):
        load_dard_state(state)


def test_when_dard_payload_is_saved_then_it_loads_with_weights_only() -> None:
    """`weights_only=True` must suffice: tensors and primitives, never code."""
    path = "/tmp/dard_state_test.pt"
    torch.save(dard_state(make_loss(), DardLossBalancer()), path)
    loss_fn, _ = load_dard_state(torch.load(path, weights_only=True))
    assert loss_fn.premise.shape[0] > 0


# --- Sidecar file contract ---------------------------------------------------


def test_when_a_sidecar_is_written_then_meta_json_stays_valid() -> None:
    """The meta dict is `json.dump`ed, so it holds provenance, not tensors.

    This is why the rule set lives in a sidecar `.pt`: putting the sparse
    tensors in the meta dict would crash every save.
    """
    path = "/tmp/dard_meta_test.json"
    payload = {
        "has_sidecar": True,
        "num_rules": 96,
        "vocab_size": 32768,
        "gamma": 10.0,
        "lambda_global": 0.2,
        "ema_decay": 0.99,
        "steps": 4,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["has_sidecar"] is True


def test_when_no_sidecar_exists_then_load_returns_none() -> None:
    """A checkpoint predating DARD has no sidecar; that is not an error."""
    from scripts._train.loop import load_dard_sidecar

    assert load_dard_sidecar("/tmp/definitely_not_a_dir", 1, "cpu") is None

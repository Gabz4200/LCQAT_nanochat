"""Regression: mined rules must actually be *violable*.

The bug this guards against was silent and total. `_premise_successors` emitted
a 4-token premise against a **1-token** conclusion. The penalty compares two
product t-norms, so `v(premise) = O(p^4)` was always smaller than
`v(conclusion) = O(p)`, the order could never be violated, and the penalty sat
at the hinge floor on every step forever.

Measured on a real 4-layer model before the fix:

```
rules=512  tok/rule: premise 4.0, concl 1.0
v_premise mean=8.4e-19   v_concl mean=3.1e-05
FRAC VIOLATED: 0.0000
penalty=0.069299   floor=0.069315
```

The loss curve looked healthy, the feature reported itself enabled, and it was
contributing no gradient at all. A test that only checks "the loss is finite"
or "the loss decreases" passes straight through this.

python -m pytest tests/test_dard_violability.py -v
"""

from __future__ import annotations

import torch

from nanochat.models.dard.atomize import (
    MAX_ARITY_GAP,
    MIN_BALANCED_ARITY,
    atomize_shard,
)
from nanochat.models.dard.compile import mine_rules
from nanochat.models.dard.loss import DardRuleLoss
from nanochat.models.dard.shard import shard_corpus


def corpus(vocab: int = 512, n: int = 4000) -> torch.Tensor:
    """Structured text: a recurring motif, so premises genuinely recur."""
    motif = [3, 5, 7, 9]
    return (
        torch.tensor(((motif * 100) + list(range(20, 200)))[:n], dtype=torch.long)
        % vocab
    )


def mined(vocab: int = 512, n_shards: int = 6, per_shard: int = 64) -> DardRuleLoss:
    shards = shard_corpus(corpus(vocab), max_shard_size=500)
    rule_set = mine_rules(
        shards[:n_shards], vocab_size=vocab, max_rules_per_shard=per_shard
    )
    assert rule_set.num_rules > 0, "mining produced no rules to test"
    return DardRuleLoss(rule_set, gamma=10.0)


def violation_fraction(loss_fn: DardRuleLoss, logits: torch.Tensor) -> float:
    """Fraction of (rule, position) pairs whose order is actually violated."""
    flat = logits.reshape(-1, logits.size(-1))
    log_p = torch.log(torch.softmax(flat, dim=-1) + loss_fn.eps)
    v_premise = torch.sparse.mm(loss_fn.premise, log_p.t()).t()
    v_conclusion = torch.sparse.mm(loss_fn.conclusion, log_p.t()).t()
    violation = torch.exp(v_premise) - torch.exp(v_conclusion)
    return float((violation > 0).to(torch.float32).mean())


# --- The regression itself ---------------------------------------------------


def test_when_rules_are_mined_then_they_are_violable() -> None:
    """A rule that can never be violated teaches nothing.

    This is the assertion that fails on the 4-token-premise / 1-token-
    conclusion bug, where the violation fraction was exactly 0.0000.
    """
    torch.manual_seed(0)
    loss_fn = mined()
    logits = torch.randn(8, loss_fn.vocab_size)
    frac = violation_fraction(loss_fn, logits)
    assert frac > 0.5, (
        f"only {frac:.1%} of mined rules are violable; the penalty is pinned at "
        "its hinge floor and contributes no gradient"
    )


def test_when_rules_are_mined_then_the_penalty_is_above_the_hinge_floor() -> None:
    """The penalty must sit above `log(2)/gamma`, not on it.

    At the floor every violation is zero, `softplus` is at its minimum, and the
    gradient w.r.t. the logits is negligible -- so the feature is inert while
    still reporting a plausible non-zero loss.
    """
    torch.manual_seed(1)
    loss_fn = mined()
    logits = torch.randn(8, loss_fn.vocab_size)
    floor = torch.log(torch.tensor(2.0)).item() / loss_fn.gamma
    assert loss_fn(logits).item() > floor


def test_when_the_model_shifts_mass_then_the_penalty_moves() -> None:
    """The penalty must respond to the distribution, not just be non-zero.

    A constant penalty equal to the floor would also pass the two tests above.
    Pushing probability onto a rule's premise has to raise it.
    """
    torch.manual_seed(2)
    loss_fn = mined()
    floor = torch.log(torch.tensor(2.0)).item() / loss_fn.gamma

    neutral = torch.zeros(1, 1, loss_fn.vocab_size)
    # Take rule 0's premise tokens and load mass onto them.
    start = int(loss_fn.premise.crow_indices()[0])
    end = int(loss_fn.premise.crow_indices()[1])
    premise_tokens = loss_fn.premise.col_indices()[start:end].tolist()

    loaded = neutral.clone()
    loaded[0, 0, premise_tokens] = 6.0
    assert loss_fn(loaded).item() > loss_fn(neutral).item()
    assert loss_fn(loaded).item() > floor


# --- The arity invariant, checked structurally -------------------------------


def test_when_rules_are_emitted_then_the_two_sides_have_comparable_arity() -> None:
    """Structural guard: `|premise| - |conclusion| < MAX_ARITY_GAP`.

    Cheaper and more direct than the behavioural test above, and it names the
    invariant rather than a symptom.
    """
    result = atomize_shard(corpus()[:500])
    assert result.rules, "no rules mined"
    for rule in result.rules:
        gap = len(rule.premise) - len(rule.conclusion)
        assert gap < MAX_ARITY_GAP, (
            f"premise has {len(rule.premise)} tokens against a conclusion of "
            f"{len(rule.conclusion)}; the premise's product t-norm is "
            "exponentially smaller, so the rule can never be violated"
        )


def test_when_rules_are_emitted_then_both_sides_have_minimum_arity() -> None:
    """A 1-token conclusion against a wider premise is the unsatisfiable case."""
    result = atomize_shard(corpus()[:500])
    for rule in result.rules:
        assert len(rule.conclusion) >= MIN_BALANCED_ARITY


def test_when_mined_then_premises_recur_enough_to_support_a_conclusion() -> None:
    """A premise must be seen more than once, or it has no continuation set.

    Mining only the widest window hit this on real text: a 4-token window
    essentially never recurs, so balancing its arity left nothing to pair it
    with and the shard mined zero rules.
    """
    result = atomize_shard(corpus()[:500])
    assert result.rules, "mining produced no rules"
    widths = {len(r.premise) for r in result.rules}
    assert min(widths) < 4, (
        f"every premise is {min(widths)} tokens wide; narrow premises are what "
        "let a rule recur often enough to carry a conclusion"
    )

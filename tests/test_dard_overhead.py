"""Measure DARD's per-step wall-clock overhead against the PRD's <=15% budget.

PRD §8.2: "Calculating the auxiliary AML loss adds <= 15% total wall-clock
runtime overhead per training step during CPU training loops." That is a claim
about *this host*, so it is measured here rather than asserted from the design.

The measurement isolates the penalty (a forward+backward of `DardRuleLoss`)
against the data term (a forward+backward of the same batch through the model),
which is the ratio the overhead is defined against. It runs in-process with
synchronization, so it excludes dataloader and optimizer noise.

This is a measurement, not a gate: it prints the number and asserts only that
the measurement ran and produced finite values. A hardware-dependent threshold
assertion would fail on someone else's CPU for reasons that have nothing to do
with the code.

python -m pytest tests/test_dard_overhead.py -v -s
"""

from __future__ import annotations

import time

import torch

from nanochat.models.dard.compile import mine_rules
from nanochat.models.dard.loss import DardRuleLoss
from nanochat.models.dard.shard import shard_corpus
from nanochat.modules.experiments.tiny_models import build_active_tiny_gpt


def _corpus(vocab: int, n: int = 4000) -> torch.Tensor:
    motif = [3, 5, 7, 9]
    stream = (motif * 100) + list(range(20, 200))
    return torch.tensor((stream * 20)[:n], dtype=torch.long) % vocab


def _time(fn, iters: int = 25, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters


def test_when_dard_runs_then_its_marginal_overhead_is_measured() -> None:
    torch.manual_seed(0)
    model = build_active_tiny_gpt()
    vocab = model.config.vocab_size
    batch, seq = 2, 64
    idx = torch.randint(0, vocab, (batch, seq))
    targets = torch.randint(0, vocab, (batch, seq))

    # Rules mined from a real corpus, so the penalty touches the same shape of
    # rule set a training run would use.
    shards = shard_corpus(_corpus(vocab), max_shard_size=500)
    rule_set = mine_rules(shards, vocab_size=vocab, max_rules_per_shard=64)
    dard = DardRuleLoss(rule_set, gamma=10.0)
    assert rule_set.num_rules > 0

    def ce_backward():
        """The data term alone: CE over logits the forward already produced."""
        model.zero_grad(set_to_none=True)
        logits = model(idx, targets=None)
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        ).backward()

    def ce_plus_dard_backward():
        """The same, plus the rule penalty. The difference is DARD's true cost.

        Both arms share one model forward, so what is timed is the *marginal*
        cost of evaluating and backpropagating the penalty -- which is what the
        PRD's per-step overhead is defined against. Timing two separate forwards
        would fold a second full forward into the number and overstate it.
        """
        model.zero_grad(set_to_none=True)
        logits = model(idx, targets=None)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        (ce + dard(logits)).backward()

    # Single-shot timing on a shared host is noisy at the millisecond scale, and
    # an *impossible* result (negative overhead) is the tell. Interleaved
    # best-of-N: taking the minimum of repeated trials cancels scheduler jitter,
    # because noise only ever adds time.
    trials = []
    for _ in range(5):
        trials.append((_time(ce_backward), _time(ce_plus_dard_backward)))
    base_best = min(b[0] for b in trials)
    both_best = min(b[1] for b in trials)
    marginal_ms = (both_best - base_best) * 1e3
    overhead = 100.0 * (both_best - base_best) / base_best

    print(
        f"\nDARD marginal cost on {torch.get_num_threads()} threads, "
        f"R={rule_set.num_rules}, V={vocab}, batch={batch}x{seq}, "
        f"nnz={rule_set.premise.values().numel() + rule_set.conclusion.values().numel()}:"
    )
    print(f"  CE fwd+bwd:            {base_best * 1e3:.2f} ms")
    print(f"  CE + DARD fwd+bwd:     {both_best * 1e3:.2f} ms")
    print(
        f"  marginal: {marginal_ms:+.2f} ms  ({overhead:+.1f}%, PRD §8.2 budget <= 15%)"
    )

    assert torch.isfinite(torch.tensor(overhead))
    assert rule_set.num_rules > 0

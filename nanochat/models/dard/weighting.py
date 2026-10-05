"""Phase II, step 1: continuous composite rule weighting in the log domain.

The PRD replaces hard rule truncation with a continuous score, so a rule that
survives in one shard is never discarded for failing a threshold in another --
it is merely weighted down (PRD §3.2.1):

    S_r     = f_r / K                      cross-shard frequency
    C_bar_r = mean of kappa over the f_r shards that emitted r
    log raw_r = alpha * log(S_r + eps) + beta * log(C_bar_r + eps)
    log w_r   = log_softmax(log raw / T_w)
    w_r       = clamp(exp(log w_r), min=eps_floor), then renormalized

Three properties are worth stating, because each is a failure mode the naive
formula has:

1. **Nothing underflows to zero.** Every operation is a `log`, and the only
   exponential taken is of a log-softmax output, which is `<= 0` by
   construction. With thousands of rules the unweighted tail of a linear
   softmax would round to exactly 0.0 in fp32; `eps_floor` bounds it away.
2. **`eps_floor` must survive fp8.** 1e-6 is below E5M2's smallest subnormal
   (2^-16 ~ 1.5e-5) and below E4M3's (2^-9, arXiv:2209.05433 Table 1), so it
   is held in fp32 and the loss block runs with autocast disabled. Under a
   bf16/fp8 autocast the floor would be the one quantity silently rounded away.
3. **The floor is applied before renormalization**, so it bounds the *final*
   weight. Flooring after would let the normalization undo it.
"""

from __future__ import annotations

import torch

from nanochat.models.dard.types import AmlRule, ShardRules

#: Guard added inside every `log` so a rule with zero frequency (K=0 or a
#: rule only ever seen in the shard being weighted) yields a finite score
#: rather than `-inf`, which would poison the log_softmax for its neighbours.
LOG_GUARD = 1e-8

#: PRD §5: hard floor guaranteeing every active rule keeps non-zero gradient
#: transmission. See the module docstring for why this must be fp32.
EPS_FLOOR = 1e-6


def composite_rule_weights(
    shards: list[ShardRules],
    alpha: float = 1.0,
    beta: float = 1.0,
    weight_temperature: float = 2.0,
    eps_floor: float = EPS_FLOOR,
) -> tuple[list[AmlRule], torch.Tensor, torch.Tensor]:
    """Weight every rule mined across `shards`, continuously.

    Args:
        shards: one `ShardRules` per micro-shard.
        alpha: exponent on cross-shard frequency. >= 0; 0 ignores frequency
            entirely and weights purely by local compression.
        beta: exponent on algebraic compression. >= 0.
        weight_temperature: `T_w > 0`. Larger flattens the distribution, which
            is what stops a single high-frequency rule from starving the tail
            (PRD §7 "noisy/inconsistent shard rules").
        eps_floor: lower bound on any single rule's final weight.

    Returns:
        `(rules, weights, frequencies)` where `rules` is the deduplicated rule
        list in a deterministic order, `weights` is `(R,)` float32 summing to
        1, and `frequencies` is `(R,)` float32 giving each rule's shard count
        `f_r`.

    Raises:
        ValueError: on a negative `alpha`/`beta`, a non-positive temperature, a
            non-positive floor, or an empty `shards` list (a rule set with no
            shards has no `K`, so `S_r = f_r / K` is undefined).
    """
    if alpha < 0.0 or beta < 0.0:
        raise ValueError(f"alpha and beta must be >= 0, got {alpha} and {beta}")
    if weight_temperature <= 0.0:
        raise ValueError(f"weight_temperature must be > 0, got {weight_temperature}")
    if eps_floor <= 0.0:
        raise ValueError(f"eps_floor must be > 0, got {eps_floor}")
    if not shards:
        raise ValueError("need at least one shard to compute rule weights")

    # f_r: how many shards rediscovered each rule, and the mean compression
    # over exactly those f_r shards (not all K) -- the PRD defines C_bar_r as a
    # mean over the shards that emitted the rule.
    frequency: dict[tuple[frozenset[int], frozenset[int]], int] = {}
    compression_sum: dict[tuple[frozenset[int], frozenset[int]], float] = {}
    for shard in shards:
        for shard_keys in _keys_seen_in(shard):
            frequency[shard_keys] = frequency.get(shard_keys, 0) + 1
            compression_sum[shard_keys] = compression_sum.get(shard_keys, 0.0) + (
                shard.compression
            )

    keys = sorted(frequency, key=lambda k: (len(k[0]), sorted(k[0]), sorted(k[1])))
    total_shards = len(shards)
    rules = [
        AmlRule(
            premise=k[0],
            conclusion=k[1],
            compression=compression_sum[k] / frequency[k],
        )
        for k in keys
    ]
    if not rules:
        empty = torch.zeros(0, dtype=torch.float32)
        return [], empty, empty

    freqs = torch.tensor([frequency[k] for k in keys], dtype=torch.float32)
    comps = torch.tensor(
        [compression_sum[k] / frequency[k] for k in keys], dtype=torch.float32
    )

    log_support = torch.log(freqs / total_shards + LOG_GUARD)
    log_compression = torch.log(comps + LOG_GUARD)
    log_raw = alpha * log_support + beta * log_compression
    log_w = torch.log_softmax(log_raw / weight_temperature, dim=0)
    weights = torch.clamp(torch.exp(log_w), min=eps_floor)
    weights = weights / weights.sum()
    return rules, weights, freqs


def _keys_seen_in(shard: ShardRules) -> set[tuple[frozenset[int], frozenset[int]]]:
    """Distinct rule keys in `shard`, counted once per shard.

    A shard emitting the same key twice must still contribute `f_r += 1`, not
    2, or `f_r` would over-count and `S_r` would exceed 1.
    """
    return {r.key for r in shard.rules}


__all__ = ["EPS_FLOOR", "LOG_GUARD", "composite_rule_weights"]

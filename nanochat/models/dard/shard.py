"""Phase I, step 1: split a tokenized corpus into non-overlapping micro-shards.

The PRD caps a shard at N < 1000 discrete sequences. The cap is not
arbitrary: AML's crossing step multiplies atoms (arXiv:1803.05252 §2.7), so
the atom count -- and with it the crossing cost -- grows superlinearly in the
number of constants contributed by a shard. Keeping each shard small bounds
both the atom count and the wall-clock cost of one `multiprocessing` worker.

Shards are **non-overlapping and deterministic**: given the same corpus and the
same `max_shard_size`, `shard_corpus` returns the same partition. That matters
because the composite weight's Global Invariance term `S_r = f_r / K` counts
how many shards rediscover a rule, and a partition that shifted between the
mining run and a re-mining run would make `f_r` incomparable across runs.
"""

from __future__ import annotations

import torch

#: PRD §3.1.1: "Each shard length |D_k| = N is capped at N < 1000 discrete sequences." Exceeding this is what the atom-growth bound in the module docstring is about, so it is a hard constant rather than a flag.
MAX_SHARD_SIZE = 999


def shard_corpus(
    tokens: torch.Tensor, max_shard_size: int = MAX_SHARD_SIZE
) -> list[torch.Tensor]:
    """Partition `tokens` into contiguous, non-overlapping micro-shards.

    Args:
        tokens: a 1-D long tensor of token ids, the flattened corpus.
        max_shard_size: maximum number of sequences per shard. Must be >= 1.

    Returns:
        A list of 1-D tensors whose concatenation is `tokens` exactly. Every
        shard but the last has `max_shard_size` sequences, so `K` is
        `ceil(N_total / max_shard_size)`.

    Raises:
        ValueError: if `max_shard_size < 1` or `tokens` is not 1-D.
    """
    if tokens.dim() != 1:
        raise ValueError(f"tokens must be 1-D, got shape {tuple(tokens.shape)}")
    if max_shard_size < 1:
        raise ValueError(f"max_shard_size must be >= 1, got {max_shard_size}")
    flat = tokens.reshape(-1)
    return [
        chunk
        for chunk in torch.split(flat, min(max_shard_size, flat.numel()))
        if chunk.numel() > 0
    ]

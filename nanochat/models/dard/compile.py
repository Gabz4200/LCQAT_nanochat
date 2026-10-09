"""Phase II, step 2/3: compile rules to sparse matrices, mine in parallel, save.

**The compiler** turns `R` rules over a vocabulary of size `V` into the two
binary CSR indicator matrices the differentiable loss reads (PRD §3.2.3):

    M_premise[r, v]    = 1 if token v is in rule r's premise
    M_conclusion[r, v] = 1 if token v is in rule r's conclusion

Row-wise CSR (not transposed) because the loss contracts `log P` -- shaped
`(..., V)` -- against `M^T`, which is a contiguous reduction down each rule's
token set. A `(R, V)` row-major CSR gives `M^T` the column-major layout that
reduction wants, so `sparse.mm` walks each row's `col_indices` without a gather.

The matrices stay **binary**. They are compiled once, offline; nothing in the
training loop writes them. That is the PRD's whole CPU-memory argument: the
rule set occupies `(nnz_premise + nnz_conclusion)` int32s, ~15-40 MB for
R=1e5 / V=32k, instead of an `R x V` float32 matrix's 12.8 GB.
"""

from __future__ import annotations

import os
import warnings

import torch

from nanochat.models.dard.atomize import atomize_shard
from nanochat.models.dard.types import AmlRule, DardRuleSet, ShardRules
from nanochat.models.dard.weighting import composite_rule_weights


def compile_rules(
    rules: list[AmlRule],
    weights: torch.Tensor,
    frequencies: torch.Tensor,
    vocab_size: int,
) -> DardRuleSet:
    """Compile `rules` into the static sparse form the loss reads.

    Raises:
        ValueError: if `vocab_size` is non-positive, a weight/frequency vector
            does not match `len(rules)`, or a rule names a token id outside
            `[0, vocab_size)`. The last check matters: an out-of-range id would
            otherwise raise deep inside a backward pass, or -- worse, if the id
            happened to be negative, silently wrap into the last vocabulary row
            and teach the model a rule about the wrong token.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if weights.shape != (len(rules),) or frequencies.shape != (len(rules),):
        raise ValueError(
            f"expected {len(rules)} weights and frequencies, got "
            f"{tuple(weights.shape)} and {tuple(frequencies.shape)}"
        )

    premise = _to_csr([r.premise for r in rules], vocab_size)
    conclusion = _to_csr([r.conclusion for r in rules], vocab_size)
    return DardRuleSet(
        premise=premise,
        conclusion=conclusion,
        weights=weights.to(torch.float32),
        frequency=frequencies.to(torch.float32),
        vocab_size=vocab_size,
    )


def _to_csr(rows: list[frozenset[int]], vocab_size: int) -> torch.Tensor:
    """Build a `(len(rows), vocab_size)` binary CSR from token-id sets.

    Columns are sorted within each row so the index array is deterministic and
    the reduction walks memory in increasing order.

    The sparse beta-state warning is silenced locally rather than globally: CSR
    is the intended representation, and `torch.sparse.check_sparse_tensor_invariants`
    is process-global state that other kernels in this repo may legitimately
    want enabled.
    """
    crow: list[int] = [0]
    col: list[int] = []
    for row in rows:
        for v in sorted(row):
            if not 0 <= v < vocab_size:
                raise ValueError(
                    f"token id {v} in a rule is outside the vocabulary [0, "
                    f"{vocab_size}); recompile with the tokenizer's vocabulary "
                    "size rather than one that does not contain it"
                )
            col.append(v)
        crow.append(len(col))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support")
        warnings.filterwarnings("ignore", message="Sparse invariant checks")
        return torch.sparse_csr_tensor(
            crow_indices=torch.tensor(crow, dtype=torch.int64),
            col_indices=torch.tensor(col, dtype=torch.int64),
            values=torch.ones(len(col), dtype=torch.float32),
            size=(len(rows), vocab_size),
        )


def mine_rules(
    shards: list[torch.Tensor],
    vocab_size: int,
    workers: int = 0,
    max_rules_per_shard: int = 4096,
    min_shard_frequency: int = 1,
    min_compression_threshold: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    weight_temperature: float = 2.0,
) -> DardRuleSet:
    """Run the whole offline pipeline: mine every shard, weight, compile.

    Workers run in separate processes (PRD §3.1.2). A worker count of 0 means
    "one process, in-thread": `shard_corpus` caps a shard at N < 1000
    sequences precisely so that serial mining is already cheap, and `spawn`ing
    processes for work that takes milliseconds costs more in import overhead
    than it saves.

    Args:
        shards: micro-shards from `shard_corpus`.
        vocab_size: the tokenizer's vocabulary size.
        workers: process count; 0 = serial.
        max_rules_per_shard: per-shard rule cap, bounding crossing cost.
        min_shard_frequency: PRD §3.2.2 optional pruning, `min_f`. Disabled by
            default (1 = keep everything).
        min_compression_threshold: PRD §3.2.2 optional pruning, `tau`.
            Disabled by default (0.0 = keep everything).
        alpha, beta, weight_temperature: passed to
            `composite_rule_weights`.

    Returns:
        The compiled rule set, ready for `DardRuleLoss`.

    Raises:
        ValueError: on a negative `min_shard_frequency`, a non-positive
            threshold, or `vocab_size <= 0`.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if min_shard_frequency < 1:
        raise ValueError(f"min_shard_frequency must be >= 1, got {min_shard_frequency}")
    if min_compression_threshold < 0.0:
        raise ValueError(
            f"min_compression_threshold must be >= 0, got {min_compression_threshold}"
        )
    mined = _mine_all(shards, workers, max_rules_per_shard)

    rules, weights, freqs = composite_rule_weights(
        mined,
        alpha=alpha,
        beta=beta,
        weight_temperature=weight_temperature,
    )
    # Pruning happens *after* weighting, so `w_r` is computed against the full rule population and the surviving weights renormalize among themselves -- pruning a threshold, not zeroing a row of an already-normalized vector.
    keep = [
        i
        for i, rule in enumerate(rules)
        if int(freqs[i]) >= min_shard_frequency
        and rule.compression >= min_compression_threshold
    ]
    if not keep:
        return compile_rules([], torch.zeros(0), torch.zeros(0), vocab_size)
    kept_weights = weights[keep]
    kept_weights = kept_weights / kept_weights.sum()
    return compile_rules(
        [rules[i] for i in keep],
        kept_weights,
        freqs[keep],
        vocab_size,
    )


def _mine_shard(payload: tuple[torch.Tensor, int]) -> ShardRules:
    """Worker entry point. Module-level because `spawn` cannot pickle a closure."""
    shard, max_rules = payload
    return atomize_shard(shard, max_rules=max_rules)


def _mine_all(
    shards: list[torch.Tensor], workers: int, max_rules_per_shard: int
) -> list[ShardRules]:
    """Atomize every shard, in a process pool when `workers > 0`."""
    if not shards:
        raise ValueError("no shards to mine")

    if workers <= 1:
        return [atomize_shard(s, max_rules=max_rules_per_shard) for s in shards]

    import multiprocessing as mp

    payloads = [(s, max_rules_per_shard) for s in shards]
    with mp.get_context("spawn").Pool(processes=min(workers, len(shards))) as pool:
        return list(pool.map(_mine_shard, payloads))


#: Stamped into every artifact. Bumping it must be a loud load-time failure, exactly as `NOISE_MAP_VERSION` is for the DiffusionBlocks block/noise map: a rule set compiled under a different weighting or pruning convention would still *load* (same shapes, same dtypes) and silently enforce different rules.
RULE_SET_VERSION = 1


def save_rule_set(
    path: str | os.PathLike[str], rule_set: DardRuleSet, meta: dict
) -> None:
    """Write a compiled rule set to `path` as a torch archive."""
    torch.save(
        {
            "version": RULE_SET_VERSION,
            "premise": rule_set.premise,
            "conclusion": rule_set.conclusion,
            "weights": rule_set.weights,
            "frequency": rule_set.frequency,
            "vocab_size": rule_set.vocab_size,
            "meta": meta,
        },
        path,
    )


def load_rule_set(path: str | os.PathLike[str]) -> tuple[DardRuleSet, dict]:
    """Load a rule set written by `save_rule_set`.

    Raises:
        ValueError: if the artifact carries a different `RULE_SET_VERSION`, or
            is missing a field. A version mismatch is rejected rather than
            loaded, for the reason `NOISE_MAP_VERSION` exists.
    """
    blob = torch.load(path, weights_only=False, map_location="cpu")
    missing = {"version", "premise", "conclusion", "weights", "frequency"} - blob.keys()
    if missing:
        raise ValueError(f"rule set at {path} is missing fields: {sorted(missing)}")
    version = blob["version"]
    if version != RULE_SET_VERSION:
        raise ValueError(
            f"rule set at {path} has version {version}, this build expects "
            f"{RULE_SET_VERSION}; the weighting or pruning convention changed, so "
            "the compiled rules mean something different -- re-mine them"
        )
    rule_set = DardRuleSet(
        premise=blob["premise"],
        conclusion=blob["conclusion"],
        weights=blob["weights"].to(torch.float32),
        frequency=blob["frequency"].to(torch.float32),
        vocab_size=int(blob["vocab_size"]),
    )
    return rule_set, dict(blob.get("meta") or {})


def rule_set_to_meta(rule_set: DardRuleSet, extra: dict | None = None) -> dict:
    """The `meta["dard"]` payload recorded in a training checkpoint.

    Descriptive only -- the tensors themselves are already in the loss module's
    state. It exists so a resumed run reports the rule set it is actually
    enforcing rather than the one its flags would imply.
    """
    payload = {
        "version": RULE_SET_VERSION,
        "num_rules": rule_set.num_rules,
        "vocab_size": rule_set.vocab_size,
        "nnz_premise": int(rule_set.premise.values().numel()),
        "nnz_conclusion": int(rule_set.conclusion.values().numel()),
    }
    if extra:
        payload.update(extra)
    return payload


__all__ = [
    "RULE_SET_VERSION",
    "compile_rules",
    "load_rule_set",
    "mine_rules",
    "rule_set_to_meta",
    "save_rule_set",
]

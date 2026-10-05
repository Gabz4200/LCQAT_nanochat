"""Typed contracts for DARD's offline mining and compiled rule set.

Everything crossing a DARD phase boundary is a frozen dataclass, so the
invariant that a rule's premise and conclusion index into the *same*
vocabulary is stated once in a type instead of being re-derived by every
unpack site.

The discrete/continuous split is the whole point of this package and is
enforced here:

* Phase I (mining) produces `AmlRule`, whose constants are **vocabulary
  indices**. It is pure integer set algebra -- no float ever touches it.
* Phase II (compilation) turns a list of `AmlRule` into a `DardRuleSet`,
  whose contents are **sparse indicator matrices**. This is the only
  representation that reaches the LLM, and it is static: nothing in Phase III
  re-runs a symbolic step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AmlRule:
    """One semilattice order relation mined from a shard: `premise <= conclusion`.

    Both sides are **sets of token ids** (the constants `c_{pos,token}` of the
    AML paper), which is what makes the meet `bigodot_{i in I} c_i` a plain set
    union. Sets rather than sequences because AML's atomized semilattice treats
    merge as set union (arXiv:1803.05252 §2.5, Eq. 7), so `c_i odot c_i = c_i`
    means duplicates are meaningless -- not merely discouraged.

    Attributes:
        premise: token ids whose conjunction implies `conclusion`.
        conclusion: token ids implied by `premise`. In AML's own orientation
            this is the *more specific* side, since the axiom reads
            `T_R odot T_L = T_R` (arXiv:2502.19944 Suppl. Def. 3); DARD reads
            the same pair as "premise licenses conclusion".
        compression: that shard's algebraic compression ratio `kappa = |R| / Z`
            (arXiv:1803.05252 §3.5, Eq. 36), i.e. relations per surviving atom.
            Shard-local by construction: it describes the model the rule came
            out of, not the rule itself.
    """

    premise: frozenset[int]
    conclusion: frozenset[int]
    compression: float

    def __post_init__(self) -> None:
        if not self.premise:
            raise ValueError("a rule needs a non-empty premise")
        if not self.conclusion:
            raise ValueError("a rule needs a non-empty conclusion")

    @property
    def key(self) -> tuple[frozenset[int], frozenset[int]]:
        """Identity for cross-shard frequency counting.

        Deliberately excludes `compression`: the same order relation mined on
        two shards is the *same* rule with two compression readings, and
        counting it twice would double-weight every invariant that happens to be
        rediscovered -- which is exactly the redundancy `S_r = f_r / K` is
        supposed to reward only once.
        """
        return (self.premise, self.conclusion)


@dataclass(frozen=True)
class ShardRules:
    """The order relations mined from one micro-shard.

    `num_relations` and `num_atoms` are kept rather than folded into the rules,
    because the AML compression ratio is a property of the *model*
    (`kappa = |R| / Z`) while a rule only carries a copy of the value that was
    current when it was emitted. Keeping the model-level counts lets the
    weighting stage recompute `kappa` from the model rather than trusting a
    per-rule copy, and lets a shard that mined nothing still report `Z > 0`.
    """

    rules: tuple[AmlRule, ...]
    num_relations: int
    num_atoms: int

    def __post_init__(self) -> None:
        if self.num_atoms <= 0:
            raise ValueError(f"num_atoms must be positive, got {self.num_atoms}")
        if self.num_relations < 0:
            raise ValueError(
                f"num_relations must be non-negative, got {self.num_relations}"
            )

    @property
    def compression(self) -> float:
        """The shard's algebraic compression ratio `kappa = |R| / Z`."""
        return self.num_relations / self.num_atoms


@dataclass(frozen=True)
class DardRuleSet:
    """Rules compiled into the static sparse form the differentiable loss reads.

    This is the single artifact that crosses from Phase II into Phase III. It is
    immutable and shape-fixed: `vocab_size` is pinned here so a mismatch
    between the compiled rules and the model's output vocabulary is a load-time
    error rather than a silent index-out-of-range deep inside a backward pass.

    Attributes:
        premise: `(R, V)` binary CSR. Row `r` is 1 at the tokens in rule `r`'s
            premise.
        conclusion: `(R, V)` binary CSR, same convention.
        weights: `(R,)` float32 rule weights. **Float32 is load-bearing, not
            incidental**: FP8 E4M3's smallest normal is 2^-6 and E5M2's is
            2^-14 ~ 6.1e-5, with smallest *subnormal* 2^-16 ~ 1.5e-5
            (arXiv:2209.05433 Table 1). A weight floored at `1e-6` therefore
            flushes to zero in either format, which would silently zero exactly
            the tail rules the floor exists to keep alive. Holding the vector in
            fp32 is the same reason the loss block is wrapped in
            `autocast(enabled=False)`.
        frequency: `(R,)` number of shards each rule appeared in. Kept because
            `S_r = f_r / K` is the PRD's Global Invariance term and a resumed
            run should be able to re-derive the weights without re-mining.
        vocab_size: the vocabulary these rows index into.
    """

    premise: torch.Tensor
    conclusion: torch.Tensor
    weights: torch.Tensor
    frequency: torch.Tensor
    vocab_size: int

    def __post_init__(self) -> None:
        v = self.vocab_size
        for name, mat in (("premise", self.premise), ("conclusion", self.conclusion)):
            if mat.layout != torch.sparse_csr:
                raise ValueError(
                    f"{name} must be a sparse CSR tensor, got {mat.layout}"
                )
            if mat.shape[1] != v:
                raise ValueError(
                    f"{name} has vocabulary width {mat.shape[1]}, expected {v}"
                )
        if self.premise.shape != self.conclusion.shape:
            raise ValueError(
                f"premise {tuple(self.premise.shape)} and conclusion "
                f"{tuple(self.conclusion.shape)} must have the same row count"
            )
        rows = self.premise.shape[0]
        if self.weights.shape != (rows,) or self.frequency.shape != (rows,):
            raise ValueError(
                f"weights {tuple(self.weights.shape)} and frequency "
                f"{tuple(self.frequency.shape)} must both be ({rows},)"
            )
        if self.weights.dtype != torch.float32 or self.frequency.dtype != torch.float32:
            raise ValueError(
                f"weights and frequency must be float32 (see DardRuleSet), got "
                f"{self.weights.dtype} and {self.frequency.dtype}"
            )

    @property
    def num_rules(self) -> int:
        return self.premise.shape[0]

    def to(self, device: torch.device | str) -> "DardRuleSet":
        """A copy with every tensor moved to `device`.

        Sparse CSR `.to(device)` keeps the layout, so the loss's
        `sparse.mm` does not silently degrade into a dense matmul that would
        allocate `R x V` floats -- 12.8 GB at the PRD's R=1e5, V=32k example.
        """
        return DardRuleSet(
            premise=self.premise.to(device),
            conclusion=self.conclusion.to(device),
            weights=self.weights.to(device),
            frequency=self.frequency.to(device),
            vocab_size=self.vocab_size,
        )

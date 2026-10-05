"""Phase I, step 2: the discrete AML atomizer (Sparse Crossing).

This module is **entirely integer set algebra**. No float appears in it, and it
is the reason DARD is not "symbolic compute in the training loop": the symbolic
work happens once, here, offline. What comes out is a handful of token-id sets
that `compile.py` freezes into sparse matrices. Nothing downstream re-derives
anything algebraically.

**The algorithm, in the papers' own terms.** AML represents a dataset of
discrete examples as an atomized semilattice over constants (arXiv:1803.05252
§2.1-2.2): the merge `odot` is commutative, associative and idempotent, and in
an atomized model merge is *set union over atoms* (§2.5, Eq. 7). The order is
induced, `a < b` iff `a odot b = b` (§2.2, Eq. 4). Enforcing an axiom
`T_L <= T_R` that the current model violates is **crossing** it: every
*discriminant* atom (one below `T_L` but not below `T_R`) is replaced by its
unions with the atoms below `T_R` (arXiv:2502.19944 Suppl. Def. 15, Full
Crossing). Sparse Crossing keeps only the subset of those unions that
preserves the atom traces (Suppl. Thms. 26-27).

**What "atom" means here, concretely.** An atom is a *minimal* group of token
ids whose co-occurrence in a shard is *not* explained by the group's parts --
i.e. it is a genuine correlation rather than an artifact of two independent
tokens happening to co-occur. Storing each atom as `frozenset` of token ids
makes idempotence (`c odot c = c`) free, and makes `is_proper_subset` the
minimality test.

**The trace step is the part that cannot be skipped.** AML's sparse variant is
trace-invariant: an atom union is only kept if the resulting atom set still
separates the same (positive, negative) duple pairs (Suppl. Thm. 27). A union
`A | B` is kept iff some (premise, conclusion) pair has all of `A | B` in its
conclusion while `A` alone would have put it in the premise, or vice versa --
checked in `atomize_shard` below. Without that check the atom set grows
without bound, which is the combinatorial blow-up the shard cap exists to bound.
"""

from __future__ import annotations

import torch

from nanochat.models.dard.types import AmlRule, ShardRules


def _find_atoms(sequences: list[frozenset[int]]) -> list[frozenset[int]]:
    """Minimal, non-redundant token groups that co-occur across `sequences`.

    A candidate group is an atom iff no *proper* subset of it is also a
    candidate: that is exactly irreducibility, and it is what stops the atom list
    from being every pair and triple that co-occurred once.

    The filter is a **size-ordered single pass** rather than the obvious
    "compare against every candidate". The quadratic version cost ~1 s per
    999-token shard, which is ~1 s of startup per shard and minutes over a real
    corpus -- the O(C^2) subset test, where C is the candidate count. Bucketing
    by size makes it O(C * MAX_ATOM_TOKENS) instead: a group of size k can only
    be explained by one of size < k, so all the smaller buckets are already
    known by the time k is reached. Same answer, orders of magnitude cheaper.
    """
    candidates: set[frozenset[int]] = set()
    for seq in sequences:
        items = sorted(seq)
        # A co-occurrence witness is at most `MAX_ATOM_TOKENS` wide. Without a
        # cap the candidate set is every subset of every sequence, which for
        # real text is the whole corpus.
        for start in range(len(items)):
            for width in range(1, MAX_ATOM_TOKENS + 1):
                end = start + width
                if end > len(items):
                    break
                candidates.add(frozenset(items[start:end]))

    by_size: list[set[frozenset[int]]] = [set() for _ in range(MAX_ATOM_TOKENS + 1)]
    for c in candidates:
        by_size[len(c)].add(c)

    atoms: list[frozenset[int]] = []
    smaller: list[frozenset[int]] = []
    for size in range(1, MAX_ATOM_TOKENS + 1):
        for c in by_size[size]:
            if not any(o <= c for o in smaller):
                atoms.append(c)
        smaller.extend(by_size[size])
    # Deterministic order: sets of ints iterate in a hash order that depends on
    # insertion history, and the atom order decides which unions are tried
    # first, so a run must not depend on it.
    return sorted(atoms, key=lambda a: (len(a), sorted(a)))


#: Cap on how many tokens one co-occurrence atom may span. AML's own examples
#: use constants per feature position; for a token stream the analogous unit is
#: a short n-gram. Four keeps atoms discriminative (the paper's 4x4 Sudoku
#: images use 4-token atoms) while bounding the candidate set.
MAX_ATOM_TOKENS = 4


def atomize_shard(shard: torch.Tensor, max_rules: int = 4096) -> ShardRules:
    """Mine the order relations of one micro-shard.

    The axioms come from the shard's own next-token structure: for each
    sequence, a *window* of leading tokens is the premise and the token
    following the window is the conclusion. That is the discrete content the
    LLM's own objective is trying to learn -- "these tokens license that next
    token" -- expressed as AML duples `T_L <= T_R`, so the mined rules are about
    the same dependency the cross-entropy is scoring, not an unrelated dataset.

    Sparse Crossing is then run per (premise, conclusion) pair: each violating
    pair contributes the unions that make it hold, and each surviving union is a
    new candidate atom. A union is kept only if it *separates* a pair, i.e. it
    discriminates a premise from a conclusion that the parts did not already
    separate -- the trace check of Suppl. Thm. 27.

    Args:
        shard: a 1-D long tensor of token ids.
        max_rules: cap on emitted rules. Crossings are taken in a deterministic
            order and the cap truncates, so the cap changes which rules survive
            but not whether mining terminates.

    Returns:
        The shard's rules plus the model-level counts the AML compression ratio
        `kappa = |R| / Z` is computed from.

    Raises:
        ValueError: if `shard` is not 1-D or `max_rules < 1`.
    """
    if shard.dim() != 1:
        raise ValueError(f"shard must be 1-D, got shape {tuple(shard.shape)}")
    if max_rules < 1:
        raise ValueError(f"max_rules must be >= 1, got {max_rules}")

    tokens = [int(t) for t in shard.tolist()]
    # Premises are built at *every* width from 1 to PREMISE_WIDTH, not just the
    # widest. A rule is only informative when both sides carry comparable
    # arity (see MIN_BALANCED_ARITY), and on real text a 4-token window rarely
    # recurs often enough to accumulate a multi-token conclusion -- so mining
    # only the widest window yields either unsatisfiable rules (narrow
    # conclusion) or none at all (no recurrence). Starting narrow and widening
    # lets the shard's own statistics choose the arity it can support.
    occurrences: dict[frozenset[int], list[int]] = {}
    for width in range(1, PREMISE_WIDTH + 1):
        for start in range(len(tokens) - width):
            window = frozenset(tokens[start : start + width])
            following = tokens[start + width]
            if following not in window:
                occurrences.setdefault(window, []).append(following)
    premises = set(occurrences)
    sequences = [frozenset(tokens[i : i + PREMISE_WIDTH]) for i in range(len(tokens))]
    atoms = _find_atoms(sequences)

    rules: list[AmlRule] = []
    seen: set[tuple[frozenset[int], frozenset[int]]] = set()
    for premise in sorted(premises, key=lambda p: (len(p), sorted(p))):
        disjoint = set(occurrences[premise]) - premise
        if len(disjoint) < MIN_BALANCED_ARITY:
            continue
        conclusion = frozenset(disjoint)
        if not _cross(premise, conclusion, atoms):
            continue
        key = (premise, conclusion)
        if key in seen:
            continue
        seen.add(key)
        rules.append(
            AmlRule(
                premise=premise,
                conclusion=conclusion,
                compression=1.0,  # replaced below, once Z is known
            )
        )
        if len(rules) >= max_rules:
            break

    num_atoms = max(len(atoms), 1)
    relations = len(premises) + len(rules)
    compression = relations / num_atoms
    return ShardRules(
        rules=tuple(AmlRule(r.premise, r.conclusion, compression) for r in rules),
        num_relations=relations,
        num_atoms=num_atoms,
    )


#: How many leading tokens form a premise. Four positions is wide enough that
#: the mined relation is a multi-token pattern (which is the case the log-domain
#: product t-norm exists for) and narrow enough that premises stay frequent
#: enough to survive the cross-shard frequency threshold.
PREMISE_WIDTH = 4


def _premise_successors(tokens: list[int]) -> dict[frozenset[int], set[frozenset[int]]]:
    """Map each premise window to the token sets that directly follow it.

    The conclusion is the token immediately after the window, **minus any token
    already in the premise**. Disjointness is not cosmetic: the loss evaluates
    each side as a product t-norm over its own token set, so a conclusion that
    contained the premise would make `v(conclusion) >= v(premise)` by
    construction and the rule's penalty would sit at its floor forever. That is
    exactly the vacuous-rule failure mode -- every "rule" trivially satisfied,
    contributing a constant and teaching nothing.

    Built once per shard rather than per premise. Scanning the shard for each
    premise separately is O(P*N) with P ~ N, so a 999-token shard did ~1e6
    window comparisons; this does N.
    """
    width = PREMISE_WIDTH
    out: dict[frozenset[int], set[frozenset[int]]] = {}
    for start in range(len(tokens) - width):
        premise = frozenset(tokens[start : start + width])
        extra = frozenset({tokens[start + width]} - premise)
        if extra:
            out.setdefault(premise, set()).add(extra)
    return out


#: A rule is only informative if its two sides carry *comparable* numbers of
#: tokens, because the penalty compares two product t-norms and a longer
#: product is exponentially smaller.
#:
#: With a 4-token premise against a 1-token conclusion, `v(premise)` is
#: O(p^4) and `v(conclusion)` is O(p): the premise is smaller by construction
#: for any p < 1, the order can never be violated, and the penalty sits at the
#: hinge floor forever. Measured on a real model: 0.0000 of rules violated,
#: penalty 0.069299 against a floor of 0.069315.
#:
#: So the conclusion is widened to the same arity as the premise, by taking the
#: tokens that follow the window across *all* its occurrences in the shard. A
#: premise that recurs is thereby paired with the distribution of what actually
#: follows it, which is both the right logical statement ("this context licenses
#: these continuations") and arity-balanced so the penalty can activate.
MIN_BALANCED_ARITY = 2


def _balanced_conclusion(
    premise: frozenset[int], tokens: list[int]
) -> set[frozenset[int]]:
    """Conclusion sets of the same arity as `premise`, disjoint from it.

    Takes the tokens following every occurrence of the premise window, then
    keeps the ones that do not overlap the premise. Returns the empty set when
    no occurrence yields a disjoint continuation, which `_cross` then rejects.
    """
    width = len(premise)
    following: set[int] = set()
    for start in range(len(tokens) - width):
        if frozenset(tokens[start : start + width]) != premise:
            continue
        following.add(tokens[start + width])
    disjoint = following - premise
    if len(disjoint) < MIN_BALANCED_ARITY:
        # Not enough distinct continuations to balance the premise. Keeping a
        # 1-token conclusion here would recreate the unsatisfiable asymmetry,
        # so the pair is dropped instead -- see MIN_BALANCED_ARITY.
        return set()
    return {frozenset(disjoint)}


def _cross(
    premise: frozenset[int], conclusion: frozenset[int], atoms: list[frozenset[int]]
) -> bool:
    """Whether crossing can make `premise <= conclusion` hold informatively.

    Crossing (arXiv:2502.19944 Suppl. Def. 15) replaces each discriminant atom
    of the premise by its unions with the atoms of the conclusion. The Sparse
    Crossing refinement keeps only unions that preserve the atom *traces*
    (Suppl. Thms. 26-27): a union is worth keeping only when it separates a
    pair that its parts did not already separate.

    Here that test reduces to a single question, because the sides are disjoint
    token sets: does the premise contain an atom that the conclusion's tokens
    do not already account for? If some premise atom is also an atom of the
    conclusion, the two sides share irreducible structure and the crossing is a
    no-op -- the rule would encode a coincidence of the window rather than an
    implication, so it is rejected. Only a premise atom *absent* from the
    conclusion's atoms can be crossed to carry the relation.

    Returns:
        True when the pair survives as a rule.
    """
    if not conclusion:
        return False
    if premise & conclusion:
        # Sides share a token: the conclusion's product already contains a
        # factor of the premise's, so v(premise) <= v(conclusion) holds by
        # construction and the rule is vacuous.
        return False
    if len(premise) - len(conclusion) >= MAX_ARITY_GAP:
        # The premise's product t-norm is exponentially smaller than the
        # conclusion's, so the order can never be violated and the penalty is
        # pinned at its floor. See MIN_BALANCED_ARITY.
        return False
    conclusion_atoms = [a for a in atoms if a <= conclusion]
    for atom in atoms:
        if atom <= premise and not any(atom <= c for c in conclusion_atoms):
            return True
    return False


#: Largest arity difference a rule may have and still be violable. One token is
#: the safe bound: with |premise| = |conclusion| + 2 the premise product is
#: O(p^2) smaller for small p, which in practice means never violated.
MAX_ARITY_GAP = 1


__all__ = ["MAX_ATOM_TOKENS", "PREMISE_WIDTH", "atomize_shard"]

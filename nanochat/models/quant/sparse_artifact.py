"""
Sparse exported artifacts: keep only the surviving weights, and shrink the
codebook to the index alphabet actually in use.

This is the concrete payoff of unifying sparsity and quantization. A pruned
SparseProp weight holds an exact 0.0 in the shadow matrix, and LC-QAT's zero
anchor puts 0.0 at index `m_neg`, so a pruned position *is* a codebook index
rather than a separate masking concept. Two consequences the dense artifact
cannot exploit:

1. **Storage.** `keep_indices` (int32 `[nnz]`) plus a CSR `row_ptr` describe
   the layer in `nnz` values instead of `m*n`. Packing those values with the
   K-selected format gives a physically smaller buffer, not just a different
   one.

2. **Alphabet compaction.** Magnitude pruning keeps the *large* weights, so a
   heavily pruned layer typically uses a strict subset of its codebook. Under
   `K = 15` a layer that only ever lands on 9 distinct indices can be exported
   with a 9-entry codebook, which changes the storage format (nibbles -> trits
   is not reachable, but 15 -> 9 still fits a nibble, and 9 -> 5 might not be
   reachable either -- what compaction buys is real whenever it crosses a
   format boundary or a bit-width).

Compaction is lossy in one direction only and that direction is safe: dropping
unused codebook levels cannot change any value, because no index refers to
them. What it does change is `K`, so the exported `weight_index_format` and the
packed buffer must be regenerated from the *compacted* alphabet, and any
consumer that hard-codes K must read it from the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nanochat.models.quant.packing import (
    FORMAT_NIBBLES,
    FORMAT_TRITS,
    TRITS_PER_BYTE,
    index_format_for_k,
    pack_weight_indices,
    unpack_weight_indices,
)

#: Smallest compacted alphabet an exported artifact may declare. `packing`
#: picks a storage format from K and the densest one (trits) starts at 3, so a
#: K=1 or K=2 artifact has no valid format and `pack_weight_indices` raises.
#: Compaction is free to *under*-report what is stored -- the padded levels are
#: never referenced -- but it is not free to report an unrepresentable K.
MIN_EXPORT_ALPHABET = 3


@dataclass
class SparseExportPlan:
    """How one layer's weight indices will be stored and read back.

    Attributes:
        keep_indices: int32 `[nnz]` index values, in CSR order (row-major).
        row_ptr: int32 `[m+1]` CSR row offsets into `keep_indices`.
        col_indices: int32 `[nnz]` input-feature indices, ascending per row.
        used: int64 `[K_used]` the original indices that survive, ascending.
        remap: int64 `[K]` original index -> compacted index, -1 when dropped.
        k_used: size of the compacted alphabet.
        compaction: whether any level was dropped (False when K was already
            tight, e.g. every level is used by some surviving weight).
    """

    keep_indices: torch.Tensor
    row_ptr: torch.Tensor
    col_indices: torch.Tensor
    used: torch.Tensor
    remap: torch.Tensor
    k_used: int
    compaction: bool

    @property
    def nnz(self) -> int:
        return int(self.keep_indices.numel())


def plan_sparse_export(
    indices: torch.Tensor, mask: torch.Tensor | None = None, k: int | None = None
) -> SparseExportPlan:
    """Build the CSR + compaction plan for one layer's quantized weights.

    Args:
        indices: `[m, n]` weight codebook indices (any integer dtype).
        mask: optional `[m, n]` boolean keep-mask. `None` means every position
            is kept, which reduces to the dense case with `nnz == m*n`.
        k: the layer's full codebook size. `None` infers it from `indices.max()`,
            which is only correct when every level is used somewhere -- a pruned
            layer can leave the top level unused. Pass it explicitly when known.

    The zero anchor is honoured implicitly rather than checked: a pruned
    position carries index `m_neg`, and if the caller pruned by magnitude it
    will appear in the dropped set. Positions whose index is unused *anywhere in
    the kept set* are what compaction removes, so a genuinely-unused codebook
    level is removed whether or not the layer is sparse.
    """
    if indices.ndim != 2:
        raise ValueError(
            f"weight indices must be [m, n], got shape {tuple(indices.shape)}"
        )
    if mask is not None and mask.shape != indices.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match indices "
            f"{tuple(indices.shape)}"
        )
    m, n = indices.shape
    keep_region = indices >= 0 if mask is None else mask.to(torch.bool)
    rows, cols = torch.nonzero(keep_region, as_tuple=True)
    original = indices[rows, cols].to(torch.int64)

    counts = torch.bincount(rows, minlength=m)
    row_ptr = torch.zeros(m + 1, dtype=torch.int64, device=indices.device)
    torch.cumsum(counts, dim=0, out=row_ptr[1:])

    # Alphabet compaction: only the levels the surviving weights actually
    # reference are kept. `used` is ascending, so `remap` is monotone and the
    # compacted codebook preserves the original level order -- no level can be
    # re-sorted past another by the remap.
    if k is None:
        k = max(3, int(indices.max()) + 1) if indices.numel() else 3
    used = torch.unique(original)
    remap = torch.full((k,), -1, dtype=torch.int64, device=indices.device)
    if used.numel() == 0:
        # An all-pruned layer keeps no alphabet at all. Emit the minimum legal
        # codebook rather than K=1: `index_format_for_k` rejects K < 3 (the
        # trit format starts at 3), so a K=1 artifact would fail to pack.
        used = torch.zeros(
            min(MIN_EXPORT_ALPHABET, k), dtype=torch.int64, device=indices.device
        )
        remap = torch.full((k,), -1, dtype=torch.int64, device=indices.device)
        remap[: used.numel()] = torch.arange(used.numel(), device=indices.device)
        return SparseExportPlan(
            keep_indices=torch.zeros(0, dtype=torch.int32, device=indices.device),
            row_ptr=torch.zeros(m + 1, dtype=torch.int32, device=indices.device),
            col_indices=torch.zeros(0, dtype=torch.int32, device=indices.device),
            used=used,
            remap=remap,
            k_used=int(used.numel()),
            compaction=int(used.numel()) < k,
        )
    remap[used] = torch.arange(used.numel(), device=indices.device)
    compacted = remap[original]
    return SparseExportPlan(
        keep_indices=compacted.to(torch.int32).contiguous(),
        row_ptr=row_ptr.to(torch.int32).contiguous(),
        col_indices=cols.to(torch.int32).contiguous(),
        used=used.contiguous(),
        remap=remap,
        k_used=max(int(used.numel()), MIN_EXPORT_ALPHABET),
        compaction=used.numel() < k,
    )


def pack_sparse_plan(plan: SparseExportPlan) -> tuple[torch.Tensor, int]:
    """Pack a plan's compacted index values into the K_used-selected format."""
    return pack_weight_indices(plan.keep_indices, plan.k_used)


def unpack_sparse_plan(
    packed: torch.Tensor, plan: SparseExportPlan, n: int | None = None
) -> torch.Tensor:
    """Inverse of `pack_sparse_plan`, returning the compacted `[nnz]` values.

    Always 1-D, and always exactly `plan.nnz` values long. The width cannot be
        inferred from the packed buffer -- a nibble-packed `[17]` buffer holds up to
        34 values but stores 17 bytes, so "how many values did I ask for" is not
        recoverable from the buffer alone. `plan.nnz` is the authoritative count.

        `n` is accepted for signature symmetry and ignored.
    """
    del n  # the count lives on the plan; a buffer cannot tell it from its size
    unpacked = unpack_weight_indices(packed, plan.nnz, plan.k_used)
    return unpacked.reshape(-1)


def packed_value_bytes(indices: torch.Tensor, k: int) -> int:
    """Bytes the *packed index values* of `indices` occupy at alphabet size `k`.

    Storage only, no structure: this is the quantity that sparsity and alphabet
    compaction shrink, so it is the one to compare across layouts.
    """
    fmt = index_format_for_k(k)
    m, n = indices.shape
    if fmt == FORMAT_TRITS:  # 5 values/byte
        return m * ((n + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE)
    if fmt == FORMAT_NIBBLES:  # 2 values/byte
        return m * ((n + 1) // 2)
    return m * n  # one value per byte for uint8


def csr_structure_bytes(plan: SparseExportPlan, index_bytes: int = 4) -> int:
    """Bytes the CSR structure costs: `row_ptr` + `col_indices`.

    This is the price of describing *which* positions survived, and it is real
    storage, not free. At 4 bytes per index it is 8 bytes per surviving weight
    on top of its packed value, which is why sparsity has to clear roughly 80%
    before the sparse layout beats the dense one (the same crossover
    SparseProp Sec. 4.1 reports for kernel selection).
    """
    return (plan.row_ptr.numel() + plan.col_indices.numel()) * index_bytes


def dense_bytes_for(indices: torch.Tensor, k: int) -> int:
    """Bytes the dense packed buffer for `indices` at codebook size `k` occupies.

    Kept as a name in its own right: it is the counterfactual every sparse
    footprint is quoted against, and `packed_value_bytes` is the primitive.
    """
    return packed_value_bytes(indices, k)


def sparse_bytes_for(plan: SparseExportPlan, index_bytes: int = 4) -> int:
    """Total sparse footprint: packed values + CSR structure.

    `keep_indices` are packed with the *compacted* alphabet, so compaction is
    included here rather than being a separate claim.
    """
    packed, _ = pack_sparse_plan(plan)
    return packed.numel() * packed.element_size() + csr_structure_bytes(
        plan, index_bytes
    )


def sparsity_break_even(indices: torch.Tensor, k: int, index_bytes: int = 4) -> float:
    """Sparsity above which the sparse layout beats the dense one, in [0, 1].

    Solves `nnz*value + nnz*index_bytes + (m+1)*index_bytes < dense` for nnz,
    where `value` is the packed bytes per surviving index. Returns the sparsity
    that makes the two equal.

    Two things follow, and both are worth stating rather than glossing:

    * at 4-byte CSR indices the crossover sits near 0.89 sparsity for a 4-bit
      alphabet, *above* SparseProp's 0.8 kernel-selection threshold (Sec. 4.1).
      The two thresholds answer different questions -- one is "is the sparse
      kernel faster", the other "is the sparse artifact smaller" -- and they do
      not coincide;
    * `index_bytes=0` (an implicit-dictionary encoding that derives columns from
      the pattern itself) breaks even at ~0.5, which is the honest reason that
      encoding exists.
    """
    m, n = indices.shape
    total = m * n
    if total == 0:
        return 1.0
    dense = packed_value_bytes(indices, k)
    value_bytes = dense / total
    per_survivor = value_bytes + index_bytes
    if per_survivor <= 0:
        return 0.0
    overhead = (m + 1) * index_bytes
    max_nnz = (dense - overhead) / per_survivor
    if max_nnz <= 0:
        return 1.0
    return float(min(1.0, max(0.0, 1.0 - max_nnz / total)))


__all__ = [
    "MIN_EXPORT_ALPHABET",
    "SparseExportPlan",
    "csr_structure_bytes",
    "dense_bytes_for",
    "pack_sparse_plan",
    "packed_value_bytes",
    "plan_sparse_export",
    "sparse_bytes_for",
    "sparsity_break_even",
    "unpack_sparse_plan",
]

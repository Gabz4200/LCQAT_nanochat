"""
Fused 2D weight-product LUT for the quantized index-linear runtime.

The inner loop of every index-weight linear is

    y[t, m] = sum_j act_lut[a[t, j]] * weight_lut[w[m, j]]

Two FP32 table fetches and one multiply per term, `m * n` of them. When the
weight alphabet is small -- which LC-QAT's K<=16 guarantees, and alphabet
compaction can shrink further -- the *product* of the two tables is itself a
small table:

    product[a, b] = act_lut[a] * weight_lut[b]

so the multiply disappears and both fetches collapse into one gather indexed by
`a * K_weight + b`. That is the whole idea: `(K_act, K_weight)` FP32 values,
typically 225 floats, replace `n` multiplies per output element.

Two things this module deliberately does *not* do:

* It does not approximate. `product` is exact FP32 arithmetic, and the fused
  result is compared against the two-fetch reference to prove it. The accuracy
  difference is float-addition ordering only, and the tests assert that
  difference stays inside an explicit tolerance rather than at zero.
* It does not change the storage format. The weight indices are still read in
  their K-selected packed form; only the *evaluation* changes.

Why the tolerance is not zero: summing `n` fused products in a different order
than `n` two-fetch products produces the same real number but a different
FP32 rounding. `fused_matches_reference` reports the deviation instead of
asserting an exact match, so a backend can state its own budget.
"""

from __future__ import annotations

import torch

#: Alphabet sizes above which the product table stops paying for itself.
#: `product` costs `K_act * K_weight` FP32 slots; a linear saves `m * n`
#: multiplies. With LC-QAT's K<=16 the table is a few hundred floats, so the
#: only case that matters is the degenerate one where a caller passes a huge
#: act alphabet (a quantizer configured far above the storage budget).
MAX_PRODUCT_TABLE_ENTRIES = 1 << 16


def product_table_size(k_act: int, k_weight: int) -> int:
    """Number of FP32 slots a fused product table for this pair would need."""
    return int(k_act) * int(k_weight)


def build_product_table(
    act_lut: torch.Tensor, weight_lut: torch.Tensor
) -> torch.Tensor:
    """Return `(K_act, K_weight)` FP32 table of `act_lut[a] * weight_lut[b]`.

    Args:
        act_lut: `(K_act,)` FP32 activation codebook.
        weight_lut: `(K_weight,)` FP32 weight codebook.

    Returns:
        `(K_act, K_weight)` FP32. Indexing it with `(a, b)` gives exactly
        `act_lut[a] * weight_lut[b]`, so a caller can reshape it to a flat
        gather table if it prefers.

    Raises:
        ValueError: if either codebook is not a 1-D FP32 tensor.
        ValueError: if the table would exceed `MAX_PRODUCT_TABLE_ENTRIES`.
    """
    for label, lut in (("act_lut", act_lut), ("weight_lut", weight_lut)):
        if lut.ndim != 1:
            raise ValueError(f"{label} must be 1-D, got shape {tuple(lut.shape)}")
        if lut.dtype != torch.float32:
            raise ValueError(f"{label} must be float32, got {lut.dtype}")
    size = product_table_size(act_lut.numel(), weight_lut.numel())
    if size > MAX_PRODUCT_TABLE_ENTRIES:
        raise ValueError(
            f"fused product table would hold {size} entries, above the "
            f"{MAX_PRODUCT_TABLE_ENTRIES} limit; use the two-fetch path instead"
        )
    # Outer product in FP32: exact, and the same rounding a fused kernel gets.
    return act_lut.unsqueeze(1) * weight_lut.unsqueeze(0)


def flatten_product_table(product: torch.Tensor, k_weight: int) -> torch.Tensor:
    """Reshape `(K_act, K_weight)` to a flat `(K_act * K_weight,)` gather table.

    Flat index is `a * k_weight + b`, which is the addressing a single fused
    lookup wants: one multiply-add on the index instead of two separate
    gathers.
    """
    if product.ndim != 2:
        raise ValueError(f"product must be 2-D, got shape {tuple(product.shape)}")
    if product.shape[1] != k_weight:
        raise ValueError(
            f"product has width {product.shape[1]}, expected k_weight={k_weight}"
        )
    return product.reshape(-1).contiguous()


def fused_index_linear(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    product: torch.Tensor | None = None,
) -> torch.Tensor:
    """`y = W @ x` evaluated through the fused product table.

    The reference oracle, spelled with the fused evaluation. Used as the oracle
    for a fused *kernel*; `reference_index_linear` remains the oracle for the
    two-fetch kernels, and `fused_matches_reference` ties the two together.

    Args:
        act_indices: `[T, n]` uint8 activation IDs.
        act_lut: `(K_act,)` FP32 activation codebook.
        weight_indices: `[m, n]` **unpacked** weight IDs.
        weight_lut: `(K_weight,)` FP32 weight codebook.
        n: input feature count.
        product: optional precomputed `(K_act, K_weight)` table from
            `build_product_table`. Recomputed when omitted.

    Returns:
        `[T, m]` FP32.
    """
    if weight_indices.ndim != 2 or weight_indices.shape[1] != n:
        raise ValueError(
            f"weight_indices must be an unpacked [m, n] matrix with n={n}, got "
            f"{tuple(weight_indices.shape)}"
        )
    table = build_product_table(act_lut, weight_lut) if product is None else product
    a = act_indices.long()  # [T, n]
    w = weight_indices.long()  # [m, n]
    # One gather per term instead of two fetches plus a multiply. `w.t()` is
    # [n, m], broadcast against `a` [T, n] to pair the j-th column of the
    # activation with the j-th column of every weight row.
    terms = table[a.unsqueeze(-1), w.t().unsqueeze(0)]  # [T, n, m]
    return terms.sum(dim=1)


def fused_matches_reference(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> tuple[bool, float]:
    """Compare fused and two-fetch evaluation; return `(ok, max_abs_deviation)`.

    Reports the deviation instead of only a boolean, so a backend can state and
    then monitor its own numerical budget. The two paths differ in FP32 addition
    order, so an exact-zero tolerance would be the wrong assertion.
    """
    from nanochat.lcqat.ops.references.index_linear_reference import (
        reference_index_linear,
    )
    from nanochat.lcqat.packing import pack_weight_indices

    packed, fmt = pack_weight_indices(weight_indices, weight_lut.numel())
    two_fetch = reference_index_linear(act_indices, act_lut, packed, weight_lut, n, fmt)
    fused = fused_index_linear(act_indices, act_lut, weight_indices, weight_lut, n)
    deviation = float((fused - two_fetch).abs().max()) if fused.numel() else 0.0
    return torch.allclose(fused, two_fetch, rtol=rtol, atol=atol), deviation


__all__ = [
    "MAX_PRODUCT_TABLE_ENTRIES",
    "build_product_table",
    "flatten_product_table",
    "fused_index_linear",
    "fused_matches_reference",
    "product_table_size",
]

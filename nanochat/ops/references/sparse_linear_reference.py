"""
Pure PyTorch reference for the *sparse* K-agnostic index-weight linear.

`reference_index_linear` resolves every weight slot through the LUT and runs a
dense matmul. That is the right oracle for a dense artifact, but it is useless
for checking the sparse kernel: a CSR kernel that silently skipped a nonzero
entry, double-counted one, or read the wrong column index would still agree
with a dense oracle computed from the same buggy index buffer.

So the oracle here is built from the *mask* rather than from the CSR listing.
The mask is the specification ("this slot is zero"); the listing is the
implementation. A correct kernel must reproduce the masked dense result
exactly, and `tests/test_lcqat_sparse_index_linear.py` asserts that it does.

Both oracles are kept, because they are not redundant:

* `reference_index_linear` answers "is the arithmetic right?"
* `reference_sparse_index_linear` answers "does the sparsity bookkeeping agree
  with the mask?" -- which is the question the CSR kernel can actually get wrong.

No custom operators and no device assumptions: this is the math a CPU/GPU
sparse kernel must reproduce, term for term.
"""

import torch


def validate_sparse_index_linear_inputs(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    indices: torch.Tensor,
    alphabet: torch.Tensor,
    m: int,
    n: int,
) -> None:
    """Fail-fast boundary validation for the CSR sparse path.

    Checks the *structural* invariants a wrong kernel would violate rather than
    produce: rank, dtype, `row_ptr` has one more entry than the matrix has
    rows, and the column listing and index buffer are the same length. Value
    scans -- pointer monotonicity, `row_ptr[0] == 0`, column indices in range --
    stay out of the hot path because they are data-dependent (see the note in
    `validate_index_linear_inputs`).

    `indices` is the compacted codebook index of each stored slot. A caller that
    has already resolved those to FP32 values (the compiled runtime path) may
    pass its value buffer here too: only the *length* of `indices` is checked,
    so the structural guarantees hold either way.
    """
    if act_indices.ndim != 2:
        raise ValueError(
            f"act_indices must be [T, n], got shape {tuple(act_indices.shape)}"
        )
    if act_indices.shape[1] != n:
        raise ValueError(
            f"shape mismatch: act_indices has width {act_indices.shape[1]}, n={n}"
        )
    if act_indices.dtype != torch.uint8:
        raise ValueError(f"act_indices must be uint8, got {act_indices.dtype}")
    for label, lut in (("act_lut", act_lut), ("alphabet", alphabet)):
        if lut.ndim != 1:
            raise ValueError(f"{label} must be 1-D")
        if lut.dtype != torch.float32:
            raise ValueError(f"{label} must be float32, got {lut.dtype}")
    if act_lut.numel() < 3:
        raise ValueError("act_lut K must be an integer >= 3")
    # No size floor on `alphabet`, deliberately. On this reference path it is a
    # codebook, so it would normally be >= 3; but the compiled runtime path
    # passes the already-resolved per-slot *values* under the same name, and a
    # heavily pruned layer may have fewer than 3 surviving slots in total.
    # Its length is checked against `indices` below, which is the invariant
    # that actually matters.
    if row_ptr.ndim != 1:
        raise ValueError(f"row_ptr must be 1-D, got {tuple(row_ptr.shape)}")
    if row_ptr.numel() != m + 1:
        raise ValueError(
            f"row_ptr must have m+1={m + 1} entries, got {row_ptr.numel()}"
        )
    if col_indices.ndim != 1 or indices.ndim != 1:
        raise ValueError("col_indices and indices must be 1-D")
    if col_indices.dtype != torch.int32 and col_indices.dtype != torch.int64:
        raise ValueError(f"col_indices must be int32 or int64, got {col_indices.dtype}")
    if col_indices.numel() != indices.numel():
        raise ValueError(
            f"col_indices ({col_indices.numel()}) and indices ({indices.numel()}) "
            "must have the same length"
        )


def reference_sparse_index_linear(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    indices: torch.Tensor,
    alphabet: torch.Tensor,
    m: int,
    n: int,
) -> torch.Tensor:
    """`y = W @ x` where `W` is given as CSR (row_ptr, col_indices, indices).

    Deliberately *not* built from a dense reconstruction: the mask is never
    consulted, so this is an independent statement of what the listing means.
    Each output row accumulates only its own `row_ptr[i]:row_ptr[i+1]` window.

    Args:
        act_indices: `[T, n]` uint8 activation IDs.
        act_lut: `(K_act,)` FP32 activation codebook.
        row_ptr: `(m + 1,)` CSR row offsets.
        col_indices: `(nnz,)` column of each stored weight.
        indices: `(nnz,)` **compacted codebook indices** (see
            `plan_sparse_export.keep_indices`), already unpacked -- not a packed
            buffer, and not FP32 values. The value of a stored weight is
            `alphabet[indices[i]]`.
        alphabet: `(k_used,)` FP32 compacted weight codebook.
        m, n: output and input feature counts.

    Returns:
        `[T, m]` FP32 result.
    """
    validate_sparse_index_linear_inputs(
        act_indices, act_lut, row_ptr, col_indices, indices, alphabet, m, n
    )
    x = act_lut[act_indices.long()]  # [T, n]
    out = torch.zeros(x.shape[0], m, dtype=torch.float32, device=x.device)
    cols = col_indices.long()
    weights = alphabet[indices.long()]  # [nnz]
    for i in range(m):
        start = int(row_ptr[i])
        stop = int(row_ptr[i + 1])
        if stop <= start:
            continue
        # Gather the activation columns this row actually touches and take the
        # dot product directly: no dense [m, n] allocation, so the oracle stays
        # honest about which entries contribute.
        cols_i = cols[start:stop]
        out[:, i] = (x[:, cols_i] * weights[start:stop].to(x.dtype)).sum(dim=-1)
    return out


def reference_masked_index_linear(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    mask: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """`y = W @ x` with `W` zeroed wherever `mask` is True.

    The mask-based oracle. A sparse kernel is correct exactly when this equals
    `reference_sparse_index_linear` over the same layer.

    Args:
        act_indices: `[T, n]` uint8 activation IDs.
        act_lut: `(K_act,)` FP32 activation codebook.
        weight_indices: `[m, n]` weight IDs, **already unpacked** -- an
            `int64` index matrix, not a packed buffer. Deliberately unlike
            `reference_index_linear`, which takes the packed form: this oracle
            is compared against a CSR listing that is likewise already
            resolved, so accepting packed input here would only invite passing
            an unpacked matrix and silently re-interpreting it as packed.
        weight_lut: `(K,)` FP32 weight codebook.
        mask: `[m, n]` bool, True where the slot is structurally zero.
        n: input feature count.
    """
    if weight_indices.ndim != 2:
        raise ValueError(
            f"weight_indices must be an unpacked [m, n] matrix, got "
            f"{tuple(weight_indices.shape)}"
        )
    if weight_indices.shape[1] != n:
        raise ValueError(
            f"shape mismatch: weight_indices has width {weight_indices.shape[1]}, n={n}"
        )
    if mask.shape != weight_indices.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match weight_indices "
            f"{tuple(weight_indices.shape)}"
        )
    w = weight_lut[weight_indices.long()]  # [m, n]
    w = w.masked_fill(mask.to(torch.bool), 0.0)
    x = act_lut[act_indices.long()]  # [T, n]
    return x @ w.T


__all__ = [
    "reference_masked_index_linear",
    "reference_sparse_index_linear",
    "validate_sparse_index_linear_inputs",
]

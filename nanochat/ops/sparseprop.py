"""
Public facade for the SparseProp AVX2 sparse backpropagation kernels.

Provides forward (SpMM) and backward (gX, gW) via custom C++ ops, with a
naive reference implementation for testing/debugging. Compiled backend loads
lazily on first use.
"""

import torch

_fake_registered = False


def _ensure_cpu_op() -> None:
    """Build/import the C++ extension once and register the FakeTensor kernels."""
    global _fake_registered
    from nanochat.ops.kernels.cpu_loader import load_cpu_sparseprop_extension

    load_cpu_sparseprop_extension()
    if not _fake_registered:

        @torch.library.register_fake("nanochat::lcqat_sparseprop_forward")
        def _sparseprop_forward_fake(x, w_val, w_col, w_ptr, bias, M):
            return torch.empty(M, x.size(1), dtype=torch.float32, device=x.device)

        @torch.library.register_fake("nanochat::lcqat_sparseprop_backward")
        def _sparseprop_backward_fake(
            gY, x, w_val, w_col, w_ptr, w_val_csc, w_row, w_cptr, M, K
        ):
            gX = torch.empty(K, gY.size(1), dtype=torch.float32, device=gY.device)
            gW_val = torch.empty_like(w_val)
            return gX, gW_val

        def _sparseprop_backward_autograd(ctx, *grad_outputs):
            gX_grad, gW_val_grad = grad_outputs
            return (
                gX_grad,
                None,
                gW_val_grad,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        torch.library.register_autograd(
            "nanochat::lcqat_sparseprop_backward", _sparseprop_backward_autograd
        )
        _fake_registered = True


def _validate_sparseprop_inputs(
    x: torch.Tensor,
    w_val: torch.Tensor,
    w_col: torch.Tensor,
    w_ptr: torch.Tensor,
    M: int,
) -> None:
    """Validate sparseprop forward inputs."""
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [K, B], got {x.dim()}D")
    if w_val.dim() != 1 or w_col.dim() != 1 or w_ptr.dim() != 1:
        raise ValueError("CSR buffers must be 1D")
    if x.dtype != torch.float32:
        raise ValueError(f"x must be float32, got {x.dtype}")
    if w_val.dtype != torch.float32:
        raise ValueError(f"w_val must be float32, got {w_val.dtype}")
    if w_col.dtype != torch.int32:
        raise ValueError(f"w_col must be int32, got {w_col.dtype}")
    if w_ptr.dtype != torch.int32:
        raise ValueError(f"w_ptr must be int32, got {w_ptr.dtype}")
    if w_ptr.size(0) != M + 1:
        raise ValueError(f"w_ptr size {w_ptr.size(0)} != M+1 = {M + 1}")


def sparseprop_forward_cpu(
    x: torch.Tensor,
    w_val: torch.Tensor,
    w_col: torch.Tensor,
    w_ptr: torch.Tensor,
    bias: torch.Tensor | None,
    M: int,
) -> torch.Tensor:
    """Run the compiled AVX2 SpMM forward on CPU."""
    _validate_sparseprop_inputs(x, w_val, w_col, w_ptr, M)
    _ensure_cpu_op()
    return torch.ops.nanochat.lcqat_sparseprop_forward(
        w_val.contiguous(),
        w_col.contiguous(),
        w_ptr.contiguous(),
        x.contiguous(),
        bias.contiguous() if bias is not None else torch.empty(0, dtype=torch.float32),
        M,
    )


def sparseprop_backward_cpu(
    gY: torch.Tensor,
    x: torch.Tensor,
    w_val: torch.Tensor,
    w_col: torch.Tensor,
    w_ptr: torch.Tensor,
    w_val_csc: torch.Tensor,
    w_row: torch.Tensor,
    w_cptr: torch.Tensor,
    M: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the compiled AVX2 backward kernels on CPU."""
    if gY.dim() != 2 or x.dim() != 2:
        raise ValueError("gY and x must be 2D [M/B, B]")
    if gY.size(0) != M or x.size(0) != K:
        raise ValueError(
            f"shape mismatch: gY[0]={gY.size(0)} != M={M} or x[0]={x.size(0)} != K={K}"
        )
    if gY.size(1) != x.size(1):
        raise ValueError(f"batch size mismatch: gY[1]={gY.size(1)} != x[1]={x.size(1)}")
    if gY.dtype != torch.float32 or x.dtype != torch.float32:
        raise ValueError("gY and x must be float32")
    _ensure_cpu_op()
    gX, gW_val = torch.ops.nanochat.lcqat_sparseprop_backward(
        gY.contiguous(),
        x.contiguous(),
        w_val.contiguous(),
        w_col.contiguous(),
        w_ptr.contiguous(),
        w_val_csc.contiguous(),
        w_row.contiguous(),
        w_cptr.contiguous(),
        M,
        K,
    )
    return gX, gW_val


# Naive Reference (for testing parity)


def _build_csr_from_mask(mask: torch.Tensor):
    """Build CSR (row_ptr, col_idx) from boolean mask [M, K].

    Fully vectorized: `nonzero` + `bincount` + `cumsum`, so the cost is O(M)
    device work and a single host sync rather than one per row. `nonzero`
    returns row-major order, which is what keeps the columns within a row
    ascending.
    """
    M = mask.shape[0]
    rows, cols = torch.nonzero(mask, as_tuple=True)
    counts = torch.bincount(rows, minlength=M)
    row_ptr = torch.zeros(M + 1, dtype=torch.int64, device=mask.device)
    torch.cumsum(counts, dim=0, out=row_ptr[1:])
    return row_ptr.to(torch.int32), cols.to(torch.int32)


def _build_csc_from_mask(mask: torch.Tensor):
    """Build CSC (col_ptr, row_idx) from boolean mask [M, K].

    Vectorized, but the **row indices within each column must be ascending**.

    That ordering is a real contract, not a cosmetic detail: the AVX2 backward
    walks column `j` over `w_row[col_ptr[j] : col_ptr[j+1]]` and accumulates
    `grad_out` rows in that order, so a different order changes the result. A
    single `torch.nonzero(mask)` emits row-major order, which groups *all* of row
    0 together, then all of row 1, etc. -- correct for CSR, wrong for CSC.

    Sorting the flat pairs by `(col, row)` restores the per-column ascending
    order. `torch.sort` on a composite key avoids materializing a second pass;
    the alternative (`mask.t().nonzero()`) is the transpose trick: nonzero on the
    transposed mask is column-major over the original, i.e. ascending rows per
    column.
    """
    K = mask.shape[1]
    # Transpose first: nonzero over the transposed mask enumerates
    # (col, row) pairs in column-major order, so within each column the row
    # indices come out ascending -- exactly the CSC layout the kernel expects.
    cols, rows = torch.nonzero(mask.t().contiguous(), as_tuple=True)
    counts = torch.bincount(cols, minlength=K)
    col_ptr = torch.zeros(K + 1, dtype=torch.int64, device=mask.device)
    torch.cumsum(counts, dim=0, out=col_ptr[1:])
    return col_ptr.to(torch.int32), rows.to(torch.int32)


def _masked_weight(weight_dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`weight_dense` with pruned slots forced to exactly zero.

    One place, because the forward and the backward must agree on whether the
    mask is applied to the weights (it is) and never to the gradient on its own.
    """
    return weight_dense * mask.to(weight_dense.dtype)


def reference_sparseprop_forward(
    x: torch.Tensor,  # [K, B]
    weight_dense: torch.Tensor,  # [M, K]
    mask: torch.Tensor,  # [M, K] bool
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense masked matmul reference: y = (W * mask) @ x + bias.

    Layout: x [K, B], W [M, K], y [M, B].
    """
    y = _masked_weight(weight_dense, mask) @ x  # [M, B]
    if bias is not None:
        y = y + bias.unsqueeze(1)
    return y


def reference_sparseprop_backward(
    gY: torch.Tensor,  # [M, B]
    x: torch.Tensor,  # [K, B]
    weight_dense: torch.Tensor,  # [M, K]
    mask: torch.Tensor,  # [M, K] bool
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Dense masked backward reference.

    Returns dense gX, the dense masked gW, and gBias -- the "only keep nnz"
    part of the sparse walk is expressed as the mask multiply, so the dense
    reference and the kernel agree on which slots carry a gradient.
    """
    # gX = W_eff.T @ gY  [K, B]
    gX = _masked_weight(weight_dense, mask).t() @ gY
    # gW = gY @ x.T  [M, K], but only keep nnz
    gW_dense = gY @ x.t()
    gW_masked = gW_dense * mask.to(gW_dense.dtype)
    # Bias grad
    gBias = gY.sum(dim=1) if bias is not None else None
    return gX, gW_masked, gBias


def build_csr_csc_from_mask(mask: torch.Tensor):
    """Build CSR and CSC structures from boolean mask [M, K]."""
    w_ptr, w_col = _build_csr_from_mask(mask)
    w_cptr, w_row = _build_csc_from_mask(mask)
    return w_ptr, w_col, w_cptr, w_row


def _expand_ptr(ptr: torch.Tensor, n: int) -> torch.Tensor:
    """Expand a CSR/CSC `ptr` into the per-nnz index it addresses, `[nnz]`.

    `ptr` is the `n+1` prefix-sum pointer of the structure and `n` its row (or
    column) count. One body serves both orientations -- CSR expands rows, CSC
    expands columns -- so the two cannot disagree about the walk order. The
    pointer is already on device, so only the final length needs a sync.
    """
    counts = ptr[1:] - ptr[:-1]
    axis = torch.arange(n, device=ptr.device)
    return torch.repeat_interleave(axis, counts.to(torch.int64)).to(torch.int32)


def _nnz_row_indices(w_row_ptr: torch.Tensor, M: int) -> torch.Tensor:
    """Expand CSR row_ptr into per-nnz row indices [nnz]."""
    return _expand_ptr(w_row_ptr, M)


def _csc_col_indices(w_cptr: torch.Tensor, K: int) -> torch.Tensor:
    """Expand CSC col_ptr into per-nnz column indices [nnz]."""
    return _expand_ptr(w_cptr, K)


def gather_values_from_dense(
    weight_dense: torch.Tensor,
    w_col: torch.Tensor,
    w_row_ptr: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """Gather weight values at nnz positions (CSR order) from dense [M, K]."""
    row_idx = _nnz_row_indices(w_row_ptr, M)
    K = weight_dense.size(1)
    lin_idx = row_idx * K + w_col
    return weight_dense.view(-1)[lin_idx.to(weight_dense.device)]

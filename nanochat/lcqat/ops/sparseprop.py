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
    from nanochat.lcqat.kernels.cpu_loader import load_cpu_sparseprop_extension

    load_cpu_sparseprop_extension()
    if not _fake_registered:

        @torch.library.register_fake("nanochat::lcqat_sparseprop_forward")
        def _sparseprop_forward_fake(x, w_val, w_col, w_ptr, bias, M):
            return torch.empty(
                M, x.size(1), dtype=torch.float32, device=x.device
            )

        @torch.library.register_fake("nanochat::lcqat_sparseprop_backward")
        def _sparseprop_backward_fake(gY, x, w_val, w_col, w_ptr,
                                      w_val_csc, w_row, w_cptr, M, K):
            gX = torch.empty(K, gY.size(1), dtype=torch.float32, device=gY.device)
            gW_val = torch.empty_like(w_val)
            return gX, gW_val

        def _sparseprop_backward_autograd(ctx, *grad_outputs):
            gX_grad, gW_val_grad = grad_outputs
            return (gX_grad, None, gW_val_grad, None, None, None, None, None, None, None)

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
        raise ValueError(f"w_ptr size {w_ptr.size(0)} != M+1 = {M+1}")


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
        raise ValueError(f"shape mismatch: gY[0]={gY.size(0)} != M={M} or x[0]={x.size(0)} != K={K}")
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
    """Build CSR (row_ptr, col_idx) from boolean mask [M, K]."""
    M, K = mask.shape
    row_ptr = [0]
    col_idx = []
    for i in range(M):
        cols = mask[i].nonzero(as_tuple=True)[0].tolist()
        col_idx.extend(cols)
        row_ptr.append(len(col_idx))
    return (
        torch.tensor(row_ptr, dtype=torch.int32, device=mask.device),
        torch.tensor(col_idx, dtype=torch.int32, device=mask.device),
    )


def _build_csc_from_mask(mask: torch.Tensor):
    """Build CSC (col_ptr, row_idx) from boolean mask [M, K]."""
    M, K = mask.shape
    col_ptr = [0]
    row_idx = []
    for j in range(K):
        rows = mask[:, j].nonzero(as_tuple=True)[0].tolist()
        row_idx.extend(rows)
        col_ptr.append(len(row_idx))
    return (
        torch.tensor(col_ptr, dtype=torch.int32, device=mask.device),
        torch.tensor(row_idx, dtype=torch.int32, device=mask.device),
    )


def _naive_sparseprop_forward(
    x: torch.Tensor,           # [K, B]
    weight_dense: torch.Tensor,  # [M, K]
    mask: torch.Tensor,         # [M, K] bool
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Dense masked matmul reference: y = (W * mask) @ x + bias.

    Layout: x [K, B], W [M, K], y [M, B].
    """
    W_eff = weight_dense * mask.to(weight_dense.dtype)
    y = W_eff @ x  # [M, B]
    if bias is not None:
        y = y + bias.unsqueeze(1)
    return y


def _naive_sparseprop_backward(
    gY: torch.Tensor,           # [M, B]
    x: torch.Tensor,            # [K, B]
    weight_dense: torch.Tensor,  # [M, K]
    mask: torch.Tensor,         # [M, K] bool
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Dense masked backward reference."""
    W_eff = weight_dense * mask.to(weight_dense.dtype)
    # gX = W_eff.T @ gY  [K, B]
    gX = W_eff.t() @ gY
    # gW = gY @ x.T  [M, K], but only keep nnz
    gW_dense = gY @ x.t()
    gW_masked = gW_dense * mask.to(gW_dense.dtype)
    # Bias grad
    gBias = gY.sum(dim=1) if bias is not None else None
    return gX, gW_masked, gBias


def reference_sparseprop_forward(
    x: torch.Tensor,
    weight_dense: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Naive reference for SpMM forward (uses dense matmul with mask)."""
    return _naive_sparseprop_forward(x, weight_dense, mask, bias)


def reference_sparseprop_backward(
    gY: torch.Tensor,
    x: torch.Tensor,
    weight_dense: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Naive reference for backward (returns dense gX, dense masked gW, gBias)."""
    return _naive_sparseprop_backward(gY, x, weight_dense, mask, bias)


def build_csr_csc_from_mask(mask: torch.Tensor):
    """Build CSR and CSC structures from boolean mask [M, K]."""
    w_ptr, w_col = _build_csr_from_mask(mask)
    w_cptr, w_row = _build_csc_from_mask(mask)
    return w_ptr, w_col, w_cptr, w_row


def _nnz_row_indices(w_row_ptr: torch.Tensor, M: int) -> torch.Tensor:
    """Expand CSR/CSC row_ptr into per-nnz row indices [nnz]."""
    nnz = int(w_row_ptr[-1].item())
    row_idx = torch.zeros(nnz, dtype=torch.int32, device=w_row_ptr.device)
    ptr = 0
    for i in range(M):
        end = int(w_row_ptr[i + 1].item())
        row_idx[ptr:end] = i
        ptr = end
    return row_idx


def _csc_col_indices(w_cptr: torch.Tensor, K: int, nnz: int, device) -> torch.Tensor:
    """Expand CSC col_ptr into per-nnz column indices [nnz]."""
    col_idx = torch.zeros(nnz, dtype=torch.int32, device=device)
    ptr = 0
    for k in range(K):
        end = int(w_cptr[k + 1].item())
        col_idx[ptr:end] = k
        ptr = end
    return col_idx


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
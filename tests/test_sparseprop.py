"""Tests for SparseProp: mathematical equivalence with dense masked Linear."""

import pytest
import torch

from nanochat.lcqat.ops.sparseprop import (
    _csc_col_indices,
    _nnz_row_indices,
    build_csr_csc_from_mask,
    gather_values_from_dense,
    reference_sparseprop_backward,
    reference_sparseprop_forward,
    sparseprop_backward_cpu,
    sparseprop_forward_cpu,
)


def _gather_csc_values(
    weight: torch.Tensor, w_row: torch.Tensor, w_cptr: torch.Tensor
) -> torch.Tensor:
    """Gather weight values at nnz positions (CSC order) from dense [M, K]."""
    nnz = w_row.numel()
    col_idx = _csc_col_indices(w_cptr, weight.size(1), nnz, weight.device)
    lin_idx = w_row.to(weight.device) * weight.size(1) + col_idx.to(weight.device)
    return weight.view(-1)[lin_idx]


@pytest.fixture
def sparse_setup():
    """Create a sparse weight, mask, and CSR/CSC structure for testing."""
    torch.manual_seed(42)
    M = 4  # output features (rows of weight matrix)
    K = 6  # input features (cols of weight matrix)
    B = 3  # batch size

    weight = torch.randn(M, K, dtype=torch.float32, requires_grad=True)
    mask = torch.rand(M, K) < 0.5
    # Ensure at least 1 nnz per row
    for i in range(M):
        if not mask[i].any():
            mask[i, 0] = True

    x = torch.randn(B, K, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(M, dtype=torch.float32)

    w_ptr, w_col, w_ptr_csc, w_row = build_csr_csc_from_mask(mask)

    # Gather nnz values from dense weight in CSR and CSC order
    w_val = gather_values_from_dense(weight, w_col, w_ptr, M)
    w_val_csc = _gather_csc_values(weight, w_row, w_ptr_csc)
    x_t = x.t().contiguous()  # [K, B]
    gY_t = torch.randn(M, B, dtype=torch.float32)  # [M, B]

    return {
        "M": M,
        "K": K,
        "B": B,
        "weight": weight,
        "mask": mask,
        "x": x,
        "x_t": x_t,
        "bias": bias,
        "gY_t": gY_t,
        "w_ptr": w_ptr,
        "w_col": w_col,
        "w_ptr_csc": w_ptr_csc,
        "w_row": w_row,
        "w_val": w_val,
        "w_val_csc": w_val_csc,
    }


class TestSparsepropForward:
    def test_forward_equivalence(self, sparse_setup):
        """Forward: custom kernel matches dense masked matmul."""
        s = sparse_setup

        y_kernel = sparseprop_forward_cpu(
            s["x_t"], s["w_val"], s["w_col"], s["w_ptr"], s["bias"], s["M"]
        )
        y_ref = reference_sparseprop_forward(
            s["x_t"], s["weight"], s["mask"], s["bias"]
        )

        assert torch.allclose(y_kernel, y_ref, atol=1e-5), (
            f"Forward mismatch: max diff = {(y_kernel - y_ref).abs().max()}"
        )

    def test_forward_no_bias(self, sparse_setup):
        """Forward without bias."""
        s = sparse_setup

        y_kernel = sparseprop_forward_cpu(
            s["x_t"], s["w_val"], s["w_col"], s["w_ptr"], None, s["M"]
        )
        y_ref = reference_sparseprop_forward(s["x_t"], s["weight"], s["mask"], None)

        assert torch.allclose(y_kernel, y_ref, atol=1e-5)


class TestSparsepropBackward:
    def test_backward_dx_equivalence(self, sparse_setup):
        """Backward input gradient: kernel matches dense reference."""
        s = sparse_setup

        gX, gW_val = sparseprop_backward_cpu(
            s["gY_t"],
            s["x_t"],
            s["w_val"],
            s["w_col"],
            s["w_ptr"],
            s["w_val_csc"],
            s["w_row"],
            s["w_ptr_csc"],
            s["M"],
            s["K"],
        )

        gX_ref, gW_ref, _ = reference_sparseprop_backward(
            s["gY_t"], s["x_t"], s["weight"], s["mask"], s["bias"]
        )

        # gX is [K, B] (kernel layout), transpose to compare
        assert torch.allclose(gX, gX_ref, atol=1e-4), (
            f"gX mismatch: max diff = {(gX - gX_ref).abs().max()}"
        )

    def test_backward_dw_equivalence(self, sparse_setup):
        """Backward weight gradient: kernel matches dense masked reference."""
        s = sparse_setup

        _, gW_val = sparseprop_backward_cpu(
            s["gY_t"],
            s["x_t"],
            s["w_val"],
            s["w_col"],
            s["w_ptr"],
            s["w_val_csc"],
            s["w_row"],
            s["w_ptr_csc"],
            s["M"],
            s["K"],
        )

        _, gW_ref, _ = reference_sparseprop_backward(
            s["gY_t"], s["x_t"], s["weight"], s["mask"], s["bias"]
        )

        # Scatter kernel gW_vals back to dense and compare
        row_idx = _nnz_row_indices(s["w_ptr"], s["M"])
        lin_idx = row_idx * s["K"] + s["w_col"]
        gW_dense = torch.zeros(s["M"], s["K"])
        gW_dense.view(-1)[lin_idx] = gW_val

        assert torch.allclose(gW_dense, gW_ref, atol=1e-4), (
            f"dW mismatch: max diff = {(gW_dense - gW_ref).abs().max()}"
        )

    def test_gradient_zero_at_masked_positions(self, sparse_setup):
        """Weight gradient is exactly zero at masked positions."""
        s = sparse_setup

        _, gW_val = sparseprop_backward_cpu(
            s["gY_t"],
            s["x_t"],
            s["w_val"],
            s["w_col"],
            s["w_ptr"],
            s["w_val_csc"],
            s["w_row"],
            s["w_ptr_csc"],
            s["M"],
            s["K"],
        )

        # Scatter to dense
        row_idx = _nnz_row_indices(s["w_ptr"], s["M"])
        lin_idx = row_idx * s["K"] + s["w_col"]
        gW_dense = torch.zeros(s["M"], s["K"])
        gW_dense.view(-1)[lin_idx] = gW_val

        # At masked positions, gW must be zero
        assert (gW_dense[~s["mask"]] == 0).all(), (
            "Non-zero gradient at masked positions!"
        )

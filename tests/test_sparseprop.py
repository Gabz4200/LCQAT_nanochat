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


class TestSparseLayoutContracts:
    """Ordering contracts the AVX2 kernels depend on.

    Neither ordering is cosmetic: the forward walks `w_col[w_ptr[m]:w_ptr[m+1]]`
    and the backward walks `w_row[w_ptr_csc[j]:w_ptr_csc[j+1]]` in the stored
    order, accumulating `grad_out` as it goes. A layout that is correct as a
    *set* but ordered differently produces different gradients without any
    error being raised -- so the orders are pinned directly.

    The CSC case is the one that a single `torch.nonzero(mask)` gets wrong:
    `nonzero` emits row-major order, which is exactly the CSR layout and the
    wrong one for CSC.
    """

    @staticmethod
    def _adversarial_mask() -> torch.Tensor:
        """Mask with a fully dense row, a fully dense column, and empty rows/cols.

        A random mask can pass an ordering check by accident. Row 1 is dense and
        column 3 is dense, so an ordering bug has to survive a run of many
        identical-row entries in the CSC stream to hide.
        """
        mask = torch.zeros(6, 8, dtype=torch.bool)
        mask[1, :] = True  # dense row: long run under row-major ordering
        mask[:, 3] = True  # dense column: long run under CSC ordering
        mask[4, 0] = True
        mask[4, 7] = True
        mask[0, 0] = True
        return mask

    def test_when_csc_built_then_rows_ascend_within_each_column(self):
        """CSC row indices are ascending inside every column."""
        mask = self._adversarial_mask()
        col_ptr, row_idx = build_csr_csc_from_mask(mask)[2:]

        row_idx_l = row_idx.long()
        for j in range(mask.shape[1]):
            column = row_idx_l[col_ptr[j].long() : col_ptr[j + 1].long()]
            assert torch.all(column[1:] > column[:-1]), (
                f"column {j} row indices are not strictly ascending: {column.tolist()}"
            )
            expected = torch.nonzero(mask[:, j], as_tuple=True)[0]
            assert torch.equal(column, expected), (
                f"column {j} holds {column.tolist()}, expected {expected.tolist()}"
            )

    def test_when_csr_built_then_cols_ascend_within_each_row(self):
        """CSR column indices are ascending inside every row."""
        mask = self._adversarial_mask()
        row_ptr, col_idx = build_csr_csc_from_mask(mask)[:2]

        col_idx_l = col_idx.long()
        for m in range(mask.shape[0]):
            row = col_idx_l[row_ptr[m].long() : row_ptr[m + 1].long()]
            assert torch.all(row[1:] > row[:-1]), (
                f"row {m} column indices are not strictly ascending: {row.tolist()}"
            )
            expected = torch.nonzero(mask[m], as_tuple=True)[0]
            assert torch.equal(row, expected), (
                f"row {m} holds {row.tolist()}, expected {expected.tolist()}"
            )

    def test_when_mask_random_then_csc_ordering_holds_for_many_seeds(self):
        """The contract holds for arbitrary masks, not just the crafted one."""
        for seed in range(8):
            torch.manual_seed(seed)
            mask = torch.rand(9, 11) < 0.5
            # Guarantee >= 1 nnz per row so no column/row degenerates.
            empty = ~mask.any(dim=1)
            mask[empty, 0] = True
            col_ptr, row_idx = build_csr_csc_from_mask(mask)[2:]
            row_idx_l = row_idx.long()
            for j in range(mask.shape[1]):
                column = row_idx_l[col_ptr[j].long() : col_ptr[j + 1].long()]
                assert torch.all(column[1:] > column[:-1]), (
                    f"seed {seed} column {j} not ascending"
                )

    def test_when_layout_built_then_pointers_and_counts_agree(self):
        """row_ptr[-1] == col_ptr[-1] == nnz, and both cover the mask exactly."""
        mask = self._adversarial_mask()
        row_ptr, col_idx, col_ptr, row_idx = build_csr_csc_from_mask(mask)

        nnz = int(mask.sum())
        assert int(row_ptr[-1]) == nnz
        assert int(col_ptr[-1]) == nnz
        assert col_idx.numel() == nnz
        assert row_idx.numel() == nnz
        # Every (row, col) in the CSR listing must be a mask entry, and the
        # CSC listing must be the same set.
        rows_csr = _nnz_row_indices(row_ptr, mask.shape[0]).long()
        cols_csc = _csc_col_indices(col_ptr, mask.shape[1], nnz, mask.device).long()
        csr_set = {(int(r), int(c)) for r, c in zip(rows_csr, col_idx.long())}
        csc_set = {(int(r), int(c)) for r, c in zip(row_idx.long(), cols_csc)}
        assert csr_set == csc_set
        assert all(mask[r, c] for r, c in csr_set)

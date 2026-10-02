"""
Sparse index-weight linear: CSR listing of surviving quantized weight slots.

`reference_sparse_index_linear` (in `ops/references/sparse_linear_reference.py`)
is the oracle. This module is the compiled path and the dispatcher.

What is different from the dense `index_linear` path:

* The weight matrix is a `(row_ptr, col_indices, alphabet)` CSR listing of
  `nnz` slots instead of a dense `[m, n]` index matrix, so a heavily pruned
  layer touches only the weights it kept.
* `alphabet` holds *resolved FP32 values*, not codebook IDs. The compaction
  step (`sparse_artifact.plan_sparse_export`) already decided which levels
  survive and already knows the value of each, so the kernel does not need the
  original codebook at inference time.

Zero-skip: a stored slot whose value is exactly `0.0` is skipped rather than
accumulated. LC-QAT's zero anchor makes `0.0` a real weight, and magnitude
pruning stores a pruned slot as a real entry rather than dropping it, so this
is on the hot path in practice. Skipping is observably equal to adding zero.

Per `AGENTS.md`, dispatch raises when the requested backend cannot run. There
is no fallback to the dense path: a silent dense fallback would turn a sparse
layer into a dense one while still reporting success, which is exactly the bug
the sparsity flags exist to make visible.
"""

from __future__ import annotations

import torch

_fake_registered = False


def _ensure_cpu_sparse_index_linear_op() -> None:
    """Build/import the C++ extension once and register the FakeTensor kernel."""
    global _fake_registered
    from nanochat.lcqat.kernels.cpu_loader import load_cpu_sparseprop_extension

    load_cpu_sparseprop_extension()
    if _fake_registered:
        return

    @torch.library.register_fake("nanochat::lcqat_sparse_index_linear")
    def _lcqat_sparse_index_linear_fake(
        act_indices, act_lut, col_indices, row_ptr, alphabet, m, n
    ):
        return torch.empty(
            (act_indices.shape[0], m), dtype=torch.float32, device=act_lut.device
        )

    def _lcqat_sparse_index_linear_backward(ctx, *grad_outputs):
        raise RuntimeError(
            "nanochat::lcqat_sparse_index_linear is inference-only and defines "
            "no gradient; training runs the STE path in F.linear instead"
        )

    torch.library.register_autograd(
        "nanochat::lcqat_sparse_index_linear",
        _lcqat_sparse_index_linear_backward,
    )
    _fake_registered = True


def validate_sparse_index_linear_inputs(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    col_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    alphabet: torch.Tensor,
    m: int,
    n: int,
) -> None:
    """Fail-fast boundary validation for the compiled CSR path.

    Kept structurally parallel to `sparse_linear_reference`'s validator so the
    two paths cannot drift: same dtypes, same `m + 1` row pointer rule, same
    `nnz` agreement between columns and values.

    Raises:
        ValueError: on any contract violation.
    """
    from nanochat.lcqat.ops.references.sparse_linear_reference import (
        validate_sparse_index_linear_inputs as _validate_reference,
    )

    # The reference validator takes both the *codebook indices* and the
    # alphabet because its kernel resolves `alphabet[indices[i]]`. This path
    # consumes pre-resolved values (compaction already did that lookup), so
    # the value buffer stands in for the index buffer -- the validator checks
    # only its *length*, and the structural guarantees are what matter here.
    _validate_reference(
        act_indices, act_lut, row_ptr, col_indices, alphabet, alphabet, m, n
    )
    # The reference tolerates int64 columns; the compiled kernel reads int32.
    if col_indices.dtype != torch.int32:
        raise ValueError(
            f"col_indices must be int32 for the compiled path, got {col_indices.dtype}"
        )
    if row_ptr.dtype != torch.int32:
        raise ValueError(
            f"row_ptr must be int32 for the compiled path, got {row_ptr.dtype}"
        )


def sparse_index_linear_cpu(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    col_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    alphabet: torch.Tensor,
    m: int,
    n: int,
) -> torch.Tensor:
    """Run the compiled CSR index-weight linear on CPU.

    Args:
        act_indices: `[T, n]` uint8 activation IDs.
        act_lut: `(K_act,)` FP32 activation codebook.
        col_indices: `[nnz]` int32 input-feature index of each stored slot.
        row_ptr: `[m + 1]` int32 CSR row offsets.
        alphabet: `[nnz]` FP32 resolved weight value of each stored slot.
        m: output features. `n`: input features.

    Returns:
        `[T, m]` FP32.
    """
    validate_sparse_index_linear_inputs(
        act_indices, act_lut, col_indices, row_ptr, alphabet, m, n
    )
    _ensure_cpu_sparse_index_linear_op()
    return torch.ops.nanochat.lcqat_sparse_index_linear(
        act_indices.contiguous(),
        act_lut.contiguous(),
        col_indices.contiguous(),
        row_ptr.contiguous(),
        alphabet.contiguous(),
        int(m),
        int(n),
    )


def sparse_index_linear(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    col_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    alphabet: torch.Tensor,
    m: int,
    n: int,
    backend: str = "cpu",
) -> torch.Tensor:
    """Dispatch the CSR index-weight linear to `backend`.

    Raises:
        ValueError: for an unknown backend. There is no dense fallback -- a
            silent downgrade would report success for a layer the caller
            believed was running sparse.
    """
    if backend != "cpu":
        raise ValueError(
            f"sparse index-linear backend {backend!r} is not available; "
            "this path has no fallback (a dense fallback would silently "
            "degrade a sparse layer)"
        )
    return sparse_index_linear_cpu(
        act_indices, act_lut, col_indices, row_ptr, alphabet, m, n
    )


__all__ = [
    "sparse_index_linear",
    "sparse_index_linear_cpu",
    "validate_sparse_index_linear_inputs",
]

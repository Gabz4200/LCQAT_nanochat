"""
Strategy dispatcher for the mul-less ternary GEMV (LC-QAT PRD section 5.2).

Models and callers never import kernels directly; they call dispatch_gemv with
an explicit backend. Error contract: a requested compiled backend that is
missing, fails to build, or is unsupported raises - it never silently falls
back to the naive reference. The naive backend is only ever reached when it
was explicitly requested (reference testing, debugging, CI parity).
"""

from typing import Literal

import torch

#: The three backends every dispatcher accepts. `naive` is the pure-PyTorch
#: oracle and is reached only when explicitly requested; `cpu` and `gpu` are
#: the compiled paths.
Backend = Literal["naive", "cpu", "gpu"]

VALID_BACKENDS: tuple[str, ...] = ("naive", "cpu", "gpu")


def _select_backend(backend: str, *, naive, cpu, gpu):
    """Run the thunk for `backend`, or raise. One implementation of the ladder.

    The thunks are zero-argument callables so each backend's imports stay
    *inside* the branch: a compiled backend's module must never be imported at
    package-import time, which is what keeps `import nanochat` working on a
    host with no compiler and no GPU.

    Raising on an unknown backend is the whole no-silent-fallback guarantee:
    there is deliberately no path from a failed `cpu`/`gpu` request back to
    `naive`.
    """
    if backend == "naive":
        return naive()
    if backend == "cpu":
        return cpu()
    if backend == "gpu":
        return gpu()
    raise ValueError(
        f"Unknown backend: {backend!r}. Valid backends: {list(VALID_BACKENDS)}"
    )


def dispatch_gemv(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    scale_neg: float,
    scale_pos: float,
    backend: Backend = "naive",
) -> torch.Tensor:
    """Dispatch the K_W=3 GEMV to the selected backend.

    Args:
        act_indices: [n] uint8 activation codebook indices.
        act_lut: [K_a] FP32 activation codebook.
        weight_indices: [m, n] uint8 trit indices in {0, 1, 2}.
        scale_neg: negative ternary level magnitude.
        scale_pos: positive ternary level magnitude.
        backend: "naive" (pure PyTorch oracle), "cpu" (AVX-512/AVX2 C++),
            "gpu" (Taichi/Vulkan kernel).
    """

    def _naive():
        from nanochat.ops.references.gemv_reference import reference_gemv_k3

        return reference_gemv_k3(
            act_indices, act_lut, weight_indices, scale_neg, scale_pos
        )

    def _cpu():
        from nanochat.ops.gemv import gemv_k3_cpu

        return gemv_k3_cpu(act_indices, act_lut, weight_indices, scale_neg, scale_pos)

    def _gpu():
        from nanochat.ops.gemv import gemv_k3_gpu

        return gemv_k3_gpu(act_indices, act_lut, weight_indices, scale_neg, scale_pos)

    return _select_backend(backend, naive=_naive, cpu=_cpu, gpu=_gpu)


def dispatch_quant_attn(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    v_idx: torch.Tensor,
    k_lut: torch.Tensor,
    v_lut: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int,
    backend: Backend = "naive",
) -> torch.Tensor:
    """Dispatch the index-native quantized-KV attention to the selected backend.

    Args:
        q: [B, Tq, H, D] FP32 queries.
        k_idx, v_idx: [B, T, H_kv, ceil(D/2)] uint8 nibble-packed cache.
        k_lut, v_lut: [H_kv, K] FP32 per-head codebooks (K odd in [3, 15]).
        cache_seqlens: [B] int32 valid rows including this step's Tq writes.
        window_left: left window size, -1 for full context (causal right).
        backend: "naive" (dequantize+SDPA oracle), "cpu" (C++ LUT-gather),
            "gpu" (Taichi/Vulkan kernel).
    """

    def _naive():
        from nanochat.ops.references.attn_reference import reference_quant_attn

        return reference_quant_attn(
            q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left
        )

    def _cpu():
        from nanochat.ops.quant_attn import quant_attn_cpu

        return quant_attn_cpu(q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left)

    def _gpu():
        from nanochat.ops.quant_attn import quant_attn_gpu

        return quant_attn_gpu(q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left)

    return _select_backend(backend, naive=_naive, cpu=_cpu, gpu=_gpu)


def dispatch_index_linear(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    format: int,
    backend: Backend = "naive",
) -> torch.Tensor:
    """Dispatch the K-agnostic index-weight linear to the selected backend.

    Weights are codebook IDs in their K-selected storage format
    (nanochat.models.quant.packing FORMAT_*), resolved through weight_lut at
    fetch time; activations are uint8 IDs resolved through act_lut.

    Args:
        act_indices: [T, n] uint8 activation codebook indices.
        act_lut: [K_a] FP32 activation codebook.
        weight_indices: packed [m, ...] weight IDs (shape per format).
        weight_lut: [K_w] FP32 weight codebook.
        n: in_features (unambiguous decode width for packed rows).
        format: FORMAT_* tag selecting the storage format.
        backend: "naive" (dequantize+matmul oracle), "cpu" (C++ LUT-gather),
            "gpu" (Taichi/Vulkan kernel).
    """

    def _naive():
        from nanochat.ops.references.index_linear_reference import (
            reference_index_linear,
        )

        return reference_index_linear(
            act_indices, act_lut, weight_indices, weight_lut, n, format
        )

    def _cpu():
        from nanochat.ops.index_linear import index_linear_cpu

        return index_linear_cpu(
            act_indices, act_lut, weight_indices, weight_lut, n, format
        )

    def _gpu():
        from nanochat.ops.index_linear import index_linear_gpu

        return index_linear_gpu(
            act_indices, act_lut, weight_indices, weight_lut, n, format
        )

    return _select_backend(backend, naive=_naive, cpu=_cpu, gpu=_gpu)


def dispatch_sparseprop_forward(
    x: torch.Tensor,
    weight_dense: torch.Tensor,
    mask: torch.Tensor,
    w_val: torch.Tensor,
    w_col: torch.Tensor,
    w_ptr: torch.Tensor,
    bias: torch.Tensor | None,
    M: int,
    backend: Backend = "naive",
) -> torch.Tensor:
    """Dispatch the SparseProp SpMM to the selected backend.

    Both representations are carried deliberately: the mask is
    the *specification* ("this slot is zero"), the CSR listing is
    the *implementation*. The naive oracle computes the dense
    masked matmul from the mask; the compiled backends walk the
    listing. A kernel that skipped or double-counted an nnz, or
    read the wrong column, agrees with neither -- which is the
    question the dispatch exists to answer.

    Args:
        x: [K, B] float32 input (transposed layout, batch-major
            storage is the caller's view of the same batch).
        weight_dense: [M, K] float32 weight (naive oracle only).
        mask: [M, K] bool keep-mask (naive oracle only).
        w_val, w_col, w_ptr: the CSR listing (compiled backends).
        bias: [M] float32 or None.
        M: out_features.
        backend: "naive" (dense masked matmul oracle), "cpu"
            (AVX2 C++ kernels), "gpu" (Taichi/Vulkan kernel).
    """

    def _naive():
        from nanochat.ops.sparseprop import reference_sparseprop_forward

        return reference_sparseprop_forward(x, weight_dense, mask, bias)

    def _cpu():
        from nanochat.ops.sparseprop import sparseprop_forward_cpu

        return sparseprop_forward_cpu(x, w_val, w_col, w_ptr, bias, M)

    def _gpu():
        from nanochat.ops.sparseprop import sparseprop_forward_gpu

        return sparseprop_forward_gpu(x, w_val, w_col, w_ptr, bias, M)

    return _select_backend(backend, naive=_naive, cpu=_cpu, gpu=_gpu)


def dispatch_sparseprop_backward(
    gY: torch.Tensor,
    x: torch.Tensor,
    weight_dense: torch.Tensor,
    mask: torch.Tensor,
    w_val: torch.Tensor,
    w_col: torch.Tensor,
    w_ptr: torch.Tensor,
    w_val_csc: torch.Tensor,
    w_row: torch.Tensor,
    w_cptr: torch.Tensor,
    M: int,
    K: int,
    backend: Backend = "naive",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch the SparseProp backward to the selected backend.

    Every backend returns the kernel layout: gX [K, B] and
    gW_val [nnz] in CSR order. The naive oracle computes the
    dense masked gradients and extracts the nnz listing through
    the same CSR structure, so the three backends agree on
    shape and semantics by construction.

    Args:
        gY: [M, B] float32 output gradient (kernel layout).
        x: [K, B] float32 input.
        weight_dense: [M, K] float32 weight (naive oracle only).
        mask: [M, K] bool keep-mask (naive oracle only).
        w_val, w_col, w_ptr: CSR listing (dW pass).
        w_val_csc, w_row, w_cptr: CSC listing (dX pass).
        M: out_features, K: in_features.
        backend: "naive" (dense masked reference), "cpu" (AVX2
            C++ kernels), "gpu" (Taichi/Vulkan kernel).
    """

    def _naive():
        from nanochat.ops.sparseprop import (
            _nnz_row_indices,
            reference_sparseprop_backward,
        )

        gX, gW_masked, _ = reference_sparseprop_backward(
            gY, x, weight_dense, mask, None
        )
        row_idx = _nnz_row_indices(w_ptr, M)
        lin_idx = row_idx.long() * K + w_col.long()
        gW_val = gW_masked.reshape(-1)[lin_idx.to(gW_masked.device)]
        return gX, gW_val

    def _cpu():
        from nanochat.ops.sparseprop import sparseprop_backward_cpu

        return sparseprop_backward_cpu(
            gY, x, w_val, w_col, w_ptr, w_val_csc, w_row, w_cptr, M, K
        )

    def _gpu():
        from nanochat.ops.sparseprop import sparseprop_backward_gpu

        return sparseprop_backward_gpu(
            gY, x, w_val, w_col, w_ptr, w_val_csc, w_row, w_cptr, M, K
        )

    return _select_backend(backend, naive=_naive, cpu=_cpu, gpu=_gpu)

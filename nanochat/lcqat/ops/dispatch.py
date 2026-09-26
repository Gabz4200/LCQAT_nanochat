"""
Strategy dispatcher for the mul-less ternary GEMV (LC-QAT PRD section 5.2).

Models and callers never import kernels directly; they call dispatch_gemv with
an explicit backend. Error contract: a requested compiled backend that is
missing, fails to build, or is unsupported raises - it never silently falls
back to the naive reference. The naive backend is only ever reached when it
was explicitly requested (reference testing, debugging, CI parity).
"""

import torch

from nanochat.lcqat.ops.references.gemv_reference import reference_gemv_k3


def dispatch_gemv(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    scale_neg: float,
    scale_pos: float,
    backend: str = "naive",
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
    if backend == "naive":
        return reference_gemv_k3(
            act_indices, act_lut, weight_indices, scale_neg, scale_pos
        )
    if backend == "cpu":
        from nanochat.lcqat.ops.gemv import gemv_k3_cpu

        return gemv_k3_cpu(act_indices, act_lut, weight_indices, scale_neg, scale_pos)
    if backend == "gpu":
        from nanochat.lcqat.ops.gemv import gemv_k3_gpu

        return gemv_k3_gpu(act_indices, act_lut, weight_indices, scale_neg, scale_pos)
    raise ValueError(
        f"Unknown backend: {backend!r}. Valid backends: ['naive', 'cpu', 'gpu']"
    )


def dispatch_quant_attn(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    v_idx: torch.Tensor,
    k_lut: torch.Tensor,
    v_lut: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int,
    backend: str = "naive",
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
    if backend == "naive":
        from nanochat.lcqat.ops.references.attn_reference import reference_quant_attn

        return reference_quant_attn(
            q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left
        )
    if backend == "cpu":
        from nanochat.lcqat.ops.quant_attn import quant_attn_cpu

        return quant_attn_cpu(q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left)
    if backend == "gpu":
        from nanochat.lcqat.ops.quant_attn import quant_attn_gpu

        return quant_attn_gpu(q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left)
    raise ValueError(
        f"Unknown backend: {backend!r}. Valid backends: ['naive', 'cpu', 'gpu']"
    )


def dispatch_index_linear(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    format: int,
    backend: str = "naive",
) -> torch.Tensor:
    """Dispatch the K-agnostic index-weight linear to the selected backend.

    Weights are codebook IDs in their K-selected storage format
    (nanochat.lcqat.packing FORMAT_*), resolved through weight_lut at
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
    if backend == "naive":
        from nanochat.lcqat.ops.references.index_linear_reference import (
            reference_index_linear,
        )

        return reference_index_linear(
            act_indices, act_lut, weight_indices, weight_lut, n, format
        )
    if backend == "cpu":
        from nanochat.lcqat.ops.index_linear import index_linear_cpu

        return index_linear_cpu(
            act_indices, act_lut, weight_indices, weight_lut, n, format
        )
    if backend == "gpu":
        from nanochat.lcqat.ops.index_linear import index_linear_gpu

        return index_linear_gpu(
            act_indices, act_lut, weight_indices, weight_lut, n, format
        )
    raise ValueError(
        f"Unknown backend: {backend!r}. Valid backends: ['naive', 'cpu', 'gpu']"
    )

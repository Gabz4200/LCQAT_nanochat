"""
Public facade for the index-native quantized-KV decode attention op
(LC-QAT PRD section 7.1).

Validates the shared contract, packs nothing (the cache already stores packed
nibbles), and invokes the registered custom operator. Compiled backends load
lazily on first use - never at package import. Inference-only: autograd raises
and points at the STE F.linear training path, mirroring lcqat_gemv_k3.
"""

import torch

from nanochat.ops.kernels.registration import register_inference_only_op
from nanochat.ops.references.attn_reference import validate_quant_attn_inputs

_fake_registered = False


def _ensure_cpu_attn_op() -> None:
    """Build/import the C++ extension once and register the FakeTensor kernel."""
    global _fake_registered
    from nanochat.ops.kernels.cpu_loader import load_cpu_attn_extension

    load_cpu_attn_extension()
    if not _fake_registered:

        def _lcqat_quant_attn_fake(
            q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left
        ):
            return torch.empty_like(q)

        register_inference_only_op("nanochat::lcqat_quant_attn", _lcqat_quant_attn_fake)
        _fake_registered = True


def quant_attn_cpu(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    v_idx: torch.Tensor,
    k_lut: torch.Tensor,
    v_lut: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int,
) -> torch.Tensor:
    """Run the compiled C++ LUT-gather attention kernel on CPU."""
    validate_quant_attn_inputs(
        q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left
    )
    _ensure_cpu_attn_op()
    return torch.ops.nanochat.lcqat_quant_attn(
        q.contiguous(),
        k_idx.contiguous(),
        v_idx.contiguous(),
        k_lut.contiguous(),
        v_lut.contiguous(),
        cache_seqlens.contiguous(),
        int(window_left),
    )


def quant_attn_gpu(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    v_idx: torch.Tensor,
    k_lut: torch.Tensor,
    v_lut: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int,
) -> torch.Tensor:
    """Run the Taichi/Vulkan attention (loaded lazily on first call)."""
    validate_quant_attn_inputs(
        q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left
    )
    from nanochat.ops.kernels.gpu_loader import GPU_MAX_HEAD_DIM

    if q.shape[-1] > GPU_MAX_HEAD_DIM:
        raise ValueError(
            f"GPU backend supports head_dim <= {GPU_MAX_HEAD_DIM}, "
            f"got {q.shape[-1]}; dispatch with backend='cpu'"
        )
    from nanochat.ops.kernels.gpu_loader import run_quant_attn

    return run_quant_attn(
        q.contiguous(),
        k_idx.contiguous(),
        v_idx.contiguous(),
        k_lut.contiguous(),
        v_lut.contiguous(),
        cache_seqlens.contiguous(),
        int(window_left),
    )

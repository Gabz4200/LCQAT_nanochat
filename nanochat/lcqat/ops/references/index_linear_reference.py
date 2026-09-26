"""
Pure PyTorch reference for the K-agnostic index-weight linear
(quantized runtime: weights are codebook IDs, dequantize-on-fetch).

Ground-truth oracle for parity tests: unpacks the K-selected weight
storage, resolves both sides through their FP32 LUTs, and runs a plain
matmul. Device-agnostic, no custom operators - the math the CPU and
GPU (Taichi/Vulkan) kernels must reproduce.
"""

import torch

from nanochat.lcqat.packing import (
    FORMAT_INT32,
    FORMAT_NIBBLES,
    FORMAT_TRITS,
    FORMAT_UINT8,
    index_format_for_k,
    unpack_weight_indices,
)


def validate_index_linear_inputs(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    format: int,
) -> None:
    """Fail-fast boundary validation shared by every index-linear backend."""
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
    if act_lut.ndim != 1 or weight_lut.ndim != 1:
        raise ValueError("act_lut and weight_lut must be 1-D")
    for label, lut in (("act_lut", act_lut), ("weight_lut", weight_lut)):
        if lut.dtype != torch.float32:
            raise ValueError(f"{label} must be float32, got {lut.dtype}")
        if lut.numel() < 3 or lut.numel() % 2 != 1:
            raise ValueError(f"{label} K must be an odd integer >= 3")
    if weight_indices.ndim != 2:
        raise ValueError(
            f"weight_indices must be [m, ...], got shape {tuple(weight_indices.shape)}"
        )
    if format not in (
        FORMAT_TRITS,
        FORMAT_NIBBLES,
        FORMAT_UINT8,
        FORMAT_INT32,
    ):
        raise ValueError(f"unknown weight index format: {format}")
    expected = index_format_for_k(weight_lut.numel())
    if format != expected:
        raise ValueError(
            f"format {format} does not match K={weight_lut.numel()} "
            f"(expected format {expected})"
        )
    if format == FORMAT_TRITS and weight_indices.shape[1] * 5 < n:
        raise ValueError(
            f"trit-packed rows hold {weight_indices.shape[1] * 5} values, need n={n}"
        )
    if format == FORMAT_NIBBLES and weight_indices.shape[1] * 2 < n:
        raise ValueError(
            f"nibble-packed rows hold {weight_indices.shape[1] * 2} values, need n={n}"
        )
    if format in (FORMAT_UINT8, FORMAT_INT32) and weight_indices.shape[1] != n:
        raise ValueError(
            f"raw weight rows have width {weight_indices.shape[1]}, need n={n}"
        )
    # Data-dependent value scans call .item()/max(), which torch.compile
    # cannot trace; the shape checks above are static and always run.
    # Inputs entering a compiled graph are guarded by the eager call.
    if torch.compiler.is_compiling():
        return
    if act_indices.numel() and int(act_indices.max()) >= act_lut.numel():
        raise ValueError(
            f"act index {int(act_indices.max())} out of range "
            f"for LUT of size {act_lut.numel()}"
        )
    raw = unpack_weight_indices(weight_indices, n, weight_lut.numel())
    if raw.numel() and int(raw.max()) >= weight_lut.numel():
        raise ValueError(
            f"weight index {int(raw.max())} out of range "
            f"for LUT of size {weight_lut.numel()}"
        )


def reference_index_linear(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    format: int,
) -> torch.Tensor:
    """y = W @ x with W rows resolved from packed IDs through weight_lut."""
    validate_index_linear_inputs(
        act_indices, act_lut, weight_indices, weight_lut, n, format
    )
    raw = unpack_weight_indices(weight_indices, n, weight_lut.numel())
    w = weight_lut[raw.long()]  # [m, n]
    x = act_lut[act_indices.long()]  # [T, n]
    return x @ w.T

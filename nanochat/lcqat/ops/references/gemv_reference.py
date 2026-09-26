"""
Pure PyTorch reference for the mul-less ternary GEMV (LC-QAT PRD sections 5.1-5.2).

Ground-truth oracle for parity tests: dequantizes the ternary weights in
floating point and runs a plain matmul. Device-agnostic, no packed inputs,
no custom operators - mathematically the same computation the CPU (AVX) and
GPU (Taichi/Vulkan) kernels must reproduce.
"""

import torch


def validate_gemv_inputs(
    act_indices: torch.Tensor, act_lut: torch.Tensor, weight_indices: torch.Tensor
) -> None:
    """Fail-fast boundary validation shared by every GEMV backend."""
    if act_indices.ndim != 1:
        raise ValueError(
            f"act_indices must be 1-D, got shape {tuple(act_indices.shape)}"
        )
    if act_lut.ndim != 1:
        raise ValueError(f"act_lut must be 1-D, got shape {tuple(act_lut.shape)}")
    if weight_indices.ndim != 2:
        raise ValueError(
            f"weight_indices must be 2-D, got shape {tuple(weight_indices.shape)}"
        )
    if act_indices.numel() != weight_indices.shape[1]:
        raise ValueError(
            f"shape mismatch: act_indices has {act_indices.numel()} entries, "
            f"weights expect {weight_indices.shape[1]}"
        )
    # Data-dependent value scans call .item(), which torch.compile cannot trace;
    # shape checks above are static and always run. Inputs entering a compiled
    # graph are guarded by the eager call that produced them.
    if torch.compiler.is_compiling():
        return
    if weight_indices.numel() and int(weight_indices.max()) > 2:
        raise ValueError("weight_indices must be trits in {0, 1, 2}")
    if act_indices.numel() and int(act_indices.max()) >= act_lut.numel():
        raise ValueError(
            f"act index {int(act_indices.max())} out of range "
            f"for LUT of size {act_lut.numel()}"
        )


def reference_gemv_k3(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    scale_neg: float,
    scale_pos: float,
) -> torch.Tensor:
    """y = W @ x with W rows in {-scale_neg, 0, +scale_pos} selected by trits.

    Args:
        act_indices: [n] uint8 activation codebook indices (< len(act_lut)).
        act_lut: [K_a] FP32 activation codebook (dequantize-on-fetch LUT).
        weight_indices: [m, n] uint8 trit indices in {0, 1, 2}
            (0 -> -scale_neg, 1 -> 0, 2 -> +scale_pos).
        scale_neg: magnitude of the negative ternary level (delta minus).
        scale_pos: magnitude of the positive ternary level (delta plus).

    Returns:
        [m] FP32 output vector.
    """
    validate_gemv_inputs(act_indices, act_lut, weight_indices)

    x = act_lut.to(torch.float32)[act_indices.long()]
    w = torch.where(
        weight_indices == 2,
        torch.tensor(
            float(scale_pos), dtype=torch.float32, device=weight_indices.device
        ),
        torch.where(
            weight_indices == 0,
            torch.tensor(
                -float(scale_neg), dtype=torch.float32, device=weight_indices.device
            ),
            torch.tensor(0.0, dtype=torch.float32, device=weight_indices.device),
        ),
    )
    return w @ x

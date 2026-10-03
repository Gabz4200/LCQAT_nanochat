"""
Pure-PyTorch reference for the DiffusionBlocks EDM denoising loss.

Ground-truth oracle for parity tests: `w * mean((pred - clean)^2)` over every
element (DiffusionBlocks trains one scalar sigma per optimizer step, so the
EDM weight `w(sigma)` is a single scalar for the whole batch). Device-agnostic,
no custom operators -- mathematically the computation the CPU (C++) and GPU
(Taichi) kernels must reproduce.
"""

import torch


def validate_db_denoise_inputs(
    pred: torch.Tensor, clean: torch.Tensor, weight: float
) -> None:
    """Fail-fast boundary validation shared by every EDM-loss backend."""
    if pred.shape != clean.shape:
        raise ValueError(
            f"pred and clean must have the same shape, got {tuple(pred.shape)} "
            f"and {tuple(clean.shape)}"
        )
    if pred.numel() == 0:
        raise ValueError("pred and clean must be non-empty (mean over zero elements)")
    if pred.dtype != torch.float32 or clean.dtype != torch.float32:
        raise ValueError(
            f"pred and clean must be float32, got {pred.dtype} and {clean.dtype}"
        )
    if not isinstance(weight, float):
        raise ValueError(f"weight must be a Python float, got {type(weight)}")
    if not (weight >= 0.0):
        raise ValueError(f"weight must be non-negative, got {weight}")


def reference_db_denoise(
    pred: torch.Tensor, clean: torch.Tensor, weight: float
) -> torch.Tensor:
    """EDM denoising loss: `weight * mean((pred - clean)^2)` as a scalar."""
    validate_db_denoise_inputs(pred, clean, weight)
    return (weight * (pred - clean).square()).mean()

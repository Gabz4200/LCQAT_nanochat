"""Compute-dtype policy.

The dtype used for matmuls and activations is a *model* decision: every
`Linear` casts its weights to it in forward, so the layers depend on it and it
cannot live in the imperative shell. Master weights stay fp32 for optimizer
precision; only the compute cast moves.

Override with `NANOCHAT_DTYPE`, one of `bfloat16`, `float16`, `float32`.
"""

from __future__ import annotations

import os

import torch

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _detect_compute_dtype() -> tuple[torch.dtype, str]:
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return (
                torch.bfloat16,
                f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)",
            )
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        return (
            torch.float32,
            f"auto-detected: CUDA SM {capability[0]}{capability[1]} "
            "(pre-Ampere, bf16 not supported, using fp32)",
        )
    # Note: MPS on recent macOS also handles bf16 fine, opt in via NANOCHAT_DTYPE=bfloat16
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"


COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

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
            "gpu" or "slang" (portable Slang shader).
    """
    if backend == "naive":
        return reference_gemv_k3(
            act_indices, act_lut, weight_indices, scale_neg, scale_pos
        )
    if backend == "cpu":
        from nanochat.lcqat.ops.gemv import gemv_k3_cpu

        return gemv_k3_cpu(act_indices, act_lut, weight_indices, scale_neg, scale_pos)
    if backend in ("gpu", "slang"):
        from nanochat.lcqat.ops.gemv import gemv_k3_gpu

        return gemv_k3_gpu(act_indices, act_lut, weight_indices, scale_neg, scale_pos)
    raise ValueError(
        f"Unknown backend: {backend!r}. Valid backends: ['naive', 'cpu', 'gpu']"
    )

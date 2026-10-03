"""
Public facade for the mul-less ternary GEMV (LC-QAT PRD section 5.2).

Takes unpacked index tensors (the natural output of the quantizers), packs
them per the PRD storage layout, and invokes the registered custom operator.
Compiled backends load lazily on first use - never at package import.
"""

import torch

from nanochat.models.quant.packing import pack_nibbles, pack_trits
from nanochat.ops.kernels.registration import register_inference_only_op
from nanochat.ops.references.gemv_reference import validate_gemv_inputs

_fake_registered = False


def ternary_scales(codebook: torch.Tensor) -> tuple[float, float]:
    """(scale_neg, scale_pos) of a K=3 codebook with levels {-d-, 0, +d+}."""
    if codebook.numel() != 3:
        raise ValueError(
            f"ternary_scales expects a K=3 codebook, got {codebook.numel()} entries"
        )
    scale_pos = float(codebook[2])
    scale_neg = float(-codebook[0])
    if not (scale_pos > 0.0 and scale_neg > 0.0):
        raise ValueError(f"degenerate ternary codebook: {codebook.tolist()}")
    return scale_neg, scale_pos


def _ensure_cpu_op() -> None:
    """Build/import the C++ extension once and register the FakeTensor kernel."""
    global _fake_registered
    from nanochat.ops.kernels.cpu_loader import load_cpu_extension

    load_cpu_extension()
    if not _fake_registered:

        def _lcqat_gemv_k3_fake(
            act_nibbles, act_lut, weight_trits, n, scale_neg, scale_pos
        ):
            return torch.empty(
                weight_trits.shape[0], dtype=torch.float32, device=act_lut.device
            )

        register_inference_only_op("nanochat::lcqat_gemv_k3", _lcqat_gemv_k3_fake)
        _fake_registered = True


def gemv_k3_cpu(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    scale_neg: float,
    scale_pos: float,
) -> torch.Tensor:
    """Run the compiled AVX-512/AVX2 mul-less GEMV on CPU."""
    validate_gemv_inputs(act_indices, act_lut, weight_indices)
    _ensure_cpu_op()
    act_nibbles = pack_nibbles(act_indices)
    weight_trits = pack_trits(weight_indices)
    return torch.ops.nanochat.lcqat_gemv_k3(
        act_nibbles.contiguous(),
        act_lut.to(torch.float32).contiguous(),
        weight_trits.contiguous(),
        int(act_indices.numel()),
        float(scale_neg),
        float(scale_pos),
    )


def gemv_k3_gpu(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    scale_neg: float,
    scale_pos: float,
) -> torch.Tensor:
    """Run the Taichi/Vulkan mul-less GEMV (loaded lazily on first call)."""
    validate_gemv_inputs(act_indices, act_lut, weight_indices)
    from nanochat.ops.kernels.gpu_loader import run_gemv

    return run_gemv(act_indices, act_lut, weight_indices, scale_neg, scale_pos)

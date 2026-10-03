"""
Public facade for the K-agnostic index-weight linear (quantized runtime:
weights as codebook IDs, dequantize-on-fetch).

Validates the shared contract, invokes the registered custom operator, and
keeps the same lifecycle as lcqat_gemv_k3 / lcqat_quant_attn: compiled
backends load lazily on first use, the op is inference-only (autograd
raises and points at the STE F.linear training path).
"""

import torch

from nanochat.ops.kernels.registration import register_inference_only_op
from nanochat.ops.references.index_linear_reference import (
    validate_index_linear_inputs,
)

_fake_registered = False


def _ensure_cpu_index_linear_op() -> None:
    """Build/import the C++ extension once and register the FakeTensor kernel."""
    global _fake_registered
    from nanochat.ops.kernels.cpu_loader import load_cpu_index_linear_extension

    load_cpu_index_linear_extension()
    if not _fake_registered:

        def _lcqat_index_linear_fake(
            act_indices, act_lut, weight_indices, weight_lut, n, format
        ):
            return torch.empty(
                (act_indices.shape[0], weight_indices.shape[0]),
                dtype=torch.float32,
                device=act_lut.device,
            )

        register_inference_only_op(
            "nanochat::lcqat_index_linear", _lcqat_index_linear_fake
        )
        _fake_registered = True


def index_linear_cpu(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    format: int,
) -> torch.Tensor:
    """Run the compiled C++ LUT-fetch linear on CPU."""
    validate_index_linear_inputs(
        act_indices, act_lut, weight_indices, weight_lut, n, format
    )
    _ensure_cpu_index_linear_op()
    return torch.ops.nanochat.lcqat_index_linear(
        act_indices.contiguous(),
        act_lut.contiguous(),
        weight_indices.contiguous(),
        weight_lut.contiguous(),
        int(n),
        int(format),
    )


def index_linear_gpu(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    format: int,
) -> torch.Tensor:
    """Run the Taichi/Vulkan LUT-fetch linear (loaded lazily)."""
    validate_index_linear_inputs(
        act_indices, act_lut, weight_indices, weight_lut, n, format
    )
    from nanochat.ops.kernels.gpu_loader import run_index_linear

    return run_index_linear(
        act_indices.contiguous(),
        act_lut.contiguous(),
        weight_indices.contiguous(),
        weight_lut.contiguous(),
        int(n),
        int(format),
    )

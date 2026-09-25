"""Lazy loader for the portable Slang mul-less GEMV (LC-QAT PRD section 5.2).

The Vulkan device and the shader module are created on first dispatch only -
never at package import. Tensor interchange uses numpy round-trips: slangpy's
native torch bridge (`Tensor.from_torch`) requires CUDA tensors, so with a
CPU-only torch build we stage through numpy (`Tensor.from_numpy`). This is a
copy path; a shared-memory fast path lands with the native bridge (see
dev/lcqat_kv_cache.md for the runtime roadmap).
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import torch

_lock = Lock()
_device = None
_module = None
_KERNEL = "lcqat_gemv_k3"


def vulkan_available() -> bool:
    """True when a Vulkan device can be created (used by test skip marks)."""
    try:
        _get_device()
        return True
    except Exception:
        return False


def _get_device():
    """Create the Vulkan device once; a missing ICD/driver is an actionable error."""
    global _device
    if _device is None:
        with _lock:
            if _device is None:
                import slangpy as spy

                try:
                    _device = spy.create_device(type=spy.DeviceType.vulkan)
                except Exception as error:
                    raise RuntimeError(
                        "LC-QAT Slang GPU backend unavailable: could not create a Vulkan "
                        "device. Install a Vulkan driver/ICD (e.g. vulkan-intel, "
                        "vulkan-radeon, nvidia drivers), or dispatch with backend='cpu' "
                        "or backend='naive'."
                    ) from error
    return _device


def _load_module():
    global _module
    if _module is None:
        with _lock:
            if _module is None:
                import slangpy as spy

                source = (
                    Path(__file__).resolve().parent / "slang" / "gemv" / "forward.slang"
                )
                if not source.is_file():
                    raise RuntimeError(f"missing Slang kernel source: {source}")
                device = _get_device()
                _module = spy.Module.load_from_file(device, str(source))
    return _module


def gemv_slang(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    scale_neg: float,
    scale_pos: float,
) -> torch.Tensor:
    """Run the Slang GEMV on the Vulkan device and return an FP32 torch tensor."""
    import slangpy as spy

    module = _load_module()
    m, n = int(weight_indices.shape[0]), int(weight_indices.shape[1])

    act_np = act_indices.reshape(-1).to(torch.int32).contiguous().numpy()
    weights_np = weight_indices.reshape(-1).to(torch.int32).contiguous().numpy()
    lut_np = act_lut.to(torch.float32).contiguous().numpy()

    device = module.device
    act_spy = spy.Tensor.from_numpy(device, act_np)
    weights_spy = spy.Tensor.from_numpy(device, weights_np)
    lut_spy = spy.Tensor.from_numpy(device, lut_np)
    out_spy = spy.Tensor.from_numpy(device, np.zeros(m, dtype=np.float32))

    getattr(module, _KERNEL).dispatch(
        spy.uint3(m, 1, 1),
        act_lut=lut_spy,
        act_indices=act_spy,
        weight_indices=weights_spy,
        outp=out_spy,
        m=m,
        n=n,
        scale_neg=float(scale_neg),
        scale_pos=float(scale_pos),
    )
    return torch.from_numpy(out_spy.to_numpy().copy())

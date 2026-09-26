"""Lazy JIT loader for the compiled CPU mul-less GEMV extension (LC-QAT PRD
section 5.2). The build is triggered on first backend="cpu" dispatch only;
a missing compiler or failed build raises instead of falling back to naive."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

_lock = Lock()
_extensions: dict[str, object] = {}


def _load(name: str, src_name: str):
    """Build (once per process) and import one C++ extension."""
    cached = _extensions.get(name)
    if cached is None:
        with _lock:
            cached = _extensions.get(name)
            if cached is None:
                from torch.utils.cpp_extension import load

                source = (
                    Path(__file__).resolve().parents[1] / "native" / "cpu" / src_name
                )
                if not source.is_file():
                    raise RuntimeError(f"missing CPU kernel source: {source}")
                cached = load(
                    name=name,
                    sources=[str(source)],
                    extra_cflags=["-O3"],
                    # TORCH_LIBRARY-only shared object: no Python init function
                    is_python_module=False,
                    verbose=os.environ.get("LCQAT_KERNEL_VERBOSE", "0") == "1",
                )
                _extensions[name] = cached
    return cached


def load_cpu_extension():
    """Build (once per process) and import the C++ extension.

    Importing the extension runs its TORCH_LIBRARY static initializers,
    registering `nanochat::lcqat_gemv_k3`. The object file build is cached by
    torch.utils.cpp_extension across processes (~/.cache/torch_extensions).
    """
    return _load("nanochat_lcqat_cpu", "gemv.cpp")


def load_cpu_attn_extension():
    """Build (once per process) the quantized-KV attention extension.

    Registers `nanochat::lcqat_quant_attn` (separate extension so the GEMV
    and attention kernels compile and cache independently).
    """
    return _load("nanochat_lcqat_cpu_attn", "quant_attn.cpp")


def load_cpu_index_linear_extension():
    """Build (once per process) the K-agnostic index-weight linear extension.

    Registers `nanochat::lcqat_index_linear` (separate extension so each
    kernel compiles and caches independently).
    """
    return _load("nanochat_lcqat_cpu_index_linear", "index_linear.cpp")

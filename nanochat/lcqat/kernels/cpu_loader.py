"""Lazy JIT loader for the compiled CPU mul-less GEMV extension (LC-QAT PRD
section 5.2). The build is triggered on first backend="cpu" dispatch only;
a missing compiler or failed build raises instead of falling back to naive."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

_lock = Lock()
_extension = None
_attn_extension = None
_index_linear_extension = None


def load_cpu_extension():
    """Build (once per process) and import the C++ extension.

    Importing the extension runs its TORCH_LIBRARY static initializers,
    registering `nanochat::lcqat_gemv_k3`. The object file build is cached by
    torch.utils.cpp_extension across processes (~/.cache/torch_extensions).
    """
    global _extension
    if _extension is None:
        with _lock:
            if _extension is None:
                from torch.utils.cpp_extension import load

                source = (
                    Path(__file__).resolve().parents[1] / "native" / "cpu" / "gemv.cpp"
                )
                if not source.is_file():
                    raise RuntimeError(f"missing CPU kernel source: {source}")
                _extension = load(
                    name="nanochat_lcqat_cpu",
                    sources=[str(source)],
                    extra_cflags=["-O3"],
                    # TORCH_LIBRARY-only shared object: no Python init function
                    is_python_module=False,
                    verbose=os.environ.get("LCQAT_KERNEL_VERBOSE", "0") == "1",
                )
    return _extension


def load_cpu_attn_extension():
    """Build (once per process) the quantized-KV attention extension.

    Registers `nanochat::lcqat_quant_attn` (separate extension so the GEMV
    and attention kernels compile and cache independently).
    """
    global _attn_extension
    if _attn_extension is None:
        with _lock:
            if _attn_extension is None:
                from torch.utils.cpp_extension import load

                source = (
                    Path(__file__).resolve().parents[1]
                    / "native"
                    / "cpu"
                    / "quant_attn.cpp"
                )
                if not source.is_file():
                    raise RuntimeError(f"missing CPU kernel source: {source}")
                _attn_extension = load(
                    name="nanochat_lcqat_cpu_attn",
                    sources=[str(source)],
                    extra_cflags=["-O3"],
                    is_python_module=False,
                    verbose=os.environ.get("LCQAT_KERNEL_VERBOSE", "0") == "1",
                )
    return _attn_extension


def load_cpu_index_linear_extension():
    """Build (once per process) the K-agnostic index-weight linear extension.

    Registers `nanochat::lcqat_index_linear` (separate extension so each
    kernel compiles and caches independently).
    """
    global _index_linear_extension
    if _index_linear_extension is None:
        with _lock:
            if _index_linear_extension is None:
                from torch.utils.cpp_extension import load

                source = (
                    Path(__file__).resolve().parents[1]
                    / "native"
                    / "cpu"
                    / "index_linear.cpp"
                )
                if not source.is_file():
                    raise RuntimeError(f"missing CPU kernel source: {source}")
                _index_linear_extension = load(
                    name="nanochat_lcqat_cpu_index_linear",
                    sources=[str(source)],
                    extra_cflags=["-O3"],
                    is_python_module=False,
                    verbose=os.environ.get("LCQAT_KERNEL_VERBOSE", "0") == "1",
                )
    return _index_linear_extension

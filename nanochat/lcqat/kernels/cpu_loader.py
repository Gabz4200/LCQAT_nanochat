"""Lazy JIT loader for the compiled CPU mul-less GEMV extension (LC-QAT PRD
section 5.2). The build is triggered on first backend="cpu" dispatch only;
a missing compiler or failed build raises instead of falling back to naive."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

_lock = Lock()
_extension = None


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

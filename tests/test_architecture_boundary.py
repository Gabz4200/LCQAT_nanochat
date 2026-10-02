"""The functional core must not import the imperative shell.

`nanochat/models/` holds pure tensor math; everything that touches I/O,
persistence or lifecycle lives in the shell packages beside it. A core module
that imports the shell inverts that direction and re-couples the math to
logging, filesystems or the optimizer.

This walks real `import` statements via `ast` rather than grepping text, so a
package path mentioned in a docstring or a comment does not count as a
violation -- that distinction is exactly what a text search gets wrong.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE = REPO_ROOT / "nanochat" / "models"

#: Shell packages the core must stay independent of.
SHELL_PACKAGES = ("modules", "training", "data", "callbacks", "utils", "tasks")

#: `ops` is the sanctioned boundary: it is how the core reaches the hardware
#: (dispatcher, reference oracles, native kernels). Everything else is a leak.
ALLOWED_CORE_DEPS = ("nanochat.ops", "nanochat.models")


def _imported_modules(path: pathlib.Path) -> list[str]:
    """Return the module paths of real `import` statements in `path`."""
    modules: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the package
                continue
            if node.module:
                modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def _core_violations() -> list[str]:
    violations = []
    for path in sorted(CORE.rglob("*.py")):
        for module in _imported_modules(path):
            parts = module.split(".")
            if parts[0] != "nanochat" or len(parts) < 2:
                continue
            if module.startswith(ALLOWED_CORE_DEPS):
                continue
            if parts[1] in SHELL_PACKAGES:
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel} imports {module}")
    return violations


def test_core_does_not_import_shell() -> None:
    """No `nanochat/models/` module may import an imperative-shell package."""
    assert _core_violations() == []


@pytest.mark.parametrize(
    "module",
    [
        "nanochat.models.backbone",
        "nanochat.models.dtype",
        "nanochat.models.flash_attention",
        "nanochat.models.fp8",
        "nanochat.models.quant.retrofit",
    ],
)
def test_core_modules_import(module: str) -> None:
    """The audited core modules import standalone, without the shell present."""
    __import__(module)


def test_compute_dtype_reexport_is_the_same_object() -> None:
    """`utils.common` re-exports the core's dtype rather than defining its own.

    A second definition would let the shell and the core disagree about
    precision, which no test would catch at runtime.
    """
    from nanochat.models import dtype as core_dtype
    from nanochat.utils import common

    assert common.COMPUTE_DTYPE is core_dtype.COMPUTE_DTYPE
    assert common.COMPUTE_DTYPE_REASON == core_dtype.COMPUTE_DTYPE_REASON


def test_experiments_live_in_the_shell_not_the_core() -> None:
    """Ablation experiments are harness, so they may import `training/`.

    They previously sat under `models/quant/`, which made the core import
    `training/diffusion_blocks.py`. The measurement must resolve in the shell
    and must no longer be importable from the core.
    """
    assert (REPO_ROOT / "nanochat" / "modules" / "experiments").is_dir()
    assert not (CORE / "quant" / "experiments").exists()

    with pytest.raises(ModuleNotFoundError):
        __import__("nanochat.models.quant.experiments")

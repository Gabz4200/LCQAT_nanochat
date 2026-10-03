"""
Tests for the DiffusionBlocks EDM loss kernel stack.

The EDM denoising objective `mean(w(sigma) * (P - C)^2)` is the exact training
loss every `denoise_step` computes, so it gets the same three-way treatment as
every other compiled op in this repo: naive == cpu == gpu parity against an
independent oracle, torch.library opcheck, torch.compile composition, and the
no-silent-fallback error contract.

The compiled forward ops are inference-only (autograd raises, pointing at the
`DbDenoiseLossFunction` STE-style wrapper that training actually uses): the
loss scalar's gradient reaches `pred`/`clean` through the wrapper's analytic
backward, never by differentiating through the kernel.

python -m pytest tests/test_db_denoise_kernel.py -v
"""

import pytest
import torch

from nanochat.ops import dispatch_db_denoise
from nanochat.ops.kernels.gpu_loader import vulkan_available

requires_vulkan = pytest.mark.skipif(
    not vulkan_available(), reason="Vulkan device unavailable for the Taichi backend"
)


def make_case(b: int = 2, t: int = 8, d: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    pred = torch.randn(b, t, d, dtype=torch.float32)
    clean = torch.randn(b, t, d, dtype=torch.float32)
    weight = float(torch.rand(()) * 4.0 + 0.01)
    return pred, clean, weight


def reference_loss(pred: torch.Tensor, clean: torch.Tensor, weight: float) -> float:
    """Independent oracle: plain Python-loop mean, no tensor ops to share bugs."""
    total = 0.0
    n = 0
    p = pred.reshape(-1).tolist()
    c = clean.reshape(-1).tolist()
    for pi, ci in zip(p, c, strict=True):
        diff = pi - ci
        total += diff * diff
        n += 1
    return weight * total / n


def test_when_naive_backend_then_matches_independent_oracle() -> None:
    pred, clean, weight = make_case()
    got = dispatch_db_denoise(pred, clean, weight, backend="naive")
    assert got.ndim == 0 and torch.isfinite(got)
    assert float(got) == pytest.approx(reference_loss(pred, clean, weight), rel=1e-5)


def test_when_cpu_backend_then_matches_naive() -> None:
    for shape, seed in [((2, 8, 16), 1), ((1, 1, 7), 2), ((3, 33, 40), 3)]:
        pred, clean, weight = make_case(*shape, seed=seed)
        expected = dispatch_db_denoise(pred, clean, weight, backend="naive")
        got = dispatch_db_denoise(pred, clean, weight, backend="cpu")
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


@requires_vulkan
def test_when_gpu_backend_then_matches_naive() -> None:
    for shape, seed in [((2, 8, 16), 1), ((1, 5, 7), 2)]:
        pred, clean, weight = make_case(*shape, seed=seed)
        expected = dispatch_db_denoise(pred, clean, weight, backend="naive")
        got = dispatch_db_denoise(pred, clean, weight, backend="gpu")
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_when_unknown_backend_then_raises() -> None:
    pred, clean, weight = make_case()
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_db_denoise(pred, clean, weight, backend="tpu")


def test_when_shapes_mismatch_then_raises() -> None:
    pred, clean, weight = make_case()
    bad = torch.randn(2, 8, 15, dtype=torch.float32)
    with pytest.raises(ValueError, match="same shape"):
        dispatch_db_denoise(pred, bad, weight, backend="naive")
    with pytest.raises(ValueError, match="same shape"):
        dispatch_db_denoise(pred, bad, weight, backend="cpu")


def test_when_dtype_not_float32_then_raises() -> None:
    pred, clean, weight = make_case()
    with pytest.raises(ValueError, match="float32"):
        dispatch_db_denoise(pred.double(), clean.double(), weight, backend="naive")


def test_when_empty_then_raises() -> None:
    pred = torch.zeros(0, 8, 16, dtype=torch.float32)
    clean = torch.zeros(0, 8, 16, dtype=torch.float32)
    with pytest.raises(ValueError, match="non-empty"):
        dispatch_db_denoise(pred, clean, 1.0, backend="naive")


def test_when_forward_op_then_opcheck_passes() -> None:
    from nanochat.ops.db_denoise import _ensure_cpu_op

    _ensure_cpu_op()
    pred, clean, weight = make_case()
    args = (pred.contiguous(), clean.contiguous(), float(weight))
    torch.library.opcheck(torch.ops.nanochat.lcqat_db_denoise, args)


def test_when_compile_naive_backend_then_composes() -> None:
    pred, clean, weight = make_case()
    expected = dispatch_db_denoise(pred, clean, weight, backend="naive")
    compiled = torch.compile(
        lambda: dispatch_db_denoise(pred, clean, weight, backend="naive"),
        fullgraph=True,
    )
    assert torch.allclose(compiled(), expected, atol=1e-6)


def test_when_backward_through_cpu_op_then_actionable_error() -> None:
    from nanochat.ops.db_denoise import _ensure_cpu_op

    _ensure_cpu_op()
    pred, clean, weight = make_case()
    pred_g = pred.clone().requires_grad_(True)
    out = torch.ops.nanochat.lcqat_db_denoise(pred_g, clean, float(weight))
    with pytest.raises(RuntimeError, match="no gradient of its own"):
        out.backward(torch.tensor(1.0))


def test_when_loss_function_backward_then_matches_autograd() -> None:
    """The wrapper's analytic backward must equal differentiating the formula."""
    from nanochat.ops.db_denoise import db_denoise_loss

    torch.manual_seed(0)
    for backend in ("naive", "cpu"):
        pred = torch.randn(2, 6, 8, dtype=torch.float32, requires_grad=True)
        clean = torch.randn(2, 6, 8, dtype=torch.float32, requires_grad=True)
        weight = 2.5
        loss = db_denoise_loss(pred, clean, weight, backend=backend)
        assert torch.isfinite(loss)
        loss.backward()
        with torch.no_grad():
            diff = pred.detach() - clean.detach()
            n = diff.numel()
            expect_p = 2.0 * weight * diff / n
        assert torch.allclose(pred.grad, expect_p, atol=1e-5)
        assert torch.allclose(clean.grad, -expect_p, atol=1e-5)


def test_when_weight_zero_then_loss_and_grads_are_zero() -> None:
    from nanochat.ops.db_denoise import db_denoise_loss

    pred = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)
    clean = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)
    loss = db_denoise_loss(pred, clean, 0.0, backend="cpu")
    assert float(loss) == 0.0
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred.grad))
    assert torch.equal(clean.grad, torch.zeros_like(clean.grad))

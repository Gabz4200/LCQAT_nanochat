"""
Tests for the mul-less ternary GEMV dispatcher, reference oracle, and compiled
kernels (LC-QAT PRD sections 5.1-5.2): three-way parity naive == cpu == gpu,
torch.library opcheck, torch.compile composition, and backend error contracts.

python -m pytest tests/test_lcqat_ops.py -v
"""

import pytest
import torch

from nanochat.lcqat.kernels.gpu_loader import vulkan_available
from nanochat.lcqat.ops import dispatch_gemv
from nanochat.lcqat.ops.gemv import _ensure_cpu_op, ternary_scales
from nanochat.lcqat.ops.references.gemv_reference import reference_gemv_k3
from nanochat.lcqat.packing import pack_nibbles, pack_trits

requires_vulkan = pytest.mark.skipif(
    not vulkan_available(), reason="Vulkan device unavailable for the Taichi backend"
)


def make_case(n: int = 64, m: int = 32, k_a: int = 15, seed: int = 0):
    torch.manual_seed(seed)
    act_indices = torch.randint(0, k_a, (n,), dtype=torch.uint8)
    act_lut = torch.linspace(-2.0, 2.0, k_a)
    weight_indices = torch.randint(0, 3, (m, n), dtype=torch.uint8)
    return act_indices, act_lut, weight_indices, 0.37, 0.42


def loop_oracle(act_indices, act_lut, weight_indices, scale_neg, scale_pos):
    """Independent pure-Python computation of y = W @ x (not reusing the reference)."""
    x = [float(act_lut[int(i)]) for i in act_indices]
    out = []
    for row in weight_indices:
        acc = 0.0
        for j, t in enumerate(row.tolist()):
            if t == 2:
                acc += scale_pos * x[j]
            elif t == 0:
                acc -= scale_neg * x[j]
        out.append(acc)
    return torch.tensor(out)


def test_when_dispatch_unknown_backend_then_value_error() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=16, m=4)
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_gemv(act, lut, w, s_neg, s_pos, backend="cuda9")


def test_when_reference_then_matches_independent_python_loop() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=37, m=7, seed=1)
    got = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="naive")
    expected = loop_oracle(act, lut, w, s_neg, s_pos)
    assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("n", [16, 37, 64, 129])
def test_when_cpu_backend_then_matches_naive_reference(n: int) -> None:
    act, lut, w, s_neg, s_pos = make_case(n=n, m=n // 2 + 3, seed=n)
    naive = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="naive")
    cpu = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="cpu")
    assert cpu.shape == naive.shape
    assert torch.isfinite(cpu).all()
    assert torch.allclose(cpu, naive, atol=1e-5, rtol=1e-5)


def test_when_cpu_backend_with_symmetric_scales_then_matches_naive() -> None:
    act, lut, w, _, _ = make_case(n=64, m=16, seed=7)
    naive = dispatch_gemv(act, lut, w, 0.5, 0.5, backend="naive")
    cpu = dispatch_gemv(act, lut, w, 0.5, 0.5, backend="cpu")
    assert torch.allclose(cpu, naive, atol=1e-5, rtol=1e-5)


@requires_vulkan
@pytest.mark.parametrize("n", [16, 37, 64])
def test_when_gpu_backend_then_matches_naive_reference(n: int) -> None:
    act, lut, w, s_neg, s_pos = make_case(n=n, m=n // 2 + 3, seed=n + 100)
    naive = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="naive")
    gpu = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="gpu")
    assert gpu.shape == naive.shape
    assert torch.isfinite(gpu).all()
    assert torch.allclose(gpu, naive, atol=1e-5, rtol=1e-5)
    # explicit three-way leg: cpu vs gpu directly
    cpu = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="cpu")
    assert torch.allclose(gpu, cpu, atol=1e-5, rtol=1e-5)


def test_when_cpu_op_then_opcheck_passes() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=32, m=8, seed=3)
    _ensure_cpu_op()
    args = (
        pack_nibbles(act),
        lut.to(torch.float32).contiguous(),
        pack_trits(w),
        32,
        float(s_neg),
        float(s_pos),
    )
    torch.library.opcheck(torch.ops.nanochat.lcqat_gemv_k3, args)


def test_when_compile_with_cpu_backend_then_composes_via_fake_tensor() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=32, m=8, seed=4)
    _ensure_cpu_op()
    expected = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="cpu")

    def run():
        return torch.ops.nanochat.lcqat_gemv_k3(
            pack_nibbles(act),
            lut.to(torch.float32),
            pack_trits(w),
            32,
            float(s_neg),
            float(s_pos),
        )

    compiled = torch.compile(run, fullgraph=True)
    got = compiled()
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_when_compile_naive_backend_then_composes() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=32, m=8, seed=5)
    expected = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="naive")
    compiled = torch.compile(
        lambda: dispatch_gemv(act, lut, w, s_neg, s_pos, backend="naive"),
        fullgraph=True,
    )
    assert torch.allclose(compiled(), expected, atol=1e-6)


def test_when_compile_codebook_quantization_then_composes() -> None:
    # base_train runs the retrofitted model under torch.compile; bucketize and
    # the STE identity must survive tracing.
    from nanochat.lcqat import MemoryEfficientLearnedCodebook

    cb = MemoryEfficientLearnedCodebook(K=15, init_min=-2.0, init_max=2.0)
    x = torch.randn(64)
    compiled = torch.compile(lambda t: cb(t).value, fullgraph=True)
    got = compiled(x)
    assert torch.allclose(got, cb(x).value, atol=0.0)


def test_when_backward_through_cpu_op_then_actionable_error() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=32, m=8, seed=6)
    lut = lut.clone().requires_grad_(True)
    with pytest.raises(RuntimeError):
        out = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="cpu")
        out.sum().backward()


def test_when_act_index_out_of_range_then_raises_on_every_backend() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=16, m=4, k_a=15, seed=8)
    act_bad = act.clone()
    act_bad[0] = 15  # LUT has 15 entries: valid indices are 0..14
    for backend in ("naive", "cpu"):
        with pytest.raises(ValueError, match="out of range"):
            dispatch_gemv(act_bad, lut, w, s_neg, s_pos, backend=backend)


def test_when_weights_contain_non_trit_then_raises() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=16, m=4, seed=9)
    w_bad = w.clone()
    w_bad[0, 0] = 3
    with pytest.raises(ValueError, match=r"\{0, 1, 2\}"):
        dispatch_gemv(act, lut, w_bad, s_neg, s_pos, backend="naive")


def test_when_ternary_scales_then_extracted_from_codebook() -> None:
    cb = torch.tensor([-0.37, 0.0, 0.42])
    assert ternary_scales(cb) == pytest.approx((0.37, 0.42))
    with pytest.raises(ValueError, match="K=3"):
        ternary_scales(torch.linspace(-1, 1, 15))
    with pytest.raises(ValueError, match="degenerate"):
        ternary_scales(torch.zeros(3))


def test_when_reference_gets_bad_shapes_then_raises() -> None:
    act, lut, w, s_neg, s_pos = make_case(n=16, m=4)
    with pytest.raises(ValueError, match="1-D"):
        reference_gemv_k3(act.unsqueeze(0), lut, w, s_neg, s_pos)
    with pytest.raises(ValueError, match="shape mismatch"):
        reference_gemv_k3(act[:8], lut, w, s_neg, s_pos)

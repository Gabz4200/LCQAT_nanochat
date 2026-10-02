"""
Tests for the K-agnostic index-weight linear op (quantized runtime:
weights as codebook IDs, dequantize-on-fetch; TODO section 1). Three-way
naive == cpu == gpu parity against an independent Python loop oracle,
torch.library opcheck, torch.compile composition, error contracts.

python -m pytest tests/test_lcqat_index_linear.py -v
"""

import pytest
import torch

from nanochat.models.quant.packing import FORMAT_TRITS, pack_weight_indices
from nanochat.ops import dispatch_index_linear
from nanochat.ops.kernels.gpu_loader import vulkan_available

requires_vulkan = pytest.mark.skipif(
    not vulkan_available(), reason="Vulkan device unavailable for the Taichi backend"
)


def make_case(k: int, m: int, n: int, t: int, seed: int = 0):
    torch.manual_seed(seed)
    k_act = 15
    act_indices = torch.randint(0, k_act, (t, n), dtype=torch.uint8)
    act_lut = torch.linspace(-2.0, 2.0, k_act)
    weight_indices = torch.randint(0, k, (m, n))
    if k > 255:
        weight_indices = weight_indices.to(torch.int32)
    else:
        weight_indices = weight_indices.to(torch.uint8)
    weight_lut = torch.sort(torch.randn(k)).values
    packed, fmt = pack_weight_indices(weight_indices, k)
    case = dict(
        act_indices=act_indices,
        act_lut=act_lut,
        weight_indices=packed,
        weight_lut=weight_lut,
        n=n,
        format=fmt,
    )
    return case, weight_indices


def loop_oracle(
    act_indices, act_lut, weight_indices, weight_lut, n, format
) -> torch.Tensor:
    """Pure-Python matmul from LUT lookups (independent of the reference)."""
    del format  # raw indices; storage format is the reference's concern
    m = weight_indices.shape[0]
    t = act_indices.shape[0]
    k_w = weight_lut.numel()

    def w_at(i, j):
        idx = int(weight_indices[i, j])
        assert 0 <= idx < k_w
        return float(weight_lut[idx])

    out = torch.zeros(t, m)
    for ti in range(t):
        x = [float(act_lut[int(act_indices[ti, j])]) for j in range(n)]
        for mi in range(m):
            out[ti, mi] = sum(x[j] * w_at(mi, j) for j in range(n))
    return out


@pytest.mark.parametrize(
    ("k", "t", "m", "n"),
    [
        (3, 1, 5, 17),
        (7, 3, 8, 16),
        (15, 7, 4, 33),
        (33, 2, 6, 16),
        (255, 4, 9, 16),
        (257, 2, 5, 16),
    ],
    ids=["k3", "k7", "k15", "k33", "k255", "k257"],
)
def test_when_naive_backend_then_matches_independent_loop(
    k: int, t: int, m: int, n: int
) -> None:
    case, raw = make_case(k, m, n, t, seed=k)
    got = dispatch_index_linear(**case, backend="naive")
    expected = loop_oracle(**{**case, "weight_indices": raw})
    assert got.shape == (t, m)
    assert torch.allclose(got, expected, atol=1e-5)


def test_when_dispatch_unknown_backend_then_value_error() -> None:
    case, _ = make_case(15, 4, 16, 2)
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_index_linear(**case, backend="cuda9")


def test_when_format_mismatch_with_k_then_value_error() -> None:
    case, _ = make_case(15, 4, 16, 2)
    case["format"] = FORMAT_TRITS  # nibble-packed rows cannot be trit-decoded
    with pytest.raises(ValueError, match="width|too small|format"):
        dispatch_index_linear(**case, backend="naive")


def test_when_act_index_out_of_range_then_value_error() -> None:
    case, _ = make_case(15, 4, 16, 2)
    case["act_indices"][0, 0] = 15
    with pytest.raises(ValueError, match="out of range"):
        dispatch_index_linear(**case, backend="naive")


def test_when_weight_index_exceeds_lut_then_value_error() -> None:
    case, raw = make_case(15, 4, 16, 2)
    # Repack with an out-of-range value for the declared K.
    bad = raw.clone()
    bad[0, 0] = 15
    from nanochat.models.quant.packing import pack_nibbles

    case["weight_indices"] = pack_nibbles(bad)
    with pytest.raises(ValueError, match="out of range"):
        dispatch_index_linear(**case, backend="naive")


def test_when_act_lut_not_float32_then_value_error() -> None:
    case, _ = make_case(15, 4, 16, 2)
    case["act_lut"] = case["act_lut"].to(torch.float64)
    with pytest.raises(ValueError, match="float32"):
        dispatch_index_linear(**case, backend="naive")


@pytest.mark.parametrize("k", [3, 15, 255])
def test_when_cpu_backend_then_matches_naive(k: int) -> None:
    case, _ = make_case(k, 8, 32, 5, seed=100 + k)
    naive = dispatch_index_linear(**case, backend="naive")
    cpu = dispatch_index_linear(**case, backend="cpu")
    assert cpu.shape == naive.shape
    assert torch.isfinite(cpu).all()
    assert torch.allclose(cpu, naive, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("k", [3, 7, 15, 33, 255, 257])
def test_when_cpu_backend_all_formats_then_matches_naive(k: int) -> None:
    case, _ = make_case(k, 6, 24, 3, seed=200 + k)
    naive = dispatch_index_linear(**case, backend="naive")
    cpu = dispatch_index_linear(**case, backend="cpu")
    assert torch.allclose(cpu, naive, atol=1e-5, rtol=1e-5)


@requires_vulkan
@pytest.mark.parametrize("k", [3, 15, 255])
def test_when_gpu_backend_then_matches_naive(k: int) -> None:
    case, _ = make_case(k, 8, 32, 5, seed=300 + k)
    naive = dispatch_index_linear(**case, backend="naive")
    gpu = dispatch_index_linear(**case, backend="gpu")
    assert torch.allclose(gpu, naive, atol=1e-5, rtol=1e-5)
    cpu = dispatch_index_linear(**case, backend="cpu")
    assert torch.allclose(gpu, cpu, atol=1e-5, rtol=1e-5)


def test_when_cpu_op_then_opcheck_passes() -> None:
    from nanochat.ops.index_linear import _ensure_cpu_index_linear_op

    _ensure_cpu_index_linear_op()
    case, _ = make_case(15, 6, 16, 3, seed=11)
    args = (
        case["act_indices"].contiguous(),
        case["act_lut"].contiguous(),
        case["weight_indices"].contiguous(),
        case["weight_lut"].contiguous(),
        int(case["n"]),
        int(case["format"]),
    )
    torch.library.opcheck(torch.ops.nanochat.lcqat_index_linear, args)


def test_when_compile_with_cpu_backend_then_composes_via_fake_tensor() -> None:
    from nanochat.ops.index_linear import _ensure_cpu_index_linear_op

    _ensure_cpu_index_linear_op()
    case, _ = make_case(15, 6, 16, 3, seed=12)
    expected = dispatch_index_linear(**case, backend="cpu")

    def run():
        return torch.ops.nanochat.lcqat_index_linear(
            case["act_indices"].contiguous(),
            case["act_lut"].contiguous(),
            case["weight_indices"].contiguous(),
            case["weight_lut"].contiguous(),
            int(case["n"]),
            int(case["format"]),
        )

    compiled = torch.compile(run, fullgraph=True)
    assert torch.allclose(compiled(), expected, atol=1e-5, rtol=1e-5)


def test_when_compile_naive_backend_then_composes() -> None:
    case, _ = make_case(3, 6, 16, 3, seed=13)
    expected = dispatch_index_linear(**case, backend="naive")
    compiled = torch.compile(
        lambda: dispatch_index_linear(**case, backend="naive"), fullgraph=True
    )
    assert torch.allclose(compiled(), expected, atol=1e-6)


def test_when_backward_through_cpu_op_then_actionable_error() -> None:
    case, _ = make_case(15, 6, 16, 3, seed=14)
    act_lut = case["act_lut"].clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="inference-only"):
        out = dispatch_index_linear(**{**case, "act_lut": act_lut}, backend="cpu")
        out.sum().backward()

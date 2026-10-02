"""
Tests for the index-native quantized-KV decode attention op (LC-QAT PRD section
7.1), seam B: three-way parity naive == cpu == gpu
against an independent pure-Python oracle, torch.library opcheck, and
torch.compile composition.

python -m pytest tests/test_lcqat_quant_attn.py -v
"""

import math

import pytest
import torch

from nanochat.lcqat.kernels.gpu_loader import vulkan_available
from nanochat.lcqat.ops import dispatch_quant_attn
from nanochat.lcqat.packing import pack_nibbles

requires_vulkan = pytest.mark.skipif(
    not vulkan_available(), reason="Vulkan device unavailable for the Taichi backend"
)

B, T, H, H_KV, D, K = 2, 16, 4, 2, 16, 15

_BACKEND_CASES = [
    (1, -1, 16),
    (1, 4, 16),
    (3, -1, 16),
    (3, 4, 7),
    (2, 0, 5),
    (1, -1, 1),
]
_BACKEND_IDS = [
    "decode-full",
    "decode-win4",
    "prefill-full",
    "prefill-win4-partial",
    "win0",
    "first-token",
]


def make_case(tq: int, window_left: int, seqlen: int, seed: int = 0):
    torch.manual_seed(seed)
    k_idx = pack_nibbles(torch.randint(0, K, (B, T, H_KV, D), dtype=torch.uint8))
    v_idx = pack_nibbles(torch.randint(0, K, (B, T, H_KV, D), dtype=torch.uint8))
    k_lut = torch.sort(torch.randn(H_KV, K), dim=-1).values
    v_lut = torch.sort(torch.randn(H_KV, K), dim=-1).values
    q = torch.randn(B, tq, H, D)
    seqlens = torch.full((B,), seqlen, dtype=torch.int32)
    return q, k_idx, v_idx, k_lut, v_lut, seqlens, window_left


def dequant_loop(idx: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Independent nibble decode + LUT gather (no shared packing helpers)."""
    b, t, h, _ = idx.shape
    out = torch.zeros(b, t, h, D)
    for bi in range(b):
        for ti in range(t):
            for hi in range(h):
                for di in range(D):
                    byte = int(idx[bi, ti, hi, di // 2])
                    nib = byte & 0x0F if di % 2 == 0 else byte >> 4
                    out[bi, ti, hi, di] = lut[hi, nib]
    return out


def loop_oracle(q, k_idx, v_idx, k_lut, v_lut, seqlens, window_left) -> torch.Tensor:
    """Pure-Python decode attention: GQA, causal + sliding window, fp32."""
    k = dequant_loop(k_idx, k_lut)
    v = dequant_loop(v_idx, v_lut)
    group = H // H_KV
    out = torch.zeros(B, q.shape[1], H, D)
    for bi in range(B):
        s = int(seqlens[bi])
        for hi in range(H):
            kv = hi // group
            for qi in range(q.shape[1]):
                g = s - q.shape[1] + qi
                lo = 0 if window_left < 0 else max(0, g - window_left)
                scores = [
                    torch.dot(q[bi, qi, hi], k[bi, j, kv]).item() / math.sqrt(D)
                    for j in range(lo, g + 1)
                ]
                scores_t = torch.tensor(scores)
                probs = torch.softmax(scores_t, dim=0)
                out[bi, qi, hi] = sum(
                    probs[j - lo] * v[bi, j, kv] for j in range(lo, g + 1)
                )
    return out


@pytest.mark.parametrize(
    ("tq", "window_left", "seqlen"),
    [
        (1, -1, 16),
        (1, 4, 16),
        (3, -1, 16),
        (3, 4, 7),
        (2, 0, 5),
    ],
    ids=["decode-full", "decode-win4", "prefill-full", "prefill-win4-partial", "win0"],
)
def test_when_naive_backend_then_matches_independent_loop(
    tq: int, window_left: int, seqlen: int
) -> None:
    case = make_case(tq, window_left, seqlen, seed=tq * 10 + window_left)
    got = dispatch_quant_attn(*case, backend="naive")
    expected = loop_oracle(*case)
    assert got.shape == (B, tq, H, D)
    assert torch.allclose(got, expected, atol=1e-5)


def test_when_dispatch_unknown_backend_then_value_error() -> None:
    case = make_case(1, -1, 16)
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_quant_attn(*case, backend="cuda9")


@pytest.mark.parametrize(
    ("tq", "window_left", "seqlen"),
    _BACKEND_CASES,
    ids=_BACKEND_IDS,
)
def test_when_cpu_backend_then_matches_naive(
    tq: int, window_left: int, seqlen: int
) -> None:
    case = make_case(tq, window_left, seqlen, seed=tq * 10 + window_left)
    naive = dispatch_quant_attn(*case, backend="naive")
    cpu = dispatch_quant_attn(*case, backend="cpu")
    assert cpu.shape == naive.shape
    assert torch.isfinite(cpu).all()
    assert torch.allclose(cpu, naive, atol=1e-5, rtol=1e-5)


@requires_vulkan
@pytest.mark.parametrize(
    ("tq", "window_left", "seqlen"),
    _BACKEND_CASES,
    ids=_BACKEND_IDS,
)
def test_when_gpu_backend_then_matches_naive(
    tq: int, window_left: int, seqlen: int
) -> None:
    case = make_case(tq, window_left, seqlen, seed=tq * 10 + window_left)
    naive = dispatch_quant_attn(*case, backend="naive")
    gpu = dispatch_quant_attn(*case, backend="gpu")
    assert gpu.shape == naive.shape
    assert torch.isfinite(gpu).all()
    assert torch.allclose(gpu, naive, atol=1e-5, rtol=1e-5)
    # explicit three-way leg: cpu vs gpu directly
    cpu = dispatch_quant_attn(*case, backend="cpu")
    assert torch.allclose(gpu, cpu, atol=1e-5, rtol=1e-5)


def test_when_cpu_op_then_opcheck_passes() -> None:
    from nanochat.lcqat.ops.quant_attn import _ensure_cpu_attn_op

    _ensure_cpu_attn_op()
    case = make_case(1, 4, 16, seed=11)
    args = tuple(t.contiguous() for t in case[:6]) + (int(case[6]),)
    torch.library.opcheck(torch.ops.nanochat.lcqat_quant_attn, args)


def test_when_compile_with_cpu_backend_then_composes_via_fake_tensor() -> None:
    from nanochat.lcqat.ops.quant_attn import _ensure_cpu_attn_op

    _ensure_cpu_attn_op()
    case = make_case(1, 4, 16, seed=12)
    expected = dispatch_quant_attn(*case, backend="cpu")

    def run():
        return torch.ops.nanochat.lcqat_quant_attn(
            *[t.contiguous() for t in case[:6]], int(case[6])
        )

    compiled = torch.compile(run, fullgraph=True)
    got = compiled()
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_when_compile_naive_backend_then_composes() -> None:
    case = make_case(2, 4, 16, seed=13)
    expected = dispatch_quant_attn(*case, backend="naive")
    compiled = torch.compile(
        lambda: dispatch_quant_attn(*case, backend="naive"), fullgraph=True
    )
    assert torch.allclose(compiled(), expected, atol=1e-6)


def test_when_backward_through_cpu_op_then_actionable_error() -> None:
    case = make_case(1, -1, 16, seed=14)
    q = case[0].clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="inference-only"):
        out = dispatch_quant_attn(q, *case[1:], backend="cpu")
        out.sum().backward()


def test_when_cache_seqlen_before_tq_then_value_error() -> None:
    case = list(make_case(3, -1, 16, seed=15))
    case[5] = torch.full((B,), 2, dtype=torch.int32)  # 2 < Tq=3
    with pytest.raises(ValueError, match=r"\[Tq, T\]"):
        dispatch_quant_attn(*case, backend="naive")


def test_when_lut_k_invalid_then_value_error() -> None:
    # K=31 is out of range because the cache is nibble-packed (K <= 15), not
    # because it is odd. Even K is legal: the asymmetric split makes one-sided
    # codebooks, and therefore K=4, 8, 14, possible.
    case = list(make_case(1, -1, 16, seed=16))
    case[3] = torch.sort(torch.randn(H_KV, 31), dim=-1).values
    case[4] = torch.sort(torch.randn(H_KV, 31), dim=-1).values
    with pytest.raises(ValueError, match=r"in \[3, 15\]"):
        dispatch_quant_attn(*case, backend="naive")
    # K=2 is below the floor and still rejected.
    case[3] = torch.sort(torch.randn(H_KV, 2), dim=-1).values
    case[4] = torch.sort(torch.randn(H_KV, 2), dim=-1).values
    with pytest.raises(ValueError, match=r"in \[3, 15\]"):
        dispatch_quant_attn(*case, backend="naive")


def test_when_lut_k_even_then_accepted() -> None:
    # K=8 is a legitimate one-sided-friendly cardinality and must pass.
    case = list(make_case(1, -1, 8, seed=18))
    out = dispatch_quant_attn(*case, backend="naive")
    assert torch.isfinite(out).all()


def test_when_q_not_float32_then_value_error() -> None:
    case = list(make_case(1, -1, 16, seed=17))
    case[0] = case[0].to(torch.float64)
    with pytest.raises(ValueError, match="float32"):
        dispatch_quant_attn(*case, backend="naive")


def test_when_n_head_not_divisible_by_h_kv_then_value_error() -> None:
    case = list(make_case(1, -1, 16, seed=18))
    case[0] = torch.randn(B, 1, H + 1, D)  # 5 heads vs 2 KV heads
    with pytest.raises(ValueError, match="divisible"):
        dispatch_quant_attn(*case, backend="naive")

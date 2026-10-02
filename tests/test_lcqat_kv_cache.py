"""
Tests for the 4-bit quantized KV cache storage layer (LC-QAT PRD section 7.1),
the KV-cache storage (seam A) and byte-budget (seam D) contracts.

python -m pytest tests/test_lcqat_kv_cache.py -v
"""

import pytest
import torch

from nanochat.engine import QuantizedKVCache, kv_codebooks_from_model

K = 15
L, B, T, H, D = 2, 2, 16, 3, 8


def make_codebooks(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform per-(layer, head) codebooks so the quantization bound is exact."""
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(-1.0, 1.0, K)
    k_cbs = base + 0.05 * torch.randn(L, H, K, generator=g)
    v_cbs = base + 0.05 * torch.randn(L, H, K, generator=g)
    k_cbs, v_cbs = k_cbs.sort(dim=-1).values, v_cbs.sort(dim=-1).values
    return k_cbs.to(torch.float32), v_cbs.to(torch.float32)


def make_cache(**kwargs) -> QuantizedKVCache:
    k_cbs, v_cbs = make_codebooks()
    defaults = dict(
        batch_size=B,
        num_heads=H,
        seq_len=T,
        head_dim=D,
        num_layers=L,
        device="cpu",
        k_codebooks=k_cbs,
        v_codebooks=v_cbs,
    )
    defaults.update(kwargs)
    return QuantizedKVCache(**defaults)


def half_gap(cb: torch.Tensor) -> float:
    """Max quantization error for a sorted codebook: half the largest level gap."""
    return float((cb[..., 1:] - cb[..., :-1]).max()) / 2.0


def test_when_write_then_dequant_window_within_half_gap() -> None:
    cache = make_cache()
    torch.manual_seed(1)
    # Uniform inside the codebook span: bucketize clamps outside it, where the
    # half-gap bound does not hold.
    k = 1.8 * torch.rand(B, 4, H, D) - 0.9
    v = 1.8 * torch.rand(B, 4, H, D) - 0.9
    cache.write(0, k, v)
    k_win, v_win = cache.dequant_window(0, 0, 4)
    k_cbs, v_cbs = make_codebooks()
    assert torch.allclose(k_win, k, atol=half_gap(k_cbs) + 1e-6)
    assert torch.allclose(v_win, v, atol=half_gap(v_cbs) + 1e-6)
    assert torch.isin(k_win.unique(), k_cbs.flatten().unique()).all()


def test_when_write_after_advance_then_chunks_land_at_sequential_positions() -> None:
    cache = make_cache()
    torch.manual_seed(2)
    k1 = 1.8 * torch.rand(B, 4, H, D) - 0.9
    v1 = 1.8 * torch.rand(B, 4, H, D) - 0.9
    cache.write(0, k1, v1)
    assert cache.get_pos() == 0  # write does not advance; attention stack does
    cache.advance(4)
    k2 = 1.8 * torch.rand(B, 3, H, D) - 0.9
    v2 = 1.8 * torch.rand(B, 3, H, D) - 0.9
    cache.write(0, k2, v2)
    assert cache.get_pos() == 4
    k_cbs, v_cbs = make_codebooks()
    k0, v0 = cache.dequant_window(0, 0, 4)
    k1b, v1b = cache.dequant_window(0, 4, 7)
    assert torch.allclose(k0, k1, atol=half_gap(k_cbs) + 1e-6)
    assert torch.allclose(k1b, k2, atol=half_gap(k_cbs) + 1e-6)
    assert torch.allclose(v1b, v2, atol=half_gap(v_cbs) + 1e-6)


def test_when_write_shape_mismatch_then_value_error() -> None:
    cache = make_cache()
    with pytest.raises(ValueError, match=r"\[B, T_cur, H, D\]"):
        cache.write(0, torch.zeros(B, 2, H + 1, D), torch.zeros(B, 2, H, D))


def test_when_write_exceeds_capacity_then_value_error() -> None:
    cache = make_cache()
    x = torch.zeros(B, T + 1, H, D)
    with pytest.raises(ValueError, match="exceeds seq_len"):
        cache.write(0, x, x)


def test_when_layer_idx_out_of_range_then_value_error() -> None:
    cache = make_cache()
    x = torch.zeros(B, 1, H, D)
    with pytest.raises(ValueError, match="out of range"):
        cache.write(L, x, x)


def test_when_dequant_window_out_of_range_then_value_error() -> None:
    cache = make_cache()
    with pytest.raises(ValueError, match="outside"):
        cache.dequant_window(0, 0, T + 1)
    with pytest.raises(ValueError, match="outside"):
        cache.dequant_window(0, 5, 5)


def test_when_codebook_k_out_of_range_then_constructor_raises() -> None:
    # The bound is the nibble packing (K <= 15), not oddness. Even K is legal:
    # the asymmetric split makes one-sided codebooks, so K=4, 8, 14 all work.
    with pytest.raises(ValueError, match=r"K in \[3, 15\]"):
        make_cache(k_codebooks=torch.zeros(L, H, 31))
    with pytest.raises(ValueError, match=r"K in \[3, 15\]"):
        make_cache(k_codebooks=torch.zeros(L, H, 2))


def test_when_codebook_k_even_then_cache_is_constructed() -> None:
    # K=8 exercises the even path end to end through the nibble packer.
    even_k = torch.linspace(-1.0, 1.0, 8)
    cbs = even_k + 0.05 * torch.randn(
        L, H, 8, generator=torch.Generator().manual_seed(4)
    )
    cbs = cbs.sort(dim=-1).values.to(torch.float32)
    cache = make_cache(k_codebooks=cbs, v_codebooks=cbs)
    k = 1.8 * torch.rand(B, 4, H, D, generator=torch.Generator().manual_seed(5)) - 0.9
    cache.write(0, k, k)
    deq_k, deq_v = cache.dequant_window(0, 0, 4)
    assert deq_k.shape == (B, 4, H, D)
    assert torch.isfinite(deq_k).all()
    assert torch.isfinite(deq_v).all()


def test_when_codebook_shape_mismatch_then_constructor_raises() -> None:
    k_cbs, v_cbs = make_codebooks()
    with pytest.raises(ValueError, match=r"\[n_layers, num_heads, K\]"):
        make_cache(k_codebooks=torch.zeros(L, H + 1, K))


def test_when_prefill_then_copies_packed_rows_and_seqlens() -> None:
    src = make_cache()
    torch.manual_seed(3)
    k = 1.8 * torch.rand(B, 5, H, D) - 0.9
    v = 1.8 * torch.rand(B, 5, H, D) - 0.9
    src.write(0, k, v)
    src.advance(5)
    dst = make_cache()
    dst.prefill(src)
    assert dst.get_pos() == 5
    k_cbs, v_cbs = make_codebooks()
    k_win, v_win = dst.dequant_window(0, 0, 5)
    assert torch.allclose(k_win, k, atol=half_gap(k_cbs) + 1e-6)
    dst.reset()
    assert dst.get_pos() == 0
    dst.write(0, k[:, :1], v[:, :1])
    dst.advance(1)
    with pytest.raises(AssertionError, match="non-empty"):
        dst.prefill(src)


def test_when_exporting_from_lcqat_model_then_matches_out_quantizers(
    tiny_gpt_lcqat,
) -> None:
    k_cbs, v_cbs = kv_codebooks_from_model(tiny_gpt_lcqat)
    config = tiny_gpt_lcqat.config
    assert k_cbs.shape == (config.n_layer, config.n_kv_head, K)
    assert v_cbs.shape == (config.n_layer, config.n_kv_head, K)
    for layer in range(config.n_layer):
        block = tiny_gpt_lcqat.transformer.h[layer]
        expected_k = block.attn.c_k.out_quantizer.get_codebook()
        expected_v = block.attn.c_v.out_quantizer.get_codebook()
        assert torch.equal(k_cbs[layer, 0], expected_k)
        assert torch.equal(v_cbs[layer, 0], expected_v)
        assert torch.equal(k_cbs[layer, 1], expected_k)  # shared across heads


def test_when_computing_storage_bytes_then_matches_real_buffers() -> None:
    cache = make_cache()
    actual = cache.k_idx.numel() + cache.v_idx.numel()  # uint8 = 1 byte each
    predicted = QuantizedKVCache.storage_bytes(
        batch_size=B,
        num_heads=H,
        seq_len=T,
        head_dim=D,
        num_layers=L,
    )
    assert predicted == actual


def test_when_prd_32k_budget_then_kv_rows_match_prd_section_7_table() -> None:
    # PRD section 7 table: 32K KV cache = 0.45 GB packed 4-bit vs 1.8 GB bf16.
    # The row implies n_layer * n_kv_head * head_dim = 13,824 (24 x 9 x 64).
    dims = dict(batch_size=1, num_heads=9, seq_len=32768, head_dim=64, num_layers=24)
    packed = QuantizedKVCache.storage_bytes(**dims)
    bf16 = (
        dims["num_layers"]
        * dims["batch_size"]
        * dims["seq_len"]
        * dims["num_heads"]
        * dims["head_dim"]
        * 2
        * 2
    )
    assert 0.44e9 <= packed <= 0.46e9, packed
    assert 1.75e9 <= bf16 <= 1.85e9, bf16
    assert bf16 == 4 * packed  # 16-bit values vs 4-bit indices


def test_when_exporting_codebooks_then_lut_bytes_match_per_head_contract(
    tiny_gpt_lcqat,
) -> None:
    k_cbs, v_cbs = kv_codebooks_from_model(tiny_gpt_lcqat)
    # FP32 codebooks: 1 LUT per key head + 1 per value head (PRD 7.1).
    lut_bytes = (k_cbs.numel() + v_cbs.numel()) * 4
    config = tiny_gpt_lcqat.config
    assert lut_bytes == 2 * config.n_layer * config.n_kv_head * K * 4

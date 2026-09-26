"""
Tests for LC-QAT index bit-packing (PRD section 1 storage pillars, 5.2 inputs).

python -m pytest tests/test_lcqat_packing.py -v
"""

import pytest
import torch

from nanochat.lcqat.packing import (
    FORMAT_INT32,
    FORMAT_NIBBLES,
    FORMAT_TRITS,
    FORMAT_UINT8,
    index_format_for_k,
    pack_nibbles,
    pack_trits,
    pack_weight_indices,
    unpack_nibbles,
    unpack_trits,
    unpack_weight_indices,
)


@pytest.mark.parametrize("shape", [(4, 16), (3, 7), (1, 5), (2, 1), (6, 33)])
def test_when_roundtripping_trits_then_indices_are_recovered(shape) -> None:
    torch.manual_seed(0)
    trits = torch.randint(0, 3, shape, dtype=torch.uint8)
    packed = pack_trits(trits)
    assert packed.dtype == torch.uint8
    assert packed.shape == (shape[0], (shape[1] + 4) // 5)
    assert torch.equal(unpack_trits(packed, shape[1]), trits)


def test_when_packing_trits_then_byte_layout_is_lsb_first_base3() -> None:
    row = torch.tensor([[0, 1, 2, 1, 0]], dtype=torch.uint8)
    packed = pack_trits(row)
    # 0*1 + 1*3 + 2*9 + 1*27 + 0*81 = 48
    assert packed.item() == 48


def test_when_trits_contain_invalid_value_then_raises() -> None:
    with pytest.raises(ValueError, match=r"\{0, 1, 2\}"):
        pack_trits(torch.tensor([[0, 3]], dtype=torch.uint8))


def test_when_trits_input_is_1d_then_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        pack_trits(torch.zeros(4, dtype=torch.uint8))


def test_when_packed_buffer_too_small_then_unpack_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        unpack_trits(torch.zeros(1, 1, dtype=torch.uint8), 6)


@pytest.mark.parametrize("n", [1, 2, 3, 8, 15, 32])
def test_when_roundtripping_nibbles_then_indices_are_recovered(n: int) -> None:
    torch.manual_seed(0)
    idx = torch.randint(0, 16, (n,), dtype=torch.uint8)
    packed = pack_nibbles(idx)
    assert packed.dtype == torch.uint8
    assert packed.shape == ((n + 1) // 2,)
    assert torch.equal(unpack_nibbles(packed, n), idx)


def test_when_packing_nibbles_then_even_index_goes_to_low_nibble() -> None:
    packed = pack_nibbles(torch.tensor([1, 2, 3], dtype=torch.uint8))
    assert packed[0].item() == 0x21  # 1 low nibble, 2 high nibble
    assert packed[1].item() == 0x03  # 3 low nibble, padded 0 high nibble


def test_when_nibbles_contain_invalid_value_then_raises() -> None:
    with pytest.raises(ValueError, match=r"\{0\.\.15\}"):
        pack_nibbles(torch.tensor([16], dtype=torch.uint8))


@pytest.mark.parametrize("shape", [(2, 3, 8), (4, 5, 1), (2, 1, 7)])
def test_when_roundtripping_batched_nibbles_then_indices_are_recovered(shape) -> None:
    # Trailing-dim contract: pack/unpack apply row-wise over any leading dims
    # (the quantized KV cache packs [B, T, H, D] in one call).
    torch.manual_seed(0)
    idx = torch.randint(0, 16, shape, dtype=torch.uint8)
    packed = pack_nibbles(idx)
    assert packed.shape == (*shape[:-1], (shape[-1] + 1) // 2)
    assert torch.equal(unpack_nibbles(packed, shape[-1]), idx)


@pytest.mark.parametrize(
    ("k", "expected_format"),
    [
        (3, FORMAT_TRITS),
        (5, FORMAT_NIBBLES),
        (15, FORMAT_NIBBLES),
        (17, FORMAT_UINT8),
        (33, FORMAT_UINT8),
        (255, FORMAT_UINT8),
        (257, FORMAT_INT32),
        (65537, FORMAT_INT32),
    ],
    ids=["k3", "k5", "k15", "k17", "k33", "k255", "k257", "k65537"],
)
def test_when_k_given_then_format_matches_dtype_table(
    k: int, expected_format: int
) -> None:
    # The TODO's K -> storage table: dtype is chosen from the codebook size.
    assert index_format_for_k(k) == expected_format


@pytest.mark.parametrize("k", [3, 7, 15, 33, 255, 257])
def test_when_packing_weight_indices_then_roundtrip_and_size_match_format(
    k: int,
) -> None:
    torch.manual_seed(k)
    m, n = 11, 130
    idx = torch.randint(0, k, (m, n))
    idx32 = idx.to(torch.int32) if k > 255 else idx.to(torch.uint8)
    packed, fmt = pack_weight_indices(idx32, k)
    assert fmt == index_format_for_k(k)
    if fmt == FORMAT_TRITS:
        assert packed.dtype == torch.uint8 and packed.shape == (m, (n + 4) // 5)
    elif fmt == FORMAT_NIBBLES:
        assert packed.dtype == torch.uint8 and packed.shape == (m, (n + 1) // 2)
    elif fmt == FORMAT_UINT8:
        assert packed.dtype == torch.uint8 and packed.shape == (m, n)
    else:
        assert packed.dtype == torch.int32 and packed.shape == (m, n)
    back = unpack_weight_indices(packed, n, k)
    assert back.dtype == idx32.dtype
    assert torch.equal(back, idx32)


def test_when_weight_index_k_invalid_then_raises() -> None:
    with pytest.raises(ValueError, match="odd integer"):
        index_format_for_k(4)
    with pytest.raises(ValueError, match="odd integer"):
        index_format_for_k(1)
    idx = torch.zeros(2, 8, dtype=torch.uint8)
    idx[0, 0] = 3  # K=3 accepts only {0, 1, 2}
    with pytest.raises(ValueError, match=r"\{0, 1, 2\}"):
        pack_weight_indices(idx, k=3)

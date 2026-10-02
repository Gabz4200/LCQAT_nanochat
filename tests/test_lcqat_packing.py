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


@pytest.mark.parametrize("shape", [(8, 16), (3, 5), (1, 7), (4, 1), (2, 2)])
def test_when_roundtripping_2d_nibbles_then_the_matrix_shape_survives(
    shape,
) -> None:
    """Exactly-2-D input must come back 2-D, not flattened.

    The 3-D case above decodes with a plain trailing-dim slice, but a 2-D
    buffer `[m, ceil(n/2)]` interleaves into `[m*ceil(n/2), 2]`, so the decode
    has to restore the row count explicitly. Without that, every exported
    weight matrix unpacks to a flat vector -- silently, since a 1-D tensor of
    the right total length still multiplies correctly under a reshape.
    """
    torch.manual_seed(0)
    idx = torch.randint(0, 16, shape, dtype=torch.uint8)
    packed = pack_nibbles(idx)
    back = unpack_nibbles(packed, shape[1])
    assert back.shape == idx.shape, (back.shape, idx.shape)
    assert torch.equal(back, idx)


def test_when_2d_nibbles_unpack_then_each_row_decodes_independently() -> None:
    """Row boundaries are preserved: a row-major decode would shift every row.

    With n odd the last nibble of each row is padding, so a decode that treats
    the buffer as one flat stream lands the padding in the wrong place and
    every row after the first is shifted by half a byte.
    """
    torch.manual_seed(1)
    idx = torch.randint(0, 16, (5, 7), dtype=torch.uint8)
    back = unpack_nibbles(pack_nibbles(idx), 7)
    for row in range(idx.shape[0]):
        assert torch.equal(back[row], idx[row]), f"row {row} decoded wrong"


@pytest.mark.parametrize(
    ("k", "expected_format"),
    [
        (3, FORMAT_TRITS),
        (4, FORMAT_NIBBLES),
        (5, FORMAT_NIBBLES),
        (8, FORMAT_NIBBLES),
        (15, FORMAT_NIBBLES),
        (16, FORMAT_UINT8),
        (17, FORMAT_UINT8),
        (33, FORMAT_UINT8),
        (255, FORMAT_UINT8),
        (256, FORMAT_INT32),
        (257, FORMAT_INT32),
        (65537, FORMAT_INT32),
    ],
    ids=[
        "k3",
        "k4",
        "k5",
        "k8",
        "k15",
        "k16",
        "k17",
        "k33",
        "k255",
        "k256",
        "k257",
        "k65537",
    ],
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
    # Even K is legal: the asymmetric split is what makes a one-sided codebook
    # (and therefore K=4, K=8, K=16) possible, and these are all real bit
    # boundaries rather than odd-numbered approximations.
    assert index_format_for_k(4) == FORMAT_NIBBLES
    assert index_format_for_k(16) == FORMAT_UINT8
    with pytest.raises(ValueError, match=">= 3"):
        index_format_for_k(2)
    with pytest.raises(ValueError, match=">= 3"):
        index_format_for_k(1)
    with pytest.raises(ValueError, match=">= 3"):
        index_format_for_k(0)
    idx = torch.zeros(2, 8, dtype=torch.uint8)
    idx[0, 0] = 3  # K=3 accepts only {0, 1, 2}
    with pytest.raises(ValueError, match=r"\{0, 1, 2\}"):
        pack_weight_indices(idx, k=3)

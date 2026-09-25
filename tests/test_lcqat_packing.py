"""
Tests for LC-QAT index bit-packing (PRD section 1 storage pillars, 5.2 inputs).

python -m pytest tests/test_lcqat_packing.py -v
"""

import pytest
import torch

from nanochat.lcqat.packing import (
    pack_nibbles,
    pack_trits,
    unpack_nibbles,
    unpack_trits,
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

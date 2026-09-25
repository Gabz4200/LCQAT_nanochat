"""
Bit-packing for LC-QAT integer index tensors (LC-QAT PRD section 1 storage
pillars and 5.2 kernel inputs).

- trits: K=3 indices {0,1,2} packed 5 per byte (3^5 = 243 <= 255), row-wise.
- nibbles: K<=15 indices {0..15} packed 2 per byte (even index in low nibble,
  matching the PRD kernel's `(k & 1) ? byte >> 4 : byte & 0xF` decode).

These layouts are the inputs to the mul-less GEMV kernel and the storage
format for exported artifacts.
"""

import torch
import torch.nn.functional as F

_TRIT_POWERS = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)
_TRITS_PER_BYTE = 5


def pack_trits(trits: torch.Tensor) -> torch.Tensor:
    """Pack K=3 indices {0,1,2} from [M, N] uint8 into [M, ceil(N/5)] bytes.

    Byte b of a row encodes flat digits d_0..d_4 (LSB-first) as sum(d_t * 3^t);
    the tail is padded with digit 0.
    """
    if trits.ndim != 2:
        raise ValueError(
            f"pack_trits expects a 2-D tensor, got shape {tuple(trits.shape)}"
        )
    if trits.numel() and not torch.compiler.is_compiling() and int(trits.max()) > 2:
        raise ValueError("pack_trits only accepts indices in {0, 1, 2}")
    m, n = trits.shape
    nbytes = (n + _TRITS_PER_BYTE - 1) // _TRITS_PER_BYTE
    padded = F.pad(trits.to(torch.int32), (0, nbytes * _TRITS_PER_BYTE - n))
    chunks = padded.view(m, nbytes, _TRITS_PER_BYTE)
    powers = _TRIT_POWERS.to(trits.device)
    return (chunks * powers).sum(dim=-1).to(torch.uint8)


def unpack_trits(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack row-wise trit bytes back into an [M, n] uint8 index tensor."""
    if packed.ndim != 2:
        raise ValueError(
            f"unpack_trits expects a 2-D tensor, got shape {tuple(packed.shape)}"
        )
    m, nbytes = packed.shape
    if nbytes * _TRITS_PER_BYTE < n:
        raise ValueError(
            f"packed buffer too small: {nbytes} bytes hold {nbytes * _TRITS_PER_BYTE} trits, need {n}"
        )
    base = packed.to(torch.int32)
    digits = torch.empty(
        m, nbytes, _TRITS_PER_BYTE, dtype=torch.int32, device=packed.device
    )
    for t in range(_TRITS_PER_BYTE):
        digits[:, :, t] = (base // (3**t)) % 3
    return digits.view(m, -1)[:, :n].to(torch.uint8)


def pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """Pack K<=15 indices from [N] uint8 into [ceil(N/2)] bytes (even=low nibble)."""
    if indices.ndim != 1:
        raise ValueError(
            f"pack_nibbles expects a 1-D tensor, got shape {tuple(indices.shape)}"
        )
    if (
        indices.numel()
        and not torch.compiler.is_compiling()
        and int(indices.max()) > 15
    ):
        raise ValueError("pack_nibbles only accepts indices in {0..15}")
    n = indices.numel()
    padded = F.pad(indices.to(torch.uint8), (0, n % 2))
    low, high = padded[0::2], padded[1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack nibble bytes back into a [n] uint8 index tensor (even=low nibble)."""
    if packed.ndim != 1:
        raise ValueError(
            f"unpack_nibbles expects a 1-D tensor, got shape {tuple(packed.shape)}"
        )
    if packed.numel() * 2 < n:
        raise ValueError(
            f"packed buffer too small: {packed.numel()} bytes hold {packed.numel() * 2} nibbles, need {n}"
        )
    low = packed & 0x0F
    high = packed >> 4
    interleaved = torch.stack([low, high], dim=-1).view(-1)
    return interleaved[:n].to(torch.uint8)

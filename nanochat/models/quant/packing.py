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
#: Values packed per byte in the K=3 trit format. Public because the sparse
#: artifact's byte accounting needs it to compare layouts.
TRITS_PER_BYTE = 5

#: Largest codebook size whose indices still fit in one unsigned byte. This is
#: the single boundary behind both the packed storage format
#: (`FORMAT_UINT8`/`FORMAT_INT32`) and the in-memory index dtype
#: (`index_dtype_for_k`), so the two can never disagree.
MAX_UINT8_CODEBOOK_K = 255

# Storage-format tags for weight indices (dtype chosen from the codebook
# size K; persisted as a 0-dim int buffer next to each packed weight).
FORMAT_TRITS = 0  # K = 3: 5 trits/byte
FORMAT_NIBBLES = 1  # K <= 15: 2 values/byte
FORMAT_UINT8 = 2  # K <= MAX_UINT8_CODEBOOK_K: one byte/value
FORMAT_INT32 = 3  # K > MAX_UINT8_CODEBOOK_K: matches the in-memory index dtype


def index_dtype_for_k(k: int) -> torch.dtype:
    """Return the dtype index tensors of cardinality `k` are stored in.

    Deliberately total (no minimum-K guard): this is the dtype half of the
    `K <= MAX_UINT8_CODEBOOK_K` rule and is called from every quantizer, where
    `K` has already been validated. Raising here would turn a dtype lookup into
    a second validation point.
    """
    return torch.int32 if int(k) > MAX_UINT8_CODEBOOK_K else torch.uint8


def index_bytes_for_k(k: int) -> float:
    """Bytes per weight index at cardinality `k`, after packing.

    Derived from the `FORMAT_*` tags rather than restated, so a format change
    here cannot leave a budget table quoting stale numbers.
    """
    fmt = index_format_for_k(k)
    if fmt == FORMAT_TRITS:
        return 1.0 / TRITS_PER_BYTE
    if fmt == FORMAT_NIBBLES:
        return 1.0 / 2
    if fmt == FORMAT_UINT8:
        return 1.0
    return 4.0  # FORMAT_INT32


def index_format_for_k(k: int) -> int:
    """Select the weight-index storage format from the codebook size K.

    K may be even: the asymmetric split `K = m_neg + 1 + m_pos` is what makes a
    one-sided codebook (m_neg = 0) and therefore true 2-bit (K=4) / 4-bit (K=16)
    boundaries possible, so an odd-K requirement would defeat the point.
    """
    k = int(k)
    if k < 3:
        raise ValueError(f"codebook size K must be an integer >= 3, got {k}")
    if k == 3:
        return FORMAT_TRITS
    if k <= 15:
        return FORMAT_NIBBLES
    if k <= MAX_UINT8_CODEBOOK_K:
        return FORMAT_UINT8
    return FORMAT_INT32


def pack_weight_indices(indices: torch.Tensor, k: int) -> tuple[torch.Tensor, int]:
    """Pack `[m, n]` weight indices into their K-selected storage format.

    Returns (packed, format_tag) where the tag is one of FORMAT_*.

    A 2-D input keeps its two-dimensional shape in every format, including
    nibbles: packing `[m, n]` into a flat vector and expecting the caller to
    reshape back makes `unpack_weight_indices(packed, n, k)` return the wrong
    shape for exactly the format the exported weights use. A 1-D input stays
    1-D, which is what the sparse artifact's `nnz` values want.
    """
    fmt = index_format_for_k(k)
    if fmt == FORMAT_TRITS:
        if indices.ndim == 1:
            return pack_trits(indices.reshape(1, -1)), fmt
        return pack_trits(indices), fmt
    if fmt == FORMAT_NIBBLES:
        return pack_nibbles(indices), fmt
    if indices.numel() and int(indices.max()) >= k:
        raise ValueError(f"weight index out of range for K={k}")
    if fmt == FORMAT_UINT8:
        return indices.to(torch.uint8).contiguous(), fmt
    return indices.to(torch.int32).contiguous(), fmt


def unpack_weight_indices(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of pack_weight_indices: recover [m, n] indices for K.

    The caller knows n (in_features) and K (codebook size), which is what
    distinguishes the formats unambiguously.
    """
    fmt = index_format_for_k(k)
    if fmt == FORMAT_TRITS:
        return unpack_trits(packed, n)
    if fmt == FORMAT_NIBBLES:
        return unpack_nibbles(packed, n)
    if packed.shape[-1] != n:
        raise ValueError(
            f"unpacked width {packed.shape[-1]} != expected n={n} for K={k}"
        )
    return packed


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
    nbytes = (n + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
    padded = F.pad(trits.to(torch.int32), (0, nbytes * TRITS_PER_BYTE - n))
    chunks = padded.view(m, nbytes, TRITS_PER_BYTE)
    powers = _TRIT_POWERS.to(trits.device)
    return (chunks * powers).sum(dim=-1).to(torch.uint8)


def unpack_trits(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack row-wise trit bytes back into an [M, n] uint8 index tensor."""
    if packed.ndim != 2:
        raise ValueError(
            f"unpack_trits expects a 2-D tensor, got shape {tuple(packed.shape)}"
        )
    m, nbytes = packed.shape
    if nbytes * TRITS_PER_BYTE < n:
        raise ValueError(
            f"packed buffer too small: {nbytes} bytes hold {nbytes * TRITS_PER_BYTE} trits, need {n}"
        )
    base = packed.to(torch.int32)
    digits = torch.empty(
        m, nbytes, TRITS_PER_BYTE, dtype=torch.int32, device=packed.device
    )
    for t in range(TRITS_PER_BYTE):
        digits[:, :, t] = (base // (3**t)) % 3
    return digits.view(m, -1)[:, :n].to(torch.uint8)


def pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """Pack K<=15 indices from [..., N] uint8 into [..., ceil(N/2)] bytes.

    Applies row-wise over any leading dims (even index of each pair goes to the
    low nibble), matching the PRD kernel's decode.
    """
    if indices.ndim == 0:
        raise ValueError("pack_nibbles expects at least a 1-D tensor")
    n = indices.shape[-1]
    if (
        indices.numel()
        and not torch.compiler.is_compiling()
        and int(indices.max()) > 15
    ):
        raise ValueError("pack_nibbles only accepts indices in {0..15}")
    padded = F.pad(indices.to(torch.uint8), (0, n % 2))
    low, high = padded[..., 0::2], padded[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack nibble bytes back into [..., n] uint8 indices (even=low nibble).

    Round-trips `pack_nibbles`: a `[m, n]` input packs to `[m, ceil(n/2)]` and
    unpacks back to `[m, n]`. The interleaving is applied per leading slice, so
    each row of a matrix decodes independently -- a row-major decode of the
    whole buffer would shift every row after the first by half a byte and is the
    failure mode this reshape guards against.
    """
    if packed.ndim == 0:
        raise ValueError("unpack_nibbles expects at least a 1-D tensor")
    if packed.shape[-1] * 2 < n:
        raise ValueError(
            f"packed buffer too small: {packed.shape[-1]} bytes hold "
            f"{packed.shape[-1] * 2} nibbles, need {n}"
        )
    stacked = torch.stack([packed & 0x0F, packed >> 4], dim=-1)  # [..., B, 2]
    if packed.ndim == 2:
        # `[m, B]` packs interleave row-wise, so decode per row and drop each
        # row's pad value from the final byte. Decoding the buffer as one flat
        # stream would shift every row after the first by half a byte.
        rows, byte_count = packed.shape
        per_row = stacked.reshape(rows, byte_count * 2)[:, :n]
        return per_row.to(torch.uint8)
    return stacked.flatten(-2)[..., :n].to(torch.uint8)

"""
Pure PyTorch reference for the index-native quantized-KV decode attention
(LC-QAT PRD section 7.1).

Ground-truth oracle for parity tests: dequantizes the packed K/V window through
the per-head FP32 LUTs and runs masked SDPA-style attention. Device-agnostic,
no custom operators - the math the CPU (C++) and GPU (Taichi) kernels must
reproduce.

Contract (mirrors flash_attn_with_kvcache's position bookkeeping, with the
cache written before attention):
    q:            [B, Tq, H, D]     FP32 queries (global positions s-Tq .. s-1)
    k_idx, v_idx: [B, T, H_kv, nb]  uint8 nibble-packed cache (nb = ceil(D/2))
    k_lut, v_lut: [H_kv, K]         FP32 per-head codebooks, K odd in [3, 15]
    cache_seqlens:[B] int32         valid rows INCLUDING this step's Tq writes
    window_left:  int               left window (-1 = full context), causal right
    returns:      [B, Tq, H, D]     FP32 attention output
GQA: query head h reads KV head h // (H // H_kv).
"""

import torch

from nanochat.lcqat.packing import unpack_nibbles


def validate_quant_attn_inputs(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    v_idx: torch.Tensor,
    k_lut: torch.Tensor,
    v_lut: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int,
) -> None:
    """Fail-fast boundary validation shared by every attention backend."""
    if q.ndim != 4:
        raise ValueError(f"q must be [B, Tq, H, D], got shape {tuple(q.shape)}")
    if k_idx.ndim != 4 or v_idx.ndim != 4:
        raise ValueError(
            "k_idx/v_idx must be [B, T, H_kv, nb], got "
            f"{tuple(k_idx.shape)} / {tuple(v_idx.shape)}"
        )
    b, tq, n_head, head_dim = q.shape
    bt, t, h_kv, nb = k_idx.shape
    if bt != b or v_idx.shape != k_idx.shape:
        raise ValueError(
            f"shape mismatch: q batch {b} vs cache {k_idx.shape} / {v_idx.shape}"
        )
    if nb != (head_dim + 1) // 2:
        raise ValueError(
            f"cache last dim {nb} does not hold head_dim {head_dim} nibbles"
        )
    if k_lut.ndim != 2 or v_lut.shape != k_lut.shape:
        raise ValueError(
            f"k_lut/v_lut must be [H_kv, K] equal shapes, got "
            f"{tuple(k_lut.shape)} / {tuple(v_lut.shape)}"
        )
    if k_lut.shape[0] != h_kv:
        raise ValueError(f"k_lut heads {k_lut.shape[0]} != cache heads {h_kv}")
    if n_head % h_kv != 0:
        raise ValueError(f"n_head {n_head} not divisible by H_kv {h_kv}")
    k = k_lut.shape[1]
    if k < 3 or k > 15 or k % 2 != 1:
        raise ValueError(f"LUT K must be odd in [3, 15], got {k}")
    for label, x in (("q", q), ("k_lut", k_lut), ("v_lut", v_lut)):
        if x.dtype != torch.float32:
            raise ValueError(f"{label} must be float32, got {x.dtype}")
    if cache_seqlens.dtype != torch.int32 or cache_seqlens.shape != (b,):
        raise ValueError(
            f"cache_seqlens must be int32 [B]={b}, got "
            f"{tuple(cache_seqlens.shape)} {cache_seqlens.dtype}"
        )
    if window_left < -1:
        raise ValueError(f"window_left must be >= -1, got {window_left}")
    # Data-dependent value scans call .item(), which torch.compile cannot trace;
    # shape checks above are static and always run. Inputs entering a compiled
    # graph are guarded by the eager call that produced them.
    if torch.compiler.is_compiling():
        return
    if int(cache_seqlens.min()) < tq or int(cache_seqlens.max()) > t:
        raise ValueError(
            f"cache_seqlens must be in [Tq, T] = [{tq}, {t}], got "
            f"[{int(cache_seqlens.min())}, {int(cache_seqlens.max())}]"
        )
    for label, idx in (("k_idx", k_idx), ("v_idx", v_idx)):
        if not idx.numel():
            continue
        # Packed bytes hold two nibbles each: bound both halves (a byte's max
        # value says nothing about its individual nibbles, e.g. 0x1F).
        nibble_max = max(int((idx & 0x0F).max()), int((idx >> 4).max()))
        if nibble_max >= k:
            raise ValueError(
                f"{label} index {nibble_max} out of range for LUT of size {k}"
            )


def dequantize_kv(
    indices: torch.Tensor, lut: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """[B, T, H_kv, nb] uint8 packed + [H_kv, K] LUT -> [B, T, H_kv, D] FP32."""
    unpacked = unpack_nibbles(indices, head_dim)
    heads = torch.arange(lut.shape[0], device=lut.device).view(1, 1, -1, 1)
    return lut[heads, unpacked.long()]


def reference_quant_attn(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    v_idx: torch.Tensor,
    k_lut: torch.Tensor,
    v_lut: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int,
) -> torch.Tensor:
    """Dequantize the cache, then causal (+ sliding window) attention."""
    validate_quant_attn_inputs(
        q, k_idx, v_idx, k_lut, v_lut, cache_seqlens, window_left
    )
    b, tq, n_head, head_dim = q.shape
    h_kv = k_lut.shape[0]
    k = dequantize_kv(k_idx, k_lut, head_dim)
    v = dequantize_kv(v_idx, v_lut, head_dim)
    group = n_head // h_kv
    if group > 1:
        k = k.repeat_interleave(group, dim=2)
        v = v.repeat_interleave(group, dim=2)

    # Query i sits at global position g_i = s - Tq + i; it may attend
    # j in [g_i - window_left, g_i] (window_left = -1 -> [0, g_i]).
    seqlens = cache_seqlens.to(torch.long)
    g = (seqlens - tq).unsqueeze(1) + torch.arange(tq, device=q.device)  # [B, Tq]
    j = torch.arange(k.shape[1], device=q.device).view(1, 1, -1)  # [1,1,T]
    lo = (
        torch.zeros_like(g) if window_left < 0 else (g - window_left).clamp_min(0)
    )  # [B, Tq]
    allowed = (j >= lo.unsqueeze(-1)) & (j <= g.unsqueeze(-1))  # [B, Tq, T]

    qh = q.transpose(1, 2)  # [B, H, Tq, D]
    kh = k.transpose(1, 2)  # [B, H, T, D]
    vh = v.transpose(1, 2)
    scores = (qh @ kh.transpose(-1, -2)) * (head_dim**-0.5)  # [B, H, Tq, T]
    scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ vh).transpose(1, 2)  # [B, Tq, H, D]

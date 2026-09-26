"""Taichi GPU backend for the LC-QAT ops (replaces the slangpy loader).

Kernels are plain Taichi (Vulkan SPIR-V at runtime, no slangc/slangpy, no
CUDA). Torch tensors are staged through Taichi-owned ndarrays
(``from_numpy``/``to_numpy``): passing host tensors straight into a Vulkan
kernel silently computes garbage, so the round-trip copy is required (same
staging cost as the previous numpy path). The runtime is initialized lazily
on the first GPU dispatch - never at package import - and Taichi's own
offline kernel cache covers the persistent shader-cache requirement.
"""

# NOTE: no `from __future__ import annotations` here - Taichi kernels need
# real annotation objects at definition time, not PEP 563 strings.

from threading import Lock

import numpy as np
import taichi as ti
import torch

_lock = Lock()
_initialized = False
_init_error: BaseException | None = None


def _ensure_init() -> None:
    """Initialize the Taichi Vulkan runtime once (actionable error if absent)."""
    global _initialized, _init_error
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        if _init_error is not None:
            raise RuntimeError(
                "LC-QAT GPU backend unavailable: Taichi could not "
                "initialize a Vulkan device. Install a Vulkan driver/ICD "
                "(e.g. vulkan-intel, vulkan-radeon, nvidia drivers), or "
                "dispatch with backend='cpu' or backend='naive'."
            ) from _init_error
        try:
            ti.init(arch=ti.vulkan, log_level="error")
        except Exception as error:
            _init_error = error
            raise RuntimeError(
                "LC-QAT GPU backend unavailable: Taichi could not initialize "
                "a Vulkan device. Install a Vulkan driver/ICD (e.g. "
                "vulkan-intel, vulkan-radeon, nvidia drivers), or dispatch "
                "with backend='cpu' or backend='naive'."
            ) from error
        _initialized = True


def vulkan_available() -> bool:
    """True when the Taichi Vulkan runtime initializes (test skip marks)."""
    try:
        _ensure_init()
        return True
    except Exception:
        return False


# Kernels.
# Taichi annotations must be NdarrayType INSTANCES; binding them to names keeps
# them valid taichi annotations without call expressions inside signatures
# (pyrefly rejects calls in annotations).
_F32_ARR = ti.types.ndarray(dtype=ti.f32)
_I32_ARR = ti.types.ndarray(dtype=ti.i32)

# Taichi compiles kernels at first launch (after _ensure_init); module import
# only decorates them, which needs no runtime.


@ti.kernel
def _gemv_k3_kernel(
    act_lut: _F32_ARR,
    act: _I32_ARR,
    w: _I32_ARR,
    out: _F32_ARR,
    m: ti.i32,
    n: ti.i32,
    scale_neg: ti.f32,
    scale_pos: ti.f32,
):
    """K_W=3 mul-less GEMV: LUT-fetch activations, conditional add per trit."""
    for i in range(m):
        pos = 0.0
        neg = 0.0
        for j in range(n):
            x = act_lut[act[j]]
            t = w[i * n + j]
            if t == 2:
                pos += x
            elif t == 0:
                neg += x
        out[i] = scale_pos * pos - scale_neg * neg


@ti.kernel
def _quant_attn_kernel(
    q: _F32_ARR,
    k_idx: _I32_ARR,
    v_idx: _I32_ARR,
    k_lut: _F32_ARR,
    v_lut: _F32_ARR,
    seqlens: _I32_ARR,
    out: _F32_ARR,
    batch: ti.i32,
    tq: ti.i32,
    n_head: ti.i32,
    head_dim: ti.i32,
    seq_len: ti.i32,
    h_kv: ti.i32,
    n_bytes: ti.i32,
    k_size: ti.i32,
    window_left: ti.i32,
):
    """Index-native decode attention: one thread per output row, two passes
    over the visible window (max, then exp-weighted value sum)."""
    for linear in range(batch * tq * n_head):
        h = linear % n_head
        i = (linear // n_head) % tq
        bi = linear // (n_head * tq)
        group = n_head // h_kv
        kv = h // group
        s = seqlens[bi]
        g = s - tq + i
        lo = 0 if window_left < 0 else ti.max(0, g - window_left)
        scale = ti.rsqrt(ti.cast(head_dim, ti.f32))
        q_base = ((bi * tq + i) * n_head + h) * head_dim
        lut_base = kv * k_size

        mx = -1.0e30
        for j in range(lo, g + 1):
            k_base = ((bi * seq_len + j) * h_kv + kv) * n_bytes
            dot = 0.0
            for dd in range(head_dim):
                byte = k_idx[k_base + (dd >> 1)]
                idx = ((byte >> 4) & 15) if ((dd & 1) != 0) else (byte & 15)
                dot += q[q_base + dd] * k_lut[lut_base + idx]
            mx = ti.max(mx, dot * scale)

        acc = ti.Vector([0.0] * 256)
        total = 0.0
        for j in range(lo, g + 1):
            k_base = ((bi * seq_len + j) * h_kv + kv) * n_bytes
            dot = 0.0
            for dd in range(head_dim):
                byte = k_idx[k_base + (dd >> 1)]
                idx = ((byte >> 4) & 15) if ((dd & 1) != 0) else (byte & 15)
                dot += q[q_base + dd] * k_lut[lut_base + idx]
            e = ti.exp(dot * scale - mx)
            total += e
            v_base = ((bi * seq_len + j) * h_kv + kv) * n_bytes
            for dd in range(head_dim):
                byte = v_idx[v_base + (dd >> 1)]
                idx = ((byte >> 4) & 15) if ((dd & 1) != 0) else (byte & 15)
                acc[dd] += e * v_lut[lut_base + idx]
        for dd in range(head_dim):
            out[q_base + dd] = acc[dd] / total


@ti.kernel
def _index_linear_kernel(
    act_lut: _F32_ARR,
    act: _I32_ARR,
    w: _I32_ARR,
    w_lut: _F32_ARR,
    out: _F32_ARR,
    t: ti.i32,
    n: ti.i32,
    m: ti.i32,
    width: ti.i32,
    format: ti.i32,
):
    """K-agnostic index-weight linear: one thread per output element,
    weight fetched through its K-selected storage format + LUT."""
    for linear in range(t * m):
        mi = linear % m
        ti_row = linear // m
        acc = 0.0
        act_base = ti_row * n
        w_base = mi * width
        for j in range(n):
            x = act_lut[act[act_base + j]]
            idx = 0
            if format == 0:
                byte = w[w_base + j // 5]
                rem = j % 5
                pow3 = 1
                if rem == 1:
                    pow3 = 3
                elif rem == 2:
                    pow3 = 9
                elif rem == 3:
                    pow3 = 27
                elif rem == 4:
                    pow3 = 81
                idx = (byte // pow3) % 3
            elif format == 1:
                byte = w[w_base + j // 2]
                idx = ((byte >> 4) & 15) if ((j & 1) != 0) else (byte & 15)
            else:
                idx = w[w_base + j]
            acc += x * w_lut[idx]
        out[linear] = acc


# Staging wrappers: torch -> numpy -> Taichi ndarray -> kernel -> torch.


def _to_ti(t: torch.Tensor, dtype) -> "ti.ndarray":
    arr = ti.ndarray(dtype, shape=(t.numel(),))
    arr.from_numpy(
        t.reshape(-1)
        .contiguous()
        .numpy()
        .astype(np.int32 if dtype == ti.i32 else np.float32, copy=False)
    )
    return arr


def run_gemv(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    scale_neg: float,
    scale_pos: float,
) -> torch.Tensor:
    """GEMV: returns [m] FP32 on the host."""
    _ensure_init()
    m, n = int(weight_indices.shape[0]), int(weight_indices.shape[1])
    out = ti.ndarray(ti.f32, shape=(m,))
    _gemv_k3_kernel(
        _to_ti(act_lut, ti.f32),
        _to_ti(act_indices, ti.i32),
        _to_ti(weight_indices, ti.i32),
        out,
        m,
        n,
        float(scale_neg),
        float(scale_pos),
    )
    ti.sync()
    return torch.from_numpy(out.to_numpy().copy())


def run_quant_attn(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    v_idx: torch.Tensor,
    k_lut: torch.Tensor,
    v_lut: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int,
) -> torch.Tensor:
    """Quantized-KV attention; returns [B, Tq, H, D]."""
    _ensure_init()
    b, tq, n_head, head_dim = (int(x) for x in q.shape)
    seq_len, h_kv, n_bytes = (int(x) for x in k_idx.shape[1:])
    k_size = int(k_lut.shape[1])
    out = ti.ndarray(ti.f32, shape=(b * tq * n_head * head_dim,))
    _quant_attn_kernel(
        _to_ti(q, ti.f32),
        _to_ti(k_idx, ti.i32),
        _to_ti(v_idx, ti.i32),
        _to_ti(k_lut, ti.f32),
        _to_ti(v_lut, ti.f32),
        _to_ti(cache_seqlens, ti.i32),
        out,
        b,
        tq,
        n_head,
        head_dim,
        seq_len,
        h_kv,
        n_bytes,
        k_size,
        int(window_left),
    )
    ti.sync()
    return torch.from_numpy(out.to_numpy().copy()).reshape(b, tq, n_head, head_dim)


def run_index_linear(
    act_indices: torch.Tensor,
    act_lut: torch.Tensor,
    weight_indices: torch.Tensor,
    weight_lut: torch.Tensor,
    n: int,
    format: int,
) -> torch.Tensor:
    """Index-weight linear; returns [T, m] FP32."""
    _ensure_init()
    t = int(act_indices.shape[0])
    m, width = int(weight_indices.shape[0]), int(weight_indices.shape[1])
    out = ti.ndarray(ti.f32, shape=(t * m,))
    _index_linear_kernel(
        _to_ti(act_lut, ti.f32),
        _to_ti(act_indices, ti.i32),
        _to_ti(weight_indices, ti.i32),
        _to_ti(weight_lut, ti.f32),
        out,
        t,
        int(n),
        m,
        width,
        int(format),
    )
    ti.sync()
    return torch.from_numpy(out.to_numpy().copy()).reshape(t, m)

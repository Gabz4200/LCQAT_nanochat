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

from nanochat.models.quant.packing import (
    FORMAT_NIBBLES,
    FORMAT_TRITS,
    TRITS_PER_BYTE,
)

#: Largest head dim the attention kernel's per-row accumulator is sized for.
#: Taichi resolves a module-level Python int as a compile-time constant inside a
#: kernel body, so the Python-side guard in `quant_attn_gpu` and the `ti.Vector`
#: width below are the same number by construction rather than by convention --
#: raising the guard alone would otherwise turn a clear ValueError into an
#: out-of-range write.
GPU_MAX_HEAD_DIM = 256

_lock = Lock()
_initialized = False
_init_error: BaseException | None = None

_VULKAN_ERROR = (
    "LC-QAT GPU backend unavailable: Taichi could not initialize a Vulkan "
    "device. Install a Vulkan driver/ICD (e.g. vulkan-intel, vulkan-radeon, "
    "nvidia drivers), or dispatch with backend='cpu' or backend='naive'."
)


def _ensure_init() -> None:
    """Initialize the Taichi Vulkan runtime once (actionable error if absent)."""
    global _initialized, _init_error
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        if _init_error is not None:
            raise RuntimeError(_VULKAN_ERROR) from _init_error
        try:
            ti.init(arch=ti.vulkan, log_level="error")
        except Exception as error:
            _init_error = error
            raise RuntimeError(_VULKAN_ERROR) from error
        _initialized = True


def vulkan_available() -> bool:
    """True when the Taichi Vulkan runtime initializes (test skip marks)."""
    try:
        _ensure_init()
        return True
    except Exception:
        return False


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

        acc = ti.Vector([0.0] * GPU_MAX_HEAD_DIM)
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
            if format == FORMAT_TRITS:
                byte = w[w_base + j // TRITS_PER_BYTE]
                rem = j % TRITS_PER_BYTE
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
            elif format == FORMAT_NIBBLES:
                byte = w[w_base + j // 2]
                idx = ((byte >> 4) & 15) if ((j & 1) != 0) else (byte & 15)
            else:
                idx = w[w_base + j]
            acc += x * w_lut[idx]
        out[linear] = acc


# SparseProp kernels (arXiv 2302.04852, Algorithms 1-2). Each mirrors
# the CPU kernel's decomposition so the backends agree by construction:
# forward and dX are element-parallel over (row, batch) and
# (column, batch) respectively -- every output element is written by
# exactly one thread, so no atomics or barriers are needed -- and dW is
# row-parallel over the CSR listing, where rows own disjoint nnz ranges.
# All loop bounds are kernel arguments: Taichi resolves Python-level
# globals at first compile, so a shape change must not require editing
# the kernel.


@ti.kernel
def _sparseprop_forward_kernel(
    w_val: _F32_ARR,
    w_col: _I32_ARR,
    w_ptr: _I32_ARR,
    x: _F32_ARR,
    bias: _F32_ARR,
    out: _F32_ARR,
    M: ti.i32,
    B: ti.i32,
    has_bias: ti.i32,
):
    """SparseProp SpMM: y[m, b] = sum over row m's nnz of w * x.

    One thread per (output row, batch element); each walks its row's
    CSR nnz list and accumulates into a scalar. `bias` may be a dummy
    one-element buffer when `has_bias` is 0 -- the guard keeps it
    unindexed.
    """
    for linear in range(M * B):
        m = linear // B
        b = linear % B
        acc = 0.0
        p_start = w_ptr[m]
        p_end = w_ptr[m + 1]
        for p in range(p_start, p_end):
            acc += w_val[p] * x[w_col[p] * B + b]
        if has_bias != 0:
            acc += bias[m]
        out[linear] = acc


@ti.kernel
def _sparseprop_backward_dw_kernel(
    gY: _F32_ARR,
    x: _F32_ARR,
    w_col: _I32_ARR,
    w_ptr: _I32_ARR,
    gW_val: _F32_ARR,
    M: ti.i32,
    B: ti.i32,
):
    """SparseProp SDDMM (dW): gW_val[p] = dot(gY[m], x[col[p]]).

    One thread per output row; the row computes a dot product over B
    for each of its nnz. The SDDMM reads the CSR structure but not
    the weight values (the gradient of an entry is the outer product
    of the two activations, independent of the entry itself), which
    is why this kernel takes no value buffer.
    """
    for m in range(M):
        p_start = w_ptr[m]
        p_end = w_ptr[m + 1]
        for p in range(p_start, p_end):
            c = w_col[p]
            acc = 0.0
            for b in range(B):
                acc += gY[m * B + b] * x[c * B + b]
            gW_val[p] = acc


@ti.kernel
def _sparseprop_backward_dx_kernel(
    gY: _F32_ARR,
    w_val_csc: _F32_ARR,
    w_row: _I32_ARR,
    w_cptr: _I32_ARR,
    gX: _F32_ARR,
    K: ti.i32,
    B: ti.i32,
):
    """SparseProp SpGEMM (dX): gX[k, b] = sum over column k's nnz.

    One thread per (input column, batch element); each walks the
    column's CSC nnz list. The transpose of a CSR matrix is the same
    matrix in CSC format, so the dX pass consumes the CSC listing the
    structure builders emit -- no sparse transpose at runtime.
    """
    for linear in range(K * B):
        k = linear // B
        b = linear % B
        acc = 0.0
        p_start = w_cptr[k]
        p_end = w_cptr[k + 1]
        for p in range(p_start, p_end):
            acc += w_val_csc[p] * gY[w_row[p] * B + b]
        gX[linear] = acc


# Staging wrappers: torch -> numpy -> Taichi ndarray -> kernel -> torch.


def _to_ti(t: torch.Tensor, dtype) -> "ti.ndarray":
    # GPU runners return host CPU tensors: detach, move to CPU, then stage.
    # Direct `.numpy()` on CUDA/requires-grad tensors raises a confusing
    # TypeError from inside staging; fail with a clear contract instead.
    if t.requires_grad:
        t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    if t.device.type != "cpu":
        raise ValueError(f"_to_ti stages CPU host tensors, got {t.device}")
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
    if head_dim > GPU_MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim {head_dim} exceeds GPU accumulator {GPU_MAX_HEAD_DIM}"
        )
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


def run_sparseprop_forward(
    w_val: torch.Tensor,
    w_col: torch.Tensor,
    w_ptr: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    M: int,
) -> torch.Tensor:
    """SparseProp SpMM; returns [M, B] FP32 on the host."""
    _ensure_init()
    B = int(x.shape[1])
    out = ti.ndarray(ti.f32, shape=(M * B,))
    has_bias = 1 if bias is not None and bias.numel() > 0 else 0
    # A one-element dummy keeps the kernel's bias argument a valid
    # ndarray when the layer has no bias; the has_bias guard means it
    # is never indexed.
    bias_arr = _to_ti(bias if has_bias else torch.zeros(1, dtype=torch.float32), ti.f32)
    _sparseprop_forward_kernel(
        _to_ti(w_val, ti.f32),
        _to_ti(w_col, ti.i32),
        _to_ti(w_ptr, ti.i32),
        _to_ti(x, ti.f32),
        bias_arr,
        out,
        int(M),
        B,
        has_bias,
    )
    ti.sync()
    return torch.from_numpy(out.to_numpy().copy()).reshape(M, B)


def run_sparseprop_backward(
    gY: torch.Tensor,
    x: torch.Tensor,
    w_val: torch.Tensor,
    w_col: torch.Tensor,
    w_ptr: torch.Tensor,
    w_val_csc: torch.Tensor,
    w_row: torch.Tensor,
    w_cptr: torch.Tensor,
    M: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SparseProp backward; returns (gX [K, B], gW_val [nnz]).

    `w_val` is not read by either kernel -- dW is the outer product
    of the activations and dX walks the CSC values -- but it sizes the
    gW_val output, and taking it keeps the signature identical to the
    CPU op so a backend swap changes one word.
    """
    _ensure_init()
    B = int(gY.shape[1])
    nnz = int(w_val.shape[0])
    # Stage the operands both kernels share once; a second _to_ti of
    # the same tensor would copy it twice.
    gY_arr = _to_ti(gY, ti.f32)
    x_arr = _to_ti(x, ti.f32)
    gX = ti.ndarray(ti.f32, shape=(K * B,))
    gW_val = ti.ndarray(ti.f32, shape=(nnz,))
    _sparseprop_backward_dw_kernel(
        gY_arr,
        x_arr,
        _to_ti(w_col, ti.i32),
        _to_ti(w_ptr, ti.i32),
        gW_val,
        int(M),
        B,
    )
    _sparseprop_backward_dx_kernel(
        gY_arr,
        _to_ti(w_val_csc, ti.f32),
        _to_ti(w_row, ti.i32),
        _to_ti(w_cptr, ti.i32),
        gX,
        int(K),
        B,
    )
    ti.sync()
    gx = torch.from_numpy(gX.to_numpy().copy()).reshape(K, B)
    gw = torch.from_numpy(gW_val.to_numpy().copy())
    return gx, gw


# DiffusionBlocks EDM denoising loss (arXiv 2506.14202, training objective).
# loss = w * mean((pred - clean)^2) over all N elements. One thread walks the
# flattened buffers and reduces into a scalar -- the reduction is serial by
# construction (parallel writes to one accumulator would race), which is the
# same shape as the GEMV kernel above: correctness first, in one fused pass.


@ti.kernel
def _db_denoise_kernel(
    pred: _F32_ARR,
    clean: _F32_ARR,
    out: _F32_ARR,
    n: ti.i32,
    w: ti.f32,
):
    """EDM loss: frontend stages flattened [N] buffers; returns scalar out[0]."""
    acc = 0.0
    for i in range(n):
        d = pred[i] - clean[i]
        acc += d * d
    out[0] = w * acc / ti.cast(n, ti.f32)


def run_db_denoise(
    pred: torch.Tensor,
    clean: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """EDM denoising loss; returns a 0-d FP32 tensor on the host."""
    _ensure_init()
    n = int(pred.numel())
    if n == 0:
        raise ValueError("run_db_denoise requires non-empty tensors")
    if tuple(pred.shape) != tuple(clean.shape):
        raise ValueError(
            f"pred shape {tuple(pred.shape)} != clean shape {tuple(clean.shape)}"
        )
    out = ti.ndarray(ti.f32, shape=(1,))
    _db_denoise_kernel(
        _to_ti(pred, ti.f32),
        _to_ti(clean, ti.f32),
        out,
        n,
        float(weight),
    )
    ti.sync()
    return torch.from_numpy(out.to_numpy().copy()).reshape(())

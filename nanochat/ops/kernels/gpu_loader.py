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

import typing
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
# them valid taichi annotations without call expressions inside signatures.
# The checker half declares them opaque: at type-check time their values are
# template instances rather than type forms, which pyrefly rejects as
# annotations. Runtime (the else half) binds the real templates, which is
# what `@ti.kernel` needs -- annotations here must stay real objects
# (no `from __future__ import annotations`).
if typing.TYPE_CHECKING:
    _F32_ARR: typing.Any
    _I32_ARR: typing.Any
else:
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


def _to_ti(t: torch.Tensor, dtype) -> typing.Any:
    # Returns a Taichi-owned ndarray. `ti.ndarray` is a factory function,
    # not a type, so the annotation is opaque; callers only pass it into
    # kernels and read it back with `to_numpy`.
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
    bias_arr = _to_ti(
        bias if bias is not None else torch.zeros(1, dtype=torch.float32), ti.f32
    )
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


# SGD-MinExpLSTM fused optimizer kernel (PRD v3.2, section 6): three Taichi launches.
# Pass 1 computes the layer-wise gradient RMS; pass 2 blends the EMAs; pass 3 runs the cell and updates.


@ti.kernel
def _sgd_minexplstm_rms_kernel(
    grad: _F32_ARR,
    rms_g_sq: _F32_ARR,
    n: ti.i32,
):
    """Pass 1: accumulate the squared gradient magnitudes.

    One thread per element; each contributes its squared
    gradient to a scalar accumulator via atomic add (the
    reduction is serial by construction, matching the GEMV
    kernel above).
    """
    for i in range(n):
        g = grad[i]
        ti.atomic_add(rms_g_sq[0], g * g)


def run_sgd_minexplstm_rms(
    grad: torch.Tensor,
    n: int,
) -> float:
    """Run pass 1: layer-wise RMS of the gradient (PRD 3.1)."""
    _ensure_init()
    # Taichi-owned accumulator: _to_ti copies data in but never
    # back, so a torch tensor passed through _to_ti would read
    # stale zeros on the host after the kernel writes into the
    # Taichi ndarray. The GEMV kernel above uses this same
    # pattern: allocate the output with ti.ndarray, launch, sync,
    # then read back with to_numpy().copy().
    rms_g_sq = ti.ndarray(ti.f32, shape=(1,))
    _sgd_minexplstm_rms_kernel(
        _to_ti(grad.contiguous(), ti.f32),
        rms_g_sq,
        n,
    )
    ti.sync()
    # Compute RMS from the sum of squares (with the PRD eps).
    return (float(rms_g_sq.to_numpy().copy()[0]) / n + 1e-8) ** 0.5


@ti.kernel
def _sgd_minexplstm_blend_kernel(
    grad: _F32_ARR,
    ema_fast_int8: _I32_ARR,
    ema_slow_int8: _I32_ARR,
    scale_fast: ti.f32,
    scale_slow: ti.f32,
    rms_g: ti.f32,
    lam: ti.f32,
    w_beta: ti.f32,
    b_beta: ti.f32,
    rms_fast_new_sq: _F32_ARR,
    rms_slow_new_sq: _F32_ARR,
    ema_fast_new: _F32_ARR,
    ema_slow_new: _F32_ARR,
    n: ti.i32,
):
    """Pass 2: blend EMAs and accumulate the new-EMA squares.

    Dequantize the INT8 states, predict the lambda-coupled
    betas from the normalized gradient (PRD 3.2), run the IIR
    EMA blend (PRD 3.3), and stash the new EMAs for pass 3.
    The squared magnitudes of the new EMAs are folded into
    scalar reduction accumulators via atomic add (serial by
    construction).
    """
    eps = 1e-8
    for i in range(n):
        g = grad[i]
        # Layer-wise RMS-normalized gradient (PRD 3.1).
        g_norm = g / (rms_g + eps)
        # Lambda-coupled beta prediction (PRD 3.2): a single
        # shared scalar affine map through a hard sigmoid.
        u = w_beta * g_norm + b_beta
        sig = 0.5 + 0.125 * u
        if sig < 0.0:
            sig = 0.0
        elif sig > 1.0:
            sig = 1.0
        min_beta_slow = 2.0 * lam
        beta_slow = min_beta_slow + (0.999 - min_beta_slow) * sig
        beta_fast = beta_slow - lam
        # Dequantize the INT8 states with their per-tensor scales.
        ef = ti.cast(ema_fast_int8[i], ti.f32) * scale_fast
        es = ti.cast(ema_slow_int8[i], ti.f32) * scale_slow
        # Dynamic EMA IIR blending (PRD 3.3).
        ef_new = beta_fast * ef + (1.0 - beta_fast) * g
        es_new = beta_slow * es + (1.0 - beta_slow) * g
        # Stash the new EMAs for pass 3.
        ema_fast_new[i] = ef_new
        ema_slow_new[i] = es_new
        # Accumulate the squared magnitudes of the new EMAs.
        ti.atomic_add(rms_fast_new_sq[0], ef_new * ef_new)
        ti.atomic_add(rms_slow_new_sq[0], es_new * es_new)


def run_sgd_minexplstm_blend(
    grad: torch.Tensor,
    ema_fast_int8: torch.Tensor,
    ema_slow_int8: torch.Tensor,
    scale_fast: float,
    scale_slow: float,
    rms_g: float,
    lam: float,
    w_beta_val: float,
    b_beta_val: float,
    n: int,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Run pass 2: blend EMAs and return the new-EMA RMS sums."""
    _ensure_init()
    # Taichi-owned accumulators and stashes (see run_sgd_minexplstm_rms:
    # _to_ti is one-way, so reduction outputs and stashes must be
    # ti.ndarray allocations read back with to_numpy().copy()).
    rms_fast_new_sq = ti.ndarray(ti.f32, shape=(1,))
    rms_slow_new_sq = ti.ndarray(ti.f32, shape=(1,))
    ema_fast_new = ti.ndarray(ti.f32, shape=(n,))
    ema_slow_new = ti.ndarray(ti.f32, shape=(n,))
    _sgd_minexplstm_blend_kernel(
        _to_ti(grad.contiguous(), ti.f32),
        _to_ti(ema_fast_int8.contiguous(), ti.i32),
        _to_ti(ema_slow_int8.contiguous(), ti.i32),
        float(scale_fast),
        float(scale_slow),
        rms_g,
        lam,
        w_beta_val,
        b_beta_val,
        rms_fast_new_sq,
        rms_slow_new_sq,
        ema_fast_new,
        ema_slow_new,
        n,
    )
    ti.sync()
    # RMS of the new EMAs (PRD 3.3 step 5), with eps.
    rms_fast_new = (float(rms_fast_new_sq.to_numpy().copy()[0]) / n + 1e-8) ** 0.5
    rms_slow_new = (float(rms_slow_new_sq.to_numpy().copy()[0]) / n + 1e-8) ** 0.5
    return (
        rms_fast_new,
        rms_slow_new,
        torch.from_numpy(ema_fast_new.to_numpy().copy()),
        torch.from_numpy(ema_slow_new.to_numpy().copy()),
    )


def run_sgd_minexplstm_step(
    grad: torch.Tensor,
    weight: torch.Tensor,
    ema_fast_int8: torch.Tensor,
    ema_slow_int8: torch.Tensor,
    h_int8: torch.Tensor,
    scale_fast: float,
    scale_slow: float,
    scale_h: float,
    # Cell parameters
    phi_w: torch.Tensor,
    phi_b: torch.Tensor,
    iota_w: torch.Tensor,
    iota_b: torch.Tensor,
    h_w: torch.Tensor,
    h_b: torch.Tensor,
    out_w: torch.Tensor,
    out_b: torch.Tensor,
    film_w: torch.Tensor,
    film_b: torch.Tensor,
    w_beta: torch.Tensor,
    b_beta: torch.Tensor,
    # Hyperparameters
    lr: float,
    wd: float,
    lam: float,
    clamp_val: float,
) -> tuple[float, float, float]:
    """Run the fused SGD-MinExpLSTM step (three passes).

    Returns the new per-tensor max-abs scales for the fast
    EMA, slow EMA, and hidden state, so the dispatch layer
    can write them back into the caller's scale tensors (the
    naive and CPU paths mutate those tensors in place; the
    GPU path receives the scales as floats and returns the
    updated values).
    """
    _ensure_init()

    grad = grad.contiguous()
    weight = weight.contiguous()
    ema_fast_int8 = ema_fast_int8.contiguous()
    ema_slow_int8 = ema_slow_int8.contiguous()
    h_int8 = h_int8.contiguous()

    # Extract scalar cell parameters. The cell's Linear layers
    # store weights as [out, in] (e.g. phi_proj.weight is [1, 3])
    # and biases as [out] (e.g. phi_proj.bias is [1]); flatten both
    # so indexing is shape-agnostic regardless of the Linear geometry.
    phi_w = phi_w.contiguous().flatten()
    phi_b = phi_b.contiguous().flatten()
    iota_w = iota_w.contiguous().flatten()
    iota_b = iota_b.contiguous().flatten()
    h_w = h_w.contiguous().flatten()
    h_b = h_b.contiguous().flatten()
    out_w = out_w.contiguous().flatten()
    out_b = out_b.contiguous().flatten()
    film_w = film_w.contiguous().flatten()
    film_b = film_b.contiguous().flatten()
    w_beta = w_beta.contiguous().flatten()
    b_beta = b_beta.contiguous().flatten()

    phi_b_val = phi_b[0].item()
    iota_b_val = iota_b[0].item()
    h_b_val = h_b[0].item()
    w_beta_val = w_beta[0].item()
    b_beta_val = b_beta[0].item()

    n = grad.numel()

    # Pass 1: layer-wise RMS of the gradient (PRD 3.1).
    rms_g = run_sgd_minexplstm_rms(grad, n)

    # Pass 2: blend the EMAs and accumulate the new-EMA squares
    # (PRD 3.2/3.3). The new EMAs are stashed in scratch buffers.
    rms_fast_new, rms_slow_new, ema_fast_new, ema_slow_new = run_sgd_minexplstm_blend(
        grad,
        ema_fast_int8,
        ema_slow_int8,
        scale_fast,
        scale_slow,
        rms_g,
        lam,
        w_beta_val,
        b_beta_val,
        n,
    )

    # Pass 3 outputs: the updated weight, the FP32 stashes
    # for the new states (h_next and the already-blended
    # EMAs), and the per-tensor max-abs scales for the new
    # states. The host re-quantizes the stashes to INT8
    # against the final scales once the kernel returns (the
    # kernel folds the max-abs reduction into the scale
    # arrays, which is serial by construction). All outputs
    # are Taichi-owned ndarrays: _to_ti is one-way, so the
    # GEMV pattern (allocate with ti.ndarray, read back with
    # to_numpy().copy()) applies to every output here.
    weight_out = ti.ndarray(ti.f32, shape=(n,))
    h_next_out = ti.ndarray(ti.f32, shape=(n,))
    scale_fast_out = ti.ndarray(ti.f32, shape=(1,))
    scale_slow_out = ti.ndarray(ti.f32, shape=(1,))
    scale_h_out = ti.ndarray(ti.f32, shape=(1,))

    # Pack the 34 cell scalars into a single constant array
    # (PRD section 6.3: load cell weights into constant memory).
    # Layout (32 floats):
    #   [0:3]   phi_w   [3]   phi_b
    #   [4:7]   iota_w  [7]   iota_b
    #   [8:11]  h_w     [11]  h_b
    #   [12:16] out_w   [16:20] out_b
    #   [20:28] film_w  [28:30] film_b
    #   [30]    w_beta  [31]  b_beta
    cell_params_host = torch.zeros(32, dtype=torch.float32, device=grad.device)
    cell_params_host[0:3] = phi_w
    cell_params_host[3] = phi_b_val
    cell_params_host[4:7] = iota_w
    cell_params_host[7] = iota_b_val
    cell_params_host[8:11] = h_w
    cell_params_host[11] = h_b_val
    cell_params_host[12:16] = out_w
    cell_params_host[16:20] = out_b
    cell_params_host[20:28] = film_w
    cell_params_host[28:30] = film_b
    cell_params_host[30] = w_beta_val
    cell_params_host[31] = b_beta_val

    _sgd_minexplstm_step_kernel(
        _to_ti(grad, ti.f32),
        _to_ti(weight, ti.f32),
        _to_ti(h_int8, ti.i32),
        float(scale_h),
        rms_g,
        rms_fast_new,
        rms_slow_new,
        _to_ti(ema_fast_new, ti.f32),
        _to_ti(ema_slow_new, ti.f32),
        _to_ti(cell_params_host, ti.f32),
        lr,
        wd,
        clamp_val,
        n,
        weight_out,
        h_next_out,
        scale_fast_out,
        scale_slow_out,
        scale_h_out,
    )
    ti.sync()

    # The kernel folded the per-tensor max-abs reduction
    # into scale_*_out[0] (serial by construction, like the
    # RMS pass). Read the final max-abs values back, convert
    # to the symmetric INT8 scale (max-abs / 127, matching
    # the naive reference's _quantize), then re-quantize the
    # stashed FP32 states to INT8 against the true final
    # scales (the kernel could not do this in a single
    # launch: the scale is unknown until every thread has
    # contributed its magnitude).
    eps = 1e-8
    scale_fast_new = float(scale_fast_out.to_numpy().copy()[0]) / 127.0 + eps
    scale_slow_new = float(scale_slow_out.to_numpy().copy()[0]) / 127.0 + eps
    scale_h_new = float(scale_h_out.to_numpy().copy()[0]) / 127.0 + eps
    # Guard against a degenerate all-zero state (the
    # scale would be 0 and the INT8 values would be 0/0).
    scale_fast_new = max(scale_fast_new, 1e-12)
    scale_slow_new = max(scale_slow_new, 1e-12)
    scale_h_new = max(scale_h_new, 1e-12)

    # Copy the updated weight back to the caller's tensor.
    weight.copy_(torch.from_numpy(weight_out.to_numpy().copy()).reshape(weight.shape))
    # Re-quantize the new states to INT8 against the final
    # scales (PRD 4.1: symmetric max-abs, int8 range).
    ema_fast_int8.copy_(
        torch.clamp(
            torch.round(ema_fast_new / scale_fast_new),
            -128,
            127,
        ).to(torch.int8)
    )
    ema_slow_int8.copy_(
        torch.clamp(
            torch.round(ema_slow_new / scale_slow_new),
            -128,
            127,
        ).to(torch.int8)
    )
    h_int8.copy_(
        torch.clamp(
            torch.round(torch.from_numpy(h_next_out.to_numpy().copy()) / scale_h_new),
            -128,
            127,
        ).to(torch.int8)
    )
    return scale_fast_new, scale_slow_new, scale_h_new


# DiffusionBlocks EDM denoising loss (arXiv 2506.14202, training objective).


@ti.kernel
def _sgd_minexplstm_step_kernel(
    grad: _F32_ARR,
    weight: _F32_ARR,
    h_int8: _I32_ARR,
    scale_h: ti.f32,
    # Layer-wise RMS values: rms_g from pass 1, the
    # new-EMA RMS sums from pass 2 (PRD 3.1/3.3).
    rms_g: ti.f32,
    rms_fast_new: ti.f32,
    rms_slow_new: ti.f32,
    # The stashed new EMAs (blended in pass 2).
    ema_fast_new: _F32_ARR,
    ema_slow_new: _F32_ARR,
    # Cell parameters packed into a single constant array.
    # Layout (32 floats, PRD section 5.1):
    #   [0:3]   phi_w   [3] phi_b
    #   [4:7]   iota_w  [7] iota_b
    #   [8:11]  h_w     [11] h_b
    #   [12:16] out_w   [16:20] out_b
    #   [20:28] film_w  [28:30] film_b
    #   [30] w_beta  [31] b_beta
    cell_params: _F32_ARR,
    # Optimizer hyperparams
    lr: ti.f32,
    wd: ti.f32,
    clamp_val: ti.f32,
    n: ti.i32,
    # Outputs
    weight_out: _F32_ARR,
    h_next_out: _F32_ARR,
    scale_fast_out: _F32_ARR,
    scale_slow_out: _F32_ARR,
    scale_h_out: _F32_ARR,
):
    """
    Fused SGD-MinExpLSTM step kernel (PRD v3.2 Section 6).
    One thread per element. Fused memory pipeline:
    1. Normalize the stashed new EMAs (RMS from pass 2)
    2. Cell forward (log-domain gates, 4D latent FiLM)
    3. FiLM-modulated SGD step + weight decay
    4. Fold the per-tensor max-abs into the scale arrays
       and stash the new hidden state in FP32 (the host
       re-quantizes to INT8 against the final scales)
    """
    # Element-parallel: one thread per element (the idiomatic
    # Taichi form, matching the GEMV / SparseProp kernels).
    for tid in range(n):
        _sgd_minexplstm_step_element(
            tid,
            grad,
            weight,
            h_int8,
            scale_h,
            rms_g,
            rms_fast_new,
            rms_slow_new,
            ema_fast_new,
            ema_slow_new,
            cell_params,
            lr,
            wd,
            clamp_val,
            n,
            weight_out,
            h_next_out,
            scale_fast_out,
            scale_slow_out,
            scale_h_out,
        )


@ti.func
def _sgd_minexplstm_step_element(
    tid: ti.i32,
    grad: _F32_ARR,
    weight: _F32_ARR,
    h_int8: _I32_ARR,
    scale_h: ti.f32,
    rms_g: ti.f32,
    rms_fast_new: ti.f32,
    rms_slow_new: ti.f32,
    ema_fast_new: _F32_ARR,
    ema_slow_new: _F32_ARR,
    cell_params: _F32_ARR,
    lr: ti.f32,
    wd: ti.f32,
    clamp_val: ti.f32,
    n: ti.i32,
    weight_out: _F32_ARR,
    h_next_out: _F32_ARR,
    scale_fast_out: _F32_ARR,
    scale_slow_out: _F32_ARR,
    scale_h_out: _F32_ARR,
) -> None:
    """One element of the fused SGD-MinExpLSTM step (pass 3)."""
    eps = 1e-8
    _ = n

    # 1. Normalize the stashed new EMAs (PRD 3.3 step 5).
    g = grad[tid]
    g_norm = g / (rms_g + eps)
    ef_new = ema_fast_new[tid]
    es_new = ema_slow_new[tid]
    fast_norm = ef_new / (rms_fast_new + eps)
    slow_norm = es_new / (rms_slow_new + eps)

    # Dequantize the hidden state with its per-tensor scale.
    h_prev = ti.cast(h_int8[tid], ti.f32) * scale_h

    # 2. Cell forward (PRD 3.4 + 3.5).
    # x = [g_norm, fast_norm, slow_norm]
    x0 = g_norm
    x1 = fast_norm
    x2 = slow_norm
    # Log-domain gates: hat_phi = sigmoid(phi - iota)
    phi_val = (
        cell_params[0] * x0 + cell_params[1] * x1 + cell_params[2] * x2 + cell_params[3]
    )
    iota_val = (
        cell_params[4] * x0 + cell_params[5] * x1 + cell_params[6] * x2 + cell_params[7]
    )
    hat_phi = 1.0 / (1.0 + ti.exp(-(phi_val - iota_val)))
    # tilde_h = h_proj(x)
    tilde_h = (
        cell_params[8] * x0
        + cell_params[9] * x1
        + cell_params[10] * x2
        + cell_params[11]
    )
    # h_next = hat_phi * h_prev + (1 - hat_phi) * tilde_h
    h_next = hat_phi * h_prev + (1.0 - hat_phi) * tilde_h
    # 4D latent: z = ReLU(out_proj(h_next))
    # out_proj.weight is [4, 1] flattened to [4] at
    # cell_params[12:16]; out_b at [16:20].
    z0 = cell_params[12] * h_next + cell_params[16]
    z1 = cell_params[13] * h_next + cell_params[17]
    z2 = cell_params[14] * h_next + cell_params[18]
    z3 = cell_params[15] * h_next + cell_params[19]
    if z0 < 0.0:
        z0 = 0.0
    if z1 < 0.0:
        z1 = 0.0
    if z2 < 0.0:
        z2 = 0.0
    if z3 < 0.0:
        z3 = 0.0
    # FiLM: [gamma, delta] = film_proj(z)
    # film_w is [2, 4] flattened row-major at
    # cell_params[20:28]; film_b at [28:30].
    gamma = (
        cell_params[20] * z0
        + cell_params[21] * z1
        + cell_params[22] * z2
        + cell_params[23] * z3
        + cell_params[28]
    )
    delta = (
        cell_params[24] * z0
        + cell_params[25] * z1
        + cell_params[26] * z2
        + cell_params[27] * z3
        + cell_params[29]
    )

    # 3. FiLM-modulated SGD step + weight decay (PRD 3.6).
    gamma_clamped = gamma
    if gamma_clamped < -0.9:
        gamma_clamped = -0.9
    elif gamma_clamped > 2.0:
        gamma_clamped = 2.0
    delta_clamped = delta
    if delta_clamped < -clamp_val:
        delta_clamped = -clamp_val
    elif delta_clamped > clamp_val:
        delta_clamped = clamp_val
    base_sgd = -lr * g
    film_scaled_sgd = (1.0 + gamma_clamped) * base_sgd
    decay_step = -lr * wd * weight[tid]
    weight_out[tid] = weight[tid] + film_scaled_sgd + delta_clamped + decay_step

    # 4. Re-quantize the new states to INT8 (symmetric
    # max-abs, PRD 4.1). The per-tensor scale is the
    # max-abs over the whole tensor; each thread folds its
    # element's magnitude into the running max via a
    # compare-and-write on the scale_out[0] slot (the
    # reduction is serial by construction, like the RMS
    # pass). The INT8 value is then quantized against the
    # *final* scale, which is unknown until every thread
    # has contributed -- so we quantize against the
    # running max here and the host publishes the final
    # scale (see run_sgd_minexplstm_step).
    af = ef_new
    if af < 0.0:
        af = -af
    aslow = es_new
    if aslow < 0.0:
        aslow = -aslow
    ah = h_next
    if ah < 0.0:
        ah = -ah
    # Fold the per-tensor max-abs into the scale
    # accumulators. ti.atomic_max is the atomic
    # reduction primitive (the plain compare-and-write
    # above races: parallel threads read the running
    # max, compute a larger value, and clobber each
    # other's update, so the final max is wrong).
    ti.atomic_max(scale_fast_out[0], af)
    ti.atomic_max(scale_slow_out[0], aslow)
    ti.atomic_max(scale_h_out[0], ah)
    # Stash the new hidden state in FP32; the host
    # re-quantizes it (and the two EMAs, which pass 2
    # already stashed) to INT8 against the final
    # per-tensor max-abs scales once the kernel returns
    # (the scale is unknown until every thread has
    # contributed its magnitude, so a single-launch
    # quantization would round against a partial max).
    h_next_out[tid] = h_next


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

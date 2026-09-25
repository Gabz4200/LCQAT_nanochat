# LC-QAT deferred phase: quantized KV cache + index-native decode runtime

Status: **design only, not implemented**. This is the single item explicitly
deferred from the LC-QAT PRD (see `dev/LOG.md`). Everything else in the PRD -
training-side codebooks, per-layer K allocation, Q/K/V output quantization,
activation LUTs, the mul-less GEMV, export - is implemented. This document is
the required follow-up plan.

## Why it was deferred

PRD §7.1 stores K/V as packed 4-bit indices (`K_A = 15`, 2 values/byte) and
resolves them to FP32 through a per-head LUT *inside the attention kernel*.
Two constraints block that today:

1. **FA3 in-place contract.** `flash_attn_with_kvcache` (nanochat/
   `flash_attention.py`) takes bf16/fp16 `k_cache`/`v_cache` tensors and writes
   new K/V into them in place at `cache_seqlens`. An index-only cache cannot be
   written to or read from by FA3 without an index-native attention kernel.
2. **Peak-RAM honesty.** The standard escape - keep indices stored, materialize
   a bf16 window each step for FA3 - still needs a `max_seq_len`-sized staging
   tensor because FA3 indexes the full cache buffer. The transient bf16 copy
   erases the peak-memory win, so shipping that would be a metric-shaped
   fake-out rather than the PRD's 0.45 GB KV budget.

Any naive "quantize before passing to FA3" path (dequantize the cache we just
quantized) is equally pointless: same peak memory, extra compute.

## What already exists (the training half, per the user's directive)

- **Q/K/V projection outputs are quantized during training**: `LCQATLinear` with
  `quantize_out=True` is installed on `attn.c_q`, `attn.c_k`, `attn.c_v` by
  `retrofit_model` (see `nanochat/lcqat/retrofit.py`). The model therefore
  learns under the same 4-bit K/V distribution the runtime will impose; only
  the *storage format* is deferred, not the quantization noise the model trains
  through.
- Per-head codebook machinery: `MemoryEfficientLearnedCodebook` instances are
  already per-module (per projection), which maps 1:1 onto the PRD's "1 FP32
  array of length K per Key/Value head" once per-head wiring is needed.
- Bit-packing primitives with tests: `nanochat/lcqat/packing.py`
  (`pack_nibbles`/`unpack_nibbles` for `K_A = 15` KV entries).
- The compiled-op pattern to imitate: `nanochat/lcqat/ops/dispatch.py` +
  `native/cpu/gemv.cpp` + `kernels/slang/gemv/forward.slang` show the exact
  dispatcher / lazy-loader / three-way-parity-test contract a quantized
  attention op must follow.

## Follow-up design (sketch, to be refined when scheduled)

1. **Storage layer.** `QuantizedKVCache` next to `KVCache` in
   `nanochat/engine.py`: `uint8` index buffers
   `[n_layers, B, T, H_kv, D]` nibble-packed along `D`, plus one FP32 codebook
   per (layer, head) exported from each `c_v`/`c_k` out-quantizer. Writes
   quantize the just-computed K/V via the existing codebook `bucketize` path.
2. **Decode attention kernel (the real work).** A mul-less attention op with
   the same three-way backend contract as `lcqat_gemv_k3`:
   - reference: dequantize the needed window in PyTorch, call SDPA (oracle);
   - CPU: native C++/AVX path (LUT-gather K/V, QK^T in registers for the
     short `window_size` rows nanochat already uses);
   - GPU: portable Slang kernel (per-head LUT fetch, masked softmax).
   Registered through `torch.library` with fake kernels so the model still
   compiles under `torch.compile`.
3. **Engine integration.** `Engine.generate`/prefill paths opt in via a flag;
   FA3 remains the default until parity + memory benchmarks pass. Sliding
   windows (`window_pattern`) keep the working set small, which is what makes
   the index-native path viable per step.
4. **Verification matrix** (same as `tests/test_lcqat_ops.py`): three-way
   output parity vs the dequantize+SDPA reference, `opcheck`, compile smoke,
   plus an end-to-end perplexity check that quantized-KV decode matches the
   bf16-cache decode within tolerance, and a peak-RAM benchmark reproducing the
   PRD §7 budget table.
5. **Also deferred alongside it:** staging tensors through numpy for Slang
   (`kernels/slang_loader.py`) exists only because slangpy's native torch
   bridge requires CUDA torch; revisit when a shared-memory path is available.

## Non-goals of the deferred phase

- No change to training: QAT stays on the STE `F.linear` path (the GEMV op is
  explicitly inference-only and its autograd raises).
- No FA3 patches: upstream FA3 keeps its bf16 contract; quantized KV lives
  beside it until the custom attention kernel earns the switch.

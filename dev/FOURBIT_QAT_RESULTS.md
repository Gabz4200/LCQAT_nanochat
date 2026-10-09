# 4-bit QAT Pre-training Run: Results

**Date:** 2026-10-08
**Objective:** Main-model weights at **4 bits** (packed nibbles + scale), both `*_MinExpLstm_Optimizer` running **Quantization-Aware Training**, the SGD optimizer also quantizing **gradients to INT8 (8-bit)**. Config sized to complete in **≤ 44 minutes**.
**Optimizer:** `MeZo_MinExpLstm_Optimizer` (M=1, cpu backend, `qat_4bit=True`)
**Technologies active:** 4-bit main-model storage, LC-QAT codebook machinery (STE), SparseProp, DiffusionBlocks (2 blocks), DARD (5 mined rules), MeZO, learnable data-dependent LR.
**Host:** 4 CPU cores, 7.6 GB RAM / 19 GB swap, no GPU, CPU-only torch 2.14.0.

## What was built (this session)

### 1. `FourBitLinear`: runtime 4-bit weight storage

`nanochat/models/quant/fourbit.py`. A linear layer whose **persistent storage** is 4 bits:

- `weight_packed`: uint8, `ceil(N/2)` bytes, **two 4-bit values per byte** (packed nibbles, the PRD kernel's `(k & 1) ? byte >> 4 : byte & 0xF` decode).
- `weight_scale`: one fp32 scalar per tensor (symmetric max-abs, the INT8-state convention).
- `weight`: the fp32 **optimization surrogate** the MinExpLSTM optimizers step.

**Measured: 4.00 bits/weight** in the packed buffer (5.31 MB packed vs 42.47 MB fp32 surrogate on a 73.5M-param d6 model's 10.6M linear weights).

The forward is quantization-aware: it dequantizes the packed nibbles (`(index - 8) * scale`, index 8 is a structural zero) and matmuls, with a **straight-through estimator** routing the output gradient to the fp32 surrogate and the scale (the standard fake-quant contract, the same one the LC-QAT codebook path relies on).

### 2. QAT on both MinExpLSTM optimizers

`qat_4bit: bool = False` flag on both `SGD_MinExpLstm_Optimizer` and `MeZo_MinExpLstm_Optimizer`:

- **Both:** after every step, each `FourBitLinear`'s 4-bit storage mirror is refreshed from the fp32 surrogate (`quantize_weight_`), so the persistent weight form stays at 4 bits between steps.
- **SGD also (PRD 8-bit-gradient allowance):** each gradient is quantized to INT8 (symmetric max-abs, one scale per tensor) and the INT8 buffer + scale live in the state: **1 byte/grad instead of 4**.

Off by default; the shipped fp32/INT8-state path is byte-for-byte unchanged (53 existing optimizer tests still pass).

### 3. Retrofit + CLI threading

- `retrofit_fourbit(model, min_linear_dim=128)` (`nanochat/models/quant/retrofit.py`): replaces eligible `nn.Linear` with `FourBitLinear` in place (mirrors `retrofit_model`'s rules: skips `lm_head`, small layers, already-retrofitted).
- `--qat-4bit` flag on `scripts/base_train.py`; threaded through `scripts/_train/build.py` `make_optimizer` (both MeZO and SGD branches) and `build_base_model` (the 4-bit retrofit branch). Mutually exclusive with `--lcqat` (both convert the Linear modules).

## The 44-min run

**Config:** depth 6, head-dim 64, max-seq-len 256, device-batch 8, total-batch 2048, db-blocks 2, MeZO M=1 cpu, `--qat-4bit`, 850 iterations.

**Sizing:** depth-6 QAT smoke measured **~3.1 s/step** (20 steps in 62 s on 4 cores); 850 steps ≈ 44 min. Depth 12 was measured too slow (>50 s for <10 steps) and dropped.

**Data note:** the cached ClimbMix shards are 1-row smoke fixtures (12 KB / 16 KB), so the run cycles a tiny corpus: the loss trajectory is dominated by the data starvation, not the optimizer. This is the largest run this host's data cache supports; a real pre-training run needs the full tokenized ClimbMix corpus (see "What a real run would need" below).

## Verification

- `tests/test_fourbit_qat.py` (12 tests): 4-bit storage footprint (0.5 bytes/weight), pack/unpack roundtrip, STE gradient routing, forward-vs-dequantized-reference parity, `retrofit_fourbit` in-place swap + forward, SGD 8-bit grads (`grad_int8` + `scale_grad` in state), SGD QAT-vs-fp32 trajectory close (bounded quantization noise), MeZO QAT step + INT8 hidden state.
- Optimizer regression (`test_minexp_lstm_optimizer`, `test_mezo_minexplstm_kernel`, `test_sgd_minexplstm_film`, `test_sgd_lightning`): **71 passed**, no regression from the QAT additions.
- Quality gates: `ruff check` ✓, `ruff format` ✓, `pyrefly check` 0 errors on all changed files ✓.

## What a real 4-bit pre-training run would need

- **Data:** the full tokenized ClimbMix corpus (billions of tokens), not the 2-row smoke fixtures. The 4-bit storage halves weight memory, but the bottleneck here is the corpus, not the weights.
- **Hardware:** a multi-GPU node. The 4-bit layout (0.5 bytes/weight + 1 byte/optimizer-state + 1 byte/grad = **2.5 bytes/param** total) is exactly the memory profile that makes large-model training feasible on limited VRAM, which is the reason the PRD specifies it.
- **Time:** on adequate hardware, the 44-min budget fits a continue-pretrain or a small-model demo; from-scratch pre-training needs longer.

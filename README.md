# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

**nanochat** is a minimal, hackable experimental harness for training LLMs. It is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) that extends it with a research focus on **quantization-aware training** for ultra-low-cost inference.

It is designed to run on a single node, covers all major LLM stages (tokenization, pretraining, finetuning, evaluation, inference), and is configured to run on commodity CPU/GPU hardware. The single complexity dial `--depth` automatically determines all other hyperparameters (width, heads, learning rate schedule, training horizon, weight decay, …) so that each model comes out compute-optimal, while a bundle of optional compression stages (LC-QAT codebooks, SparseProp sparse backprop, diffusion-block block-wise training) ship **always-on by default**.

The headline result is the [Time-to-GPT-2 leaderboard](#time-to-gpt-2-leaderboard): a GPT-2 capability model (~4e19 FLOPs) that previously cost ~$43,000 (8×H100, 2019) can now be trained in ~1.5 hours for ~$48 on modern hardware. With this fork's quantization stages enabled, the trained model fits in tiny memory footprints and supports mul-less / mul-light runtime kernels for fast CPU decode.

> **This is a fork, not the upstream repo.** Upstream discussion live links (DeepWiki, Discord, Discussions) point at `karpathy/nanochat`. Fork-specific work and the LC-QAT/SparseProp/quantization experiments live in [Gabz4200/LCQAT_nanochat](https://github.com/Gabz4200/LCQAT_nanochat).

**TL;DR** — get the most common paths:

| What | Run |
|------|-----|
| CPU/CPU install | `uv sync --extra cpu` |
| Train a tiny model (CPU, ~5 min) | `bash runs/runcpu.sh` |
| Full GPT-2 speedrun (8×H100) | `bash runs/speedrun.sh` |
| Quantized inference (after `--lcqat` train) | `uv run python -m scripts.export_lcqat --source sft` |

---

## Table of contents

- [Time-to-GPT-2 leaderboard](#time-to-gpt-2-leaderboard)
- [Getting started](#getting-started)
- [Stages](#stages)
- [Quantization: LC-QAT, KD, EfQAT, activation LUTs](#quantization-lc-qat-kd-efqat-activation-luts)
- [SparseProp sparse backprop](#sparseprop-sparse-backprop)
- [DiffusionBlocks (block-wise training + diffusion inference)](#diffusionblocks-block-wise-training--diffusion-inference)
- [Running on CPU / MPS](#running-on-cpu--mps)
- [Precision / dtype](#precision--dtype)
- [Benchmarks](#benchmarks)
- [Research](#research)
- [File structure](#file-structure)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Cite](#cite)
- [License](#license)
## Time-to-GPT-2 leaderboard

Presently, the main focus of development is on tuning the pretraining stage, which takes the most amount of compute. Inspired by the modded-nanogpt repo and to incentivise progress and community collaboration, nanochat maintains a leaderboard for a "GPT-2 speedrun", which is the wall-clock time required to train a nanochat model to GPT-2 grade capability, as measured by the DCLM CORE score. The [runs/speedrun.sh](runs/speedrun.sh) script always reflects the reference way to train a GPT-2 grade model. The current leaderboard looks as follows:

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

The primary metric we care about is "time to GPT-2" - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an 8×H100 GPU node. The GPT-2 CORE score is 0.256525. In 2019, the training of GPT-2 cost approximately $43,000 so it is incredible that due to many advances over 7 years across the stack, we can now do so much faster and for well below $100 (e.g. at the current ~$3/GPU/hr, an 8×H100 node is ~$24/hr, so 2 hours is ~$48).

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more docs on how to interpret and contribute to the leaderboard.

> **Note on quantization defaults:** this fork enables **LC-QAT and SparseProp *always-on by default*** (see [Quantization](#quantization-lc-qat-kd-efqat-activation-luts)); the historical speedrun rows above were produced with `--fp8` float training. The reference script has been updated to match current defaults — see [Getting started](#getting-started).

### Getting started

#### Setup

nanochat uses [uv](https://docs.astral.sh/uv/) for dependency management. **This fork ships CPU-only PyTorch wheels** (the CUDA extra was dropped for the LC-QAT runtime; the comment in `pyproject.toml` explains how to re-add a GPU extra + CUDA index if you need GPU training). To install:

```bash
uv sync --extra cpu    # CPU wheels (CUDA extra is dropped; re-add per pyproject.toml
                       # if you need GPU training)
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, pyrefly, ruff, taichi, aislop, …):

```bash
uv sync --extra cpu --group dev
```

#### Reproduce and talk to GPT-2

The most fun you can have is to train your own GPT-2 and talk to it. The entire pipeline to do so is contained in the single file [runs/speedrun.sh](runs/speedrun.sh), which is designed to run on an 8×H100 GPU node. Boot up a new 8×H100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take ~1.5 hours to run. Once it's done, you can talk to your model over the CLI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and chat:

```bash
python -m scripts.chat_cli
```

Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. An example conversation with a speedrun model:

> **User:** Hello whats up?
>
> **Assistant:** Hello! How can I help you today?
>
> **User:** Why is the sky blue?
>
> **Assistant:** The sky is blue due to an optical illusion called the Rayleigh Scattering of Light, which occurs when sunlight passes through tiny particles in the air. These particles are made up of tiny blue and violet particles that cause the light to bend in a particular way.
>
> **User:** That's cool. Write a poem about it.
>
> **Assistant:** In the twilight's glow, the sky is blue,<br>
> A canvas of shimmering blue, where skies unfold,<br>
> A world of blue, where blue and more blue,<br>
> Are woven together in a tapestry so fine,<br>
> Where every hue seems to sing a story.<br>
> ...

A few more notes:

- This fork trains with **LC-QAT + SparseProp always-on by default**. To reproduce the historical float/fp8 speedrun, the reference script uses `--no-lcqat --no-sparseprop` (and drops `--fp8`, which requires CUDA and conflicts with LC-QAT's default-on state).
- The code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device-batch-size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

### Stages

nanochat is a single cohesive pipeline, not a configurable framework: there are no giant config objects, model factories, or if-then-else monsters. The entry points live in `scripts/` and all share the global `COMPUTE_DTYPE` and the `--depth` complexity dial:

| Stage | Entry point | Description |
|-------|-------------|-------------|
| Tokenizer | `scripts/tok_train.py` | Train BPE tokenizer (vocab 2**15 = 32768) |
| Tokenizer eval | `scripts/tok_eval.py` | Report compression ratio, vocab coverage |
| Pretrain | `scripts/base_train.py` | Block-wise (DiffusionBlocks) pretraining; LC-QAT + SparseProp default on |
| Base eval | `scripts/base_eval.py` | CORE metric, bits-per-byte, sampling |
| SFT | `scripts/chat_sft.py` | Supervised finetune on the DiffEngine; default on LC-QAT/SparseProp |
| RL | `scripts/chat_rl.py` | PPO-style RL finetune (LC-QAT/SparseProp default on) |
| Export | `scripts/export_lcqat.py` | Freeze a LC-QAT checkpoint into a stripped quantized inference artifact |
| Chat | `scripts/chat_cli.py` | Talk to a trained model over CLI |

### Quantization: LC-QAT, KD, EfQAT, activation LUTs

Learned Codebook Quantization-Aware Training (LC-QAT) is the fork's core contribution. Every retrofitted `Linear` gets an **asymmetric odd-size codebook** `K = 2M + 1` with index `M` anchored **exactly to 0.0**, so zero-initialized weights and sparse activations quantize without noise. Levels are cumulative `softplus` steps (monotonic under gradient descent); the forward pass uses a straight-through estimator that trains both the input and the codebook; and the codebook parameters (`raw_pos_deltas` / `raw_neg_deltas`) get their own AdamW group with a dedicated learning rate (`--codebook-lr`, default `1e-3`, no weight decay).

**LC-QAT and SparseProp are always-on by default** in `base_train`, `chat_sft`, and `chat_rl`. Disable them with `--no-lcqat` / `--no-sparseprop`.

```bash
python -m scripts.base_train --depth=12 --no-lcqat   # plain float training
python -m scripts.base_train --lcqat --lcqat-preset prd   # PRD table: 8-bit down_proj
torchrun -m scripts.chat_sft -- --run=sft            # LC-QAT + SparseProp on by default
python -m scripts.chat_rl --no-lcqat --no-sparseprop  # plain RL
```

| Flag | Meaning |
|------|---------|
| `--lcqat` / `--no-lcqat` | LC-QAT is on by default; `--no-lcqat` runs plain float training |
| `--lcqat-preset small` | **Default, max compression**: q/k weights K=3 (mul-less ternary), everything else K=15 |
| `--lcqat-preset prd` | PRD table: `mlp.c_proj` (down_proj) at K=255/255, rest as small |
| `--lcqat-k-map substr:KW/KA,...` | Per-module overrides, e.g. `mlp.c_proj:255/255,attn.c_v:15/15` |
| `--codebook-lr` | Codebook AdamW LR (PRD: 10–50× network weights), no weight decay |
| `--fp8` | FP8 training for the float path. **Mutually exclusive with `--lcqat`** (both convert `Linear`) |

Per-layer roles (`nanochat/lcqat/retrofit.py`): `attn.c_q`/`c_k` get ternary weights, Q/K/V **outputs** are quantized during training and the quantized KV-cache runtime is implemented (`Engine.generate(quantized_kv=True)`), `mlp.c_fc` output is quantized as the input side of the fused relu² LUT, `lm_head` and linears under 128 dims stay in floating point.

**Activation LUTs.** Beyond weights, activations are quantized through fused lookup tables registered in an activation registry (`nanochat/lcqat/lut.py`). All common nonlinearities are covered: `relu2`, `silu`, `gelu`, `tanh`, `sigmoid`. The export step bakes these into static FP32 LUTs via `wire_activation_luts`.

**Knowledge Distillation (KD) anchoring.** High compression ratios compress the loss manifold into sharp local minima. LC-QAT anchors the student QAT optimization using a KL-divergence against a frozen, detached FP32/BF16 teacher (the pre-quantization model, or any other unquantized reference). The teacher is never optimized.

`L_KD = tau^2 * D_KL( softmax(Z_teacher / tau) || softmax(Z_student / tau) )`
`L_total = (1 - alpha) * L_CE(Y, Y_hat_quant) + alpha * L_KD`

Enabled via `--kd-alpha` (default 0.0 = off; ~0.1 typical), `--kd-teacher-source` and `--kd-teacher-tag` (the float checkpoint to load as teacher). Implemented in `nanochat/lcqat/kd.py` (`KDLoss`).

**EfQAT selective layer freezing.** To keep memory flat at scale, LC-QAT can selectively freeze middle-layer codebook deltas and weight gradients after a warm-up, keeping only "critical outlier layers" (input embedding projections, attention q/k, final output) trainable. Enabled with `--efqat-freeze-after N` (default -1 = off). Implemented in `nanochat/lcqat/efqat.py` (`SelectiveFreezer`). The optimizer simply skips params whose `.grad is None`, so momentum buffers are unaffected.

#### Mul-less GEMV + index-linear kernels

For quantized decode (`K_W = 3` ternary), the runtime routes through a backend dispatcher:

- `naive` — pure PyTorch oracle (`nanochat/lcqat/ops/references/`), for parity tests and debugging;
- `cpu` — C++ kernel under `nanochat/lcqat/native/cpu/` (`gemv.cpp`, `index_linear.cpp`, `sparseprop.cpp`), JIT-built by torch on first use (needs `g++`/`clang++`; build cached under `~/.cache/torch_extensions`);
- `gpu` — Taichi kernel (`nanochat/lcqat/kernels/gpu_loader.py`), loaded lazily (needs `taichi` + a Vulkan ICD).

`dispatch_index_linear` extends the same pattern to arbitrary storage widths `K ∈ {3, 15, 255, 257}` (see `nanochat/lcqat/ops/index_linear.py`). A requested backend that cannot run raises — it never silently falls back to naive. Three-way parity (`naive == cpu == gpu`), `torch.library.opcheck`, and `torch.compile` composition are covered by `tests/test_lcqat_ops.py`.

#### Export

```bash
uv run python -m scripts.export_lcqat --source sft --out exports/lcqat_sft.pt
```

Freezes codebooks into static FP32 LUTs, replaces FP32 weight matrices with `uint8` index buffers, and saves a minimal artifact for the quantized inference runtime (not resumable for training). After export, `load_model` auto-detects the LC-QAT state from the artifact.

### SparseProp sparse backprop

[SparseProp](https://arxiv.org/abs/2408.08525) injects unstructured sparsity into the backward pass: a static sparsity mask (default `--sparseprop-sparsity=0.75`, fraction of weights pruned) is applied to `Linear` layers, and the backward pass routes through AVX2 C++ kernels (`nanochat/lcqat/native/cpu/sparseprop.cpp`) that compute gradients over `nnz` entries only — `O(nnz)` instead of `O(M·K)`. Forward is standard masked SpMM; backward is the mul-less/sparse win.

SparseProp is **always-on by default** alongside LC-QAT. When LC-QAT is active, LCQATLinear is wrapped as `SparsePropLinearLCQAT` so the two coexist in one module (`nanochat/lcqat/sparseprop.py`). Disable with `--no-sparseprop`.

| Flag | Meaning |
|------|---------|
| `--no-sparseprop` | Disable SparseProp (default: on) |
| `--sparseprop-sparsity` | Sparsity level 0.0–1.0 (default `0.75`) |

Parity (`naive == cpu`) and integration with the full training pipeline are in `tests/test_sparseprop.py` and `tests/test_sparseprop_integration.py`.

### DiffusionBlocks (block-wise training + diffusion inference)

`nanochat/diffusion_blocks.py` implements SakanaAI DiffusionBlocks (ICLR 2026): depth is partitioned into `B` independent blocks (`EquiProbabilityPartitioner` distributes layers equi-probably across blocks), each block gets a noise-conditioned adapter (AdaLN from an EDM sinusoidal sigma embedding, `denoise_step`), and an equi-probable cycling scheduler picks one block active per micro-step. Only the active block holds gradients + optimizer state, cutting grad/optimizer memory ~`B`×.

The `DiffusionBlockEngine` is the **default training engine** for `base_train`, `chat_sft`, and `chat_rl` (`--db-blocks`, default `4`). It wraps a `GPT` and exposes `train_step` / `denoise_step` / `generate`. LC-QAT and SparseProp are applied through the engine so the whole pipeline — adapters, denoise head, KV cache — is quantized and sparse.

```bash
# DiffusionBlocks on CPU (toy, d4/256-wide, seq 64, batch 2)
python -m scripts.base_train --depth=4 --db-blocks=4 --no-lcqat --no-sparseprop --num-iterations=500
```

### Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a simple example of exercising the code paths on CPU or Apple Silicon. It shrinks the model to fit into a reasonable time interval (a few ten minutes of training). You will not get strong results this way — think of it as an educational/demo run.

Because DiffusionBlocks is always-on, the run also exercises the block-wise training + diffusion inference path (`denoise_step` scales ~`B`× as it runs only the active block; `train_step` still forwards the full model and saves only backward/optimizer work).

Measured on CPU (d4/256-wide toy, seq 64, batch 2, peak RSS 406 MB):

| B | CE `train_step` | EDM `denoise_step` |
|---|-----------------|-------------------|
| 1 | 1108 tok/s | 1171 tok/s |
| 2 | 1298 tok/s | 2365 tok/s |
| 4 | 1284 tok/s | 4045 tok/s |
### Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, …) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, …) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default. On recent macOS, MPS also runs `NANOCHAT_DTYPE=bfloat16` fine (~25% less memory, similar speed) |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. bf16/fp32 don't need it — bf16 has the same exponent range as fp32. Inference in fp16 works fine everywhere.

## Benchmarks

Two microbench scripts live in `scripts/`:

- `scripts/gemv_bench.py` — LC-QAT index-fetch matmul paths vs `torch.nn.functional.linear`, at the PRD's suggested shapes (m ∈ {768, 4096}, n ∈ {768, 2048}). Sanity-checks every backend against the naive oracle before timing. `uv run python -m scripts.gemv_bench`.
- `scripts/kv_budget_bench.py` — reproduces the LC-QAT PRD section 7 memory-budget table (heterogeneous packed weights, FP32 codebook LUTs, 32K packed 4-bit KV cache, C++ workspace constant). With `--alloc-kv`, it allocates and page-touches a real `QuantizedKVCache` at the PRD's implied dims to check resident growth against `storage_bytes`. `uv run python -m scripts.kv_budget_bench [--alloc-kv]`.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See the [Jan 7 miniseries v1 discussion](https://github.com/karpathy/nanochat/discussions/420) (upstream) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. To see if a run helps, monitor the wandb plots for:

1. `val_bpb` (validation loss in vocab-size-invariant units of bits per byte) as a function of `step`, `total_training_time` and `total_training_flops`.
2. `core_metric` (the DCLM CORE score)
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), `train/tok_per_sec` (training throughput)

See an example [here](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044).

The important thing to note is that nanochat is written and configured around one single dial of complexity - the depth of the transformer. This single integer automatically determines all other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, …) so that the trained model comes out compute optimal. The idea is that the user doesn't have to think about or set any of this, they are simply asking for a smaller or bigger model using `--depth`, and everything "just works". By sweeping out the depth, you achieve the nanochat miniseries of compute optimal models at various sizes. GPT-2 capability model happens to be somewhere around d24–d26 range with the current code. Any candidate changes to the repo have to be principled enough that they work for all settings of depth.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── LEADERBOARD.md                  # Time-to-GPT-2 leaderboard docs
│   ├── LOG.md                          # Training experiment log
│   ├── nanochat.png                    # project logo
│   ├── repackage_data_reference.py     # Pretraining data shard generation
│   ├── scaling_analysis.ipynb          # scaling-laws analysis notebooks
│   ├── scaling_laws_jan26.png
│   └── estimate_gpt3_core.ipynb
├── nanochat
│   ├── __init__.py                     # empty
│   ├── checkpoint_manager.py           # Save/Load model checkpoints
│   ├── common.py                       # Misc small utilities, quality of life (COMPUTE_DTYPE)
│   ├── core_eval.py                    # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py                   # Tokenizing Distributed Data Loader
│   ├── dataset.py                      # Download/read utils for pretraining data
│   ├── diffusion_blocks.py             # DiffusionBlockEngine: block-wise AR + diffusion inference
│   ├── engine.py                       # Efficient model inference with KV Cache
│   ├── execution.py                    # Allows the LLM to execute Python code as tool
│   ├── flash_attention.py              # Flash Attention 3 / SDPA dispatch
│   ├── fp8.py                          # Float8Linear conversion
│   ├── gpt.py                          # The GPT nn.Module Transformer
│   ├── loss_eval.py                    # Evaluate bits per byte (instead of loss)
│   ├── lcqat                           # LC-QAT: codebooks, KD, EfQAT, retrofit, ops/kernels, export
│   │   ├── __init__.py
│   │   ├── codebook.py                 # AsymmetricLearnedCodebook
│   │   ├── efqat.py                    # SelectiveFreezer
│   │   ├── export.py                   # export_lcqat_checkpoint, wire_activation_luts
│   │   ├── kd.py                       # KDLoss knowledge distillation
│   │   ├── linear.py                   # LCQATLinear module
│   │   ├── lut.py                      # Activation LUTs (relu2/silu/gelu/tanh/sigmoid)
│   │   ├── packing.py                  # Bit packing utilities
│   │   ├── retrofit.py                 # LayerKConfig, PRESETS, retrofit_model, parse_k_map
│   │   ├── sparseprop.py               # SparsePropLinear, inject_sparseprop_layers
│   │   ├── kernels
│   │   │   ├── __init__.py
│   │   │   ├── cpu_loader.py           # JIT build of C++ ops
│   │   │   └── gpu_loader.py           # Taichi/Vulkan GPU kernels
│   │   ├── native
│   │   │   └── cpu
│   │   │       ├── gemv.cpp            # mul-less GEMV (AVX-512/AVX2/scalar)
│   │   │       ├── index_linear.cpp    # multi-width index-fetch matmul
│   │   │       ├── quant_attn.cpp      # quantized attention
│   │   │       └── sparseprop.cpp      # AVX2 sparse backward
│   │   └── ops
│   │       ├── __init__.py
│   │       ├── dispatch.py
│   │       ├── gemv.py                 # dispatch_gemv (naive|cpu|gpu)
│   │       ├── index_linear.py         # dispatch_index_linear
│   │       ├── quant_attn.py           # quantized KV-cache attention
│   │       ├── sparseprop.py           # sparseprop_forward_cpu / _backward_cpu
│   │       └── references              # PyTorch oracle implementations for parity
│   │           ├── __init__.py
│   │           ├── attn_reference.py
│   │           ├── gemv_reference.py
│   │           └── index_linear_reference.py
│   ├── optim.py                        # AdamW + Muon optimizer, 1GPU and distributed
│   └── tokenizer.py                    # BPE Tokenizer wrapper in style of GPT-4
├── pyproject.toml                      # CPU-only torch wheels; CUDA extra dropped
├── runs
│   ├── miniseries.sh                   # Miniseries training script
│   ├── runcpu.sh                       # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh                 # Scaling laws experiments
│   └── speedrun.sh                     # Train the ~$100 nanochat GPT-2 speedrun
├── scripts
│   ├── base_eval.py                    # Base model: CORE score, bits per byte, samples
│   ├── base_train.py                   # Base model: pretrain (DiffusionBlockEngine)
│   ├── chat_cli.py                     # Chat model: talk to over CLI
│   ├── chat_eval.py                    # Chat model: eval tasks
│   ├── chat_rl.py                      # Chat model: reinforcement learning
│   ├── chat_sft.py                     # Chat model: train SFT
│   ├── export_lcqat.py                 # Export stripped LC-QAT artifact
│   ├── gemv_bench.py                   # Mul-less GEMV / index-linear microbench
│   ├── infer_bench.py                  # Inference: latency/throughput/VRAM bench
│   ├── kv_budget_bench.py              # LC-QAT memory-budget repro
│   ├── tok_eval.py                     # Tokenizer: evaluate compression rate
│   └── tok_train.py                    # Tokenizer: train it
├── tasks
│   ├── arc.py                          # Multiple choice science questions
│   ├── common.py                       # TaskMixture | TaskSequence
│   ├── gsm8k.py                        # 8K Grade School Math questions
│   ├── humaneval.py                    # Misnomer; Simple Python coding task
│   ├── mmlu.py                         # Multiple choice, broad topics
│   └── smoltalk.py                     # Conglomerate dataset of SmolTalk from HF
├── tests
│   ├── conftest.py
│   ├── test_attention_fallback.py      # FA3/SDPA attention fallback
│   ├── test_adamw_cpu.py               # CPU AdamW optimizer
│   ├── test_calculator.py              # Sandboxed code execution smoke
│   ├── test_dbcpu_*.py                 # DiffusionBlocks CPU: engine, train/denoise, packing, ...
│   ├── test_engine.py                  # Inference engine, KV cache
│   ├── test_execution.py               # Sandboxed code execution
│   ├── test_lcqat_codebook.py          # LC-QAT codebooks
│   ├── test_lcqat_export.py            # Codebook export, activation LUT wiring
│   ├── test_lcqat_kd.py                # KD anchoring loss
│   ├── test_lcqat_efqat.py             # EfQAT selective freezing
│   ├── test_lcqat_kv_cache.py          # Quantized KV-cache runtime
│   ├── test_lcqat_linear.py            # LCQATLinear module
│   ├── test_lcqat_linear_runtime.py    # Quantized inference runtime
│   ├── test_lcqat_lut.py               # Activation LUT registry
│   ├── test_lcqat_ops.py               # GEMV / index-linear parity + opcheck
│   ├── test_lcqat_packing.py           # Bit packing
│   ├── test_lcqat_quant_attn.py        # Quantized attention parity
│   ├── test_lcqat_retrofit.py          # retrofit_model / parse_k_map
│   ├── test_sparseprop.py              # SparseProp kernel parity
│   ├── test_sparseprop_integration.py  # SparseProp + LC-QAT + DiffusionBlocks
│   ├── test_tasks.py                   # Task slicing, mixtures, HubDataset
│   └── test_tokenizer.py               # BPE round-trips, chat rendering
└── uv.lock
```

Run the test suite with `pytest tests/` (add `-m "not slow"` to skip slow runs). The LC-QAT ops parity tests (`tests/test_lcqat_ops.py`) require a C++ compiler to JIT the native kernels; on macOS MPS, set `NANOCHAT_DTYPE=bfloat16` for the runtime tests.

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part is the quantization story: LC-QAT codebooks + SparseProp + mul-less CPU GEMV kernels that make a capability model fit in tiny memory and decode without floating-point multiplies.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- This repo is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat); the name (*nanochat*) derives from Andrej Karpathy's earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- LC-QAT (learned codebook quantization), SparseProp sparse backprop, DiffusionBlocks, the mul-less GEMV kernels, and this fork are the work of [Gabz4200](https://github.com/Gabz4200) and contributors.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research, cite the upstream work:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

and, if you use the LC-QAT / SparseProp / DiffusionBlocks quantization work from this fork:

```bibtex
@misc{lcqat-nanochat,
  author = {Gabz and contributors},
  title = {nanochat fork: LC-QAT, SparseProp, and DiffusionBlocks},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Gabz4200/LCQAT_nanochat}
}
```

## License

MIT

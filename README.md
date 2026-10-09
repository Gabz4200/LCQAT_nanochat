# lcqat-nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

**lcqat-nanochat** is a minimal, hackable fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) for one question: how far can training, storage, and decoding of a capable model move onto a commodity CPU? It keeps the full LLM pipeline (tokenization, pretraining, finetuning, evaluation, inference) behind a single complexity dial, `--depth`, which sets width, heads, schedule, horizon, and decay to compute-optimal values.

No capability model has been trained and measured on CPU in this fork yet. The machinery for it exists and each piece carries a benchmark: LC-QAT codebooks shrink the weights, SparseProp sparsity removes most of them, DiffusionBlocks training fits the remainder into a small memory budget, DARD adds corpus-mined rules as gradient, and packed-index kernels decode without materializing fp32 matrices. LC-QAT, SparseProp, and DARD ship on by default. Every number below comes from toy models (d4/d6, largest checkpoint d6 at step 2). Read them as evidence about mechanisms and their interaction, not about a capability model.

> **This is a fork, not the upstream repo.** Upstream links (DeepWiki, Discord, Discussions) point at `karpathy/nanochat`. Fork work on LC-QAT, SparseProp, quantization, and CPU kernels lives in [Gabz4200/LCQAT_nanochat](https://github.com/Gabz4200/LCQAT_nanochat).

**TL;DR** (CPU path is the supported one):

| What | Run |
|------|-----|
| Install (CPU wheels) | `uv sync --extra cpu --group dev` |
| Train a tiny model end to end (CPU, ~30 min) | `bash runs/runcpu.sh` |
| Exercise one stage on CPU (~1.5 min, 200 steps) | `python -m scripts.base_train --depth=4 --head-dim=64 --window-pattern L --max-seq-len=256 --device-batch-size=2 --total-batch-size=512 --num-iterations=200 --db-blocks=2 --run=dummy` |
| Run the test suite | `pytest tests/` |
| Full GPT-2 speedrun (needs 8×H100 + a GPU extra) | `bash runs/speedrun.sh` |
| Freeze a trained model into a quantized artifact | `uv run python -m scripts.export_lcqat --source sft` |

---

## Table of contents

- [Install](#install)
- [Usage](#usage)
  - [Stages](#stages)
  - [Train GPT-2 on GPU](#train-gpt-2-on-gpu)
  - [Run on CPU](#run-on-cpu)
  - [Example: a CPU run end to end](#example-a-cpu-run-end-to-end)
  - [Precision / dtype](#precision--dtype)
- [What's in the box](#whats-in-the-box)
- [Research sweeps](#research-sweeps)
- [Time-to-GPT-2 leaderboard](#time-to-gpt-2-leaderboard)
- [Development](#development)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Cite](#cite)
- [License](#license)

---

Deeper material lives in [`docs/`](docs/README.md): the
[quantization reference](docs/quantization.md),
[kernels](docs/kernels.md), [SparseProp](docs/sparseprop.md),
[DiffusionBlocks](docs/diffusionblocks.md), [DARD](docs/dard.md) and
[architecture](docs/architecture.md).

---

## Install

lcqat-nanochat uses [uv](https://docs.astral.sh/uv/) for dependencies. The fork pins CPU-only PyTorch wheels: torch lives under the `cpu` extra with an explicit CPU index, so a bare `uv sync` installs no torch and `import nanochat.models.backbone` fails until you request it. That failure is the intended signal, not a packaging bug.

```bash
uv sync --extra cpu    # CPU wheels. The CUDA extra was dropped for the LC-QAT
                       # runtime; pyproject.toml comments explain how to
                       # re-add a GPU extra + CUDA index if you need one.
source .venv/bin/activate
```

For development (adds pytest, pyrefly, ruff, taichi, aislop, datasets, huggingface-hub, matplotlib, ipykernel):

```bash
uv sync --extra cpu --group dev
```

Two repo-specific warnings. `uv add` resolves torch from PyPI and pulls a CUDA build, so follow every dependency change with `uv sync --extra cpu --group dev` to restore the pinned CPU wheel. The LC-QAT parity tests JIT a C++ kernel, so a compiler (`g++`/`clang++`) must sit on `PATH`; builds cache per user under `~/.cache/torch_extensions`.

---

## Usage

### Stages

Entry points live in `scripts/` and share the global `COMPUTE_DTYPE` and the `--depth` dial:

| Stage | Entry point | Description |
|-------|-------------|-------------|
| Tokenizer | `scripts/tok_train.py` | Train BPE tokenizer (vocab 2**15 = 32768) |
| Tokenizer eval | `scripts/tok_eval.py` | Report compression ratio, vocab coverage |
| Pretrain | `scripts/base_train.py` | Block-wise (DiffusionBlocks) pretraining; LC-QAT + SparseProp on |
| Base eval | `scripts/base_eval.py` | CORE metric, bits-per-byte, sampling |
| SFT | `scripts/chat_sft.py` | Supervised finetune; LC-QAT/SparseProp on |
| RL | `scripts/chat_rl.py` | PPO-style RL finetune; LC-QAT/SparseProp on |
| Chat eval | `scripts/chat_eval.py` | ChatCore score for an SFT/RL checkpoint |
| Export | `scripts/export_lcqat.py` | Freeze an LC-QAT checkpoint into a stripped quantized inference artifact |
| DARD mining | `scripts/dard_mine.py` | Mine AML rules from the corpus and save a compiled rule set (not a stage) |
| Chat | `scripts/chat_cli.py` | Talk to a trained model over CLI |

**Backbones.** `--backbone gpt` (default, all-attention Transformer) or `--backbone lfm2` (the LFM2 hybrid: gated short convolutions interleaved with a small fraction of GQA blocks, SwiGLU MLPs, pre-norm RMSNorm, RoPE with QK-norm; arXiv 2511.23404). Both speak the same interface, so every stage, optimizer (SGD_MinExpLstm, MeZo_MinExpLstm, AdamW), and quantization stage (LC-QAT, SparseProp, 4-bit QAT, DARD, DiffusionBlocks) works on either. `base_eval` / `chat_cli` detect the backbone from the saved config, so a checkpoint loads the stack it was trained with.

`speedrun.sh` runs the first six back to back. The ablation harness, kernel microbenchmarks, and inference benchmark also live in `scripts/`; the ablation is summarized under [What's in the box](#whats-in-the-box) and the benchmarks under [docs/kernels.md](docs/kernels.md#benchmarks).

### Train GPT-2 on GPU

The full pipeline to train your own GPT-2 sits in [runs/speedrun.sh](runs/speedrun.sh), written for an 8×H100 node. Boot the box (e.g. [Lambda](https://lambda.ai/service/gpu-cloud)) and run:

```bash
bash runs/speedrun.sh
```

Run it in screen; it takes about 1.5 hours. Then activate the venv (`source .venv/bin/activate`) and chat:

```bash
python -m scripts.chat_cli
```

Ask it to write stories or poems, or to explain why the sky is blue. One speedrun-model exchange:

> **User:** Why is the sky blue?
>
> **Assistant:** The sky is blue due to an optical illusion called the Rayleigh Scattering of Light, which occurs when sunlight passes through tiny particles in the air. [...]
>
> **User:** Write a poem about it.
>
> **Assistant:** In the twilight's glow, the sky is blue, / A canvas of shimmering blue, where skies unfold, / [...] (model output, unedited apart from truncation)

Notes on reproducing it:

- This fork trains with LC-QAT + SparseProp + DiffusionBlocks on by default. The reference script passes `--no-lcqat --no-sparseprop` to recover the historical float recipe, and drops `--fp8`, which needs CUDA and conflicts with default-on LC-QAT. It keeps the DiffusionBlocks engine; add `--db-blocks=0` for a conventional autoregressive baseline.
- Omitting `torchrun` runs on a single GPU with gradient accumulation and near-identical results, only slower.
- On GPUs under 80 GB, lower `--device-batch-size` (32 to 16, 8, 4, 2, or 1) until the run fits.
- Most code is vanilla PyTorch and should run on xpu, MPS, or similar, but only CPU and CUDA paths see regular exercise.

### Run on CPU

`pyproject.toml` pins torch from the PyTorch CPU index with no CUDA extra. The GPU path still works (`torchrun`, distributed dataloader, `--fp8` float recipe), but defaults target CPU. Measurements below were taken on a 4-core box with 7.6 GB of RAM.

**Quickstart.** [runs/runcpu.sh](runs/runcpu.sh) runs every stage end to end on CPU or Apple Silicon: dataset download, tokenizer, d6 pretraining, SFT. It shrinks the model to fit ~30 minutes of training on an M3 Max. Expect a wiring check and demo, not strong results.

```bash
bash runs/runcpu.sh
```

It passes `--db-blocks=2` instead of the default 4. On a toy depth the per-block overhead dominates the step, and 2 exercises the block-wise path without letting overhead set the runtime.

**CPU-specific code:**

| Area | Where | Note |
|---|---|---|
| SGD_MinExpLstm_Optimizer | `nanochat/training/minexp_lstm.py` | Default optimizer (`--optimizer minexplstm`): elementwise SGDMinExpLSTMCellFiLM4D predicts FiLM gain/shift on top of SGD, lambda-coupled dual-EMA horizons, INT8 fast/slow/hidden states (3 bytes/param), joint online co-training via `meta_step` (AdamW over the cell + the learnable LR predictor). The LR is a **learnable, data-dependent** value: a tiny predictor maps the RMS-normalized gradient magnitude through a sigmoid to (0, 1), co-trained by the meta-optimizer; the AdamW meta-optimizer's own LR (`meta_lr`) stays a plain hyperparameter, tuned by Lightning's LR finder before training starts. `--optimizer adamw` keeps the baseline |
| MeZo_MinExpLstm_Optimizer | `nanochat/training/mezo_minexplstm.py` | Zeroth-order variant (`--optimizer mezo`): the same cell and FiLM head, but the update direction is a SPSA estimate `(L(θ+εz) - L(θ-εz)) / (2ε)` from dual forward passes, so no gradient is materialized. `--mezo-probes M` (default 1) runs M independent probes per step, averaged order-invariantly (PRD v2.1); M=4 is the target. `naive`/`cpu`/`gpu` backends with three-way parity |
| Fused AdamW | `nanochat/training/optim.py`, `cpu_adamw_for` | `fused=True`. Non-fused AdamW walks the parameter list in Python, a real cost at this size |
| Thread pools | `configure_cpu_training(num_threads)` | Sets intra- and inter-op pools; the inter-op pool is the one people forget |
| ISA dispatch | `ops/native/cpu/gemv.cpp` | AVX-512 / AVX2 / scalar chosen at runtime via cpuid, so the extension builds and runs without AVX-512 |
| Storage-bound kernels | `ops/native/cpu/index_linear.cpp`, `quant_attn.cpp` | Deliberately `-O3` scalar. Both read packed indices in place with no materialized fp32 matrix, so LUT gathers and memory set the pace, not FMA throughput |
| Zero-skip | `ops/sparse_index_linear.py` | A stored slot of exactly `0.0` is skipped, not accumulated; it sits on the hot path because the LC-QAT zero anchor makes `0.0` a common weight |
| Sparse training path | `nanochat/models/quant/sparseprop.py` | Measured crossover per layer: masked dense GEMM below 80% sparsity (the default regime), `nnz` walk above. See [docs/sparseprop.md](docs/sparseprop.md) |
| KV cache | `QuantizedKVCache` | K/V stay nibble-packed uint8 codebook indices, resolved through per-(layer, head) FP32 codebooks inside the kernel, so the cache never materializes as bf16 |

CPU work here is memory-bandwidth-bound, so FLOP counts mislead. The `nnz`-walking sparse kernel did 4x fewer multiply-accumulates than the dense GEMM it replaced and still lost by an order of magnitude: a per-`nnz` gather of a batch-float row cannot vectorize the way a packed GEMM micro-kernel does. The index-linear kernels stay scalar for the same reason. With weights read in place in packed form, no arithmetic exists to hide gathers behind.

### Example: a CPU run end to end

The default configuration (LC-QAT + SparseProp + DiffusionBlocks on) on the 4-core box, d4/`n_embd` 256, seq 256, 200 steps. One command, about 70 seconds:

```bash
python -m scripts.base_train \
    --depth=4 --head-dim=64 --window-pattern L \
    --max-seq-len=256 --device-batch-size=2 --total-batch-size=512 \
    --num-iterations=200 --db-blocks=2 --run=dummy \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1
```

`--run=dummy` keeps wandb offline. The four trailing flags disable eval, CORE, sampling, and checkpointing for a smoke run; keep `--sample-every` on to see the samples below.

The startup banner confirms all three stages engaged. Check the codebook table, the exact-zero count, and the block count:

```
Learned activation LUTs attached: 4
SparseProp injected 24 sparse Linear layers; 0.7500 of their weights are exact zeros
LC-QAT enabled: {'Kw=3(m1,p1),Ka=15(m6,p8)': 8, 'Kw=15(m6,p8),Ka=15(m6,p8)': 12, 'Kw=15(m6,p8),Ka=8(m0,p7)': 4}
Initialized DiffusionBlocks Engine with 2 independent blocks
LC-QAT retrofitted 6 diffusion-engine Linear layers
Parameter counts:
transformer_matrices    : 3,146,080
codebooks               : 744
total                   : 36,701,290
```

The loop runs at 1,200-2,000 tok/s on four cores:

```
step 00000/00200 (0.00%) | loss: 0.061637 | lrm: 0.01 | dt: 981.47ms | tok/sec: 521 | epoch: 1
step 00020/00200 (10.00%) | loss: 0.856505 | lrm: 0.51 | dt: 682.23ms | tok/sec: 750 | epoch: 1
step 00199/00200 (99.50%) | loss: 0.109227 | lrm: 0.06 | dt: 757.69ms | tok/sec: 675 | epoch: 1
```

Per-step loss jumps by design: sigma is resampled each step and `w(sigma)` reweights it, which is the EDM objective working as specified, not instability (see [docs/diffusionblocks.md](docs/diffusionblocks.md)). The overfit test pins the loss-decreases assertion; do not eyeball a three-step trace.

Two steps of the same 200 decode as prose:

```
<|bos|>The capital of France is ensemble comes leafy Dual OySep thereafter cyclone quartz
publish lingering wattageolor Slide Created Adams
```

That is what 102k tokens buys: the smallest run that shows the pipeline end to end with every default stage on. It carries no capability claim.

**One gotcha before copying the command.** `--total-batch-size` must be a multiple of `device-batch-size x max-seq-len x world-size`, or startup asserts:

```
AssertionError: total_batch_size (64) must be a multiple of 512.
```

CPU runs hit this often because the instinct is to keep both numbers small.

**Exporting.** Freeze a checkpoint into the quantized artifact with:

```bash
python -m scripts.export_lcqat --source base --model-tag d4exp --step 40
# Exported LC-QAT artifact to exports/lcqat_base_40.pt (30 quantized modules, 143.8 MiB)
```

That checkpoint comes from a 40-step run of the same command with `--save-every=40`. The 30 modules are the 24 transformer Linears plus the 6 DiffusionBlocks-engine Linears (4 adapters, 2 denoise heads); adapters and heads quantize too, and the artifact reloads as a `DiffusionBlockEngine` with all three subtrees still quantized. See [docs/kernels.md](docs/kernels.md#export).

Read 143.8 MiB against the 151.9 MiB fp32 source and compression looks weak. The parameter table explains it: at d4 the quantized `transformer_matrices` are 3.1M of 36.7M parameters, and the untouched embedding tables hold the other 91%. The win scales with depth, not parameter count. At GPT-2 scale the transformer dominates and the same code path compresses it hard; nobody has measured that here yet.

### Precision / dtype

lcqat-nanochat avoids `torch.amp.autocast` and manages precision through one global, `COMPUTE_DTYPE` (defined in `nanochat/models/dtype.py`; every `Linear` casts to it in forward). It auto-detects from hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default. Recent macOS runs `NANOCHAT_DTYPE=bfloat16` fine (~25% less memory, similar speed) |

Override with `NANOCHAT_DTYPE`:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

Weights stay in fp32 for optimizer precision; the custom `Linear` casts to `COMPUTE_DTYPE` in forward. Embeddings store directly in `COMPUTE_DTYPE` to save memory. The result matches autocast mixed-precision benefits with explicit control over which precision each op uses.

`float16` training enables a `GradScaler` in `base_train.py` against gradient underflow. bf16 and fp32 skip it; bf16 shares fp32's exponent range. fp16 inference works everywhere.

---

## What's in the box

Four mechanisms ship on by default and compose:

- **LC-QAT** equips each retrofitted `Linear` with an asymmetric learned codebook and an exact zero anchor, so weights compress and pruned weights still dequantize to exactly `0.0`. The packed artifact stores codebook indices, not fp32 matrices.
- **SparseProp** prunes 75% of weights by magnitude and routes the backward pass over the survivors.
- **DiffusionBlocks** partitions depth into blocks and trains one per optimizer step, which fits training into a small memory budget.
- **DARD** mines algebraic rules from the corpus offline, compiles them into a differentiable auxiliary loss, and surfaces rule violations as gradient instead of relearning them statistically each step.
- **Fused CPU kernels** read those packed indices in place at decode time.

Two opt-in quantization / execution stages:

- **4-bit QAT** (`--qat-4bit`, mutually exclusive with `--lcqat`) retrofits every eligible `Linear` to `FourBitLinear`: an fp32 master weight plus a packed-nibble (uint8, two weights per byte) mirror that the straight-through-estimator forward reads, refreshed after each step. All weights are real 4 bits at inference; the fp32 master exists only for the optimizer update. See [dev/FOURBIT_QAT_RESULTS.md](dev/FOURBIT_QAT_RESULTS.md).
- **Parallel DiffusionBlocks** (`--db-block-execution parallel`, MeZO-only) runs every block's denoiser concurrently over the same `[clean | noisy]` input and sums their EDM losses, so all B blocks get a gradient signal each step instead of 1/B. The memory-traffic trade: sequential (default) holds O(L/B) activation memory; parallel holds O(L) transiently, which is only safe under MeZo_MinExpLstm_Optimizer because its SPSA estimate consumes the loss *value* and never backprops the positive-point graph. On CPU the concurrent blocks overlap because torch releases the GIL inside matmuls. See [docs/diffusionblocks.md](docs/diffusionblocks.md#sequential-vs-parallel-block-execution).

One reference page per mechanism:

| | |
|---|---|
| [docs/quantization.md](docs/quantization.md) | Codebook mechanics, full flag table, KD anchoring, EfQAT freezing and per-block latching, denoiser distillation, sigma-conditioned codebooks, per-channel tables, activation LUTs |
| [docs/kernels.md](docs/kernels.md) | Mul-less GEMV, index-linear and sparse-CSR kernels, quantized export, the three benchmark scripts |
| [docs/sparseprop.md](docs/sparseprop.md) | Sparsity flags, why the training path uses a masked dense GEMM instead of an `nnz` walk, the four defect fixes, measured cost |
| [docs/diffusionblocks.md](docs/diffusionblocks.md) | Block-wise engine, versioned block/noise map, compiled EDM loss kernel, clean-past conditioning, `--db-blocks=0`, why EDM and AR losses never compare |
| [docs/dard.md](docs/dard.md) | Offline AML rule mining, log-domain rule weighting, differentiable order-violation loss, how it attaches to a DiffusionBlocks objective without logits |
| [docs/architecture.md](docs/architecture.md) | Package layout and the functional-core boundary |

The LFM2 backbone (`nanochat/models/lfm2.py`) and the 4-bit QAT stage are summarized under [What's in the box](#whats-in-the-box); the LFM2 + MeZO + parallel-DiffusionBlocks + 4-bit QAT run results live in [`dev/LFM2_MEZO_DB_RESULTS.md`](dev/LFM2_MEZO_DB_RESULTS.md).

The ablation harness (`scripts/lcqat_ablation.py`) tests nine claims from this repo so runs can falsify them instead of asserting them. It runs nine experiments and writes 20 rows into the generated table in [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md); the driver exits non-zero when an advertised claim fails. Each row is a micro-probe over random weights and a few SGD steps, not a training run, so none states an accuracy claim.

---

## Research sweeps

Two sweeps cover capability rather than mechanisms: [`runs/scaling_laws.sh`](runs/scaling_laws.sh) and [`runs/miniseries.sh`](runs/miniseries.sh). Both read depth-dependent flags from [`runs/lib_depth.sh`](runs/lib_depth.sh), so a threshold change lands once and the sweeps stay comparable. Upstream's [miniseries discussion](https://github.com/karpathy/nanochat/discussions/420) gives background.

Run history and negative results live in [`dev/LOG.md`](dev/LOG.md).

---

## Time-to-GPT-2 leaderboard

Development focuses on pretraining, which consumes the most compute. Following modded-nanogpt, nanochat tracks a "GPT-2 speedrun": wall-clock time to train a nanochat model past GPT-2-grade capability, measured by DCLM CORE score. [runs/speedrun.sh](runs/speedrun.sh) always holds the reference recipe. Current table:

> **This fork has not posted a speedrun row.** Each row below is upstream `@karpathy`'s, trained with the float (`--fp8`) recipe. The reference script here no longer produces that configuration by default (see the quantization note below). Treat "time to GPT-2" as an upstream metric, not a claim about this fork.

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

The primary metric is "time to GPT-2": wall-clock hours on an 8xH100 node to beat the GPT-2 (1.6B) CORE score of 0.256525. GPT-2 cost about $43,000 to train in 2019; the same grade now takes about 2 hours, or about $48 at ~$3/GPU/hr ($24/hr for the node).

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for interpretation and contribution rules.

> **Note on quantization defaults:** this fork enables LC-QAT and SparseProp on by default (see [docs/quantization.md](docs/quantization.md)); the rows above used `--fp8` float training. `runs/speedrun.sh` now passes `--no-lcqat --no-sparseprop` to recover the float recipe and drops `--fp8`, which is CUDA-only and conflicts with LC-QAT. Re-adding GPU training means re-adding the CUDA extra (see [Install](#install)); that is why the speedrun is the one script that does not run out of the box.

---

## Development

Five of the six gates run from `.pre-commit-config.yaml` (`fail_fast: true`, canonical hook repos pinned to immutable dot-tags), so `pre-commit run --all-files` covers them before push:

```bash
uv run ruff check .                      # lint
uv run ruff format --check .             # format
uv run pyrefly check                     # types
uv run pytest                            # tests
uv run aislop scan .                     # code-quality scan
```

The sixth, `uv lock --check`, is not a hook; run it by hand.

`pytest` is the gate contributors run most while iterating. It needs a C++ compiler on `PATH` (ops tests JIT `ops/native/cpu/`) and, for environment-touching tests, a trained tokenizer plus two flat ClimbMix shards in `~/.cache/base_data_climbmix`. The last shard in that set is the validation split.

`[tool.pytest.ini_options] pythonpath = ["."]` in `pyproject.toml` makes bare `pytest` work from the repo root; it is load-bearing, not cosmetic.

---

## Contributing

lcqat-nanochat targets micro-models a single researcher can train end to end for under $1,000. Accessibility means low cost and low cognitive load: no giant configuration objects, no model factories, no branching mazes. The codebase stays a single cohesive baseline you can read, fork, and run start to end to get a ChatGPT you can talk to.

In this fork the budget question is mostly "what is the cheapest machine this runs on?", so the quantization story carries the work: LC-QAT codebooks, SparseProp, block-wise DiffusionBlocks training, and mul-less CPU kernels, composed by default, so a capable model fits a small footprint and decodes on CPU without materializing fp32 weights. Two rules follow from taking CPU seriously:

- **Measure memory bandwidth, not FLOPs.** Several plausible optimizations here lowered op counts and worsened access patterns. Both kernel microbenchmarks check against a naive oracle first, because a measured speedup from a kernel computing the wrong function is worse than no measurement.
- **A path faster than the one it replaced may skip work.** One SparseProp path ran about 3x cheaper than the dense path because it had silently dropped the straight-through relaxation; it looked like a win until both arms proved they compute the same function.

Repo-specific guardrails live in [AGENTS.md](AGENTS.md): the functional-core boundary, the no-silent-backend-fallback rule, and which flags stay always-on versus opt-in.

AI policy: disclose. When you submit a PR, declare parts with substantial LLM contribution that you have not written or do not fully understand.

**For the record**: I do use AI often. For repetitive code tasks or cleared scope changes/additions, I write a PRD first, think it through, and then use AI to write much of the code. And for both the AI code and the code I do write, I ask AI to write the commit messages, because I am too lazy to write them myself.

---

## Acknowledgements

- This repo forks [karpathy/nanochat](https://github.com/karpathy/nanochat); the name (*nanochat*) derives from Andrej Karpathy's earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which covered pretraining only.
- lcqat-nanochat also follows [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified nanoGPT with metrics and a leaderboard, and reuses several pretraining ideas and parts of the implementation.
- LC-QAT (learned codebook quantization), SparseProp sparse backprop, DiffusionBlocks, the mul-less GEMV kernels, and this fork are the work of [Gabz4200](https://github.com/Gabz4200) and contributors.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for development compute.
- Thank you to chief LLM whisperer Alec Radford for advice and guidance.
- Thank you to repo czar Sofie [@svlandeg](https://github.com/svlandeg) for managing issues, pull requests, and discussions of nanochat.

---

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
  title = {lcqat-nanochat: LC-QAT, SparseProp, and DiffusionBlocks},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Gabz4200/LCQAT_nanochat}
}
```

---

## License

MIT

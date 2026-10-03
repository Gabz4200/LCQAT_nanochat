# lcqat-nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

**lcqat-nanochat** is a minimal, hackable experimental harness for training LLMs. It is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) retargeted at a narrower question: how small can a model get before it stops being a model? Concretely, how much of a capability model can be trained, stored, and decoded on a commodity CPU rather than a GPU node?

**That end-to-end number does not exist yet.** No capability model has been trained and measured on CPU in this fork. What is here is the machinery for it, built and benchmarked one piece at a time: LC-QAT codebooks that shrink the weights, SparseProp sparsity that removes most of them, DiffusionBlocks training that fits what is left into a small memory budget, and kernels that read packed indices in place instead of materializing fp32 matrices. All four ship **on by default**.

Every number in this README comes from a toy model. The end-to-end sweeps run at d4 or d6, and the largest trained checkpoint on the development host is a d6 at step 2. Read the measurements below as evidence about mechanisms and about how they interact, not as evidence about a capability model. Pretraining one on CPU is the result still outstanding.

**CPU is the target.** `pyproject.toml` pins torch from the PyTorch CPU index and the CUDA extra was removed outright. The GPU path still works (`torchrun`, the distributed dataloader, `--fp8`), but the defaults are tuned for the CPU case, and the two engineering facts that shaped the most code here are both about memory bandwidth rather than FLOPs. See [Running on CPU](#running-on-cpu). The measurements here were taken on a 4-core box with 7.6 GB of RAM.

The rest of the harness is unchanged: a single node, all major LLM stages (tokenization, pretraining, finetuning, evaluation, inference), and one complexity dial. `--depth` automatically determines every other hyperparameter (width, heads, learning rate schedule, training horizon, weight decay, …) so each model comes out compute-optimal.

Upstream's headline result still stands as context: a GPT-2 capability model (~4e19 FLOPs) that cost ~$43,000 to train in 2019 can be trained in ~1.5 hours for ~$48 on an 8×H100 node. That is the [Time-to-GPT-2 leaderboard](#time-to-gpt-2-leaderboard), and every row in it is upstream's — this fork has not posted one. What this fork works on is the second half: making the result cheap to run.

> **This is a fork, not the upstream repo.** Upstream discussion live links (DeepWiki, Discord, Discussions) point at `karpathy/nanochat`. Fork-specific work and the LC-QAT/SparseProp/quantization experiments live in [Gabz4200/LCQAT_nanochat](https://github.com/Gabz4200/LCQAT_nanochat).

**TL;DR** — the CPU path is the supported one:

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

- [Getting started](#getting-started)
  - [Setup](#setup)
  - [Reproduce and talk to GPT-2 (GPU)](#reproduce-and-talk-to-gpt-2-gpu)
- [Stages](#stages)
- [Running on CPU](#running-on-cpu)
- [Example: a CPU run end to end](#example-a-cpu-run-end-to-end)
- [Precision / dtype](#precision--dtype)
- [What's in the box](#whats-in-the-box)
- [Research](#research)
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
[DiffusionBlocks](docs/diffusionblocks.md) and
[architecture](docs/architecture.md).

---

## Getting started

### Setup

lcqat-nanochat uses [uv](https://docs.astral.sh/uv/) for dependency management. **This fork ships CPU-only PyTorch wheels**: torch is declared under the `cpu` extra with an explicit CPU index, so a bare `uv sync` deliberately installs *no* torch and `import nanochat.models.backbone` fails until you ask for it. That is the intended signal, not a packaging bug.

```bash
uv sync --extra cpu    # CPU wheels. The CUDA extra was dropped for the LC-QAT
                       # runtime; pyproject.toml has a comment explaining how to
                       # re-add a GPU extra + CUDA index if you need one.
source .venv/bin/activate
```

For development (adds pytest, pyrefly, ruff, taichi, aislop, datasets, huggingface-hub, matplotlib, ipykernel):

```bash
uv sync --extra cpu --group dev
```

Two warnings specific to this repo. `uv add` resolves torch from PyPI and will pull a CUDA build, so follow any dependency change with `uv sync --extra cpu --group dev` to get back to the pinned CPU wheel. And the LC-QAT ops parity tests JIT a C++ kernel, so a compiler (`g++`/`clang++`) must be on `PATH` for the suite to pass; the build is cached per user under `~/.cache/torch_extensions`.

### Reproduce and talk to GPT-2 (GPU)

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

- This fork trains with **LC-QAT + SparseProp + DiffusionBlocks always-on by default**. To reproduce the historical float/fp8 speedrun, the reference script passes `--no-lcqat --no-sparseprop` (and drops `--fp8`, which requires CUDA and conflicts with LC-QAT's default-on state). It does **not** pass `--db-blocks=0`, so the pretraining stage still runs through the DiffusionBlocks engine; add that flag for a conventional autoregressive baseline.
- The code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device-batch-size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

---

## Stages

lcqat-nanochat is a single cohesive pipeline, not a configurable framework: there are no giant config objects, model factories, or if-then-else monsters. The entry points live in `scripts/` and all share the global `COMPUTE_DTYPE` and the `--depth` complexity dial:

| Stage | Entry point | Description |
|-------|-------------|-------------|
| Tokenizer | `scripts/tok_train.py` | Train BPE tokenizer (vocab 2**15 = 32768) |
| Tokenizer eval | `scripts/tok_eval.py` | Report compression ratio, vocab coverage |
| Pretrain | `scripts/base_train.py` | Block-wise (DiffusionBlocks) pretraining; LC-QAT + SparseProp default on |
| Base eval | `scripts/base_eval.py` | CORE metric, bits-per-byte, sampling |
| SFT | `scripts/chat_sft.py` | Supervised finetune on the DiffEngine; default on LC-QAT/SparseProp |
| RL | `scripts/chat_rl.py` | PPO-style RL finetune (LC-QAT/SparseProp default on) |
| Chat eval | `scripts/chat_eval.py` | ChatCORE score for an SFT/RL checkpoint |
| Export | `scripts/export_lcqat.py` | Freeze a LC-QAT checkpoint into a stripped quantized inference artifact |
| Chat | `scripts/chat_cli.py` | Talk to a trained model over CLI |

`speedrun.sh` runs the first six back to back. Three other tools in `scripts/` are not stages: the ablation harness, the kernel microbenchmarks, and the inference benchmark. The first is summarized under [What's in the box](#whats-in-the-box); the benchmarks are in [docs/kernels.md](docs/kernels.md#benchmarks).

---

## Running on CPU

`pyproject.toml` pins torch from the PyTorch CPU index, the CUDA extra was removed outright, and most of the engineering here went into making a quantized model small and fast enough to run on a commodity box. The GPU path still works: `torchrun`, the distributed dataloader, and the `--fp8` float recipe are all intact. The defaults are just tuned for the CPU case.

**Quickstart.** [runs/runcpu.sh](runs/runcpu.sh) exercises every stage end to end on CPU or Apple Silicon: dataset download, tokenizer, d6 pretraining, SFT. It shrinks the model to fit a reasonable time interval (~30 minutes of training on an M3 Max). You will not get strong results this way. Treat it as a demo run and a smoke test that the whole pipeline is wired.

```bash
bash runs/runcpu.sh
```

It uses `--db-blocks=2` rather than the default 4, deliberately: on a toy depth the per-block overhead is a large fraction of the step, and 2 keeps the block-wise path exercised without dominating the run.

**What is CPU-specific in the code:**

| Area | Where | Note |
|---|---|---|
| Fused AdamW | `nanochat/training/optim.py`, `cpu_adamw_for` | `fused=True`. On CPU a non-fused AdamW walks the parameter list in Python, which is a real cost at this model size |
| Thread pools | `configure_cpu_training(num_threads)` | Sets intra- **and** inter-op pools; the inter-op one is the one that gets forgotten |
| ISA dispatch | `ops/native/cpu/gemv.cpp` | AVX-512 / AVX2 / scalar chosen at runtime via cpuid, so the extension builds and runs on machines with no AVX-512 |
| Storage-bound kernels | `ops/native/cpu/index_linear.cpp`, `quant_attn.cpp` | Deliberately `-O3` scalar. Both read packed indices in place with no materialized fp32 matrix, so the bottleneck is memory and LUT gathers rather than FMA throughput |
| Zero-skip | `ops/sparse_index_linear.py` | A stored slot whose value is exactly `0.0` is skipped, not accumulated — on the hot path because LC-QAT's zero anchor makes `0.0` a real weight |
| Sparse training path | `nanochat/models/quant/sparseprop.py` | Masked dense GEMM, *not* an nnz walk — see [docs/sparseprop.md](docs/sparseprop.md) |
| KV cache | `QuantizedKVCache` | K/V stored as nibble-packed uint8 codebook indices, resolved through per-(layer, head) FP32 codebooks *inside* the kernel, so the cache never materializes as bf16 |

**CPU is memory-bandwidth-bound, so FLOP reductions are the wrong thing to optimize for.** Two of the findings in this README come down to that. The nnz-walking sparse kernel does 4× *fewer* multiply-accumulates than the dense GEMM it replaced and still lost by an order of magnitude, because a per-nnz gather of a batch-float row cannot vectorize the way a packed GEMM micro-kernel does. And the index-linear kernels are scalar on purpose: with the weights read in place in packed form there is no arithmetic to hide the gathers behind.

---

## Example: a CPU run end to end

Real output from the default configuration (LC-QAT + SparseProp + DiffusionBlocks all on) on the 4-core box, d4/`n_embd` 256, seq 256, 200 steps. The whole run is one command and takes about 70 seconds:

```bash
python -m scripts.base_train \
    --depth=4 --head-dim=64 --window-pattern L \
    --max-seq-len=256 --device-batch-size=2 --total-batch-size=512 \
    --num-iterations=200 --db-blocks=2 --run=dummy \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1
```

`--run=dummy` keeps wandb offline. The four trailing flags turn off eval, CORE, sampling and checkpointing, which is what you want for a smoke run; leave `--sample-every` on if you want the samples shown below.

The startup banner is the fastest way to confirm the three stages actually engaged — the codebook table, the exact-zero count, and the block count all print:

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

and the loop itself, at 1,200–2,000 tok/s on four cores:

```
step 00000/00200 (0.00%) | loss: 0.466746 | lrm: 0.01 | dt: 376.55ms | tok/sec: 1,359 | epoch: 1
step 00020/00200 (10.00%) | loss: 0.414501 | lrm: 0.51 | dt: 309.57ms | tok/sec: 1,653 | epoch: 1
step 00199/00200 (99.50%) | loss: 0.304096 | lrm: 0.06 | dt: 409.06ms | tok/sec: 1,251 | epoch: 1
```

Two steps of the same 200 come out as prose:

```
<|bos|>The capital of France is ensemble comes leafy Dual OySep thereafter cyclone quartz
publish lingering wattageolor Slide Created Adams
```

That is what 102k tokens buys. Nothing here is a capability claim; it is the smallest run that shows the pipeline running end to end with every default stage engaged.

**One gotcha before you copy the command.** `--total-batch-size` must be a multiple of `device-batch-size × max-seq-len × world-size`, or startup asserts:

```
AssertionError: total_batch_size (64) must be a multiple of 512.
```

It is easy to hit on CPU, where the natural instinct is to keep both numbers small.

**Exporting.** Freeze a checkpoint into the quantized artifact with:

```bash
python -m scripts.export_lcqat --source base --model-tag d4lm --step 40
# Exported LC-QAT artifact to exports/lcqat_base_40.pt (24 quantized modules, 141.9 MiB)
```

Read that 141.9 MiB against the 149.2 MiB fp32 checkpoint it came from and the compression looks unimpressive. It is not, and the parameter table above says why: at d4 the quantized `transformer_matrices` are 3.1M of 36.7M parameters, and the other 91% is the embedding tables, which this format does not touch. The win scales with depth, not with parameter count. At GPT-2 scale the transformer is the overwhelming majority of the model and the same code path compresses it hard; nobody has measured that here yet.

---

## Precision / dtype

lcqat-nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/models/dtype.py`, since every `Linear` casts to it in forward). By default this is auto-detected based on your hardware:

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

---

## What's in the box

Four mechanisms ship **on by default**, and they compose:

- **LC-QAT** gives every retrofitted `Linear` an asymmetric learned codebook with an
  exact zero anchor, so weights compress and a pruned weight still dequantizes to
  exactly `0.0`. Terse on the memory story: the packed artifact holds codebook
  indices, not fp32 matrices.
- **SparseProp** prunes 75% of weights by magnitude and routes the backward pass
  over the survivors.
- **DiffusionBlocks** partitions depth into blocks and trains one per optimizer
  step, which is what makes the whole thing fit in a small memory budget.
- **Fused CPU kernels** read those packed indices in place at decode time.

Reference pages, one per mechanism:

| | |
|---|---|
| [docs/quantization.md](docs/quantization.md) | Codebook mechanics, the full flag table, KD anchoring, EfQAT freezing and per-block latching, denoiser distillation, sigma-conditioned codebooks, per-channel tables, activation LUTs |
| [docs/kernels.md](docs/kernels.md) | Mul-less GEMV, index-linear and sparse-CSR kernels, quantized export, the three benchmark scripts |
| [docs/sparseprop.md](docs/sparseprop.md) | Sparsity flags, why the training path is a masked dense GEMM rather than an `nnz` walk, the four defect fixes, measured cost |
| [docs/diffusionblocks.md](docs/diffusionblocks.md) | The block-wise engine, `--db-blocks=0`, and the two caveats about the EDM default |
| [docs/architecture.md](docs/architecture.md) | Package layout and the functional-core boundary |

The ablation harness (`scripts/lcqat_ablation.py`) measures nine claims this repo
makes so they can be falsified rather than asserted. It runs nine experiments and
writes 20 rows into the generated table in
[`dev/LEADERBOARD.md`](dev/LEADERBOARD.md); the driver exits non-zero when a claim
it advertises comes out wrong. **Every row is a micro-probe** — random weights, a
few SGD steps — not a training run, so none of them is an accuracy claim.

---

## Research

Two sweeps are worth knowing about if you are chasing capability rather than
mechanisms: [`runs/scaling_laws.sh`](runs/scaling_laws.sh) and
[`runs/miniseries.sh`](runs/miniseries.sh). Both derive their depth-dependent
flags from [`runs/lib_depth.sh`](runs/lib_depth.sh), so a threshold change is made
once and the two sweeps cannot silently drift out of comparability. Upstream's
[miniseries discussion](https://github.com/karpathy/nanochat/discussions/420)
has the background.

Run history and negative results are in [`dev/LOG.md`](dev/LOG.md).

---

## Time-to-GPT-2 leaderboard

Presently, the main focus of development is on tuning the pretraining stage, which takes the most amount of compute. Inspired by the modded-nanogpt repo and to incentivise progress and community collaboration, nanochat maintains a leaderboard for a "GPT-2 speedrun", which is the wall-clock time required to train a nanochat model to GPT-2 grade capability, as measured by the DCLM CORE score. The [runs/speedrun.sh](runs/speedrun.sh) script always reflects the reference way to train a GPT-2 grade model. The current leaderboard looks as follows:

> **This fork has not posted a speedrun row.** Every row below is upstream `@karpathy`'s, produced by the float (`--fp8`) recipe — see the note on quantization defaults at the end of this section for why the reference script here no longer produces that configuration by default. Treat "time to GPT-2" as an upstream metric rather than a claim about this fork.

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

> **Note on quantization defaults:** this fork enables **LC-QAT and SparseProp *always-on by default*** (see [docs/quantization.md](docs/quantization.md)); the historical speedrun rows above were produced with `--fp8` float training. `runs/speedrun.sh` has been updated to match — it now passes `--no-lcqat --no-sparseprop` to reproduce the float recipe, and drops `--fp8` entirely, because `--fp8` is CUDA-only and conflicts with LC-QAT. Re-adding GPU training means re-adding the CUDA extra (see [Setup](#setup)); that is why the speedrun is the one script here that does not run out of the box.

---

## Development

Five of the six gates run from `.pre-commit-config.yaml` (`fail_fast: true`, canonical hook repos pinned to immutable dot-tags), so `pre-commit run --all-files` is enough before you push:

```bash
uv run ruff check .                      # lint
uv run ruff format --check .             # format
uv run pyrefly check                     # types
uv run pytest                            # tests
uv run aislop scan .                     # code-quality scan
```

The sixth, `uv lock --check`, is not a hook and has to be run by hand.

`pytest` alone is the gate most contributors run while iterating. It needs a C++ compiler on `PATH` (the ops tests JIT `ops/native/cpu/`), and the environment-touching tests need a trained tokenizer plus two flat ClimbMix shards in `~/.cache/base_data_climbmix`. The last shard in that set is the validation split.

`[tool.pytest.ini_options] pythonpath = ["."]` in `pyproject.toml` is what makes a bare `pytest` work from the repo root; it is load-bearing, not cosmetic.

---

## Contributing

The goal of lcqat-nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - lcqat-nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to.

In this fork that budget question has mostly been answered "what is the cheapest machine this will run on?", so the interesting part is the quantization story: LC-QAT codebooks + SparseProp + block-wise DiffusionBlocks training + mul-less CPU kernels, all composed by default, so that a capability model fits in a small memory footprint and decodes on a CPU without materializing fp32 weight matrices. Two engineering rules follow from taking CPU seriously:

- **Measure memory bandwidth, not FLOPs.** Several plausible-looking optimizations here lost because they improved the operation count and worsened the access pattern. Both kernel microbenchmarks sanity-check against a naive oracle first, because a measured speedup from a kernel computing the wrong function is worse than no measurement.
- **A path that is faster than the thing it replaced may not be doing the work.** One SparseProp path was ~3× cheaper than the dense path precisely because it had silently dropped the straight-through relaxation; it looked like a win until both arms were shown to compute the same function.

The full set of repo-specific guardrails lives in [AGENTS.md](AGENTS.md) — the functional-core boundary, the no-silent-backend-fallback rule, and which flags are always-on versus opt-in.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

---

## Acknowledgements

- This repo is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat); the name (*nanochat*) derives from Andrej Karpathy's earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- lcqat-nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- LC-QAT (learned codebook quantization), SparseProp sparse backprop, DiffusionBlocks, the mul-less GEMV kernels, and this fork are the work of [Gabz4200](https://github.com/Gabz4200) and contributors.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

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

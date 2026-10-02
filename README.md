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
| Quantized inference (after LC-QAT training) | `uv run python -m scripts.export_lcqat --source sft` |

---

## Table of contents

- [Time-to-GPT-2 leaderboard](#time-to-gpt-2-leaderboard)
- [Getting started](#getting-started)
- [Stages](#stages)
- [Quantization: LC-QAT, KD, EfQAT, activation LUTs](#quantization-lc-qat-kd-efqat-activation-luts)
- [SparseProp sparse backprop](#sparseprop-sparse-backprop)
- [Per-channel value-centered quantization](#per-channel-value-centered-quantization)
- [Ablation harness](#ablation-harness)
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
python -m scripts.base_train --depth=12 --no-lcqat              # plain float training
python -m scripts.base_train --lcqat-preset prd                 # PRD table: 8-bit down_proj
python -m scripts.base_train --codebook-grad-scale none         # disable the 1/sqrt(N) codebook scaling
python -m scripts.base_train --db-objective ce                 # next-token CE instead of the EDM objective
torchrun -m scripts.chat_sft -- --run=sft                      # LC-QAT + SparseProp on by default
python -m scripts.chat_rl --no-lcqat --no-sparseprop          # plain RL
```

| Flag | Meaning |
|------|---------|
| `--no-lcqat` | LC-QAT is **on by default**; this flag runs plain float training. There is no `--lcqat` flag |
| `--lcqat-preset asym` | **Default.** Same total level counts as `small`, but split by sign: the two MLP tensors that see `relu(x).square()` (>= 0) get `m_neg=0`, so `K=8` levels are all usable instead of 7 of 15 being dead |
| `--lcqat-preset small` | Symmetric max compression: q/k weights K=3 (mul-less ternary), everything else K=15 |
| `--lcqat-preset prd` | PRD table: `mlp.c_proj` (down_proj) at K=255/255, rest as small |
| `--lcqat-k-map substr:KW/KA,...` | Per-module overrides. Each K is a total level count (`15`) or an explicit split (`0-7`), e.g. `mlp.c_proj:0-7/0-7` |
| `--codebook-lr` | Codebook AdamW LR (PRD: 10–50× network weights), no weight decay |
| `--codebook-grad-scale {inv_sqrt_n,none}` | PRD 2.4 codebook gradient scaling. `inv_sqrt_n` (default) scales the codebook gradient by `1/sqrt(numel)`; with `N = B·T·D` in the millions this is ~1000× smaller, so it interacts multiplicatively with `--codebook-lr` |
| `--db-objective {edm,ce}` | DiffusionBlocks objective. `edm` (default) trains each block as a denoiser over its own noise range, so only L/B layers run and activations are `O(L/B)`. `ce` is full-depth next-token cross-entropy with block-isolated gradients (saves backward memory, no forward FLOPs). **`ce` is not the same as turning DiffusionBlocks off** — it still runs through the engine and still gradients one block per step. Use `--db-blocks=0` for a conventional baseline |
| `--db-blocks` | Number of independent diffusion blocks (checkpoint provenance: cannot change on resume). **`0` disables DiffusionBlocks entirely** and trains a plain autoregressive LM: no partitioner, no denoise heads, no block isolation, every layer trains every step |
| `--db-overlap` | Log-σ overlap between adjacent blocks (DiffusionBlocks App. C). 0.1 for text, 0.05 for vision |
| `--db-block-sampling {step,micro}` | Draw the active block once per optimizer step (default) or per micro-step. `micro` is **lossy** with gradient accumulation: `_apply_requires_grad` clears gradients the active block does not own, so each micro-step erases the previous one and only the last block sampled reaches the optimizer |
| `--fp8` | FP8 training for the float path. **Mutually exclusive with LC-QAT** (both convert `Linear`) |
| `--lcqat-channel-center` | Per-output-channel weight codebooks instead of one shared table (PRD 3.4), so channels with very different scales are not forced onto a compromise grid. Off by default; costs `out_features × K` levels and makes the layer **un-exportable**, so it cannot be combined with the fused inference path. Persisted in checkpoint meta, so a resume does not silently drop it |
| bias quantization | Opt-in bias codebook (`nanochat/models/quant/bias_quant.py`), set via `LayerKConfig.quantize_bias` / `LCQATLinear(quantize_bias=True)`. **Off by default and with no CLI flag yet**, so existing checkpoints and outputs are bit-identical. Measured cost and error in the [ablation table](#ablation-harness) |

Per-layer roles (`nanochat/models/quant/retrofit.py`): `attn.c_q`/`c_k` get ternary weights, Q/K/V **outputs** are quantized during training and the quantized KV-cache runtime is implemented (`Engine.generate(quantized_kv=True)`), `mlp.c_fc` output is quantized as the input side of the fused relu² LUT, `lm_head` and linears under 128 dims stay in floating point.

### Two things to know about the DiffusionBlocks default

- **`lm_head` is not trained by `--db-objective edm`.** The EDM objective trains a denoiser that predicts a clean *embedding* (`denoise_heads[b]`), never tokens, so `lm_head` has no gradient on that path. It is still used when sampling. This is a property of the objective, not a wiring bug — the loss has no logits to attach a gradient to. Use `--db-objective ce` if you need a token-level objective (and remember the two objectives must not be mixed within a run).
- **The per-step loss is jittery by design.** σ is resampled every step and `w(σ) = (σ²+σ_d²)/(σ·σ_d)²` reweights it, so the loss varies substantially between steps even with fixed weights. That is EDM working as specified, not instability.

**Activation LUTs.** Beyond weights, activations are quantized through fused lookup tables registered in an activation registry (`nanochat/models/quant/lut.py`). All common nonlinearities are covered: `relu2`, `silu`, `gelu`, `tanh`, `sigmoid`. The export step bakes these into static FP32 LUTs via `wire_activation_luts`.

**Knowledge Distillation (KD) anchoring.** High compression ratios compress the loss manifold into sharp local minima. LC-QAT anchors the student QAT optimization using a KL-divergence against a frozen, detached FP32/BF16 teacher (the pre-quantization model, or any other unquantized reference). The teacher is never optimized.

`L_KD = tau^2 * D_KL( softmax(Z_teacher / tau) || softmax(Z_student / tau) )`
`L_total = (1 - alpha) * L_CE(Y, Y_hat_quant) + alpha * L_KD`

Enabled via `--kd-alpha` (default 0.0 = off; ~0.1 typical), `--kd-teacher-source` and `--kd-teacher-tag` (the float checkpoint to load as teacher). Implemented in `nanochat/models/quant/kd.py` (`KDLoss`).

**EfQAT selective layer freezing.** To keep memory flat at scale, LC-QAT can selectively freeze middle-layer codebook deltas and weight gradients after a warm-up, keeping only "critical outlier layers" (input embedding projections, attention q/k, final output) trainable. Enabled with `--efqat-freeze-after N` (default -1 = off). Implemented in `nanochat/models/quant/efqat.py` (`SelectiveFreezer`). The optimizer simply skips params whose `.grad is None`, so momentum buffers are unaffected.

**EfQAT per-block permanent freezing.** `SelectiveFreezer` above freezes by *layer role* (a global middle-layer band). `--efqat-latch-blocks N --efqat-latch-after M` instead retires the `N` *highest-index* diffusion blocks at step `M`, permanently, via `BlockLatchFreezer`. This is the block-wise analogue: a block that has converged on its noise range is latched and its parameters stop receiving gradients, which flattens optimizer state as the block count grows.

Latched blocks are excluded from sampling (`DiffusionBlockEngine.live_blocks` / `sample_block`). This is load-bearing, not tidiness: the EDM objective precomputes `clean` under `no_grad`, so a block's own parameters are the only differentiable path through it — sampling a latched block yields a loss with `grad_fn is None` and `backward()` raises. The latch is written to checkpoint metadata and re-applied on resume.

**Denoiser distillation (PRD 2.5).** `--kd-denoiser-alpha A` adds a teacher term to the DiffusionBlocks objective, anchoring the quantized denoiser to its float twin on the *same* noisy input and noise level:

`L_KD = w(sigma) * || D_quant(x_t, sigma) - D_float(x_t, sigma) ||^2`

Sharing `(x_t, sigma)` is what isolates the anchor to quantization error; a different noise draw would measure two different problems. The float twin is a deepcopy of the model taken *before* LC-QAT retrofit, with SparseProp's forward stripped so the teacher is genuinely dense. `--kd-denoiser-alpha 0` (default) is off. Implemented in `nanochat/models/quant/kd.py` (`DenoiserDistiller`).

**Sigma-conditioned codebooks (PRD 3.2).** A DiffusionBlocks engine partitions sigma into disjoint equi-probability ranges and trains one block per range, so a single activation codebook has to span every noise level and spends most of its levels on values that never occur. Two mechanisms, both opt-in:

- `--db-sigma-codebook conditioned` (`SigmaConditionedCodebook`) — one codebook per sigma anchor, selected by a hard log-space bucket. Each block gets a codebook tuned to its own noise range. Costs `num_anchors ×` the codebook parameters and the inference LUT; `--db-sigma-anchors` defaults to `--db-blocks`.
- `--db-sigma-codebook modulated` (`SigmaModulatedCodebook`) — one codebook scaled by a learned positive gain on `log(sigma)`. Same artifact size as the unconditional codebook, so it is the option that does not change the storage contract.

The modulation is a *gain* rather than a shift because a shift cannot satisfy both of the contracts the level table has to honour: the exact zero anchor (SparseProp prunes weights to `0.0` and the CPU kernels skip on `w == 0.0`; a shifted anchor dequantized a pruned weight to 0.014 / 0.19 / 12.0 across three noise levels) and strict monotonicity (pinning the anchor back to `0.0` after a shift of 3.0 pushed the top negative level to 2.857, past the anchor). A positive gain satisfies both by construction.

Both variants initialize to *exactly* the unconditional codebook (`gain == 1.0`, identical anchor tables), so enabling one does not perturb step 0. Neither composes with the fused export path — a conditioned codebook needs one LUT per noise level and the exported `activation_lut` is a single static table — so the combination raises at the first fused forward rather than silently quantizing with the wrong levels.

#### Per-channel value-centered quantization

`PerChannelValueCenteredQuantizer` (`nanochat/models/quant/per_channel.py`) gives each output channel its own asymmetric codebook, instead of one alphabet shared across channels. The mechanism is the scalar `ValueCenteredQuantizationLUT` decomposition extended with a per-channel axis, so the exact-zero anchor holds per channel: `x == center_c` reconstructs to exactly `center_c`, and with the default zero centres that is exactly `0.0`, which is the SparseProp structural-zero contract.

Two things about using it:

- **Call `init_from_tensor(x)` before training.** A hand-supplied `init_max` that overshoots the data fails silently and permanently: every value bucketizes onto the zero anchor, the anchor is the only level ever gathered, so the codebook parameters receive exactly zero gradient and the table never moves. Measured with `init_max=100` on data of magnitude 8 — NMSE stayed at exactly 1.0 for 400 steps on every channel. Fitting the outer levels to the data's own extremes sidesteps it.
- **It costs `C × K` levels**, versus `K` for a shared codebook. That is a table-size change, not a rounding change, so it is opt-in and the fused inference paths do not consume it.

What is measured: given one channel 100× larger than the others, a fitted shared codebook flattens the small channels to NMSE ≈ 1.0 (they round entirely onto the anchor), while per-channel rescues one to ≈ 0.4. The effect is **per channel, not on the pooled mean** — the pooled mean is dominated by the large channel, where the two are near-identical, so a mean-based comparison re-tests the big channel and hides the whole effect. It is also the only loss under which a per-channel table is the right tool: under a pooled MSE the gradient is dominated by the largest channel and the small channels' tables never move. No end-to-end accuracy claim is made; the tests in `tests/test_lcqat_per_channel.py` establish the mechanism, not a training win.

#### Mul-less GEMV + index-linear kernels

For quantized decode (`K_W = 3` ternary), the runtime routes through a backend dispatcher:

- `naive` — pure PyTorch oracle (`nanochat/ops/references/`), for parity tests and debugging;
- `cpu` — C++ kernel under `nanochat/ops/native/cpu/` (`gemv.cpp`, `index_linear.cpp`, `sparseprop.cpp`), JIT-built by torch on first use (needs `g++`/`clang++`; build cached under `~/.cache/torch_extensions`);
- `gpu` — Taichi kernel (`nanochat/ops/kernels/gpu_loader.py`), loaded lazily (needs `taichi` + a Vulkan ICD).

`dispatch_index_linear` extends the same pattern to arbitrary storage widths `K ∈ {3, 15, 255, 257}` (see `nanochat/ops/index_linear.py`). A requested backend that cannot run raises — it never silently falls back to naive. Three-way parity (`naive == cpu == gpu`), `torch.library.opcheck`, and `torch.compile` composition are covered by `tests/test_lcqat_ops.py`.

#### Export

```bash
uv run python -m scripts.export_lcqat --source sft --out exports/lcqat_sft.pt
```

Freezes codebooks into static FP32 LUTs, replaces FP32 weight matrices with `uint8` index buffers, and saves a minimal artifact for the quantized inference runtime (not resumable for training). After export, `load_model` auto-detects the LC-QAT state from the artifact.

### SparseProp sparse backprop

[SparseProp](https://arxiv.org/abs/2408.08525) injects unstructured sparsity into the backward pass: a static sparsity mask (default `--sparseprop-sparsity=0.75`, fraction of weights pruned) is applied to `Linear` layers, and the backward pass routes through AVX2 C++ kernels (`nanochat/ops/native/cpu/sparseprop.cpp`) that compute gradients over `nnz` entries only — `O(nnz)` instead of `O(M·K)`. Forward is standard masked SpMM; backward is the mul-less/sparse win.

SparseProp is **always-on by default** alongside LC-QAT. When LC-QAT is active, LCQATLinear is wrapped as `SparsePropLinearLCQAT` so the two coexist in one module (`nanochat/models/quant/sparseprop.py`). Disable with `--no-sparseprop`.

| Flag | Meaning |
|------|---------|
| `--no-sparseprop` | Disable SparseProp (default: on) |
| `--sparseprop-sparsity` | Sparsity level 0.0–1.0 (default `0.75`) |
| `--sparseprop-scope {layer,global}` | Rank magnitudes within each layer, or across all prunable layers (SparseProp Global-GMP) |
| `--sparseprop-start-frac` | When gradual pruning begins, as a fraction of total steps |
| `--sparseprop-every` | Prune every N steps; `0` (default) disables gradual pruning entirely |
| `--sparseprop-ramp-steps` | Steps over which the target ramps from 0 to the configured sparsity |
| `--sparseprop-dense-threshold` | Layers below this many parameters are left dense (CSR bookkeeping is not worth it) |

Parity (`naive == cpu`) and integration with the full training pipeline are in `tests/test_sparseprop.py` and `tests/test_sparseprop_integration.py`; pruning behaviour in `tests/test_sparseprop_pruning.py`.

**SparseProp's training path is a masked dense GEMM, not an nnz walk.** The `AVX2` CSR/CSC kernels in `sparseprop.cpp` are retained for parity tests and as the reference oracle, but they are no longer on the training hot path. Both the forward and the backward of `SparsePropLinearFunction` now compute a dense matmul over the pruned weight:

```python
out = x_flat @ weight.t()  # forward
grad_x = grad_y @ weight  # backward, weight already holds exact zeros
grad_w = (grad_y.t() @ x_flat) * mask  # backward, masked back onto the sparse pattern
```

This is exact, not an approximation: pruned slots hold the *exact* zero anchor, so the dense product sums the same surviving terms as the SpMM. Measured agreement is ~1e-6 relative — fp32 summation-order noise. Pruned slots stay exactly `0.0`, which is the contract the sparse export and the mul-less kernels read.

The reason is arithmetic intensity, not arithmetic count. At `out=256, in=1024, batch=2048, sparsity=0.75` the nnz-walking kernel does **4x fewer** multiply-accumulates than the GEMM and still lost by an order of magnitude, because a per-nnz gather of a `batch`-float row cannot be vectorized the way a packed GEMM micro-kernel is:

| | forward | backward |
|---|---|---|
| AVX2 `nnz` walk | 80.9 ms | 73.9 ms |
| masked dense GEMM | 7.4 ms | 15.6 ms |
| speedup | **10.9x** | **4.7x** |

Pruning still removes the memory, not the FLOPs: the win is in the exported/packed artifact and the inference kernels, not in the training step.

**The training step is now essentially free.** Three defects were found and fixed; end-to-end layer timing (dense LC-QAT vs LC-QAT + SparseProp at 75% sparsity) went from **4.27x slower to 1.01x — parity**:

1. The `nnz`-walking forward and backward kernels (above).
2. `weight.detach().view(-1)[csr_gidx]` — a *differentiable* gather, so autograd recorded an `IndexBackward0` whose backward scattered gradients back over the whole dense buffer via `index_put`, at 301 ms of a 532 ms step.
3. **A real correctness bug, not just a slowdown.** `SparsePropLinearLCQAT` re-parents the LC-QAT quantizers but does not inherit `LCQATLinear`'s methods, so calling `weight_quantizer(self.weight)` directly bypassed `_quantize`. That (a) reintroduced the same `IndexBackward0` `index_put`, this time at 486 ms — over half the step — and (b) silently skipped the PRD 2.4 `inv_sqrt_n` gradient scale, so a SparseProp layer's codebook received gradients `sqrt(numel)` larger than the same layer un-sparsified. Both are fixed by giving the wrapper the same `_quantize` helper the dense layer has.

**A fourth bug: `out_quantizer` was never applied.** `__init__` re-parents `out_quantizer` (it must, or `c_q`/`c_k`/`c_v` and `c_fc` lose it and the KV-cache quantization path dies), but the forward never called it. Every SparseProp run trained the 24 layers that carry an `out_quantizer` with an **unquantized** output, unlike the dense layer. `SparsePropLinearLCQAT.forward` now mirrors `LCQATLinear.forward` and quantizes its output. This is guarded by tests that assert the output values lie on the codebook's levels and that the codebook receives a gradient — an idempotence check would *not* have caught it.

### Measured: SparseProp's end-to-end cost

The four-combination comparison (LC-QAT, LC-QAT+DiffusionBlocks, LC-QAT+SparseProp,
and all three) is written up in [`dev/STACK_COMPARE.md`](dev/STACK_COMPARE.md),
generated by [`runs/stackcompare.sh`](runs/stackcompare.sh). That file also states
which of the four pairwise comparisons isolate a single mechanism and which do not,
and what these runs cannot establish.

`runs/stackcompare.sh`, four arms, depth 6, identical FLOP estimate and parameter count across arms:

| pair | before | after |
|---|---|---|
| plain LM, +/- SparseProp | **+34.7%** (17791 -> 23974 ms) | **-9.8%** (17600 -> 15874 ms) |
| DiffusionBlocks, +/- SparseProp | **+59.0%** (5750 -> 9145 ms) | **-14.7%** (5719 -> 4876 ms) |

**On the residual: in-process micro-benchmarks say parity, the full trainer says
SparseProp is faster.** Both were run; the gap between them is documented rather
than smoothed over. Measured on this host, all at the sweep's own shapes
(depth 6, n_embd 384, vocab 32768, device-batch 4, seq 512, 2 grad-accum
micro-steps):

| scope | dense | sparse | delta |
|---|---|---|---|
| `base_train` step, 6 samples each | 17352-17949 ms | 15904-16348 ms | **-10%** |
| in-process fwd+bwd, one micro-step | 7534 ms | 7592 ms | +0.8% |
| in-process `AdamW(fused=True)` step alone | 251 ms | 245 ms | -2.4% |
| in-process backward onto existing grads | 7424 ms | 7733 ms | +4.2% |
| full step, 2 micro-steps, no dataloader | 17591 ms | 15839 ms | **-10%** |

The optimizer's parameter list is byte-identical between arms (same roles, same
shapes, same tensor count), so this is not the optimizer doing less work, and the
profiler agrees the op counts match (`mm` 111/111, `bucketize` 198/198, `index`
114/114) — SparseProp is not skipping work. The last row reproduces the trainer's
gap outside the trainer, which points at per-step host effects (allocator state,
thread scheduling) rather than at the kernels. It is not attributed to a mechanism
here, because attributing it to one would be a guess.

Whatever the cause, the claim this section has to support is the negative one:
SparseProp is **not** slower than not having it, on either pair. Removal is not
justified.

Validation bpb is unchanged by the optimization, as expected — these were performance and gradient-scale fixes, not modelling changes (plain LM: 2.365218 dense vs 2.364602 sparse).

**A fifth bug, found by chasing an anomaly: the learned activation LUT was dead on the SparseProp path.** After the four fixes above, the four-arm sweep showed SparseProp *faster* end to end (-20% plain LM, -37% denoiser) while every isolated measurement showed parity. The cause was `SparsePropLinearLCQAT.apply_trained_activation`, which re-implemented the gather as

```python
lut.resolved_table()[indices]  # wrong: hard values only
```

instead of calling the table's own forward. `LearnableIndexLut.forward` returns `soft + hard - soft.detach()` — the same forward value, but with the straight-through relaxation attached, so the gradient reaches `logits` / `relaxed()`. The manual gather returns `hard` alone, so **`--lcqat-lut-relaxation` and `--lcqat-act-body` silently did nothing on every SparseProp run** and the trained table received no gradient. It was also ~3x cheaper (14 ms vs 45 ms at `[4,512,1536]`), which is exactly why the sparse arm looked faster: it was skipping the `bucketize` plus relaxation work that `c_fc` legitimately owes.

Isolating one layer made it unambiguous: `c_fc` alone measured 164 ms dense vs 120 ms sparse, with identical op counts except two extra `aten::index` gathers in the dense arm. After the fix, the arms match within float32 noise, and reverting the fix makes **100% of activation elements differ** (max abs 12.0) rather than by rounding.

The lesson generalizes: a path can be *faster than the thing it replaces* precisely because it is not doing the work. Measured speedups are only trustworthy once both arms are shown to compute the same function.

Guarded by tests asserting the sparse and dense activations agree, that the LUT receives a gradient, and that injection does not drop the table.

Pruning is **magnitude-based per row**, not random: a keep-mask (`True` = retained) is built by taking the top-`k` by magnitude in each row, with all-zero rows fully pruned. Gradual pruning intersects each new mask with the previous one, so a slot that has been pruned is never revived. Pruned slots hold the *exact* zero anchor rather than a mask multiply, which is what the CSR kernels and the sparse export rely on.

**Gather indices are precomputed, not rebuilt per step.** The AVX2 kernels consume the weight in CSR/CSC nnz order, so the dense weight has to be gathered into that order on every forward and backward. The index mapping depends only on the sparsity pattern, so it is built once in `_build_sparse_structure` (where the mask actually changes) and cached in two non-persistent buffers, `_csr_gather_index` and `_csc_gather_index`. Recomputing it per call cost a `repeat_interleave` over the full row dimension three times per step per layer. `persistent=False` keeps them out of the checkpoint — they are fully derivable from `w_ptr`/`w_col`. Since the training path is now a dense GEMM, the gather itself happens only when the AVX2 kernels or the sparse export read it.

### Ablation harness

`scripts/lcqat_ablation.py` measures the claims the write-up makes, so they can be falsified rather than asserted:

```bash
uv run python scripts/lcqat_ablation.py --experiment all --seeds 8     # all nine, 20 rows
uv run python scripts/lcqat_ablation.py --seeds 5 --dry-run            # print
uv run python scripts/lcqat_ablation.py --experiment grad_scale --json-out r.json
```

Nine experiments ship; `dev/LEADERBOARD.md`'s generated table holds all 20 rows:

| `--experiment` | what it measures | result |
|---|---|---|
| `asym_vs_small` | `c_proj`'s **activation** quantizer on non-negative `relu(x)^2` | level utilization 0.533 → 1.000; absolute level count ties at 8, so the win is headroom, not resolution |
| `grad_scale` | `inv_sqrt_n` against `none` | observed 0.00196 vs predicted 0.00195 |
| `objective` | CE vs EDM block-output NMSE | 1.726 → 1.592; loss is a diagnostic only (EDM resamples sigma, so it is jittery by design) |
| `block_sampling` | accumulated-step vs per-micro-step gradient magnitude | micro/step 0.352 and 0.332 — directional, strongly seed-dependent |
| `overlap` | out-of-nominal-band sigma fraction vs overlap `g` | 0 → 0.293 → 0.420; **rises**, because overlap widens the draw interval by construction |
| `sparsity` | layer vs global scope, dequant NMSE | NMSE ties by construction; mask **disagreement** 0.3746 (1 seed) is the real result |
| `act_lut` | proximity relaxation and SmoothPWL body | 225 → 30 params matched-budget; 120 → 16 per preset; fp8 knots bit-identical |
| `sigma_cond` | sigma-conditioned codebooks | structural only — 2× params, and the distributional premise is UNVERIFIED |
| `bias_quant` | opt-in bias codebook | NMSE 0.0130 for 14 params / 116 B per layer, codebook live |

`asym_vs_small` deserves the note it carries in the leaderboard: on that data
`small` (symmetric, 15 levels) places only 8 levels in range while `asym`
(one-sided, 8 levels) places all 8. Measuring *weights* instead inverts the
result — `asym` is measurably worse on signed weights.

The driver exits non-zero when a claim it advertises comes out wrong, and writes only between `<!-- BEGIN GENERATED: lcqat_ablation.py -->` markers so hand-written analysis in `dev/LEADERBOARD.md` survives a re-run. Measurement protocol is in `nanochat/models/quant/ablation_metrics.py`, tests in `tests/test_lcqat_ablation_harness.py` and `tests/test_lcqat_ablation_driver.py`.

**Every row is a micro-probe**, not a training run: random weights, a few SGD
steps, no end-to-end quality claim. `bias_quant` in particular reports the
reconstruction error and the storage cost of a bias codebook — whether
*encoding* bias is worth it on a trained model is not measured.

### DiffusionBlocks (block-wise training + diffusion inference)

`nanochat/training/diffusion_blocks.py` implements SakanaAI DiffusionBlocks (ICLR 2026): depth is partitioned into `B` independent blocks (`EquiProbabilityPartitioner` distributes layers equi-probably across blocks), each block gets a noise-conditioned adapter (AdaLN from an EDM sinusoidal sigma embedding, `denoise_step`), and an equi-probable cycling scheduler picks one block active per micro-step. Only the active block holds gradients + optimizer state, cutting grad/optimizer memory ~`B`×.

The `DiffusionBlockEngine` is the **default training engine** for `base_train`, `chat_sft`, and `chat_rl` (`--db-blocks`, default `4`). It wraps a `GPT` and exposes `train_step` / `denoise_step` / `generate`. LC-QAT and SparseProp are applied through the engine so the whole pipeline — adapters, denoise head, KV cache — is quantized and sparse.

```bash
# DiffusionBlocks on CPU (toy, d4/256-wide, seq 64, batch 2)
python -m scripts.base_train --depth=4 --db-blocks=4 --no-lcqat --no-sparseprop --num-iterations=500
```

**Turning it off.** `--db-blocks=0` trains a plain autoregressive LM: no partitioner, no adapters, no denoise heads, no block isolation. Every transformer layer trains on every step by next-token cross-entropy, and the checkpoint records `meta["db"] = None` so `base_eval` loads it as a bare `GPT` with a strict load rather than building a zero-initialized engine.

This is the switch to reach for when you want a conventional LM baseline, and it is *not* equivalent to `--db-objective ce`: `ce` keeps the engine, the block partition and the block-isolated gradients, so only one block is trained per step. `runs/stackcompare.sh` uses `--db-blocks=0` for exactly that reason. The denoiser-only flags (`--db-sigma-codebook`, `--kd-denoiser-alpha`, `--efqat-latch-blocks`) are rejected at startup in this mode rather than silently ignored.

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

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/models/dtype.py`, since every `Linear` casts to it in forward). By default this is auto-detected based on your hardware:

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

The package is split along a **functional-core / imperative-shell** line. `nanochat/models/` holds pure tensor math with no I/O, optimizer, or logging; everything that touches hardware, persistence, or lifecycle lives in the shell packages beside it. `nanochat/ops/` sits between them, so the math never imports a kernel directly.

```
.
├── nanochat
│   ├── models/                          # functional core: pure tensor math
│   │   ├── backbone.py                  # the GPT nn.Module Transformer
│   │   ├── dtype.py                     # COMPUTE_DTYPE policy (model-layer decision)
│   │   ├── io.py                        # typed contracts (LayerQuantSpec)
│   │   ├── flash_attention.py           # Flash Attention 3 / SDPA dispatch
│   │   ├── fp8.py                       # Float8Linear conversion
│   │   └── quant/                       # LC-QAT: codebooks, layers, export
│   │       ├── codebook.py              # asymmetric learned codebook (STE)
│   │       ├── linear.py                # LCQATLinear module
│   │       ├── lut.py                   # Activation LUTs (relu2/silu/gelu/tanh/sigmoid)
│   │       ├── learnable_lut.py         # LearnableIndexLut (softmax-relaxed LUT)
│   │       ├── per_channel.py           # PerChannelValueCenteredQuantizer
│   │       ├── product_lut.py           # fused 2-D product LUT
│   │       ├── activation.py            # SmoothPWL activation bodies
│   │       ├── bias_quant.py            # opt-in bias codebook
│   │       ├── sigma_codebook.py        # sigma-conditioned / modulated codebooks
│   │       ├── retrofit.py              # LayerKConfig, PRESETS, retrofit_model
│   │       ├── sparseprop.py            # SparsePropLinear, magnitude pruning
│   │       ├── pruning.py               # gradual pruning schedule
│   │       ├── sparse_artifact.py       # CSR sparse export planning + packing
│   │       ├── kd.py                    # KDLoss / DenoiserDistiller
│   │       ├── efqat.py                 # SelectiveFreezer, BlockLatchFreezer
│   │       ├── optimizer.py             # build_qat_param_groups
│   │       ├── export.py                # export_lcqat_checkpoint, wire_activation_luts
│   │       ├── ablation.py              # ablation experiment implementations
│   │       ├── ablation_metrics.py      # ablation measurement protocol
│   │       └── reference/               # pure-PyTorch reference implementations
│   ├── ops/                             # kernel layer: dispatcher + backends
│   │   ├── dispatch.py                  # dispatch_gemv / dispatch_index_linear
│   │   ├── gemv.py                      # mul-less GEMV backends
│   │   ├── index_linear.py              # index-fetch matmul, arbitrary K
│   │   ├── quant_attn.py                # quantized KV-cache attention
│   │   ├── sparseprop.py                # AVX2 sparse forward/backward
│   │   ├── references/                  # naive PyTorch oracles (CI ground truth)
│   │   ├── native/cpu/                  # C++: gemv, index_linear, quant_attn, sparseprop
│   │   └── kernels/                     # cpu_loader.py, gpu_loader.py (Taichi)
│   ├── modules/                         # imperative shell: runtime + persistence
│   │   ├── checkpoint_manager.py        # save/load model checkpoints
│   │   ├── experiments/                  # one module per ablation experiment
│   │   ├── engine.py                    # inference engine, KV cache, calculator tool
│   │   ├── execution.py                 # sandboxed Python execution
│   │   ├── core_eval.py                 # DCLM CORE score
│   │   └── loss_eval.py                 # bits-per-byte evaluation
│   ├── training/                        # training-time machinery
│   │   ├── diffusion_blocks.py          # DiffusionBlockEngine, EquiProbabilityPartitioner
│   │   └── optim.py                     # AdamW + Muon optimizer
│   ├── data/                            # ingestion boundary
│   │   ├── dataloader.py                # tokenizing distributed data loader
│   │   ├── dataset.py                   # download/read utils for pretraining data
│   │   └── tokenizer.py                 # BPE tokenizer wrapper in GPT-4 style
│   ├── callbacks/                       # side-effect observers for the loop
│   │   └── training.py                  # W&B logging, GC management, run summary
│   ├── utils/
│   │   └── common.py                    # COMPUTE_DTYPE and misc utilities
│   └── tasks/                           # evaluation task mixtures
│       ├── common.py                    # TaskMixture | TaskSequence
│       └── arc.py / gsm8k.py / humaneval.py / mmlu.py / smoltalk.py
├── scripts                              # entry points (the imperative shell)
│   ├── _train/                          # base_train split by concern
│   │   ├── build.py                     # model construction, LC-QAT + SparseProp wiring
│   │   ├── loop.py                      # the training loop
│   │   └── eval.py                      # validation, CORE metric, sampling, checkpoints
│   ├── base_train.py  base_eval.py  chat_sft.py  chat_rl.py  chat_eval.py
│   ├── chat_cli.py  export_lcqat.py  tok_train.py  tok_eval.py
│   └── lcqat_ablation.py  gemv_bench.py  infer_bench.py  kv_budget_bench.py
├── runs                                 # shell harnesses
│   ├── speedrun.sh  miniseries.sh  scaling_laws.sh  runcpu.sh  stackcompare.sh
├── tests                                # mirrors the source layout
│   ├── test_lcqat_*.py                  # parity, opcheck, runtime, wiring
│   ├── test_sparseprop*.py              # kernel parity, pruning, integration
│   ├── test_dbcpu_*.py                  # DiffusionBlocks CPU
│   ├── test_lm_mode.py                  # plain-LM path (--db-blocks=0)
│   ├── test_numerical_fingerprint.py    # behavior-preservation gate
│   ├── test_architecture_boundary.py    # core must not import the shell
│   └── test_w0_*.py  test_w2_*.py  ...
├── dev
│   ├── LEADERBOARD.md                   # Time-to-GPT-2 leaderboard docs
│   ├── STACK_COMPARE.md                 # four-arm SparseProp/DiffusionBlocks comparison
│   ├── LOG.md                           # training experiment log
│   └── *_analysis.ipynb, *.png
├── pyproject.toml                       # CPU-only torch wheel (the `cpu` extra)
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

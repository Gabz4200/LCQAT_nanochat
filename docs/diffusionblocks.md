# DiffusionBlocks

This page covers the block-wise training engine in `nanochat/training/diffusion_blocks.py`: how it partitions depth, what it saves, and how to turn it off. Read it before changing `--db-*` flags, the partitioner, or the EDM loss path. Kernel details live in [kernels.md](kernels.md); the quantized stages that run through the engine live in [quantization.md](quantization.md) and [sparseprop.md](sparseprop.md).

The engine trains by default in `base_train`, `chat_sft`, and `chat_rl` (`--db-blocks`, default `4`).

```bash
# Toy CPU run: d4/256-wide, seq 64, batch 2
python -m scripts.base_train --depth=4 --db-blocks=4 --no-lcqat --no-sparseprop --num-iterations=500
```

## How it trains

Depth splits into `B` independent blocks. `EquiProbabilityPartitioner` distributes layers equi-probably across blocks. Each block carries a noise-conditioned adapter (per-layer AdaLN from an EDM sinusoidal sigma embedding, one `(gamma, beta)` pair per layer, because the paper conditions inside the block and one pair per group cannot express that). An equi-probable scheduler selects one active block per **optimizer step**. Only the active block holds gradients and optimizer state, which cuts grad/optimizer memory ~`B`x.

`DiffusionBlockEngine` wraps a `GPT` and exposes `train_step` / `denoise_step` / `generate`. LC-QAT and SparseProp apply through the engine, so adapters, denoise head, and KV cache all quantize and sparsify with the base model.

**Turning it off.** `--db-blocks=0` trains a plain autoregressive LM: no partitioner, no adapters, no denoise heads, no block isolation. Every layer trains on every step with next-token cross-entropy, and the checkpoint records `meta["db"] = None` so `base_eval` loads it as a bare `GPT` with a strict load instead of building a zero-initialized engine.

Use `--db-blocks=0` for a conventional LM baseline. It differs from `--db-objective ce`: `ce` keeps the engine, the partition, and block-isolated gradients, so it still trains one block per step. `runs/stackcompare.sh` uses `--db-blocks=0` for that reason. The denoiser-only flags (`--db-sigma-codebook`, `--kd-denoiser-alpha`, `--efqat-latch-blocks`) fail at startup in this mode instead of running silently ignored.

## Two properties of the default

**EDM does not train `lm_head`.** The EDM objective predicts a clean *embedding* (`denoise_heads[b]`), never tokens, so `lm_head` receives no gradient on that path. It still serves sampling. The loss supplies no logits to attach a gradient to, so this follows from the objective rather than from a wiring fault. Use `--db-objective ce` for a token-level objective, and never mix the two objectives within one run.

**The per-step loss jitters by design.** The sampler redraws sigma each step and reweights with `w(sigma) = (sigma^2+sigma_d^2)/(sigma*sigma_d)^2`, so the loss moves between steps at fixed weights. That movement is EDM as specified, not instability.

## On CPU

DiffusionBlocks makes this fork viable without a GPU.

`denoise_step` runs only the active block, L/B of the layers, and that throughput win is where the CPU savings come from. `train_step` still forwards the whole model and saves only backward and optimizer work:

| B | CE `train_step` | EDM `denoise_step` |
|---|-----------------|-------------------|
| 1 | 1108 tok/s | 1171 tok/s |
| 2 | 1298 tok/s | 2365 tok/s |
| 4 | 1284 tok/s | 4045 tok/s |

(d4/256-wide toy, seq 64, batch 2, peak RSS 406 MB.)

Two mistakes recur here:

- **The ~Bx scaling sits in `denoise_step`, not `train_step`.** A loop that calls `train_step` gains no speed from raising `--db-blocks`; it only shrinks the gradient and optimizer working set.
- **Memory comes first, FLOPs second.** Cutting grad/optimizer state Bx is what fits a model on a small host at all; `train_step` forward FLOPs stay untouched.

Two helpers in the same module serve CPU work: `configure_cpu_training(num_threads)` sets both intra- and inter-op thread pools, and `cpu_adamw_for(engine)` builds a fused AdamW over `engine.parameters()`. Fusion matters because non-fused AdamW walks the parameter list in Python, a real cost at this model size.

## Block/noise map (v2)

Block `0` holds the earliest layers and owns the HIGHEST noise range; block `B-1` (latest layers) owns the lowest (paper Fig. 6, App. C). `EquiProbabilityPartitioner.range_for_block` owns that map alone; the training sampler and the Euler sampler both route through it. Checkpoints stamp `meta["db"]["noise_map_version"] = 2`, and the loader rejects anything else loudly: a v1 checkpoint (block 0 on the lowest range) resumed under v2 would retarget every block's noise range without warning.

## The EDM loss kernel

`denoise_step` computes `weight * mean((pred - clean)^2)` through `dispatch_db_denoise` (`nanochat/ops/`), with three agreeing backends: `naive` (pure-PyTorch oracle), `cpu` (compiled C++, the default via `--db-denoise-backend`), `gpu` (Taichi/Vulkan). The compiled forward op stays inference-only; gradients reach the denoiser through `DbDenoiseLossFunction`'s analytic backward, the same split the SparseProp training path uses. `tests/test_db_denoise_kernel.py` pins parity (naive == cpu == gpu), opcheck, and `torch.compile` composition.

## Clean-past conditioning (paper App. E.4)

The denoiser input concatenates `[clean | noisy]`: noisy future tokens condition on clean past tokens in one forward over the 2T sequence, and the loss covers the noisy suffix only. The 2T attention mask is always built explicitly. A plain causal mask over 2T would leak clean FUTURE into each noisy token (row `T+i` would see clean columns up to `T+i`), and the denoiser would copy instead of denoise. Packed block-diagonal masks expand the same way, keeping each noisy token inside its own document past.

## EDM versus plain AR (do not compare raw losses)

Same batch, same weights, two kinds of training (`tests/test_db_edm_behavior.py` pins this):

| | EDM `denoise_step` | Plain AR (`--db-blocks=0` / `train_step`) |
|---|---|---|
| Loss | weighted embedding MSE, no logits | next-token cross-entropy |
| `lm_head` | untouched (nothing to attach to) | trained |
| Forward | active block only (L/B layers) | full depth (all L layers) |

Raw loss values sit on different scales, so never compare them across the two. `--db-blocks=0` stays the conventional LM baseline: no partitioner, no adapters, no heads, every layer trains every step. None of this pass changes that path (`train_step` does not depend on the denoise backend).

## Per-block heads, AdaLN, arbitration, latch

- `denoise_heads`: one `ModuleList` entry per block (`db_denoise_heads.{b}.`); legacy `db_denoise_head.` migrates to block 0 and raises if num_blocks>1.
- Per-layer AdaLN: `Block(cond=(gamma,beta))`, default None, which reproduces the plain block bit-identically.
- `_requires_grad_for` arbitrates every requires_grad decision (`parts[1]` for `db_*`, `parts[2]` for `transformer.h.`).
- `live_blocks` / latch exclusion freezes inactive blocks (EfQAT).

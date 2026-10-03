# DiffusionBlocks

Block-wise training and diffusion inference, implemented in
`nanochat/training/diffusion_blocks.py`. The engine is the default for
`base_train`, `chat_sft` and `chat_rl`.

## DiffusionBlocks (block-wise training + diffusion inference)

`nanochat/training/diffusion_blocks.py` implements SakanaAI DiffusionBlocks: depth is partitioned into `B` independent blocks (`EquiProbabilityPartitioner` distributes layers equi-probably across blocks), each block gets a noise-conditioned adapter (per-layer AdaLN from an EDM sinusoidal sigma embedding — one `(gamma, beta)` pair *per layer*, because the paper conditions inside the block and a single pair for the whole group cannot express that), and an equi-probable scheduler picks one block active per **optimizer step**. Only the active block holds gradients + optimizer state, cutting grad/optimizer memory ~`B`×.

The `DiffusionBlockEngine` is the **default training engine** for `base_train`, `chat_sft`, and `chat_rl` (`--db-blocks`, default `4`). It wraps a `GPT` and exposes `train_step` / `denoise_step` / `generate`. LC-QAT and SparseProp are applied through the engine so the whole pipeline — adapters, denoise head, KV cache — is quantized and sparse.

```bash
# DiffusionBlocks on CPU (toy, d4/256-wide, seq 64, batch 2)
python -m scripts.base_train --depth=4 --db-blocks=4 --no-lcqat --no-sparseprop --num-iterations=500
```

**Turning it off.** `--db-blocks=0` trains a plain autoregressive LM: no partitioner, no adapters, no denoise heads, no block isolation. Every transformer layer trains on every step by next-token cross-entropy, and the checkpoint records `meta["db"] = None` so `base_eval` loads it as a bare `GPT` with a strict load rather than building a zero-initialized engine.

This is the switch to reach for when you want a conventional LM baseline, and it is *not* equivalent to `--db-objective ce`: `ce` keeps the engine, the block partition and the block-isolated gradients, so only one block is trained per step. `runs/stackcompare.sh` uses `--db-blocks=0` for exactly that reason. The denoiser-only flags (`--db-sigma-codebook`, `--kd-denoiser-alpha`, `--efqat-latch-blocks`) are rejected at startup in this mode rather than silently ignored.

## Two things to know about the DiffusionBlocks default

- **`lm_head` is not trained by `--db-objective edm`.** The EDM objective trains a denoiser that predicts a clean *embedding* (`denoise_heads[b]`), never tokens, so `lm_head` has no gradient on that path. It is still used when sampling. This is a property of the objective, not a wiring bug — the loss has no logits to attach a gradient to. Use `--db-objective ce` if you need a token-level objective (and remember the two objectives must not be mixed within a run).
- **The per-step loss is jittery by design.** σ is resampled every step and `w(σ) = (σ²+σ_d²)/(σ·σ_d)²` reweights it, so the loss varies substantially between steps even with fixed weights. That is EDM working as specified, not instability.

## On CPU

DiffusionBlocks is the reason this fork is viable without a GPU.

`denoise_step` runs **only the active block**, L/B of the layers, and that is where the CPU throughput win comes from. `train_step` still forwards the whole model and saves only the backward and optimizer work:

| B | CE `train_step` | EDM `denoise_step` |
|---|-----------------|-------------------|
| 1 | 1108 tok/s | 1171 tok/s |
| 2 | 1298 tok/s | 2365 tok/s |
| 4 | 1284 tok/s | 4045 tok/s |

(d4/256-wide toy, seq 64, batch 2, peak RSS 406 MB.)

Two caveats, both of which are easy to get wrong:

- **The ~B× scaling is in `denoise_step`, not `train_step`.** If your loop is calling `train_step`, raising `--db-blocks` will not speed it up — it only shrinks the gradient and optimizer working set.
- **It is a memory argument before it is a FLOPs argument.** Cutting grad/optimizer state by B× is what makes a model fit on a small host at all; the forward FLOPs are untouched in `train_step`.

Helper entry points for CPU work live in the same module: `configure_cpu_training(num_threads)` sets both the intra- and inter-op thread pools, and `cpu_adamw_for(engine)` builds a fused AdamW over `engine.parameters()` — fused specifically because on CPU a non-fused AdamW walks the parameter list in Python, which is a real cost at this model size.

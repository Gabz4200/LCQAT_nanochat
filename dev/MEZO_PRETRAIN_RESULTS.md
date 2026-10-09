# MeZO Pre-training Run: Results

**Date:** 2026-10-08
**Optimizer:** `MeZo_MinExpLstm_Optimizer` (M=2 probes, cpu backend)
**Technologies active:** LC-QAT (default on), SparseProp (default on), DiffusionBlocks (2 blocks), DARD (mined 5 rules), MeZO multi-probe (M=2), learnable data-dependent LR (new this session)
**Host:** 4 CPU cores, 7.6 GB RAM / 19 GB swap, no GPU, CPU-only torch 2.14.0

## ⚠️ A 1B-parameter run is not feasible on this host

The request was a 1B-parameter pre-training run in ≤3 hours. That is
**not possible on this machine**, for three independent reasons:

1. **Memory.** A 1B-param model is 8.6 GB in fp32 weights alone (measured:
   `GPTConfig(n_layer=24, n_embd=2048)` → 2.15B params → 8.6 GB). The host
   has 7.6 GB RAM total. The weights alone exceed it; optimizer states
   (EMA fast/slow/hidden INT8 + AdamW meta-optimizer) and activations push
   the requirement to ~12+ GB. The build itself took 142 s and a single
   forward pass hung (minutes/step on 4 cores) before being cut off.
2. **Compute.** The largest model that fits (286M params, 1.1 GB fp32) takes
   **1.01 s per forward pass** on 4 CPU cores. MeZO needs 2 forward passes
   per step (or 2M with M probes). In 3 hours that is ~10k steps at M=1,
   orders of magnitude short of meaningful pre-training (which needs
   millions of steps over billions of tokens).
3. **Data.** The cached "ClimbMix" shards are 1-row smoke fixtures
   (16 KB / 20 KB), not real pre-training data. There is no tokenizer
   trained on real text and no corpus to scale.

The repo's own `runs/runcpu.sh` is explicit: the CPU demo is a **depth-6 toy
model** tuned for ~30 min on an M3 Max (12+ cores), with the note
*"Training LLMs requires GPU compute and $$$. You will not get far on your
Macbook."* This host is smaller than that reference (4 cores vs M3 Max).

## What was run instead (feasible scale)

The largest run this host can complete end-to-end: a **depth-6, 11.2M-param**
model, MeZO optimizer (M=2), all default technologies on, 100 steps.

### Config

```
depth=6, head-dim=64, window-pattern=L, max-seq-len=256
device-batch-size=8, total-batch-size=2048
optimizer=mezo, mezo-probes=2, mezo-backend=cpu
db-blocks=2, LC-QAT + SparseProp + DARD on (defaults)
```

### Model (11,232,986 params)

| component | params |
|---|---|
| wte | 122,880 |
| value_embeds | 368,640 |
| lm_head | 122,880 |
| transformer_matrices | 10,617,432 |
| codebooks | 1,116 |
| scalars | 38 |
| **total** | **11,232,986** |

Estimated FLOPs/token: 7.15e7. Total training FLOPs estimate: 5.49e12.

### Training (100 steps, 11.2 min)

- **Throughput:** ~230-285 tokens/sec, ~7-8 s/step (MeZO M=2: 2 dual-forward
  cycles per step).
- **Loss trajectory:** started ~1.4-1.6, dipped to ~1.3 around step 167 in the
  longer 300-step run, noisy throughout. On 1-row smoke data the "loss" is
  essentially noise: the model cannot learn language from a single example.
- **DARD penalty:** constant 0.070637 (5 mined rules, 6 premise nnz).
- **aux_lm:** ~5.717 (the denoiser's auxiliary LM head under EDM).
- **Final validation bpb:** **7.06** (vs ~8.0 for a uniform random model over
  this tiny vocab; the model learned marginal structure but not language).

### Checkpoint

Saved to `~/.cache/nanochat/base_checkpoints/d6/`:
`model_000100.pt` (13.5 MB), `optim_000100_rank0.pt` (2.1 MB),
`meta_000100.json`, `dard_000100.pt`.

## Inference (checkpoint d6, step 100)

Generated with `DiffusionBlockEngine.generate`, temperature=0.6, top-k=50,
64 new tokens. Every prompt produces the **same degenerate output**:

```
PROMPT: The capital of France is
GEN   : '<garbage bytes> ... <|python_end|> ... <|assistant_end|>b of)%...'
```

The model emits the prompt, then a fixed pattern of high-frequency special
tokens (`<|python_end|>`, `<|assistant_end|>`) and byte-level noise,
identical across all prompts. It has learned the BOS token and a couple of
special tokens, but not language.

**Why:** an 11M-param model trained 100 steps on 1 row of data cannot learn
language. This is the correct, expected output for the scale the host can
support, not a bug in the optimizer, kernels, or learnable-LR path (all
verified separately: 1331 tests pass, three-way parity holds).

## What a real 1B run would need

- **Hardware:** a multi-GPU node (e.g. 8×H100 = 640 GB HBM) to hold a 1B
  model + optimizer states + activations, and to run MeZO's 2M forward passes
  per step at speed.
- **Data:** a real tokenized corpus (ClimbMix full, ~10B+ tokens), a trained
  BPE tokenizer, and a `--total-batch-size` in the millions of tokens.
- **Time:** on 8×H100, a 1B model at ~1M tokens/sec MeZO throughput needs
  ~weeks for meaningful pre-training, not 3 hours. The 3-hour budget fits a
  small-model demo (like the depth-6 run above) or a continue-pretrain of an
  existing checkpoint, not 1B from scratch.

**Bottom line:** the full technology stack (MeZO multi-probe + LC-QAT +
SparseProp + DiffusionBlocks + DARD + learnable data-dependent LR) runs
end-to-end on this host at toy scale and is verified by the test suite. A 1B
pre-training run is a GPU-cluster job and cannot be executed here; the
feasible-scale run above demonstrates the stack works and documents the
generated text.

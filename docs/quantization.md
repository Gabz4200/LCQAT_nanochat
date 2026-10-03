# Quantization reference

LC-QAT and the opt-in stages layered on it. For what each stage is *for* and why it
ships on by default, see [What's in the box](../README.md#whats-in-the-box).

Code lives in `nanochat/models/quant/`. Every flag below is registered once in
`scripts/_cli.py`, `retrofit.py` or `w6.py`, so `base_train`, `chat_sft` and
`chat_rl` cannot drift apart on the same knob.

## Quantization: LC-QAT, KD, EfQAT, activation LUTs

Learned Codebook Quantization-Aware Training (LC-QAT) is the fork's core contribution. Every retrofitted `Linear` gets an **asymmetric codebook** of cardinality `K = m_neg + 1 + m_pos` with index `m_neg` anchored **exactly to 0.0**, so zero-initialized weights and sparse activations quantize without noise. `K` is *not* required to be odd: the asymmetric split is what makes a one-sided codebook (`m_neg = 0`, so K=4, K=8, K=16) expressible, and an odd-K requirement would defeat the point. For a plain integer `K`, `split_from_k` splits symmetrically when it can and gives the extra level to the positive side. Levels are cumulative `softplus` steps (monotonic under gradient descent); the forward pass uses a straight-through estimator that trains both the input and the codebook; and the codebook parameters (`raw_pos_deltas` / `raw_neg_deltas`) get their own AdamW group with a dedicated learning rate (`--codebook-lr`, default `1e-3`, no weight decay).

**LC-QAT and SparseProp are always-on by default** in `base_train`, `chat_sft`, and `chat_rl`. Disable them with `--no-lcqat` / `--no-sparseprop`.

```bash
python -m scripts.base_train --depth=12 --no-lcqat              # plain float training
python -m scripts.base_train --lcqat-preset prd                 # PRD table: 8-bit down_proj
python -m scripts.base_train --codebook-grad-scale none         # disable the 1/sqrt(N) codebook scaling
python -m scripts.base_train --db-objective ce                 # next-token CE instead of the EDM objective
torchrun -m scripts.chat_sft -- --run=sft                      # LC-QAT + SparseProp on by default
python -m scripts.chat_rl --no-lcqat --no-sparseprop           # plain RL
```

| Flag | Meaning |
|------|---------|
| `--no-lcqat` | LC-QAT is **on by default**; this flag runs plain float training. There is no `--lcqat` flag |
| `--lcqat-preset asym` | **Default.** Same level *counts* as `small` for the signed tensors, but the split is chosen per tensor by sign: the two tensors that see `relu(x).square()` (>= 0) — `down_act` and `fc_out` — drop from 15 levels to **8**, and all 8 land in range instead of 7 of 15 being stranded below the data. A smaller table that is fully used beats a bigger one that is half dead. See the ablation table in [dev/LEADERBOARD.md](../dev/LEADERBOARD.md) |
| `--lcqat-preset small` | Symmetric max compression: q/k weights K=3 (mul-less ternary), everything else K=15 |
| `--lcqat-preset prd` | PRD table: `mlp.c_proj` (down_proj) at K=255/255, rest as small |
| `--lcqat-k-map substr:KW/KA,...` | Per-module overrides. Each K is a total level count (`15`) or an explicit split (`0-7`), e.g. `mlp.c_proj:0-7/0-7` |
| `--codebook-lr` | Codebook AdamW LR (PRD: 10–50× network weights), no weight decay |
| `--codebook-grad-scale {inv_sqrt_n,none}` | PRD 2.4 codebook gradient scaling. `inv_sqrt_n` (default) scales the codebook gradient by `1/sqrt(numel)`; with `N = B·T·D` in the millions this is ~1000× smaller, so it interacts multiplicatively with `--codebook-lr` |
| `--db-objective {edm,ce}` | DiffusionBlocks objective. `edm` (default) trains each block as a denoiser over its own noise range, so only L/B layers run and activations are `O(L/B)`. `ce` is full-depth next-token cross-entropy with block-isolated gradients (saves backward memory, no forward FLOPs). **`ce` is not the same as turning DiffusionBlocks off** — it still runs through the engine and still gradients one block per step. Use `--db-blocks=0` for a conventional baseline |
| `--db-blocks` | Number of independent diffusion blocks (checkpoint provenance: cannot change on resume). **`0` disables DiffusionBlocks entirely** and trains a plain autoregressive LM: no partitioner, no denoise heads, no block isolation, every layer trains every step |
| `--db-overlap` | Log-σ overlap between adjacent blocks (DiffusionBlocks App. C). 0.1 for text, 0.05 for vision |
| `--db-block-sampling {step,micro}` | Draw the active block once per optimizer step (default) or per micro-step. `micro` is **lossy** with gradient accumulation: `_apply_requires_grad` clears gradients the active block does not own, so each micro-step erases the previous one and only the last block sampled reaches the optimizer |
| `--lcqat-lut-relaxation {logits,proximity}` | How the *trained* activation LUT picks its output level. `logits` (default) = a free `K_in × K_out` logit matrix with a straight-through round. `proximity` = a learnable knot grid + output levels selected by inverse-square distance: `K_in + K_out` parameters instead of `K_in × K_out`, with no softmax-saturation cliff. Init is bit-identical to the bake either way. Measured: 225 → 30 params at matched budget, 120 → 16 on the shipped presets |
| `--lcqat-act-body {pwl,smoothpwl}` | The learnable body approximating the elementwise activation between two quantized layers. `pwl` (default) = the shipped frozen `K_in → K_out` index table, exact at the knots and linear between. `smoothpwl` = a radial-basis map with free knots/slopes/intercepts, which fits `relu^2` far better at matched parameter count but must be re-baked into an integer table for inference |
| `--fp8` | FP8 training for the float path. **Mutually exclusive with LC-QAT** (both convert `Linear`) |
| `--lcqat-channel-center` | Per-output-channel weight codebooks instead of one shared table (PRD 3.4), so channels with very different scales are not forced onto a compromise grid. Off by default; costs `out_features × K` levels and makes the layer **un-exportable**, so it cannot be combined with the fused inference path. Persisted in checkpoint meta, so a resume does not silently drop it |
| bias quantization | Opt-in bias codebook (`nanochat/models/quant/bias_quant.py`), set via `LayerKConfig.quantize_bias` / `LCQATLinear(quantize_bias=True)`. **Off by default and with no CLI flag yet**, so existing checkpoints and outputs are bit-identical. Measured cost and error in the ablation table in [dev/LEADERBOARD.md](../dev/LEADERBOARD.md) |

Per-layer roles (`nanochat/models/quant/retrofit.py`): `attn.c_q`/`c_k` get ternary weights, Q/K/V **outputs** are quantized during training and the quantized KV-cache runtime is implemented (`Engine.generate(quantized_kv=True)`), `mlp.c_fc` output is quantized as the input side of the fused relu² LUT, `lm_head` and linears under 128 dims stay in floating point.

## Activation LUTs

Beyond weights, activations are quantized through fused lookup tables registered in an activation registry (`nanochat/models/quant/lut.py`). All common nonlinearities are covered: `relu2`, `silu`, `gelu`, `tanh`, `sigmoid`. The export step bakes these into static FP32 LUTs via `wire_activation_luts`.

## Knowledge Distillation (KD) anchoring

High compression ratios compress the loss manifold into sharp local minima. LC-QAT anchors the student QAT optimization using a KL-divergence against a frozen, detached FP32/BF16 teacher (the pre-quantization model, or any other unquantized reference). The teacher is never optimized.

`L_KD = tau^2 * D_KL( softmax(Z_teacher / tau) || softmax(Z_student / tau) )`
`L_total = (1 - alpha) * L_CE(Y, Y_hat_quant) + alpha * L_KD`

Enabled via `--kd-alpha` (default 0.0 = off; ~0.1 typical), `--kd-teacher-source` and `--kd-teacher-tag` (the float checkpoint to load as teacher), `--kd-temperature` (default 2.0). Implemented in `nanochat/models/quant/kd.py` (`KDLoss`).

`--kd-alpha` is **CE-only and rejected under the default `--db-objective edm`**: it needs logits, and the denoising objective never produces any. Use the denoiser form below instead. Startup rejects the combination rather than silently dropping the anchor.

## EfQAT selective layer freezing

To keep memory flat at scale, LC-QAT can selectively freeze middle-layer codebook deltas and weight gradients after a warm-up, keeping only "critical outlier layers" (input embedding projections, attention q/k, final output) trainable. Enabled with `--efqat-freeze-after N` (default -1 = off). Implemented in `nanochat/models/quant/efqat.py` (`SelectiveFreezer`). The optimizer simply skips params whose `.grad is None`, so momentum buffers are unaffected.

## EfQAT per-block permanent freezing

`SelectiveFreezer` above freezes by *layer role* (a global middle-layer band). `--efqat-latch-blocks N --efqat-latch-after M` instead retires the `N` *highest-index* diffusion blocks at step `M`, permanently, via `BlockLatchFreezer`. This is the block-wise analogue: a block that has converged on its noise range is latched and its parameters stop receiving gradients, which flattens optimizer state as the block count grows.

Latched blocks are excluded from sampling (`DiffusionBlockEngine.live_blocks` / `sample_block`). This is load-bearing, not tidiness: the EDM objective precomputes `clean` under `no_grad`, so a block's own parameters are the only differentiable path through it — sampling a latched block yields a loss with `grad_fn is None` and `backward()` raises. The latch is written to checkpoint metadata and re-applied on resume.

## Denoiser distillation (PRD 3.1, EDM form)

`--kd-denoiser-alpha A` adds a teacher term to the DiffusionBlocks objective, anchoring the quantized denoiser to its float twin on the *same* noisy input and noise level:

`L_KD = w(sigma) * || D_quant(x_t, sigma) - D_float(x_t, sigma) ||^2`

Sharing `(x_t, sigma)` is what isolates the anchor to quantization error; a different noise draw would measure two different problems. The float twin is a deepcopy of the model taken *before* LC-QAT retrofit, with SparseProp's forward stripped so the teacher is dense. `--kd-denoiser-alpha 0` (default) is off. Implemented in `nanochat/models/quant/kd.py` (`DenoiserDistiller`).

## Sigma-conditioned codebooks (PRD 3.2)

A DiffusionBlocks engine partitions sigma into disjoint equi-probability ranges and trains one block per range, so a single activation codebook has to span every noise level and spends most of its levels on values that never occur. Two mechanisms, both opt-in:

- `--db-sigma-codebook conditioned` (`SigmaConditionedCodebook`) — one codebook per sigma anchor, selected by a hard log-space bucket. Each block gets a codebook tuned to its own noise range. Costs `num_anchors ×` the codebook parameters and the inference LUT; `--db-sigma-anchors` defaults to `--db-blocks`.
- `--db-sigma-codebook modulated` (`SigmaModulatedCodebook`) — one codebook scaled by a learned positive gain on `log(sigma)`. Same artifact size as the unconditional codebook, so it is the option that does not change the storage contract.

The modulation is a *gain* rather than a shift because a shift cannot satisfy both of the contracts the level table has to honour: the exact zero anchor (SparseProp prunes weights to `0.0` and the CPU kernels skip on `w == 0.0`; a shifted anchor dequantized a pruned weight to 0.014 / 0.19 / 12.0 across three noise levels) and strict monotonicity (pinning the anchor back to `0.0` after a shift of 3.0 pushed the top negative level to 2.857, past the anchor). A positive gain satisfies both by construction.

Both variants initialize to *exactly* the unconditional codebook (`gain == 1.0`, identical anchor tables), so enabling one does not perturb step 0. Neither composes with the fused export path — a conditioned codebook needs one LUT per noise level and the exported `activation_lut` is a single static table — so the combination raises at the first fused forward rather than silently quantizing with the wrong levels.

## Per-channel value-centered quantization

`PerChannelValueCenteredQuantizer` (`nanochat/models/quant/per_channel.py`) gives each output channel its own asymmetric codebook, instead of one alphabet shared across channels. The mechanism is the scalar `ValueCenteredQuantizationLUT` decomposition extended with a per-channel axis, so the exact-zero anchor holds per channel: `x == center_c` reconstructs to exactly `center_c`, and with the default zero centres that is exactly `0.0`, which is the SparseProp structural-zero contract.

Two things about using it:

- **Call `init_from_tensor(x)` before training.** A hand-supplied `init_max` that overshoots the data fails silently and permanently: every value bucketizes onto the zero anchor, the anchor is the only level ever gathered, so the codebook parameters receive exactly zero gradient and the table never moves. Measured with `init_max=100` on data of magnitude 8 — NMSE stayed at exactly 1.0 for 400 steps on every channel. Fitting the outer levels to the data's own extremes sidesteps it.
- **It costs `C × K` levels**, versus `K` for a shared codebook. That is a table-size change, not a rounding change, so it is opt-in and the fused inference paths do not consume it.

What is measured: given one channel 100× larger than the others, a fitted shared codebook flattens the small channels to NMSE ≈ 1.0 (they round entirely onto the anchor), while per-channel rescues one to ≈ 0.4. The effect is **per channel, not on the pooled mean** — the pooled mean is dominated by the large channel, where the two are near-identical, so a mean-based comparison re-tests the big channel and hides the whole effect. It is also the only loss under which a per-channel table is the right tool: under a pooled MSE the gradient is dominated by the largest channel and the small channels' tables never move. No end-to-end accuracy claim is made; the tests in `tests/test_lcqat_per_channel.py` establish the mechanism, not a training win.

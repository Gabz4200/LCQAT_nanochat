# Quantization reference

This page covers LC-QAT and the opt-in stages layered on it: codebooks, KD anchoring, EfQAT freezing, denoiser distillation, sigma-conditioned codebooks, per-channel tables, activation LUTs. Read it before touching `nanochat/models/quant/` or any `--lcqat`, `--kd`, or `--efqat` flag. For what each stage is *for* and why it ships on by default, see [What's in the box](../README.md#whats-in-the-box). The runtime those codebooks compile to lives in [kernels.md](kernels.md).

Code lives in `nanochat/models/quant/`. Every flag below registers once in `scripts/_cli.py`, `retrofit.py`, or `w6.py`, so `base_train`, `chat_sft`, and `chat_rl` cannot drift apart on the same knob.

## LC-QAT

Learned Codebook Quantization-Aware Training (LC-QAT) is the fork's core contribution. Every retrofitted `Linear` gets an **asymmetric codebook** of cardinality `K = m_neg + 1 + m_pos` with index `m_neg` anchored **exactly to 0.0**, so zero-initialized weights and sparse activations quantize without noise. `K` need not be odd: the asymmetric split is what makes a one-sided codebook (`m_neg = 0`, so K=4, K=8, K=16) expressible, and an odd-K requirement would defeat the point. For a plain integer `K`, `split_from_k` splits symmetrically when it can and gives the extra level to the positive side. Levels are cumulative `softplus` steps (monotonic under gradient descent); the forward pass uses a straight-through estimator that trains both input and codebook; and the codebook parameters (`raw_pos_deltas` / `raw_neg_deltas`) get their own AdamW group with a dedicated learning rate (`--codebook-lr`, default `1e-3`, no weight decay).

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
| `--lcqat-preset asym` | **Default.** Same level *counts* as `small` for the signed tensors, but the split is chosen per tensor by sign: the two tensors that see `relu(x).square()` (>= 0), `down_act` and `fc_out`, drop from 15 levels to **8**, and all 8 land in range instead of 7 of 15 being stranded below the data. A smaller table that is fully used beats a bigger one that is half dead. See the ablation table in [dev/LEADERBOARD.md](../dev/LEADERBOARD.md) |
| `--lcqat-preset small` | Symmetric max compression: q/k weights K=3 (mul-less ternary), everything else K=15 |
| `--lcqat-preset prd` | PRD table: `mlp.c_proj` (down_proj) at K=255/255, rest as small |
| `--lcqat-k-map substr:KW/KA,...` | Per-module overrides. Each K is a total level count (`15`) or an explicit split (`0-7`), e.g. `mlp.c_proj:0-7/0-7` |
| `--codebook-lr` | Codebook AdamW LR (PRD: 10-50× network weights), no weight decay |
| `--codebook-grad-scale {inv_sqrt_n,none}` | PRD 2.4 codebook gradient scaling. `inv_sqrt_n` (default) scales the codebook gradient by `1/sqrt(numel)`; with `N = B·T·D` in the millions this is ~1000× smaller, so it interacts multiplicatively with `--codebook-lr` |
| `--db-objective {edm,ce}` | DiffusionBlocks objective. `edm` (default) trains each block as a denoiser over its own noise range, so only L/B layers run and activations are `O(L/B)`. `ce` is full-depth next-token cross-entropy with block-isolated gradients (saves backward memory, no forward FLOPs). **`ce` is not the same as turning DiffusionBlocks off**: it still runs through the engine and still gradients one block per step. Use `--db-blocks=0` for a conventional baseline |
| `--db-blocks` | Number of independent diffusion blocks (checkpoint provenance: cannot change on resume). **`0` disables DiffusionBlocks entirely** and trains a plain autoregressive LM: no partitioner, no denoise heads, no block isolation, every layer trains every step |
| `--db-overlap` | Log-σ overlap between adjacent blocks (DiffusionBlocks App. C). 0.1 for text, 0.05 for vision |
| `--db-block-sampling {step,micro}` | Draw the active block once per optimizer step (default) or per micro-step. `micro` is **lossy** with gradient accumulation: `_apply_requires_grad` clears gradients the active block does not own, so each micro-step erases the previous one and only the last block sampled reaches the optimizer |
| `--lcqat-lut-relaxation {logits,proximity}` | How the *trained* activation LUT picks its output level. `logits` (default) = a free `K_in × K_out` logit matrix with a straight-through round. `proximity` = a learnable knot grid + output levels selected by inverse-square distance: `K_in + K_out` parameters instead of `K_in × K_out`, with no softmax-saturation cliff. Init is bit-identical to the bake either way. Measured: 225 → 30 params at matched budget, 120 → 16 on the shipped presets |
| `--lcqat-act-body {pwl,smoothpwl}` | The learnable body approximating the elementwise activation between two quantized layers. `pwl` (default) = the shipped frozen `K_in → K_out` index table, exact at the knots and linear between. `smoothpwl` = a radial-basis map with free knots/slopes/intercepts, which fits `relu^2` far better at matched parameter count but must be re-baked into an integer table for inference |
| `--fp8` | FP8 training for the float path. **Mutually exclusive with LC-QAT** (both convert `Linear`) |
| `--lcqat-channel-center` | Per-output-channel weight codebooks instead of one shared table (PRD 3.4), so channels with very different scales are not forced onto a compromise grid. Off by default; costs `out_features × K` levels and makes the layer **un-exportable**, so it cannot be combined with the fused inference path. Persisted in checkpoint meta, so a resume does not silently drop it |
| bias quantization | Opt-in bias codebook (`nanochat/models/quant/bias_quant.py`), set via `LayerKConfig.quantize_bias` / `LCQATLinear(quantize_bias=True)`. **Off by default and with no CLI flag yet**, so existing checkpoints and outputs are bit-identical. Measured cost and error in the ablation table in [dev/LEADERBOARD.md](../dev/LEADERBOARD.md) |

Per-layer roles (`nanochat/models/quant/retrofit.py`): `attn.c_q`/`c_k` get ternary weights, Q/K/V **outputs** quantize during training and the quantized KV-cache runtime is implemented (`Engine.generate(quantized_kv=True)`), `mlp.c_fc` output quantizes as the input side of the fused relu² LUT, `lm_head` and linears under 128 dims stay in floating point.

## Activation LUTs

Activations quantize through fused lookup tables in an activation registry (`nanochat/models/quant/lut.py`). All common nonlinearities are covered: `relu2`, `silu`, `gelu`, `tanh`, `sigmoid`. Export bakes them into static FP32 LUTs via `wire_activation_luts`.

## Knowledge Distillation (KD) anchoring

High compression ratios compress the loss manifold into sharp local minima. LC-QAT anchors the student QAT optimization with KL-divergence against a frozen, detached FP32/BF16 teacher (the pre-quantization model, or another unquantized reference). The teacher never trains.

`L_KD = tau^2 * D_KL( softmax(Z_teacher / tau) || softmax(Z_student / tau) )`
`L_total = (1 - alpha) * L_CE(Y, Y_hat_quant) + alpha * L_KD`

Enable with `--kd-alpha` (default 0.0 = off; ~0.1 typical), `--kd-teacher-source` and `--kd-teacher-tag` (the float checkpoint to load as teacher), `--kd-temperature` (default 2.0). Implementation: `nanochat/models/quant/kd.py` (`KDLoss`).

`--kd-alpha` is **CE-only and rejected under the default `--db-objective edm`**: it needs logits, and the denoising objective never produces any. Use the denoiser form below instead. Startup rejects the combination rather than silently dropping the anchor.

## EfQAT selective layer freezing

To keep memory flat at scale, LC-QAT can freeze middle-layer codebook deltas and weight gradients after warm-up, keeping only critical outlier layers (input embedding projections, attention q/k, final output) trainable. Enable with `--efqat-freeze-after N` (default -1 = off). Implementation: `nanochat/models/quant/efqat.py` (`SelectiveFreezer`). The optimizer skips params whose `.grad is None`, so momentum buffers are unaffected.

## EfQAT per-block permanent freezing

`SelectiveFreezer` freezes by *layer role* (a global middle-layer band). `--efqat-latch-blocks N --efqat-latch-after M` instead retires the `N` *highest-index* diffusion blocks at step `M`, permanently, via `BlockLatchFreezer`. This is the block-wise analogue: a block that converged on its noise range latches, and its parameters stop receiving gradients, which flattens optimizer state as the block count grows.

Latched blocks leave the sampling pool (`DiffusionBlockEngine.live_blocks` / `sample_block`). This is load-bearing, not tidiness: the EDM objective precomputes `clean` under `no_grad`, so a block's own parameters are the only differentiable path through it. Sampling a latched block yields a loss with `grad_fn is None`, and `backward()` raises. The latch persists in checkpoint metadata and re-applies on resume.

## Denoiser distillation (PRD 3.1, EDM form)

`--kd-denoiser-alpha A` adds a teacher term to the DiffusionBlocks objective, anchoring the quantized denoiser to its float twin on the *same* noisy input and noise level:

`L_KD = w(sigma) * || D_quant(x_t, sigma) - D_float(x_t, sigma) ||^2`

Sharing `(x_t, sigma)` isolates the anchor to quantization error; a different noise draw would measure two different problems. The float twin is a deepcopy of the model taken *before* LC-QAT retrofit, with SparseProp's forward stripped so the teacher stays dense. `--kd-denoiser-alpha 0` (default) is off. Implementation: `nanochat/models/quant/kd.py` (`DenoiserDistiller`).

## Sigma-conditioned codebooks (PRD 3.2)

A DiffusionBlocks engine partitions sigma into disjoint equi-probability ranges and trains one block per range, so a single activation codebook must span every noise level and spends most levels on values that never occur. Two mechanisms address this, both opt-in:

- `--db-sigma-codebook conditioned` (`SigmaConditionedCodebook`): one codebook per sigma anchor, selected by a hard log-space bucket. Each block gets a codebook tuned to its own noise range. Costs `num_anchors ×` the codebook parameters and the inference LUT; `--db-sigma-anchors` defaults to `--db-blocks`.
- `--db-sigma-codebook modulated` (`SigmaModulatedCodebook`): one codebook scaled by a learned positive gain on `log(sigma)`. Same artifact size as the unconditional codebook, so the storage contract does not change.

The modulation is a *gain* rather than a shift because a shift cannot satisfy both contracts on the level table: the exact zero anchor (SparseProp prunes weights to `0.0` and the CPU kernels skip on `w == 0.0`; a shifted anchor dequantized a pruned weight to 0.014 / 0.19 / 12.0 across three noise levels) and strict monotonicity (pinning the anchor back to `0.0` after a shift of 3.0 pushed the top negative level to 2.857, past the anchor). A positive gain satisfies both by construction.

Both variants initialize to *exactly* the unconditional codebook (`gain == 1.0`, identical anchor tables), so enabling one does not perturb step 0. Neither composes with the fused export path: a conditioned codebook needs one LUT per noise level and the exported `activation_lut` is a single static table, so the combination raises at the first fused forward rather than quantizing silently with the wrong levels.

## Per-channel value-centered quantization

`PerChannelValueCenteredQuantizer` (`nanochat/models/quant/per_channel.py`) gives each output channel its own asymmetric codebook instead of one shared alphabet. The mechanism extends the scalar `ValueCenteredQuantizationLUT` decomposition with a per-channel axis, so the exact-zero anchor holds per channel: `x == center_c` reconstructs to exactly `center_c`, and with default zero centres that is exactly `0.0`, the SparseProp structural-zero contract.

Two facts govern use:

- **Call `init_from_tensor(x)` before training.** A hand-supplied `init_max` that overshoots the data fails silently and permanently: every value bucketizes onto the zero anchor, the anchor is the only level ever gathered, so the codebook parameters receive exactly zero gradient and the table never moves. Measured with `init_max=100` on data of magnitude 8, NMSE stayed at exactly 1.0 for 400 steps on every channel. Fitting the outer levels to the data's own extremes avoids it.
- **It costs `C × K` levels** versus `K` for a shared codebook. That is a table-size change, not a rounding change, so it stays opt-in and the fused inference paths do not consume it.

Measured result: given one channel 100x larger than the others, a fitted shared codebook flattens the small channels to NMSE ~= 1.0 (they round entirely onto the anchor), while per-channel rescues one to ~= 0.4. The effect is **per channel, not on the pooled mean**: the pooled mean is dominated by the large channel, where the two are near-identical, so a mean-based comparison re-tests the big channel and hides the effect. It is also the only loss under which a per-channel table is the right tool: under pooled MSE the gradient is dominated by the largest channel and the small channels' tables never move. No end-to-end accuracy claim follows; the tests in `tests/test_lcqat_per_channel.py` establish the mechanism, not a training win.

## 4-bit QAT (`--qat-4bit`)

`FourBitLinear` (`nanochat/models/quant/fourbit.py`) is the second quantization stage, and it is **mutually exclusive with `--lcqat`**: a weight is either a codebook index (LC-QAT) or a packed 4-bit nibble (4-bit QAT), never both.

Mechanism:

- **fp32 master weight + packed-nibble mirror.** The master stays fp32 so the optimizer updates in full precision; the mirror is a `uint8` tensor holding two 4-bit weights per byte (16 levels, symmetric around the per-tensor scale). The forward reads the *mirror* through a straight-through estimator, so every weight is real 4 bits at inference.
- **`quantize_weight_()` refreshes the mirror** after each optimizer step, so the mirror tracks the master as training moves it. The STE gradient flows to the master as if the forward had been identity.
- **Retrofit is structural.** `retrofit_fourbit(model, min_linear_dim)` replaces every eligible `Linear` with `FourBitLinear`, in place, after backbone construction. Because `FourBitLinear` is not a plain `nn.Linear`, the LC-QAT and SparseProp retrofit paths skip it, so `--qat-4bit` composes with SparseProp at the *engine-owned* layers (the DiffusionBlocks adapters / denoise heads) but leaves the base transformer in 4-bit nibble form.

Parity and storage are pinned in `tests/test_fourbit_qat.py` (12 tests): the packed mirror round-trips to the quantized weight, the STE forward matches the quantized value, and the packed storage is a quarter of fp32. Measured layer counts on the LFM2 clone: depth 6 → 20 layers, depth 8 → 28 layers, depth 12 → 40 layers. Run results: [`dev/FOURBIT_QAT_RESULTS.md`](../dev/FOURBIT_QAT_RESULTS.md).

The interaction with checkpoint loading matters: a `--qat-4bit` checkpoint's base transformer has no plain `nn.Linear` to wrap with SparseProp, so `checkpoint_manager.build_model` gates SparseProp injection on the checkpoint's *own* sparse-state keys rather than the always-on `meta["sparseprop"]["enabled"]` flag (per the SparseProp paper, the CSR structures exist only to route gradients; the forward function is the dense linear with pruned weights zeroed). See [docs/diffusionblocks.md](diffusionblocks.md#sequential-vs-parallel-block-execution) and the checkpoint-loading note in [`dev/LFM2_MEZO_DB_RESULTS.md`](../dev/LFM2_MEZO_DB_RESULTS.md).

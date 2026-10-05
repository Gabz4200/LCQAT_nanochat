# Leaderboard

How to take part in nanochat's "Time-to-GPT-2" leaderboard.

The metric is "time to GPT-2": wall-clock time to beat the GPT-2 (1.6B) CORE score on an 8xH100 node. GPT-2 itself trained on 32 TPU v3 for 168 hours (7 days) at $8/hour/TPUv3 in 2019, about $43K total, and scores 0.256525 CORE, an ensemble metric from the DCLM paper over 22 evaluations such as ARC and MMLU. (To re-verify the baseline: `base_eval.py` once evaluated HuggingFace models via `--hf-path openai-community/gpt2-xl`, removed for simplicity; see commit `21a7774` or earlier.)

## How to

The script [runs/speedrun.sh](../runs/speedrun.sh) always holds the current state of the art.

Once setup and tokenizer training are done, a typical tuning command looks like:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26-feb2-fp8-ratio8.25" \
    --model-tag="d26_feb2_fp8_ratio8.25" \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.25 \
    --fp8
```

Notes:

- `depth` sets the Transformer size.
- `run` is the wandb name.
- `model-tag` is the checkpoint location on disk.
- `device-batch-size`: ideally 32, since sequence length 2048 (the default) over 8 GPUs gives `32 X 2048 X 8 = 524,288`, the total batch size that works well at this scale. But 32 OOMs on bigger models (e.g. d26), so halve to 16. `base_train.py` compensates with gradient accumulation of 2 to reach the total batch size: two forward+backward passes, then one step. Keep powers of two so the accumulation math stays neat.
- `sample-every = -1` turns off periodic sampling.
- `core-metric-max-per-task=-1` runs the entire CORE eval.
- `core-metric-every=999999` is a hack to run the CORE eval once, at the end.
- `target-param-data-ratio=8.25` sets the training horizon: non-embedding parameter count times this number. The compute-optimal Tokens:Params ratio is the `base_train.py` default (10.5). GPT-2 capability sits between d24 and d26, so reaching it exactly means overtraining d24 or undertraining d26; here a d26 is slightly undertrained. Odd depths (e.g. d25) are not recommended because the head-dimension sizing comes out uneven.
- `--fp8` turns on fp8 training. Without fp8-capable GPUs, leave it out and train in bf16. bf16 is higher precision than fp8, so you can expect to lower the `target-param-data-ratio` for the same capability.

After about 1.5 hours the run ends with something like:

```
wandb: Run summary:
wandb:          core_metric 0.25851
wandb:                 step 16704
wandb: total_training_flops 4.330784131228946e+19
wandb:  total_training_time 10949.46713
```

CORE must exceed GPT-2's 0.256525. Report `total_training_time` (here 10949 seconds, about 10949/60/60 ~= 3.04 hours), which covers training iterations only, excluding eval and logging. Also report validation bpb, since CORE is noisy.

Beating GPT-2 in less than the current record earns a PR. Raw gains are not the whole decision: gnarly or bloated code, or something too esoteric, weighs against the improvement. Nanochat also wants a miniseries of models, not one target, so the change must generalize across depths for a sweep.

After committing, get the short hash with:

```
git log -1 --format="%h"
```

## Run 1

Achieved Jan 29 2026 on commit `348fbb3`. Launch command:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run=d24-jan29 \
    --model-tag=d24_jan29 \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=3000 \
    --target-param-data-ratio=12
```

Result:

```
wandb: Run summary:
wandb:          core_metric 0.25851
wandb:                 step 16704
wandb: total_training_flops 4.330784131228946e+19
wandb:  total_training_time 10949.46713
```

Validation bpb was 0.74833.

Detailed writeup: [Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)

## Run 2

Achieved Feb 2 2026 on commit `a67eba3`. Launch command:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26-feb2-fp8-ratio8.5" \
    --model-tag="d26_feb2_fp8_ratio8.5" \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.5 \
    --fp8
```

Result:

```
core_metric 0.2578
step 14889
total_training_time 10493
Minimum validation bpb: 0.745036
```

The change is `--fp8`: all Linear layers except the gates train in fp8 via `torchao` with tensorwise scaling. Each step is slightly lower quality, but steps run much faster, net ahead. Without fp8 hardware, drop the flag and train in bfloat16; the result beats GPT-2 by a wider margin from the precision upgrade. Finer tuning (keeping small matmuls in bf16) may help further.

Previous record 3.04 hours, now 2.91 hours: `(3.04 - 2.91)/3.04*100` ~= 4.3% faster.

## Run 3

Achieved Feb 5 2026 on commit `2c062aa`. Launch command:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26_feb4_double_batch_ratio8.25" \
    --model-tag="d26_feb4_double_batch_ratio8.25" \
    --device-batch-size=16 \
    --total-batch-size=1048576 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.25 \
    --fp8
```

Result:

```
core_metric 0.26024
step 7226
total_training_time 9922
Minimum validation bpb: 0.74645
```

The change is the doubled batch size, 0.5M to 1M: better for d26, so fewer steps via `--target-param-data-ratio` 8.5 to 8.25. The original 0.5M batch suited d12; bigger models (e.g. d26) prefer 1M. That measurement became a principled auto-batch-size rule by depth, so all depths benefit. See [dev/LOG.md](dev/LOG.md) entry "2026-02-05: Auto Batch Size Scaling".

## Run 4

Achived Mar 3 2026 on commit `324e69c`. The change is the switch from HuggingFace FineWeb-EDU to the NVIDIA ClimbMix dataset. `@karpathy` had tried dataset swaps many times, each a negative result (FineWeb, DCLM, Olmo); ClimbMix gave clear, immediate gains. Credit to `@ddudek` for finding ClimbMix for nanochat and reporting the improvements that started the followup.

To reproduce on the commit above, download at least 150 data shards and train the tokenizer:

```
python -m nanochat.data.dataset -n 150
python -m scripts.tok_train
```

Then run as usual, with ratio 9.5 (below the 10.5 compute optimum), i.e. a slightly undertrained d24:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run="d24-climbmix" \
    --model-tag="d24-climbmix" \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=9.5 \
    --device-batch-size=16 \
    --fp8
```

This ran 7 times. Training is mildly non-deterministic, so CORE spreads:

```
0.25373
0.2584
0.25489
0.2568
0.25732
0.26765
0.25119
```

Mean 0.25714 (above the 0.256525 threshold), max-min 0.01646. Better scores come from shuffling the data shards (a different order). That is unexpected: the documents were fully shuffled during data construction, so the distribution should look uniform. The default order is unfortunately among the worse ("unlucky") shuffles, but it beats GPT-2, so it merged. TODO: invest more later.

NOTE: `val_bpb` is *NOT* comparable to the previous 3 runs after the data change. This run sits at `0.71854` validation bpb. With a fixed dataset, `val_bpb` is a smooth relative metric with less noise than CORE.

## Run 5

Achieved Mar 9, 2026 on commit `6ed7d1d`. Same launch command as Run 4 except `--target-param-data-ratio=8.7`. Five identical runs averaged CORE 0.2690, well above the 0.2565 threshold. The ratio stopped there because the val_loss safety gap is tighter than the CORE gap: 0.71808 against Run 4's 0.71854. Possibly 8.6 would work, but not worth splitting hairs.

This commit is special: every improvement in [this commit](https://github.com/karpathy/nanochat/commit/6ed7d1d82cee16c2e26f45d559ad3338447a6c1b) came from autonomous "research" by a private version of [autoresearch](https://github.com/karpathy/autoresearch) on a d12 model. More in [this tweet](https://x.com/karpathy/status/2031135152349524125). The changes carried from d12 to d24 directly: a new record, 2.02 hours down to 1.80 hours "time to GPT-2".

## Run 6

Achieved Mar 14, 2026 on commit `a825e63`. Same launch command as Run 4 except `--target-param-data-ratio=8`. Shorter training keeps working as the architecture improves. An overtrained d22 was tried instead of the undertrained d24 and was worse. These changes came from autoresearch round 2, pointed at the modded-nanogpt repo for inspiration. It tried several ideas and found a way to make backout and smear help (a manual attempt long before had regressed). Smear in particular costs some code complexity: an "early fusion" of context across tokens, a kind of bigram input letting the network reach higher ngrams sooner. Inference needed changes to match, verified by a unit test of Engine inference against naive `GPT.generate()`. Five runs averaged CORE 0.262634 at 1.65 hours (99 minutes) each.

<!-- BEGIN GENERATED: lcqat_ablation.py -->
_Generated by `scripts/lcqat_ablation.py` on 2026-10-01 14:44:59 - do not edit by hand._

### Paired ablation results

| experiment | metric | baseline | variant | baseline | variant | delta | better | seeds |
|---|---|---|---|---|---|---|---|---|
| asym_vs_small_levels | level_utilization | small | asym | 0.533333 | 1 | +0.466667 | variant | 8 |
| grad_scale | codebook_grad_ratio | none | inv_sqrt_n | 0.276437 | 0.000541822 | -0.275895 | variant | 8 |
| objective | block_out_nmse | ce | edm | 1.72562 | 1.59242 | -0.133194 | variant | 8 |
| block_sampling_transformer.h | grad_magnitude_ratio | step | micro | 11.9371 | 4.20075 | -7.73637 | baseline | 8 |
| block_sampling_db_denoise_heads | grad_magnitude_ratio | step | micro | 7.58253 | 2.51898 | -5.06355 | baseline | 8 |
| overlap_g0 | sigma_out_of_nominal_band | g=0 | g=0 | 0 | 0 | +0 | baseline | 8 |
| overlap_g0.125 | sigma_out_of_nominal_band | g=0 | g=0.125 | 0 | 0.292572 | +0.292572 | baseline | 8 |
| overlap_g0.25 | sigma_out_of_nominal_band | g=0 | g=0.25 | 0 | 0.420044 | +0.420044 | baseline | 8 |
| sparsity_layer | dequant_nmse | layer@0.5 | layer@0.75 | 0.00532126 | 0.00390735 | -0.00141391 | variant | 8 |
| sparsity_global | dequant_nmse | global@0.5 | global@0.75 | 0.00532126 | 0.00390735 | -0.00141391 | variant | 8 |
| sparsity_scope_mask_disagreement | scope_mask_disagreement | layer | global | 0 | 0.374567 | +0.374567 | variant | 1 |
| act_lut_matched_budget | lut_params | logits_pwl | proximity_pwl | 225 | 30 | -195 | variant | 8 |
| act_lut_logits_pwl | lut_params | logits_pwl | logits_pwl | 120 | 120 | +0 | baseline | 8 |
| act_lut_logits_smoothpwl | lut_params | logits_pwl | logits_smoothpwl | 120 | 120 | +0 | baseline | 8 |
| act_lut_proximity_pwl | lut_params | logits_pwl | proximity_pwl | 120 | 16 | -104 | variant | 8 |
| act_lut_proximity_smoothpwl | lut_params | logits_pwl | proximity_smoothpwl | 120 | 16 | -104 | variant | 8 |
| sigma_cond | codebook_params | static | conditioned@2 | 14 | 28 | +14 | baseline | 8 |
| bias_quant_nmse | bias_nmse | fp32 bias | bias_quant@15 | 0 | 0.0130425 | +0.0130425 | baseline | 8 |
| bias_quant_cost | bias_codebook_params | fp32 bias | bias_quant@15 | 0 | 14 | +14 | baseline | 8 |
| bias_quant_liveness | codebook_param_delta | zero gradient | codebook gradient live: True | 0 | 0.000172562 | +0.000172562 | variant | 8 |

**Notes**

- `asym_vs_small_levels`: fraction of codebook levels actually hit on a non-negative relu^2 probe; absolute level count ties at 8 for both arms, so the gain is headroom (asym spends 8 levels, small spends 15 for the same 8)
- `grad_scale`: observed ratio 0.00196002 vs predicted 1/sqrt(N) = 0.00195312 for N=262144
- `objective`: mean block-output NMSE over 3 SGD steps on a 4-layer random model with 2 diffusion blocks; mean train loss ce 4.83464 vs edm 28.0369, reported as a diagnostic only: the edm loss is jittery by design, so neither arm is monotone and no quality claim is made from it. No lm_head gradient assertion is made; the edm objective predicts an embedding, not tokens.
- `block_sampling_transformer.h`: ||accumulated gradient|| / ||isolated single-block reference gradient|| over transformer.h.* parameters after one optimizer step of 4 accumulated micro-steps; the two families are never pooled. step 11.9371 (spread 1.49), micro 4.20075 (spread 2.61), micro/step 0.3519. NOT a fraction in [0,1] and NOT calibrated to 1: the reference is a single micro-step while both arms accumulate 4 of them, so the absolute scale is set by how much the micro-steps' gradients agree, which is a property of the sigma draw rather than of the sampling mode. Only the micro/step ratio carries the claim. That ratio is strongly seed-dependent: the same step configuration read 42.7 at --seeds 1 and 11.9 at --seeds 8; sigma is resampled per seed and the step arm's magnitudes follow that lottery. Treat the ratio as directional evidence at a fixed seed count, not as a constant; re-running at a different seed count will move both arms. The finding the ratio supports: `micro` retains strictly LESS signal than `step` for the same wall-clock step, because the block drawn last displaces the one the reference owns. Gradients accumulate across micro-steps (zero_grad once, not per micro-step), each loss is divided by the micro-step count, sigma is redrawn per sample, and the reference is built from a separate engine before any accumulation runs.
- `block_sampling_db_denoise_heads`: ||accumulated gradient|| / ||isolated single-block reference gradient|| over db_denoise_heads.* parameters after one optimizer step of 4 accumulated micro-steps; the two families are never pooled. step 7.58253 (spread 1.38), micro 2.51898 (spread 2.53), micro/step 0.3322. NOT a fraction in [0,1] and NOT calibrated to 1: the reference is a single micro-step while both arms accumulate 4 of them, so the absolute scale is set by how much the micro-steps' gradients agree, which is a property of the sigma draw rather than of the sampling mode. Only the micro/step ratio carries the claim. That ratio is strongly seed-dependent: the same step configuration read 42.7 at --seeds 1 and 11.9 at --seeds 8; sigma is resampled per seed and the step arm's magnitudes follow that lottery. Treat the ratio as directional evidence at a fixed seed count, not as a constant; re-running at a different seed count will move both arms. The finding the ratio supports: `micro` retains strictly LESS signal than `step` for the same wall-clock step, because the block drawn last displaces the one the reference owns. Gradients accumulate across micro-steps (zero_grad once, not per micro-step), each loss is divided by the micro-step count, sigma is redrawn per sample, and the reference is built from a separate engine before any accumulation runs.
- `overlap_g0`: overlap g=0 over 4096 draws per seed; the band is the partitioner's nominal equi-probability range, not the widened draw interval. CONTRADICTS the 'strictly decreases with overlap' framing: this fraction RISES with g, because sample_sigma draws from [lo/alpha, hi*alpha] and alpha widens the interval by construction. That widening IS the mechanism by which overlap absorbs out-of-range mass (handoff 12.1). Measuring against the widened interval would be circular (10.3). No model runs for this row.
- `overlap_g0.125`: overlap g=0.125 over 4096 draws per seed; the band is the partitioner's nominal equi-probability range, not the widened draw interval. CONTRADICTS the 'strictly decreases with overlap' framing: this fraction RISES with g, because sample_sigma draws from [lo/alpha, hi*alpha] and alpha widens the interval by construction. That widening IS the mechanism by which overlap absorbs out-of-range mass (handoff 12.1). Measuring against the widened interval would be circular (10.3). No model runs for this row.
- `overlap_g0.25`: overlap g=0.25 over 4096 draws per seed; the band is the partitioner's nominal equi-probability range, not the widened draw interval. CONTRADICTS the 'strictly decreases with overlap' framing: this fraction RISES with g, because sample_sigma draws from [lo/alpha, hi*alpha] and alpha widens the interval by construction. That widening IS the mechanism by which overlap absorbs out-of-range mass (handoff 12.1). Measuring against the widened interval would be circular (10.3). No model runs for this row.
- `sparsity_layer`: dequantized NMSE of a retrofitted attn.c_proj, layer scope, sparsity 0.5 -> 0.75, reached by the gradual schedule at its last ramp event. Masks are KEEP-masks (True = retained) and structural zeros dequantize to exactly 0.0 (verified at the target; verified at the floor). NMSE rising with sparsity is expected (pruning removes weight magnitude), so this row records the mask's cost, not a quality win. No training runs.
- `sparsity_global`: dequantized NMSE of a retrofitted attn.c_proj, global scope, sparsity 0.5 -> 0.75, reached by the gradual schedule at its last ramp event. Masks are KEEP-masks (True = retained) and structural zeros dequantize to exactly 0.0 (verified at the target; verified at the floor). NMSE rising with sparsity is expected (pruning removes weight magnitude), so this row records the mask's cost, not a quality win. No training runs.
- `sparsity_scope_mask_disagreement`: fraction of positions where the layer-scope and global-scope masks disagree at sparsity 0.75, pooled over attn.c_proj and mlp.c_fc (0.3755 in the worse of the two layers). Both scopes prune to the SAME per-layer fraction at every setting measured here, because these two layers' magnitude distributions overlap enough that one global threshold cuts both equally. So the per-scope NMSE rows above cannot distinguish them and would tie by construction. The scopes disagree about WHICH weights survive, and that is what this row measures. Whether global's selection is better after training is NOT measured here: no training runs. Single seed, because the two masks are a deterministic function of the weight tensor, not a stochastic draw.
- `act_lut_matched_budget`: matched-budget parameter count at K_in=K_out=15: 30 proximity parameters (K_in + K_out) vs 225 for a free K_in x K_out logit matrix, a 7.50x reduction. fp8 knot export bit-identical to fp32: True. This is a structural count that depends only on K, so it is exact rather than a fit result. The model's own codebooks have K_in=8 != K_out=15, so this row is the matched-budget statement and the per-preset rows below are the real codebooks.
- `act_lut_logits_pwl`: relaxation=logits body=pwl: 120 table parameters vs 120 for a free K_in x K_out logit matrix at K_in=8, K_out=15 (proximity is K_in + K_out = 23). no fp8 export: the free-logit relaxation stores no knot positions, so the fp8 knot saving does not apply to this preset. Measured this run at K=8: body max abs error on relu^2 over a 400-point grid 0.0808781, resolved-table error 0.0808781. These are re-measured per preset, not copied from the handoff, whose figures came from one setting. No model runs.
- `act_lut_logits_smoothpwl`: relaxation=logits body=smoothpwl: 120 table parameters vs 120 for a free K_in x K_out logit matrix at K_in=8, K_out=15 (proximity is K_in + K_out = 23). no fp8 export: the free-logit relaxation stores no knot positions, so the fp8 knot saving does not apply to this preset. Measured this run at K=8: body max abs error on relu^2 over a 400-point grid 0.146674, resolved-table error 0.0808781. These are re-measured per preset, not copied from the handoff, whose figures came from one setting. No model runs.
- `act_lut_proximity_pwl`: relaxation=proximity body=pwl: 16 table parameters vs 120 for a free K_in x K_out logit matrix at K_in=8, K_out=15 (proximity is K_in + K_out = 23). fp8 knot export bit-identical to fp32: True. Measured this run at K=8: body max abs error on relu^2 over a 400-point grid 0.0808781, resolved-table error 0.0808781. These are re-measured per preset, not copied from the handoff, whose figures came from one setting. No model runs.
- `act_lut_proximity_smoothpwl`: relaxation=proximity body=smoothpwl: 16 table parameters vs 120 for a free K_in x K_out logit matrix at K_in=8, K_out=15 (proximity is K_in + K_out = 23). fp8 knot export bit-identical to fp32: True. Measured this run at K=8: body max abs error on relu^2 over a 400-point grid 0.146674, resolved-table error 0.0808781. These are re-measured per preset, not copied from the handoff, whose figures came from one setting. No model runs.
- `sigma_cond`: STRUCTURAL ONLY. Conditioned stores 2 codebooks (one per anchor) vs one shared: 28 parameters / 240 B vs 14 / 116 B, a 2.0x parameter blow-up. The distributional premise that a block's activation distribution varies with sigma in an exploitable way is UNVERIFIED (handoff 12.2 disproved it on c_proj's relu^2 input). NO accuracy claim is made for this experiment and none should be read into this row.
- `bias_quant_nmse`: BIAS NMSE ONLY, reported separately from the weight and activation errors on purpose: the bias is 256 numbers against a 1024x256 weight matrix, so a pooled relative error would be the weight's by element count and would hide the bias term entirely (the 9.8 pooled-NMSE mistake). Quantized bias NMSE 0.0130425 vs 0 for the FP32 bias, which round-trips through nothing and so scores 0 exactly. 14/15 levels hit. init_from_tensor() was called on the bias before this was measured, because the quantizer's default act_init span (-2..2, fitted to relu^2 outputs) is ~40x wider than a std-0.02 bias: every entry then buckets onto the zero anchor, 1 level is hit, the error is exactly 1.0 and the gradient is identically zero. from_float() performs the same fit, so this is a precondition the probe states and checks, not a fix it depends on.
- `bias_quant_cost`: EXTRA COST PER LAYER: 14 parameters / 116 B for one shared K=15 table, independent of out_features=256 (one shared 1-D table, not one per output channel). The FP32 baseline carries the bias inline at no extra parameters, which is why its value is 0 and `better` is always `baseline` here: this row is a price tag, not a comparison anyone can win. Judge it against the bias NMSE row.
- `bias_quant_liveness`: codebook gradient live: True. L2 distance the bias codebook's parameters moved over 5 SGD steps at lr=1.0 (mean over 8 seeds), from a real backward pass through LCQATLinear.forward: observed 0.000172562 against a representable floor of 5.76705e-07 (one fp32 ULP at these latents' magnitude, times the step count). Liveness requires both a non-zero gradient and movement at or above that floor: the gradient alone would call the §9.8 dead-codebook case live, and movement alone calls a codebook that is merely too small to update at this learning rate dead. A codebook with a live gradient that does not move at all would mean the learning rate or the gradient scale is wrong; zero gradient means every entry bucketizes to the zero anchor and the error is pinned at exactly 1.0, which is the §9.8 trap this row exists to catch.

<!-- END GENERATED: lcqat_ablation.py -->

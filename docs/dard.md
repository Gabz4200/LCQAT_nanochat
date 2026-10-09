# DARD: Differentiable Algebraic Rule Distillation

This page covers DARD: offline AML rule mining, log-domain rule weighting, the differentiable order-violation loss, and how it attaches to DiffusionBlocks. Read it before touching `nanochat/models/dard/` or any `--dard-*` flag. For the engine it attaches to, see [diffusionblocks.md](diffusionblocks.md).

DARD mines discrete algebraic rules offline and compiles them into a differentiable auxiliary loss for the LM training loop. It is **always on by default** (like LC-QAT and SparseProp); disable with `--no-dard`.

The premise is sample efficiency: instead of relying solely on cross-entropy's statistical trial-and-error, DARD extracts exact order relations from the corpus and penalizes the model when its next-token distribution contradicts them.

## The discrete/continuous boundary

Understand this boundary before reading any code, because it keeps DARD out of the training hot path.

```
PHASE I  (offline, integer set algebra)   corpus to micro-shards to AML atoms to rules
PHASE II (offline, one pass)              rules to weights to two sparse CSR matrices
PHASE III (online, differentiable)        logits to L_AML (a scalar) added to L_CE
```

**Phase I is not differentiable and never runs during training.** It runs once, in `atomize_shard`, and emits token-id sets. **Phase III is not symbolic at all**: it is `softmax`, `log`, one sparse matmul and a `softplus`, all autograd. The two meet at exactly one artifact: a `DardRuleSet` holding two binary `(R, V)` CSR matrices and a weight vector.

There is no online symbolic compute. `denoise_step` and the CE loop touch nothing but tensors.

## Phase I: AML mining (`atomize.py`, `shard.py`)

The corpus splits into non-overlapping micro-shards capped at **N < 1000 tokens** (the PRD's cap; crossing cost grows superlinearly in the atoms a shard contributes).

AML represents a discrete dataset as an atomized semilattice ([arXiv:1803.05252](https://arxiv.org/abs/1803.05252) Sections 2.1-2.2): merge is commutative, associative and idempotent, and in an atomized model merge is set union over atoms (Section 2.5, Eq. 7). The order is induced: `a < b` iff `a` merged with `b` equals `b` (Section 2.2, Eq. 4).

Here an **atom is a minimal group of token ids that co-occur**, where minimal means no proper subset of the group is also a candidate. Storing atoms as `frozenset` makes idempotence free.

**The premise and conclusion must be disjoint token sets, and their arities must be comparable.** Both are silent-failure modes: violate either and the feature reports itself enabled, logs a plausible loss, and contributes no gradient.

*Disjointness.* `_successors` originally unioned the following token into the premise, making `premise` a subset of `conclusion` *by construction*, with every penalty at its floor. `_premise_successors` now subtracts the premise.

*Arity.* The premise is a 4-token window and the conclusion was the single following token. The penalty compares two **product t-norms**, so `v(premise) = O(p^4)` is always smaller than `v(conclusion) = O(p)`: the order can never be violated. Measured on a real 4-layer model before the fix:

```
rules=512  tok/rule: premise 4.0, concl 1.0
v_premise mean=8.4e-19   v_concl mean=3.1e-05
FRAC VIOLATED: 0.0000
penalty=0.069299   floor=0.069315
```

**0% of rules violable**, and the loss curve looked perfectly healthy. Two changes fix it:

1. The conclusion is the set of tokens following *every* occurrence of the premise, not just one. It carries comparable arity and states the real logical claim ("this context licenses these continuations").
2. Premises are mined at **every width from 1 to `PREMISE_WIDTH`**, not just the widest. A 4-token window essentially never recurs in real text, so after (1) it had nothing to pair with and the shard mined *zero* rules. Starting narrow lets the corpus's own statistics pick the arity it can support.

`MIN_BALANCED_ARITY` (2) and `MAX_ARITY_GAP` (1) encode the invariant, and `tests/test_dard_violability.py` pins the behaviour: **98% of 4,114 rules violable on real text** after the fix. Reintroducing the bug fails 5 of its 6 tests.

**Crossing** ([arXiv:2502.19944](https://arxiv.org/abs/2502.19944) Suppl. Def. 15) makes `premise` a subset of `conclusion` by unioning each discriminant atom of the premise with the atoms of the conclusion. Sparse Crossing keeps only unions that preserve the atom *traces* (Suppl. Thms. 26-27). With disjoint sides that test collapses to one question: does the premise contain an atom the conclusion's atoms do not already explain? If not, the pair encodes a coincidence of the window rather than an implication, and `_cross` drops it.

### The compression ratio is AML's, not a per-rule quantity

AML defines **only** a model-level compression ratio:

```
kappa = |R| / Z        (arXiv:1803.05252 Section 3.5, Eq. 36)
```

where `|R|` is the number of relations and `Z` the number of surviving atoms. The PRD's per-rule `C_r,k` is a DARD-specific reading of that: each rule carries the `kappa` of the shard it was mined from. This is stated rather than papered over. There is no per-rule compression in the papers, and `test_when_atomize_then_every_rule_carries_the_shard_compression` pins the reading actually implemented.

## Phase II: weighting and compilation

Every rule is weighted **continuously** rather than thresholded (PRD Section 3.2.1):

```
S_r       = f_r / K                              cross-shard frequency
C_bar_r   = mean of kappa over the f_r shards     local compression
log raw_r = alpha*log(S_r + eps) + beta*log(C_bar_r + eps)
log w_r   = log_softmax(log raw_r / T_w)
w_r       = clamp(exp(log w_r), min=eps_floor), then renormalized
```

Three properties, each a failure mode the naive formula has:

1. **Nothing underflows to zero.** Every operation is a `log`, and the only exponential is of a log-softmax output, which is `<= 0` by construction. With thousands of rules an unweighted softmax would round the tail to exactly 0.0 in fp32.
2. **`eps_floor = 1e-6` must be fp32.** It sits below fp8 E5M2's smallest *subnormal* (`2^-16`, about 1.5e-5) and E4M3's (`2^-9`; Table 1 of [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)). Under a bf16/fp8 autocast the floor is the one quantity silently rounded away, and it is the quantity keeping the tail rules alive. Hence the fp32 buffer, enforced by `DardRuleSet.__post_init__`.
3. **The floor applies before renormalization**, so it bounds the *final* weight. Flooring after would let the normalization undo it.

`f_r` counts **shards**, not emissions: a shard emitting the same key twice still contributes 1, or `S_r` would exceed 1.

Pruning (`--dard-min-shard-frequency`, `--dard-min-compression`) is **off by default** and happens *after* weighting, so `w_r` is computed against the full population and survivors renormalize among themselves. Pruning thresholds, it does not zero a row of an already-normalized vector.

The rules compile to two binary `(R, V)` CSR matrices. **Binary and static** is the CPU-memory argument: `R=186,866` over `V=32768` stores 930,619 int32 entries at **3.7 MB**, against 24 GB for the dense `R x V` fp32 form.

## Phase III: the loss (`loss.py`)

```
log v(T)     = M_T · log(P + eps)                    product t-norm, log domain
violation    = v(premise) - v(conclusion)
L_AML        = (1/gamma) * sum_r w_r * softplus(gamma * violation_r)
```

Evaluating the product t-norm in the log domain is the point: a direct product over more than 5 near-1e-3 tokens underflows, and with it the gradient.

The whole block runs under `torch.autocast(enabled=False)` for three independent reasons, each a documented PRD Section 5 failure mode:

- `log(P + eps)` with `P` near `eps = 1e-8`. bf16 has 8 mantissa bits, so `p + 1e-8` rounds to `p` for any `p > ~1e-6` and the `+eps` guard stops guarding.
- the compiled weights (above);
- `softplus(gamma * violation)` with `gamma = 10`, whose tail gradient is coarse enough in bf16 to be lost.

This matches mixed-precision practice ([arXiv:1710.03740](https://arxiv.org/abs/1710.03740) Section 3.3): reductions and accumulation stay fp32 while the matmuls could in principle run lower. This implementation keeps the whole block in fp32 because the rule count is small relative to the model.

### Loss balancing

```
mu^t     = rho*mu^{t-1} + (1-rho)*L^t
L_total  = L_CE/max(mu_CE, eps) + lambda*L_AML/max(mu_AML, eps)
```

`lambda_global = 0.20` is the share of gradient norm assigned to rule enforcement. The EMA runs over loss *values*, not gradients, so it costs one scalar update per step and needs no extra backward. That is a deliberate departure from [GradNorm](https://arxiv.org/abs/1711.02257), whose `w_i` come from gradient norms at a shared trunk (an extra backward per step). GradNorm's `alpha` restoring force is not provided here and is not claimed.

The first step bypasses normalization (`mu` is still 0, so dividing would multiply CE by 1e5), but **both** terms still pass through unweighted. Returning `loss_ce` alone would leave `loss_aml.grad is None` for exactly one step, silently.

## DiffusionBlocks: how DARD attaches to a loss with no logits

This is the integration problem, and the resolution is worth stating.

The penalty is a function of `softmax(logits)`. `denoise_step` returns denoised **embeddings**; the EDM objective has no logits anywhere. Rejecting DARD under `--db-objective edm`, this repo's default, would make the feature unusable in the default configuration.

The denoiser is already a classifier: `generate()` projects the denoised state through `lm_head` and samples from the result. `denoiser_logits` (`engine_hook.py`) reproduces exactly that projection (slice the padded vocabulary, cast fp32, softcap at 15.0), and DARD attaches there. The distribution being penalized is the same one the sampler's next-token decision uses. No new inference concept, no second head. `test_when_denoiser_logits_are_projected_then_they_match_the_generation_path` pins the agreement.

### A deliberate behaviour change

`lm_head` currently receives **no** gradient from the EDM objective: its target is embeddings, and `tests/test_db_edm_behavior.py` pins that deliberately. DARD under EDM creates a new, legitimate gradient path into `lm_head`: it now trains, through the rule penalty, to put mass on rule-consistent tokens. The data term is unchanged. Both halves are pinned in `test_when_dard_is_on_then_lm_head_finally_receives_a_gradient_under_edm`.

The engine publishes `last_dard_logits` and does **not** add the penalty itself; the loop owns the balancing. Folding it in at the engine would make the lambda the engine sees and the lambda the loop applies two different numbers.

### The per-block head, and what it costs

With only the above, the penalty reads the *final* block's hidden state. But `denoise_step` executes one block per step, and `_activate_block` froze the others, so the rule penalty reached **no** transformer layer at all. Measured before the fix: layers with non-zero DARD gradient = `[]`. The rules were being enforced against a representation no rule could shape.

The fix projects **every** block's own hidden state through the **same** `lm_head` (`_run_block_denoiser(collect_block_logits=True)`) and makes every block require grad, so the penalty reaches all of them. One weight does the work: no second head, and no new parameters or checkpoint keys. Measured after: every layer of every block, non-zero, for every sampled block index.

**This trades away forward isolation.** Running every block is L layers per step, not L/B, the saving DiffusionBlocks exists for. So it is opt-in (`set_dard(..., per_block_head=False)` to decline), not a silent default. `tests/test_dbcpu_denoise.py::test_when_ce_and_edm_objectives_then_only_edm_isolates_the_forward` caught the regression when it was on by default, and two tests now pin both sides: `test_when_the_per_block_head_is_off_then_block_isolation_is_preserved` and `test_when_dard_is_off_then_no_block_logits_are_collected`.

Block-isolation in the *noise-specialization* sense is untouched either way: the data loss still trains one block per step. What per-block adds is that the auxiliary rule term updates every block too.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-dard` | off | Disable DARD (default: on) |
| `--dard-rules PATH` | empty | Load a compiled rule set instead of mining |
| `--dard-shards N` | `8` | Aim for N micro-shards, capped at 999 tokens each |
| `--dard-mine-tokens N` | `0` | Mine from the first N tokens of the batch (0 = all) |
| `--dard-lambda` | `0.20` | `lambda_global`, share of gradient norm for rule enforcement |
| `--dard-gamma` | `10.0` | Softplus hinge temperature |
| `--dard-ema-decay` | `0.99` | EMA decay `rho` for the balancer |
| `--dard-weight-temperature` | `2.0` | `T_w`, rule-weight temperature |
| `--dard-alpha` / `--dard-beta` | `1.0` / `1.0` | Exponents on frequency / compression |
| `--dard-min-shard-frequency` | `1` | Prune rules seen in fewer shards (1 = keep all) |
| `--dard-min-compression` | `0.0` | Prune rules below this `kappa` (0 = keep all) |
| `--dard-max-rules` | `2048` | Cap on rules per shard |
| `--dard-workers` | `0` | Mining worker processes (0 = serial) |

`--dard-shards` matters more than it looks: `S_r = f_r / K` is the Global Invariance term, and at `K = 1` every rule has `f_r = K`, so `S_r = 1` for all of them and the term carries no signal. The shard *size* therefore derives from the requested shard *count* and caps at the PRD's 999. Fixing it at the cap would produce exactly one shard on a small batch.

DARD is deliberately **absent** from the `--db-blocks<=0` rejection tuple in `build_engine`: it is a vocabulary-space loss and both objectives define one.

## Mining offline

```bash
python -m scripts.dard_mine --tokens 2000000 --out ~/.cache/nanochat/dard.pt
python -m scripts.base_train --dard-rules ~/.cache/nanochat/dard.pt
```

Mining reads the same parquet shards the dataloader reads, so the rules describe the corpus the model will train on. Without `--out` it mines and reports without saving. Startup mining (the default, no `--dard-rules`) reads the first training batch instead.

Artifacts are version-stamped (`RULE_SET_VERSION`) and a stale version is **rejected loudly**, for the same reason `NOISE_MAP_VERSION` exists: a rule set compiled under a different weighting convention has identical shapes and dtypes, so it would load and silently enforce different rules.

### The rule set is part of the checkpoint

DARD's modules are **optimizer-side attachments**, so they are *not* in `engine.state_dict()`. Measured before this was fixed: 33 keys in the saved state dict, none of them DARD's.

That made a resume re-mine from whichever batch it happened to see, enforcing a *different* rule set than the loss curve up to that point was measured under, with nothing in the logs to say so. Each save therefore writes a sidecar `dard_{step:06d}.pt` beside the checkpoint, holding the rule matrices and the balancer EMA, and `base_train` prefers it on resume:

```
DARD: restored 96 compiled rules and the balancer EMA from the checkpoint sidecar
```

A sidecar rather than a meta field because `save_checkpoint` does `json.dump(meta_data)`: putting sparse tensors there would crash every save. The JSON carries provenance only (`num_rules`, `vocab_size`, `gamma`, `steps`), which is enough to tell whether a sidecar applies.

Both payloads are version-stamped (`RULE_SET_VERSION`, `DARD_STATE_VERSION`) and a stale one is rejected rather than loaded. Covered by `tests/test_dard_checkpoint.py`.

## Measured

Overhead, in-process on this host (4 threads, R=90, V=128, batch 2x64), sharing one model forward so the number is DARD's **marginal** cost. Two runs:

```
CE fwd+bwd:         31.50 ms          CE fwd+bwd:         32.87 ms
CE + DARD fwd+bwd:  34.69 ms          CE + DARD fwd+bwd:  34.88 ms
marginal: +3.19 ms  (+10.1%)          marginal: +2.01 ms  (+6.1%)
```

Both sit inside the PRD Section 8.2 budget of 15% or less. Regenerate with `pytest tests/test_dard_overhead.py -s`. It is a measurement, not a gate: a hardware-dependent threshold would fail on someone else's CPU for reasons unrelated to the code.

The measurement is interleaved best-of-N rather than single-shot. Single-shot timing at the millisecond scale on a shared host produced an *impossible* negative overhead (-9.5%, i.e. the "penalty" made the step faster); taking the minimum of repeated trials cancels scheduler jitter, because noise only ever adds time.

Mining 200,000 tokens takes **2m05s**: 186,866 rules over V=32768, 930,619 stored entries (3.7 MB), weights summing to 1.0 with min 4.41e-6 (above the 1e-6 floor).

`atomize_shard` ran **30x faster** after two fixes (0.98s down to 0.032s per 999-token shard): the atom filter was O(C²) over candidates and is now a size-ordered single pass (O(C · MAX_ATOM_TOKENS), same answer), and the premise-to-successor map builds in one pass instead of rescanning the shard per premise.

## Not implemented, and why

**No native C++/Taichi kernel.** The repo's kernel contract asks for three faces per op, but DARD's penalty composes entirely of existing differentiable primitives (`softmax`, `log`, `torch.sparse.mm`, `softplus`), and `torch.sparse.mm` *is* the single-pass BLAS SpMM the PRD asks for. A custom op would add a JIT-cache-busting `.cpp` (a cache-bust for every downstream user, per AGENTS.md) with no computational benefit. The independent oracle in `nanochat/ops/references/dard_reference.py` is a plain Python loop over rules, so parity is tested against a differently-written implementation.

**No torch.library op / analytic backward.** Same reason: there is no compiled op to wrap. The penalty is differentiable through the standard autograd graph, and `tests/test_dard_loss.py` pins gradient parity against the oracle.

## Tests

| File | Covers |
|------|--------|
| `tests/test_dard_mining.py` | Sharding, atomization, compression, weighting, compilation, artifacts |
| `tests/test_dard_loss.py` | Value **and gradient** parity vs the oracle, underflow, validation, balancer |
| `tests/test_dard_diffusion_blocks.py` | lm_head projection, EDM/AR/RL paths, LC-QAT coexistence, flags |
| `tests/test_dard_checkpoint.py` | Rule-set + EMA round-trip, sidecar contract, version rejection |
| `tests/test_dard_smoke.py` | Real `base_train` with DARD on / off / `--db-blocks=0` |
| `tests/test_dard_overhead.py` | Marginal cost measurement |

## Running all four technologies

`runs/night4.sh` is a worked example with LC-QAT, SparseProp, DiffusionBlocks and DARD all on. `scripts/dard_report.py` generates from the result and reports repetition rate and validation bpb:

```bash
bash runs/night4.sh
python -m scripts.dard_report --model-tag night4
```

### What this host can actually do

Measured on the 4-core i5-8250U this was tuned on, not assumed:

| | measured |
|---|---|
| bf16 matmul ceiling | **0.7 TFLOP/s** |
| full-depth AR micro-step, depth 6 | 5.2 s |
| DiffusionBlocks B=2 / B=3 | 2.9 s / **2.0 s** (1.8-2.7× faster) |
| with LC-QAT + SparseProp on top | 8.2 s per micro-step |
| end-to-end | ~135 s/step at a 32k batch, ~230 tok/s |

DiffusionBlocks is a real and worthwhile **1.8-2.7× wall-clock** win. It is not a token-throughput win in the sense that changes the budget: every token still passes through every layer over a full block cycle, so it cuts activation memory and FLOPs per micro-step, not the number of tokens the machine can afford to train on. Both facts matter when sizing a run.

At 230 tok/s a 6.5-hour budget is ~5.4M tokens. For a 37M-parameter model that is **0.15 tokens/parameter**, against the ~20 that Chinchilla calls optimal. The compute-optimal size for this budget is therefore closer to 8M parameters, and any run here is heavily data-starved. Expect a real loss curve and locally well-formed English, not GPT-2 quality. GPT-2 small (124M) on this box would need ~120x the budget.

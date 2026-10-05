# DARD verification: the five questions

Measured, not asserted. Run 2026-10-05 against the current tree.

## Q1. Is the AML training correct? **YES**

The discrete half must satisfy AML's own algebra before anything is worth
compiling.

```
check idempotence (AML: c ⊙ c = c)      : True
check merge = set union (2018 Eq. 7)     : True
check induced order a<b iff a⊙b == b     : True
```

On a motif corpus (`[1,2,3,4]*60 + [9,9,9,9]*10`) the mined rules are
semantically right, not just well-formed:

```
P=[4]    C=[1,9]   disjoint=True  gap=-1
P=[3,4]  C=[1,9]   disjoint=True  gap=0
```

Premise `[4] → conclusion [1,9]` is the ABAB-pattern implication, and
every rule is disjoint with `|premise| - |conclusion| < MAX_ARITY_GAP`, the
invariant that makes a rule violable at all (`dev/DARD_AUDIT.md` §0, where
violating it gave a **0.0000** violation fraction).

## Q2. Is the conversion to a continuous loss correct? **YES**

Value *and* gradient, against the independent Python-loop oracle in
`nanochat/ops/references/dard_reference.py`:

```
compiled = 0.06930930   oracle = 0.06930930   rel err = 0.00e+00
gradient max|diff| = 4.5e-13
```

Exact to fp32 round-off: the sparse `M · log P` path and the explicit
per-rule sum are the same function, not merely close.

## Q3. Does the aggregation avoid the "average is zero" problem? **YES**

The penalty is a weighted sum with a partition of unity, not an RMS, MSE,
or mean-over-rules plus rescale:

```
L = (1/gamma) * Σ_r  w_r · softplus(gamma · violation_r)      Σ_r w_r = 1
```

- `weights sum to 1.000000`: a partition of unity, so the penalty's scale
  comes from the rules' *shape*, not their count;
- up-weighting one rule moves the total (`w=[0.700,0.300] → 0.06930930`,
  `w=[0.824,0.176] → 0.06931026`), so the weighting matters rather than decorates;
- at uniform logits every violation is zero and the value is the hinge floor
  `log(2)/gamma = 0.06931472`, matching the analytic value exactly.

No mean shrinks a sparse signal toward zero: `w` is normalized once, and each
rule contributes in proportion to its own violation.

## Q4. Is there a global scalar making it compete with cross-entropy? **YES**

`gradient_scale`, applied through `_ScaleGradient` (identity forward, scaled
backward) so the logged value stays the PRD formula while the gradient
is corrected:

| `gradient_scale` | DARD grad norm | CE grad norm | ratio |
|---|---|---|---|
| 1 (literal PRD) | 1.13e-05 | 5.87e-01 | 0.0000 |
| 100 | 1.13e-03 | 5.88e-01 | 0.0019 |
| 6.6e+05 (derived default) | 7.47e+00 | 5.85e-01 | **12.77** |

The PRD's own EMA balancer cannot do this: it rescales loss *values*, and
dividing both terms by scalars leaves the gradient ratio unchanged. That is
why the correction applies to the gradient. At `lambda_global = 0.20` the
auxiliary term receives ~20% of the cross-entropy gradient, the only reading
under which "the proportion of gradient norm assigned to rule enforcement"
means anything.

The default is derived, not guessed: `V ** mean_premise_arity`, because the
gradient scales with `v(premise)`, a product of probabilities, which decays
geometrically in the rule's arity.

## Q5. Under DiffusionBlocks, does DARD reach every block? **NO. This was a real bug.**

Measured on a clean engine, before any fix:

```
active block 0: transformer layers with DARD grad = []
active block 1: transformer layers with DARD grad = []
```

**Cause.** `denoise_step` runs only the active block's layers, and
`_activate_block` sets `requires_grad=False` on every other block. The DARD
penalty reached neither the active block's layers nor any other block's,
only the shared `lm_head` it reads through. Every rule constrained a
representation no rule could shape.

**Fix.** A per-block language-model head sharing the single `lm_head` weight,
applied alongside each block's own forward, so the penalty's gradient flows
from the head back into the block that produced it, for every block.

### After the fix

Per-block gradient, measured on a 4-layer / 2-block model, both block indices:

```
block 0: L0=1.26e+04  L1=8.73e+03  L2=4.37e+03  L3=3.72e+03
block 1: L0=7.61e+03  L1=4.80e+03  L2=2.16e+03  L3=1.89e+03
lm_head: 2.53e+02
```

**Cost, honestly.** Running every block is L layers per step, not L/B. That is
forward isolation — the property DiffusionBlocks exists for — so the per-block
head is opt-in via `set_dard(..., per_block_head=False)`. This was not free:
the first version was on by default and broke
`test_when_ce_and_edm_objectives_then_only_edm_isolates_the_forward`
(`4 == 2`). Both behaviours are now pinned by tests in both directions.

## Proof of concept: `runs/poc15.sh`

All four technologies, one run, 13.03 min of step time on a 4-core i5-8250U
(depth 3, n_embd 256, seq 256, batch 12288, 16 steps).

```
LC-QAT enabled
SparseProp injected 18 sparse Linear layers   (model)
SparseProp injected 24 sparse Linear layers   (engine-owned)
DiffusionBlocks Engine with 2 independent blocks
DARD: mined 152 rules over 8 shards (178 premise nnz)
LM_HEAD_DARD_GRAD=2.532060e+02 nonzero=True
```

An earlier 24-step / batch-16384 configuration took **20.69 min** and missed the
budget; it was resized to 16 steps / batch 12288, which measures 13.03 min.

| metric | value |
|---|---|
| final train loss | 2.154 |
| DARD penalty | 0.069321 |
| validation bpb | 3.312605 |
| bits/char on held-out text | **1.6237** |
| repetition-4, 5/5 prompts | **0.00** |

The text is not yet English — a 3-layer 15M-parameter model trained for 13
minutes on CPU is well below the fluency threshold — but repetition-4 is 0.00 on
all five prompts, so it is not degenerating into loops. `night4` (EDM
objective) produced 2.12 bits/char for comparison.

### A correction worth recording

An earlier revision of this report stated that all five prompts returned the
same continuation, and attributed it to the greedy decoder. Both halves were
wrong. Top-k sampling was already in use (`temperature=0.8, top_k=40`); the
real cause was that `GPT.generate` seeds its own generator from a `seed`
argument that defaults to the constant **42**, so every prompt drew from the
same RNG stream.

Measured, with sampling held constant and only the seed varied:

| | distinct continuations |
|---|---|
| `seed=42` for every prompt | 2 of 5 |
| `seed=100+i` per prompt | 5 of 5 |

`scripts/dard_report.py` now passes a distinct seed per prompt. The corrected
run keeps repetition-4 at 0.00 on all five prompts with genuinely distinct
continuations. Reporting the identical-output observation without first
varying the seed would have blamed the decoder for an RNG artifact.

## Gates

| gate | result |
|---|---|
| `ruff check .` | clean |
| `ruff format --check .` | 219 files already formatted |
| `pyrefly check` | 615 errors — exactly the baseline |
| `pytest -m "not slow"` | 6 failed + 28 errors = 34, exactly baseline; **0 new** |
| `pytest tests/test_dard_diffusion_blocks.py` | 29 passed |

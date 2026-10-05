# night4: all four technologies, results

`bash runs/night4.sh`, 2026-10-05. Depth 4, 36.7M params, seq 512, batch 32768,
100 steps, `--db-blocks=2`. LC-QAT, SparseProp, DiffusionBlocks and DARD all on
and confirmed active in the log.

## Configuration, and why it is small

| | measured on this host |
|---|---|
| CPU | Intel i5-8250U, 4 cores / 8 threads, 7 GB RAM |
| bf16 matmul ceiling | **0.7 TFLOP/s** |
| full-depth AR micro-step (depth 6) | 5.2 s |
| DiffusionBlocks B=2 / B=3 | 2.9 s / **2.0 s**, a real 1.8–2.7× win |
| with LC-QAT + SparseProp on top | 8.2 s per micro-step |
| end-to-end | **127 s/step**, ~258 tok/s |
| total | 100 steps in **188 min** (3h08) |

DiffusionBlocks wins on wall clock, and SparseProp holds the LC-QAT path at
parity instead of slowing it (per `docs/sparseprop.md`). Neither changes the
*token* budget: every token still passes through every layer over a full block
cycle, so at 258 tok/s a 6.5h window is ~6M tokens.

0.5B parameters was requested. **It is not reachable here**: 0.5B needs 3.0
GFLOP/tok, so 6h at 100% of this machine's measured peak is 5.0M tokens against
Chinchilla's 10,000M, short by ~2000×. GPT-2 small (124M) is short by ~120×.
Sizing to the budget instead gives depth 4 / 36.7M, still ~4× above the
compute-optimal ~9M for 24M tokens.

## Results

| | |
|---|---|
| train loss | 4.52 → **0.55** |
| trainer val "bpb" | 3.208 → **3.679** |
| nats/byte on fresh text (correct metric) | **3.06** |
| bits/char | **2.12** (random ≈ 11.5, good English 1.0–1.5) |
| 4-gram repetition | **0.00** |
| DARD penalty | 0.069326 → 0.069321, live throughout |
| rules | 353, 98% violable |

### The two findings that matter

**1. Validation loss rose monotonically: 3.208 → 3.679.** Training loss fell
8× while validation worsened. That is a run memorizing rather than learning
general structure, consistent with 3.3M tokens seen against 36.7M parameters
(0.09 tokens/param, vs Chinchilla's ~20).

**2. Generation is word-salad, not GPT-2 level.**

```
'The capital of France is Litpalistically silver Shark not cascventh microscopic
 Later oxidation DoctorMV FS oats ...'
```

Real English fragments with `rep4=0.00`: local structure learned (no
repetition, 2.12 bits/char), nothing compositional. Expected at 0.09
tokens/param, not a bug, but below the "GPT-2 level coherence" bar, and not
described as meeting it.

## A measurement bug I hit and fixed

My first report said 15.2 bpb, contradicting the trainer's 3.68. The trainer
divides summed nats by summed target **bytes** (`evaluate_bpb`); my script
divided by token **count**. On this tokenizer that overstates by ~4×. Fixed in
`scripts/dard_report.py` to use `get_token_bytes`: 3.06 nats/byte =
2.12 bits/char.

Had I reported the first number unchecked against the trainer's, I would have
reported a 4× worse model than exists.

## On the AML / DARD convergence claim

**Untested, and this run cannot test it.** No CE-only baseline arm exists here,
so nothing isolates DARD's contribution. What holds:

- the penalty is *live*: 98% of rules violable, gradient ratio vs CE = **1.09**
  after the `gradient_scale` correction, so `lambda_global=0.2` puts ~22% of
  the CE gradient behind rule enforcement;
- overhead is +6% to +10% marginal (PRD budget ≤15%).

What does **not** hold: that this accelerates convergence. On AML's argument
that needs ~20–100× fewer tokens; at 0.09 tokens/param there was no headroom
for it to show, and a single arm cannot separate it from noise.

The next experiment to run: the same config with `--no-dard`, same seed and
corpus, compared on val loss at matched steps.

## The two bugs that made DARD inert

Both were silent: the loss curve looked healthy in each case.

1. **Every rule was permanently satisfied.** A 4-token premise against a
   1-token conclusion gives `v(premise) = O(p⁴) < v(conclusion) = O(p)`
   always. Violation fraction: **0.0000**. Fixed by balancing premise and
   conclusion arity and mining premises at every width; now **98% violable**.

2. **The gradient was ~5 orders of magnitude too small.** CE 5.0e-01 vs DARD
   9.4e-07. The PRD's EMA balancer cannot fix this: it rescales loss *values*,
   and dividing both by scalars leaves the gradient ratio unchanged. Fixed with
   a `gradient_scale` in a custom autograd Function (identity forward, scaled
   backward, logged value stays the PRD formula). Now 1.09.

Both are covered by `tests/test_dard_violability.py` (6 tests); reintroducing
either fails them.

## night4 vs night5: the `lm_head` question, settled

**Solid.** Under `--db-objective edm`, `denoise_step` produces **no gradient**
for `lm_head`. Measured on a clean engine with no checkpoint interference:

```
EDM backward: lm_head max delta = 0.000e+00
lm_head .grad is None? True
```

`lm_head` is 8.39M of 36.70M params (22.9%) and the only map from hidden state
to token ids, so a purely-EDM-trained checkpoint projects through an untrained
head.

**Not established.** Whether night4's *saved* head moved. Checkpoint-vs-fresh-init
is not a valid test here: init is seed-dependent (two fresh inits differ by
max 0.125), and per-layer ckpt-vs-init distances came out non-monotonically
(`wte` 6.98, `lm_head` 0.74, `c_q` 0.16); those numbers track init scale more
than training. I over-read them twice and walked both claims back.

Instead of arguing it out, night5 runs with explicit `--db-objective ce`,
where the head provably trains (verified before launch:
`lm_head.weight.std()` moves off init after 3 steps). Generation quality
decides the question, a measurement, not an inference.

**The config change stands either way.** `--db-objective ce` trains `lm_head`
*by the data term*, not by a side-channel. Depending on an auxiliary loss as
the sole trainer of 23% of the model is not a configuration to ship, whatever
night4's head did.

Guardrail added: `tests/test_db_objective_generates.py` (6 tests) pins both
branches' `lm_head` gradient behaviour, the 15%-of-parameters floor, and that
`runs/night5.sh` states its objective explicitly. `AGENTS.md` records the
consequence alongside the other engine guardrails.

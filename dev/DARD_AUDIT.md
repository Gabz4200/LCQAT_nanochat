# DARD audit: gaps found and fixed

An audit of the DARD implementation asked "is anything missing?" **Six** real
gaps surfaced; all are fixed. Four are silent failures that would leave a
multi-hour run looking healthy while doing nothing useful.

## 0. Every mined rule was permanently satisfied: silent, worst of all

Found by noticing that `dard: 0.069299` was **byte-identical on every logged step**, then that `0.069315` is the hinge floor `log(2)/γ`. The penalty sat at
its minimum.

Cause: the premise was a 4-token window and the conclusion a **single token**.
The penalty compares two product t-norms, so `v(premise) = O(p⁴)` always
undercuts `v(conclusion) = O(p)`. The order could never break.

```
before:  rules=512  premise 4.0 tok/rule, concl 1.0 tok/rule
         v_premise mean=8.4e-19   v_concl mean=3.1e-05
         FRAC VIOLATED: 0.0000      penalty=0.069299  floor=0.069315
after:   rules=4114 premise 1.06, concl 3.73  (real text, 60k tokens)
         FRAC VIOLATED: 0.9810      penalty=0.069329
```

The first fix overcorrected to zero rules: balancing arity requires the
premise to recur, and a 4-token window almost never does in real text. Premises
are now mined at *every* width from 1 to `PREMISE_WIDTH`, letting the corpus's
own statistics choose the arity it supports.

`MIN_BALANCED_ARITY = 2` and `MAX_ARITY_GAP = 1` encode the invariant.
`tests/test_dard_violability.py` (6 tests) pins it; reintroducing the bug fails
5 of the 6.

Why this matters most: the other gaps would degrade a run. This one made
DARD *inert* while every observable said it worked: rules mined, penalty
logged, loss finite, curve falling. No original assertion caught it, because
each checked structure or finiteness. The checks that catch it are
behavioural: violation fraction, penalty above floor, penalty responds to a
distribution shift.

## 1. The rule set was not in the checkpoint: silent, serious

`DardRuleLoss` is an optimizer-side attachment, so it is **not** in
`engine.state_dict()`. Measured before the fix:

```
DARD buffers in engine.state_dict(): NONE   (33 keys, none of them ours)
```

A resume re-mined from whichever batch it saw next and enforced a *different*
rule set: the loss curve measured one objective before the resume and another
after, with nothing in the logs to say so. This is the failure
`NOISE_MAP_VERSION` prevents for the DiffusionBlocks noise map.

**Fix.** `dard_state` / `load_dard_state` (version-stamped, `RULE_SET_VERSION`
style) plus a sidecar `dard_{step:06d}.pt` beside each checkpoint. The tensors
cannot live in the meta dict because `save_checkpoint` runs `json.dump(meta_data)`,
and sparse tensors there would crash every save. The JSON carries provenance only
(`num_rules`, `vocab_size`, `gamma`, `steps`).

`base_train` prefers the sidecar on resume and re-mines only when none exists.
Verified: *"DARD: restored 96 compiled rules and the balancer EMA from the
checkpoint sidecar"*. Guarded by `tests/test_dard_checkpoint.py` (9 tests).

Two bugs surfaced during the build, both mine:
- a 0-row template cannot accept a real rule set (`[0,V] <- [96,V]` rejected);
- `load_state_dict` cannot `copy_` into a CSR with a different nnz count, so
  the buffers are assigned directly rather than merged.

## 2. The balancer EMA was never checkpointed: silent

`mu_ce` / `mu_aml` restart at 0 on resume, so the first step takes the warmup
bypass and the first steps run unbalanced: one step of drift, small and
avoidable, now three scalars in the same sidecar.

## 3. `rule_set_to_meta` was defined, documented, and never used

Its docstring says *"The `meta["dard"]` payload recorded in a training
checkpoint"*. Nothing called it except its own test, a defined-but-unused
contract, which AGENTS.md calls a violation. Every other always-on stage writes
meta (`kd`, `db`, `efqat_latch`, `sigma_codebook`, `sparseprop`); DARD wrote
none. Fixed as part of #1.

## 4. `apply_lcqat` crashed on any resume: pre-existing, now fixed

`chat_sft` could not start on this host at all:

```
ValueError: from_float expects a plain float Linear, got an LCQATLinear
  chat_sft.py:322 -> diffusion_blocks.py:704 -> _retrofit_plain_linear
```

Confirmed pre-existing by stashing all DARD work and reproducing on the clean
tree. Cause: `apply_lcqat` re-retrofits already-quantized layers.

- the **denoise-head loop** had no guard at all;
- the **adapter-MLP loop** guarded with `isinstance(linear, LCQATLinear)`,
  which misses `SparsePropLinearLCQAT`: functionally LC-QAT but not a subclass. The same trap `_wrap_sparseprop`'s own comment warns about.

Both now use duck-typed `is_lcqat_layer`, matching the documented convention.
`chat_sft` starts.

## 5. `chat_sft` mined 0 rules: data-dependent, warning fires correctly

With a 128-token SFT batch split into 8 shards, each shard is 16 tokens and no
atom survives irreducibility filtering. The penalty is then exactly 0.0 while
DARD still reports itself enabled.

The **warning works**: this is the "enabled-but-dead anchor is visible"
property the balanced loss curve alone cannot show, and why the raw penalty is
logged every step rather than only the normalized total. Not fixed by lowering
`PREMISE_WIDTH`: at that width the rules stop being multi-token implications.
The remedy is more tokens at mining time (`--dard-mine-tokens`, or
`scripts/dard_mine.py`).

## Verified working

- `base_train` with DARD on / off / `--db-blocks=0`: all four technologies
  confirmed active in the log, penalty non-zero, `--no-dard` a clean removal.
- `chat_sft` with DARD: starts and attaches the penalty (after fix #4).
- `chat_rl`: wiring reviewed; `last_dard_logits` is published by `logprobs`
  and read immediately after, and both engine call sites reset it.
- Loss parity vs the independent Python-loop oracle, values *and* gradients.

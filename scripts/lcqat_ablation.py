#!/usr/bin/env python

"""
Ablation driver for the LC-QAT claims.

Runs the nine paired sweeps §6.5 of the handoff asks for -- codebook split,
codebook gradient scaling, training objective, block sampling, sigma overlap,
SparseProp sparsity/scope/gradual pruning, activation-LUT relaxation/body,
sigma-conditioned codebooks, and bias quantization -- and writes a leaderboard
to `dev/LEADERBOARD.md`.

## What this measures, and what it does not

The first two measure what can be measured on a static probe:

* **Codebook split quality.** The NMSE of a weight-quantizer round trip, plus
  how many of the codebook's levels are actually hit. `asym` exists because
  `relu^2` outputs are non-negative, so a symmetric codebook wastes half its
  levels; level utilization is the direct observation of that claim.
* **Gradient scaling.** The ratio of the codebook gradient with and without the
  PRD 2.4 `1/sqrt(N)` scale, plus how far the codebook moves per optimizer
  step. This is the gradient-level evidence, and it should agree to `1/sqrt(N)`
  to numerical precision.

The other six measure *structure* rather than quality, because the questions
they answer are about wiring and cost, not about accuracy:

* **Objective** (`ce` vs `edm`): how many transformer layers execute, and how
  far a block's denoiser output moves from a non-negative `relu^2`-shaped probe
  after a fixed number of steps.
* **Block sampling** (`step` vs `micro`): what fraction of the isolated
  per-block reference gradient actually reaches the optimizer.
* **Overlap** (0.0 / 0.125 / `--db-overlap`): the fraction of sampled
  `(sigma, block)` pairs that land outside the block's nominal sigma band.
* **Sparsity** (level x `{layer, global}`, with gradual pruning): the NMSE of
  the dequantized weight after pruning, and whether pruned positions
  dequantize to exactly `0.0`.
* **Activation LUT** (`{logits, proximity}` x `{pwl, smoothpwl}`): the table's
  parameter count, whether the fp8 knot export resolves to a bit-identical
  index table, and the fitted body's error against `relu^2`.
* **Sigma conditioning** (`shared` vs `conditioned`): codebook parameter count
  and artifact bytes. **Structural only** -- see the note in `run_sigma_cond`.

The ninth measures a real fidelity number, on the bias alone:

* **Bias quantization** (`fp32 bias` vs `bias through its own codebook`): the
  bias round-trip NMSE **on its own**, the extra table's parameter count and
  bytes per layer, and whether that table's parameters receive a non-zero
  gradient after a few steps. The bias error is reported separately from the
  weight and activation errors on purpose: pooling the three would let the
  weight term mask the bias term, which is the §9.8 pooled-NMSE mistake this
  package already made once.

It does **not** measure end-task perplexity, and it does not measure whether
sigma conditioning would help accuracy. No row in the leaderboard is an
accuracy number, and every row states its metric for that reason.

## Protocol

Every comparison is **paired**: the seed, the input tensor, and the model init
are identical across arms and exactly one factor varies. Unpaired runs at these
effect sizes measure seed variance instead. Deltas are computed per seed and
pooled afterwards, so the per-seed offset cancels.

## Reduced protocol

`dev/HANDOFF_symbiosis.md` §6.5 specifies a "fixed d6/200-step protocol". **This
driver does not run that**, and no row here is a 200-step measurement. The host
has 7.6 GB of RAM, and a real 200-step d6 run per arm across six experiments is
not affordable on it. The reduction is stated here rather than left implicit:

* **Seeds**: `--seeds` (default 5). Every arm at a given seed sees byte-identical
  model init, probe tensors, noise draws and generator seeds.
* **Optimization steps**: `--grad-scale-steps` (default 5) for `grad_scale`,
  `--objective-steps` (default 3) for `objective`, and
  `--block-sampling-micro-steps` (default 4) micro-steps accumulated into one
  step for `block_sampling`. No other experiment runs an optimizer.
* **Probe sizes**: `--n` (default 1024) rows for the codebook probes; a
  `--ablation-batch` x `--ablation-seq` (default 2 x 8) non-negative block
  probe; `--overlap-samples` (default 2048) sigma draws per block; a 2-layer
  engine over `--ablation-n-layer` (default 4) transformer layers in
  `--ablation-blocks` (default 2) diffusion blocks; and
  `attn.c_proj` + `mlp.c_proj` as the two pruned layers.

Every one of these is a micro-probe on a tiny random model. None of them is a
training run, and none of them establishes an end-to-end quality claim:

* `asym_vs_small` and `grad_scale` are probes on a random tiny model, not a
  training run, and do not establish an end-to-end quality claim.
* `objective` runs 3 SGD steps on a 2-layer random model and reports a
  block-output drift, not a converged quality difference.
* `block_sampling` runs 4 micro-steps on a 2-layer random model and reports
  gradient retention, not a convergence property.
* `overlap` samples the partitioner's sigma distribution; no model runs at all.
* `sparsity` prunes a freshly-retrofitted random model once per ramp event and
  reports dequantized NMSE; no training happens.
* `act_lut` fits a `SmoothPWL` body to `relu^2` and counts table parameters; no
  model runs.
* `sigma_cond` counts codebook parameters and bytes. Its distributional premise
  -- that a block's activation distribution varies with sigma in a way a
  conditioned codebook could exploit -- is recorded as **UNVERIFIED**, because
  the handoff's own probe (§12.2) failed to find it. No accuracy claim is made
  for it and none should be read into its row.
* `bias_quant` runs `--bias-quant-steps` (default 5) SGD steps on one random
  layer's bias codebook and reports its round-trip NMSE, its per-layer cost, and
  whether the table moved. That is a micro-probe, not a training run: it says
  nothing about what the extra codebook does to a model's loss after real
  training, and it measures a *randomly initialized* layer's bias rather than a
  trained one, so the magnitude it reports is the pessimistic end of the range.

## Usage

```bash
# Default sweep, 5 paired seeds, tiny model
uv run python scripts/lcqat_ablation.py

# Wider sweep
uv run python scripts/lcqat_ablation.py --seeds 8 --n 4096 --out dev/LEADERBOARD.md

# One experiment
uv run python scripts/lcqat_ablation.py --experiment asym_vs_small
uv run python scripts/lcqat_ablation.py --experiment act_lut --lcqat-lut-relaxation proximity
uv run python scripts/lcqat_ablation.py --experiment bias_quant --bias-quant-k-bias 15

# The DiffusionBlocks probes, with the knobs they measure
uv run python scripts/lcqat_ablation.py --experiment overlap --db-overlap 0.25
uv run python scripts/lcqat_ablation.py --experiment block_sampling --db-block-sampling micro

# Print to stdout without touching the leaderboard
uv run python scripts/lcqat_ablation.py --dry-run
```

Exits non-zero if a measured claim comes out the wrong way for a variant the
write-up advertises, so CI catches a regression in the method itself rather
than leaving it to be noticed. Only *structural* claims are gated. A
measurement that contradicts a documented expectation -- a losing `smoothpwl`,
a growing overlap error rate -- is reported in the row's `note` and is not a
failure, because a probe finding is a result and not a regression.
"""

from __future__ import annotations

import argparse
import json
import math  # noqa: F401
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# The published table. `--out` defaults here for the documented command line;
# a programmatic `main()` call must name its own destination instead.
PUBLISHED_LEADERBOARD = REPO_ROOT / "dev" / "LEADERBOARD.md"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.models.quant.ablation_metrics import (  # noqa: E402
    AblationRow,
    GradScaleObservation,
    measure_reconstruction,
    observe_grad_scale,
    quantization_error,
    reconstruct_layer,
    render_leaderboard,
    row_to_dict,
    select_quantizer,
)
from nanochat.models.quant.activation import (  # noqa: E402
    ACT_BODIES,
    ACT_BODY_PWL,
    ACT_BODY_SMOOTHPWL,
    SmoothPWL,
    get_activation,
)
from nanochat.models.quant.bias_quant import DEFAULT_K_BIAS  # noqa: E402
from nanochat.models.quant.learnable_lut import (  # noqa: E402
    RELAXATION_LOGITS,
    RELAXATION_PROXIMITY,
    RELAXATIONS,
    LearnableIndexLut,
)
from nanochat.models.quant.linear import (  # noqa: E402
    GRAD_SCALE_INV_SQRT_N,
    GRAD_SCALE_NONE,
    LCQATLinear,
)
from nanochat.models.quant.pruning import (  # noqa: E402
    add_sparseprop_pruning_args,
    schedule_from_args,
)
from nanochat.models.quant.retrofit import (  # noqa: E402
    PRESETS,
    CodebookSpec,
    retrofit_model,
)
from nanochat.models.quant.sigma_codebook import SigmaConditionedCodebook  # noqa: E402
from nanochat.models.quant.sparseprop import (  # noqa: E402
    SCOPE_GLOBAL,
    SCOPE_LAYER,
    inject_sparseprop_layers,
)

# The measurements live in `nanochat.modules.experiments`, one module per
# experiment. Every name below was defined in this file before the split and is
# re-imported from that implementation, so any caller that did
# `from scripts.lcqat_ablation import X` still resolves to the same object.
from nanochat.modules.experiments import (  # noqa: E402
    BASELINE_PRESET,
    BIAS_CODEBOOK_DEAD,
    BIAS_CODEBOOK_LIVE,
    K_MATCHED_BUDGET,
    OBJECTIVE_CE,
    OBJECTIVE_EDM,
    OVERLAP_FLOORS,
    SAMPLING_FAMILIES,
    SAMPLING_MICRO,
    SAMPLING_STEP,
    SCOPES,
    SMOOTHPWL_DEFAULT_INIT_RANGE,
    SPARSITY_FLOORS,
    STEP_ARM_BLOCK,
    VARIANT_PRESET,
    _accumulate_one_step,
    _act_kwargs,
    _length_only_idx,
    _per_block_reference,
    _run_objective_arm,
    _spread,
    _weight_kwargs,
    bias_probe,
    block_probe_tensors,
    build_activation_lut,
    build_probe_engine,
    codebook_of,
    count_codebook_cost,
    engine_named_parameters,
    expected_1_over_sqrt_n,
    grad_retained_fraction,
    matched_budget_codebooks,
    measure_overlap_out_of_band,
    measure_sparsity_nmse,
    non_negative_probe,
    piecewise_linear,
    probe_layer,
    pruned_lcqat_layers,
    run_act_lut,
    run_asym_vs_small,
    run_bias_quant,
    run_block_sampling,
    run_grad_scale,
    run_objective,
    run_overlap,
    run_sigma_cond,
    run_sparsity,
    scope_mask_disagreement,
    sorted_overlap_sweep,
    sparsity_sweep,
)
from nanochat.training.diffusion_blocks import (  # noqa: E402
    EquiProbabilityPartitioner,
    edm_preconditioning,
)

#: This module is consumed as a namespace: `tests/test_lcqat_ablation_driver.py`
#: loads it by path and reaches the measurements through it. Module objects
#: (`argparse`, `json`, `math`, `torch`, ...) are not part of that surface,
#: and the five driver entry points were each listed twice.
__all__ = [
    "PUBLISHED_LEADERBOARD",
    "REPO_ROOT",
    "check_claims",
    "main",
    "register_args",
    "validate_args",
    "write_leaderboard",
    "ACT_BODIES",
    "ACT_BODY_PWL",
    "ACT_BODY_SMOOTHPWL",
    "AblationRow",
    "BASELINE_PRESET",
    "BIAS_CODEBOOK_DEAD",
    "BIAS_CODEBOOK_LIVE",
    "CodebookSpec",
    "DEFAULT_K_BIAS",
    "EquiProbabilityPartitioner",
    "GRAD_SCALE_INV_SQRT_N",
    "GRAD_SCALE_NONE",
    "GradScaleObservation",
    "K_MATCHED_BUDGET",
    "LCQATLinear",
    "LearnableIndexLut",
    "OBJECTIVE_CE",
    "OBJECTIVE_EDM",
    "OVERLAP_FLOORS",
    "PRESETS",
    "Path",
    "RELAXATIONS",
    "RELAXATION_LOGITS",
    "RELAXATION_PROXIMITY",
    "SAMPLING_FAMILIES",
    "SAMPLING_MICRO",
    "SAMPLING_STEP",
    "SCOPES",
    "SCOPE_GLOBAL",
    "SCOPE_LAYER",
    "SMOOTHPWL_DEFAULT_INIT_RANGE",
    "SPARSITY_FLOORS",
    "STEP_ARM_BLOCK",
    "SigmaConditionedCodebook",
    "SmoothPWL",
    "VARIANT_PRESET",
    "_accumulate_one_step",
    "_act_kwargs",
    "_length_only_idx",
    "_per_block_reference",
    "_run_objective_arm",
    "_spread",
    "_weight_kwargs",
    "add_sparseprop_pruning_args",
    "bias_probe",
    "block_probe_tensors",
    "build_activation_lut",
    "build_probe_engine",
    "codebook_of",
    "count_codebook_cost",
    "edm_preconditioning",
    "engine_named_parameters",
    "expected_1_over_sqrt_n",
    "get_activation",
    "grad_retained_fraction",
    "inject_sparseprop_layers",
    "matched_budget_codebooks",
    "measure_overlap_out_of_band",
    "measure_reconstruction",
    "measure_sparsity_nmse",
    "non_negative_probe",
    "observe_grad_scale",
    "piecewise_linear",
    "probe_layer",
    "pruned_lcqat_layers",
    "quantization_error",
    "reconstruct_layer",
    "render_leaderboard",
    "retrofit_model",
    "row_to_dict",
    "run_act_lut",
    "run_asym_vs_small",
    "run_bias_quant",
    "run_block_sampling",
    "run_grad_scale",
    "run_objective",
    "run_overlap",
    "run_sigma_cond",
    "run_sparsity",
    "schedule_from_args",
    "scope_mask_disagreement",
    "select_quantizer",
    "sorted_overlap_sweep",
    "sparsity_sweep",
]


def register_args(parser: argparse.ArgumentParser) -> None:
    """Register the ablation flags."""
    group = parser.add_argument_group("ablation sweep")
    group.add_argument(
        "--experiment",
        default="all",
        choices=[
            "all",
            "asym_vs_small",
            "grad_scale",
            "objective",
            "block_sampling",
            "overlap",
            "sparsity",
            "act_lut",
            "sigma_cond",
            "bias_quant",
        ],
        help="Which experiment to run. 'all' runs every experiment.",
    )
    group.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of paired seeds. More seeds shrink the confidence, not the "
        "per-seed variance; 5 is enough to see the effect is not seed luck.",
    )
    group.add_argument(
        "--n",
        type=int,
        default=1024,
        help="Rows in the probe tensor. Larger N shrinks the codebook gradient, "
        "which is what the inv_sqrt_n scaling tracks.",
    )
    group.add_argument(
        "--grad-scale-steps",
        type=int,
        default=5,
        help="Optimizer steps used to measure codebook parameter movement.",
    )
    group.add_argument(
        "--grad-scale-lr",
        type=float,
        default=1e-2,
        help="Learning rate for the parameter-movement measurement.",
    )
    group.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "dev" / "LEADERBOARD.md",
        help="Leaderboard output path. Defaults to the published dev/LEADERBOARD.md.",
    )
    group.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON dump of the raw measurements.",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the leaderboard instead of writing it.",
    )

    # The knobs the six added experiments measure. Names and semantics match the
    # training entry points, so a reader who knows `base_train --help` knows
    # these; the defaults are the *probe's* defaults, not necessarily the
    # trainer's, and the help text says which.
    group.add_argument(
        "--db-objective",
        choices=[OBJECTIVE_CE, OBJECTIVE_EDM],
        default=OBJECTIVE_EDM,
        help=(
            "objective for the `objective` experiment. 'edm' (default here) "
            "denoises one block and is the DiffusionBlocks objective; 'ce' is "
            "the whole-depth next-token escape hatch, kept as its baseline."
        ),
    )
    group.add_argument(
        "--db-block-sampling",
        choices=[SAMPLING_STEP, SAMPLING_MICRO],
        default=SAMPLING_MICRO,
        help=(
            "block-sampling mode for the `block_sampling` experiment. 'micro' "
            "(default here) redraws the block every micro-step, which is the "
            "mode the handoff found lossy; 'step' holds one block for the whole "
            "optimizer step and is the reference."
        ),
    )
    group.add_argument(
        "--db-overlap",
        type=float,
        default=0.25,
        help=(
            "log-sigma overlap for the `overlap` experiment (DiffusionBlocks "
            "App. C; the text recipe quotes 0.1). Swept alongside the fixed "
            "overlap floors, and the largest swept value is the variant."
        ),
    )
    group.add_argument(
        "--lcqat-lut-relaxation",
        choices=list(RELAXATIONS),
        default="logits",
        help=(
            "activation-LUT relaxation for the `act_lut` experiment. 'logits' "
            "(default) is the free (K_in x K_out) matrix; 'proximity' is the "
            "K_in + K_in knot/level parameterization."
        ),
    )
    group.add_argument(
        "--lcqat-act-body",
        choices=list(ACT_BODIES),
        default=ACT_BODY_PWL,
        help=(
            "activation body for the `act_lut` experiment. 'pwl' (default) is "
            "the frozen per-codebook bake; 'smoothpwl' fits a radial-basis body "
            "first and uses it to seed the table."
        ),
    )
    group.add_argument(
        "--db-sigma-codebook",
        choices=["", "conditioned", "modulated"],
        default="conditioned",
        help=(
            "sigma-conditioning mode for the `sigma_cond` experiment. "
            "'conditioned' (default here) is the B-fold codebook; '' is the "
            "shared static codebook it is measured against. 'modulated' shares "
            "one codebook and is not a separate arm, so it is accepted but not "
            "swept -- its artifact size equals the shared one by construction."
        ),
    )
    group.add_argument(
        "--db-sigma-anchors",
        type=int,
        default=0,
        help=(
            "anchor count for --db-sigma-codebook=conditioned. 0 (default) "
            "means one codebook per diffusion block, so the parameter blow-up "
            "measured by `sigma_cond` is exactly the block count."
        ),
    )

    # The bias codebook's size and the short training run that proves it moves.
    # Steps and lr are the *probe's*, not a trainer's; see `run_bias_quant`.
    group.add_argument(
        "--bias-quant-k-bias",
        type=int,
        default=DEFAULT_K_BIAS,
        help=(
            "level count for the per-layer bias codebook the `bias_quant` "
            "experiment measures. The shipped default of BiasQuantizer, so the "
            "cost measured here is the cost of turning the flag on with no other "
            "change."
        ),
    )
    group.add_argument(
        "--bias-quant-steps",
        type=int,
        default=5,
        help=(
            "SGD steps the `bias_quant` experiment takes on the bias codebook. "
            "Enough to establish the table is live (its parameters move), not "
            "enough to fit it -- this is a micro-probe, not a training run."
        ),
    )
    group.add_argument(
        "--bias-quant-lr",
        type=float,
        default=1.0,
        help=(
            "Learning rate for the `bias_quant` experiment's SGD steps. The "
            "default is 1.0 rather than something weight-scale because these "
            "latents are not weights: the codebook step parameters sit at "
            "magnitude ~5, where one fp32 ULP is ~6e-7, while the gradient this "
            "probe produces is ~3e-5. A learning rate of 1e-2 therefore asks for "
            "an update of ~3e-7, which is at or below the spacing between "
            "representable values -- the parameter comes back bit-identical and "
            "the row reports a live codebook as dead. Measured, not assumed: at "
            "1e-2 two of three seeds moved 0.0, at 1.0 every seed moves."
        ),
    )

    # Probe sizes. Every one of these is a micro-probe; see the "Reduced
    # protocol" section of the module docstring for what that does and does not
    # license anyone to conclude from the result.
    group.add_argument(
        "--ablation-blocks",
        type=int,
        default=2,
        help="Diffusion blocks in the engine the `objective`/`block_sampling` probes build.",
    )
    group.add_argument(
        "--ablation-n-layer",
        type=int,
        default=4,
        help="Transformer layers in the engine the DiffusionBlocks probes build.",
    )
    group.add_argument(
        "--ablation-batch",
        type=int,
        default=2,
        help="Batch rows in the `objective` block probe.",
    )
    group.add_argument(
        "--ablation-seq",
        type=int,
        default=8,
        help="Sequence positions in the `objective` block probe.",
    )
    group.add_argument(
        "--objective-steps",
        type=int,
        default=3,
        help="SGD steps the `objective` experiment runs per arm.",
    )
    group.add_argument(
        "--objective-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the `objective` experiment's SGD steps.",
    )
    group.add_argument(
        "--block-sampling-micro-steps",
        type=int,
        default=4,
        help="Micro-steps accumulated into one optimizer step by `block_sampling`.",
    )
    group.add_argument(
        "--overlap-samples",
        type=int,
        default=2048,
        help="Sigma draws per diffusion block in the `overlap` experiment.",
    )
    group.add_argument(
        "--sparseprop-sparsity",
        type=float,
        default=0.75,
        help="Target sparsity for the `sparsity` experiment.",
    )
    # Scope, gradual schedule, dense threshold. Registered through the shared
    # helper, so this driver cannot drift from `base_train` on a flag that
    # changes what a pruning mask means.
    add_sparseprop_pruning_args(group)


def validate_args(args: argparse.Namespace) -> None:
    """Fail fast on a configuration that cannot produce a measurement."""
    if args.seeds < 1:
        raise ValueError(f"--seeds must be >= 1, got {args.seeds}")
    if args.n < 1:
        raise ValueError(f"--n must be >= 1, got {args.n}")
    if args.grad_scale_steps < 1:
        raise ValueError(
            f"--grad-scale-steps must be >= 1, got {args.grad_scale_steps}"
        )
    if args.grad_scale_lr <= 0.0:
        raise ValueError(f"--grad-scale-lr must be positive, got {args.grad_scale_lr}")

    if args.ablation_blocks < 1:
        raise ValueError(f"--ablation-blocks must be >= 1, got {args.ablation_blocks}")
    if args.ablation_n_layer < args.ablation_blocks:
        # The engine asserts `partitioner.num_blocks <= n_layer`, so asking for
        # more blocks than layers dies inside `DiffusionBlockEngine.__init__`
        # with a bare AssertionError. Said here, with the numbers, instead.
        raise ValueError(
            f"--ablation-n-layer ({args.ablation_n_layer}) must be >= "
            f"--ablation-blocks ({args.ablation_blocks}): every block needs at "
            "least one transformer layer"
        )
    if args.ablation_batch < 1:
        raise ValueError(f"--ablation-batch must be >= 1, got {args.ablation_batch}")
    if args.ablation_seq < 1:
        raise ValueError(f"--ablation-seq must be >= 1, got {args.ablation_seq}")
    if args.objective_steps < 1:
        raise ValueError(f"--objective-steps must be >= 1, got {args.objective_steps}")
    if args.objective_lr <= 0.0:
        raise ValueError(f"--objective-lr must be positive, got {args.objective_lr}")
    if args.block_sampling_micro_steps < 2:
        # One micro-step is step-level sampling with extra steps in the name,
        # and it is the case in which the retention metric is degenerate: with
        # nothing accumulated, both arms trivially retain everything sampled.
        raise ValueError(
            f"--block-sampling-micro-steps must be >= 2 for a mode comparison, "
            f"got {args.block_sampling_micro_steps}"
        )
    if args.overlap_samples < 1:
        raise ValueError(f"--overlap-samples must be >= 1, got {args.overlap_samples}")
    if args.db_overlap < 0.0:
        raise ValueError(
            f"--db-overlap must be >= 0 (0.0 is the disjoint partition), got "
            f"{args.db_overlap}"
        )
    if not 0.0 <= args.sparseprop_sparsity < 1.0:
        # 1.0 is excluded on purpose: `magnitude_mask` at sparsity 1.0 keeps
        # exactly one entry per row, so the "layer is pruned to zero" NMSE
        # comparison would be measuring an all-ones layer the sweep invented.
        raise ValueError(
            f"--sparseprop-sparsity must be in [0.0, 1.0), got "
            f"{args.sparseprop_sparsity}"
        )
    if args.sparseprop_every < 0:
        raise ValueError(
            f"--sparseprop-every must be >= 0, got {args.sparseprop_every}"
        )
    if not 0.0 <= args.sparseprop_start_frac <= 1.0:
        raise ValueError(
            f"--sparseprop-start-frac must be in [0.0, 1.0], got "
            f"{args.sparseprop_start_frac}"
        )
    if not 0.0 <= args.sparseprop_dense_threshold <= 1.0:
        raise ValueError(
            f"--sparseprop-dense-threshold must be in [0.0, 1.0], got "
            f"{args.sparseprop_dense_threshold}"
        )
    if args.sparseprop_scope not in (SCOPE_LAYER, SCOPE_GLOBAL):
        raise ValueError(
            f"--sparseprop-scope must be one of "
            f"{(SCOPE_LAYER, SCOPE_GLOBAL)}, got {args.sparseprop_scope!r}"
        )
    if any(not 0.0 <= floor < 1.0 for floor in SPARSITY_FLOORS):
        raise ValueError(
            f"sparsity floors must be in [0.0, 1.0), got {SPARSITY_FLOORS}"
        )
    if args.db_sigma_codebook not in ("", "conditioned", "modulated"):
        raise ValueError(
            f"--db-sigma-codebook must be '', 'conditioned' or 'modulated', got "
            f"{args.db_sigma_codebook!r}"
        )
    # The anchor count is validated *after* resolution, because 0 is not a
    # request for zero anchors -- it is the documented "one per diffusion block"
    # shorthand, and `run_sigma_cond` resolves it the same way. Testing the raw
    # flag would reject the default outright.
    resolved_anchors = args.db_sigma_anchors or args.ablation_blocks
    if (
        args.experiment in ("all", "sigma_cond")
        and args.db_sigma_codebook == "conditioned"
        and resolved_anchors < 2
    ):
        # `SigmaConditionedCodebook.__init__` raises on a single anchor, and
        # `install_sigma_codebooks` raises SystemExit. Caught at parse time
        # instead, with the same numbers. Only checked when the experiment runs:
        # the flag is inert for every other experiment, and rejecting an inert
        # flag would make `--experiment all` fail for a knob it never reads.
        raise ValueError(
            f"--db-sigma-anchors must resolve to >= 2 with "
            f"--db-sigma-codebook=conditioned, got {resolved_anchors} "
            f"(--db-sigma-anchors={args.db_sigma_anchors} with "
            f"--ablation-blocks={args.ablation_blocks}; 0 means 'one per "
            "diffusion block')"
        )
    if args.bias_quant_k_bias < 2:
        # `MemoryEfficientLearnedCodebook.validate_split` requires
        # `m_neg + m_pos >= 2`, so a K of 1 or 0 is unconstructible rather than
        # merely a poor choice. Said here with the number, rather than as a
        # `ValueError` from inside the codebook with no idea which flag set it.
        raise ValueError(
            f"--bias-quant-k-bias must be >= 2 (the zero anchor plus at least "
            f"one level), got {args.bias_quant_k_bias}"
        )
    if args.bias_quant_steps < 1:
        raise ValueError(
            f"--bias-quant-steps must be >= 1, got {args.bias_quant_steps}"
        )
    if args.bias_quant_lr <= 0.0:
        raise ValueError(f"--bias-quant-lr must be positive, got {args.bias_quant_lr}")
    if args.lcqat_lut_relaxation not in RELAXATIONS:
        raise ValueError(
            f"--lcqat-lut-relaxation must be one of {RELAXATIONS}, got "
            f"{args.lcqat_lut_relaxation!r}"
        )
    if args.lcqat_act_body not in ACT_BODIES:
        raise ValueError(
            f"--lcqat-act-body must be one of {ACT_BODIES}, got {args.lcqat_act_body!r}"
        )
    for preset in (BASELINE_PRESET, VARIANT_PRESET):
        if preset not in PRESETS:
            raise ValueError(
                f"preset {preset!r} is not registered; available: {sorted(PRESETS)}"
            )


def check_claims(rows: list[AblationRow], n_elements: int) -> list[str]:
    """Return human-readable problems with a measured result. Empty means clean.

    These are checks on the *method*, not on performance: if the advertised
    scaling stops matching `1/sqrt(N)`, the implementation drifted from the
    claim and no downstream accuracy number should be trusted until it is
    re-checked. Performance itself is not gated -- a sweep on a random model is
    not a convergence benchmark.
    """
    problems = []
    for row in rows:
        if row.experiment == "grad_scale":
            predicted = expected_1_over_sqrt_n(n_elements)
            notes = row.notes or ""
            try:
                observed = float(notes.split("observed ratio ")[1].split(" ")[0])
            except (IndexError, ValueError):
                problems.append(
                    f"grad_scale: could not read the observed ratio from {notes!r}"
                )
                continue
            if abs(observed - predicted) > 1e-3:
                problems.append(
                    f"grad_scale: observed ratio {observed:.6g} does not match the "
                    f"claimed 1/sqrt(N) = {predicted:.6g} (N={n_elements})"
                )
        elif row.experiment.startswith("asym_vs_small"):
            if row.better == "baseline":
                problems.append(
                    f"{row.experiment}: baseline {row.baseline} beat the advertised "
                    f"variant {row.variant} (delta {row.delta:+.6g}). On a "
                    "non-negative relu^2 probe the asymmetric split must waste "
                    "strictly fewer levels, so this is a real regression."
                )
        elif row.experiment == "objective":
            # Only the measurement's own soundness is gated. Neither objective
            # is "better" -- they optimize different functions -- so a
            # non-monotone or jittery loss is not a failure and is not asserted
            # either way. What must hold is that both arms produced a finite
            # number at all: a `nan` here means the block-output probe diverged
            # or the loss never propagated, and rendering it as a number would
            # be a false reading rather than a surprising one.
            for name, value in (
                (row.baseline, row.value_baseline),
                (row.variant, row.value_variant),
            ):
                if value != value or value == float("inf"):
                    problems.append(
                        f"objective: arm {name} produced a non-finite block-output "
                        f"NMSE ({value!r}); the probe diverged or the loss never "
                        "propagated"
                    )
        elif row.experiment.startswith("block_sampling"):
            # The reference must be non-empty and the two modes must be
            # *distinguishable*. A tie means the metric stopped discriminating
            # (the §12.3 failure: holding sigma fixed makes every block tie), so
            # the comparison is reporting nothing even though it "passed".
            if row.value_baseline <= 0.0:
                problems.append(
                    f"{row.experiment}: the isolated per-block reference gradient is "
                    f"empty over {row.variant}* parameters (retention "
                    f"{row.value_baseline:.6g}). Either the reference was built from "
                    "an already-erased state, or the prefix no longer names anything."
                )
            if abs(row.delta) <= 0.0:
                problems.append(
                    f"{row.experiment}: step and micro sampling are indistinguishable "
                    f"(delta {row.delta:+.6g}). Sigma must vary per sample or every "
                    "block ties and the metric stops discriminating (handoff 12.3)."
                )
        elif row.experiment.startswith("sparsity_scope"):
            # The two scopes prune to the same per-layer fraction on these two
            # layers, so the NMSE rows tie by construction and the disagreement
            # row is what carries the comparison. A disagreement of zero would
            # mean the scope flag reached neither path, which is a wiring bug
            # this row is the only thing that would catch.
            if row.value_variant <= 0.0:
                problems.append(
                    f"{row.experiment}: the layer-scope and global-scope masks are "
                    f"identical (disagreement {row.value_variant:.6g}). The two scopes "
                    "then select the same weights, so the comparison reports nothing; "
                    "check that the schedule actually branched on `scope`."
                )
        elif row.experiment.startswith("overlap_g"):
            # The gate is the invariant that can actually be defended, not the
            # handoff's "strictly decreases" framing, which this measurement
            # contradicts by construction: `sample_sigma` draws from
            # `[lo/alpha, hi*alpha]`, so the out-of-*nominal*-band fraction rises
            # with g. Measuring against the widened interval would fall by
            # construction and test nothing. What must hold is that the disjoint
            # partition is genuinely disjoint -- at g=0 essentially nothing
            # should leave its band, and a large value there means the sampler is
            # not respecting the partition at all.
            if row.variant == "g=0" and row.value_variant > 0.01:
                problems.append(
                    f"{row.experiment}: at overlap 0.0 (the disjoint partition) "
                    f"{row.value_variant:.4%} of draws still land outside their "
                    "block's nominal band. The partition is supposed to be disjoint, "
                    "so this is a sampler or boundary defect, not an overlap effect."
                )
        elif row.experiment.startswith("sparsity_"):
            # Pruning is expected to *raise* NMSE, so the sign is not gated.
            # What is gated is the structural-zero contract: a pruned position
            # must dequantize to exactly 0.0, or SparseProp is not legal on top
            # of LC-QAT. The check reads the flag recorded in the note, because
            # a silently-broken exact-zero would otherwise show up only as a
            # slightly worse NMSE.
            if "FAILED" in (row.notes or ""):
                problems.append(
                    f"{row.experiment}: a pruned position did not dequantize to "
                    "exactly 0.0. The LC-QAT zero anchor is what makes structural "
                    "sparsity legal, so this is a real regression."
                )
            if "KEEP-masks" not in (row.notes or ""):
                problems.append(
                    f"{row.experiment}: the note does not record the KEEP-mask "
                    "convention (True = retained), so the row cannot be audited."
                )
        elif row.experiment.startswith("act_lut_"):
            # Two exact, free claims, gated only where they are actually
            # asserted. The fp8 identity applies to the proximity relaxation
            # alone: the free-logit matrix stores no knot positions, so there is
            # nothing to export, and gating it there would demand a result the
            # run never produced and fail a correct sweep.
            notes = row.notes or ""
            if (
                "no fp8 export" not in notes
                and "bit-identical to fp32: True" not in notes
            ):
                problems.append(
                    f"{row.experiment}: the fp8 knot export does not resolve to a "
                    "bit-identical index table. `resolved_table()` reads levels only, "
                    "so it must be identical for any knot dtype."
                )
            # The parameter count is structural: proximity is K_in + K_out
            # against a free K_in x K_out logit matrix. A proximity row that does
            # not shrink the free-logit baseline is not the parameterization the
            # count describes.
            if (
                row.variant.startswith(RELAXATION_PROXIMITY)
                and row.value_variant >= row.value_baseline
            ):
                problems.append(
                    f"{row.experiment}: the proximity parameterization reports "
                    f"{row.value_variant:g} parameters against a free "
                    f"{row.value_baseline:g}-parameter logit matrix, so it did not "
                    "shrink. K_in + K_out must be smaller than K_in * K_out."
                )
        elif row.experiment == "sigma_cond":
            # Structural only, by design. The gate is that the structural claim
            # holds -- the conditioned codebook is the anchor count times the
            # shared one -- and that the unverified premise is still declared
            # unverified. A row that quietly dropped the disclaimer would be
            # reading as an accuracy result.
            if "UNVERIFIED" not in (row.notes or ""):
                problems.append(
                    "sigma_cond: the note must record the distributional premise as "
                    "UNVERIFIED (handoff 12.2). Without it the row reads as an "
                    "accuracy result, which this experiment does not make."
                )
            if row.value_variant <= row.value_baseline:
                problems.append(
                    f"sigma_cond: a {row.value_variant:g}-parameter conditioned "
                    f"codebook is not larger than the {row.value_baseline:g}-parameter "
                    "shared one. One codebook per anchor must cost proportionally more."
                )
        elif row.experiment == "bias_quant_nmse":
            # The probe's own soundness, not a verdict on the capability. Two
            # things must hold for this number to mean anything, and both are
            # falsifiable:
            #
            # * The quantized arm must be *worse* than the FP32 baseline. The
            #   baseline scores 0 exactly, so an NMSE of 0 here means the table
            #   reproduced the bias exactly -- which a K-level table over a
            #   spread bias does not do. That is a broken measurement, not a
            #   perfect one, and taking it at face value would report a "free
            #   win" for a probe that never quantized anything.
            # * It must be *below* 1.0. An NMSE of exactly 1.0 is the signature
            #   of every entry bucketizing onto the zero anchor: the
            #   reconstruction is all zeros, so the error equals the signal
            #   power. `init_from_tensor` is called precisely to avoid it, and a
            #   run that reports 1.0 has silently lost that call -- the
            #   `per_channel.py` measurement `BiasQuantizer` cites.
            #
            # The row deliberately does NOT gate "is bias quantization worth
            # it". That verdict needs a trained model, and the note says so;
            # encoding either answer here would be a check that can only pass.
            if row.value_variant <= 0.0:
                problems.append(
                    f"{row.experiment}: the quantized bias reconstructs with NMSE "
                    f"{row.value_variant:.6g} against an FP32 baseline of "
                    f"{row.value_baseline:.6g}. A K-level codebook over a spread "
                    "bias cannot reproduce it exactly, so the table is not being "
                    "read -- the probe is degenerate, not free."
                )
            if row.value_variant >= 1.0:
                problems.append(
                    f"{row.experiment}: bias NMSE is {row.value_variant:.6g}, i.e. "
                    "the reconstruction is no better than all zeros. That is the "
                    "silent permanent failure BiasQuantizer.init_from_tensor "
                    "documents: every entry bucketed to the zero anchor, so the "
                    "gather is the anchor and the gradient is identically zero. "
                    "Check that init_from_tensor() is still being called on the "
                    "bias before this is measured."
                )
        elif row.experiment == "bias_quant_cost":
            # Structural, and falsifiable in one direction only: the FP32
            # baseline carries the bias inline, so a bias codebook must add
            # parameters. Zero would mean the flag never reached the layer and
            # the whole experiment measured a bias-free module.
            if row.value_variant <= 0.0:
                problems.append(
                    f"{row.experiment}: the bias codebook reports "
                    f"{row.value_variant:g} parameters over {row.n_seeds} seeds. "
                    "The FP32 baseline adds none by construction, so zero means "
                    "quantize_bias never reached LCQATLinear.from_float and the "
                    "rest of this experiment measured a bias-free layer."
                )
        elif row.experiment == "bias_quant_liveness":
            # The 9.8 trap, gated. A bias codebook that does not move is inert
            # storage that looks like a feature on the parameter-count row, and
            # the failure is silent: it prints no error and the error metric
            # reads as a plausible number rather than as a sign of anything
            # wrong. Read from the note's flag so the rendered leaderboard stays
            # self-describing, exactly as the `sparsity` rows do.
            if BIAS_CODEBOOK_DEAD in (row.notes or ""):
                problems.append(
                    f"{row.experiment}: the bias codebook's parameters received no "
                    "usable gradient and did not move. Either the table's span "
                    "overshoots the bias (the too-wide init that pins NMSE at "
                    "exactly 1.0) or nothing selects a non-anchor level, so the "
                    "per-layer codebook is cost with no effect."
                )
    return problems


def write_leaderboard(path: Path, table: str) -> None:
    """Replace the leaderboard's generated section with `table`, in place.

    Takes the finished table text rather than the rows, so the dry-run output
    and the file content are the same string by construction. Accepting rows
    here as well would mean rendering in two places, and a caller that passed
    both would silently emit the table twice.

    Text outside the markers is preserved, so hand-written analysis in the
    leaderboard survives a re-run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = "<!-- BEGIN GENERATED: lcqat_ablation.py -->"
    end_marker = "<!-- END GENERATED: lcqat_ablation.py -->"
    section = (
        f"{marker}\n"
        f"_Generated by `scripts/lcqat_ablation.py` on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} -- do not edit by hand._\n\n"
        f"{table}"
        f"\n{end_marker}\n"
    )

    existing = ""
    if path.exists():
        text = path.read_text()
        if marker in text and end_marker in text:
            head = text.split(marker)[0]
            # `tail` carries the hand-written prose that follows the end marker,
            # including whatever trailing blank lines the previous run left.
            # Re-appending it verbatim grows the file by one blank line per run,
            # so normalize to exactly one final newline.
            tail = text.split(end_marker, 1)[1].strip("\n")
            path.write_text(head + section + tail + "\n" if tail else head + section)
            return
        existing = text

    path.write_text(existing.rstrip("\n") + "\n\n" + section)


def main(argv: list[str] | None = None, *, from_cli: bool = False) -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired LC-QAT ablation sweeps and write a leaderboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    register_args(parser)
    args = parser.parse_args(argv)
    validate_args(args)

    rows: list[AblationRow] = []
    n_elements = 0

    if args.experiment in ("all", "asym_vs_small"):
        rows.append(run_asym_vs_small(args))
    if args.experiment in ("all", "grad_scale"):
        grad_row, _ = run_grad_scale(args)
        rows.append(grad_row)
        torch.manual_seed(0)
        n_elements = probe_layer(VARIANT_PRESET).weight.numel()
    if args.experiment in ("all", "objective"):
        rows.extend(run_objective(args))
    if args.experiment in ("all", "block_sampling"):
        rows.extend(run_block_sampling(args))
    if args.experiment in ("all", "overlap"):
        rows.extend(run_overlap(args))
    if args.experiment in ("all", "sparsity"):
        rows.extend(run_sparsity(args))
    if args.experiment in ("all", "act_lut"):
        rows.extend(run_act_lut(args))
    if args.experiment in ("all", "sigma_cond"):
        rows.extend(run_sigma_cond(args))
    if args.experiment in ("all", "bias_quant"):
        rows.extend(run_bias_quant(args))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([row_to_dict(r) for r in rows], indent=2) + "\n"
        )

    # Rendered once, in `main`, because the dry-run path prints exactly the text
    # the file path writes. `write_leaderboard` takes the finished text rather
    # than the rows, so passing both a rendered table *and* the rows made it
    # render a second time and emit the table twice.
    table = render_leaderboard(rows, "Paired ablation results")
    if args.dry_run:
        print(table)
    else:
        if not from_cli and args.out == PUBLISHED_LEADERBOARD:
            # `main()` is called both from the command line (where the
            # documented default is right) and from tests, which pass an
            # explicit argv. A caller that supplies its own argv but no `--out`
            # would otherwise overwrite the published table with whatever
            # ad-hoc arguments it used -- that is how a 1-seed `grad_scale` run
            # replaced the full sweep, and the next full-suite run did it again.
            # Only the real command line may write the published file by
            # default; a programmatic caller must name its destination.
            raise SystemExit(
                f"refusing to write the published {PUBLISHED_LEADERBOARD} from a "
                "programmatic main() call without --out; pass an explicit path "
                "(use --dry-run to print instead)"
            )
        write_leaderboard(args.out, table)
        print(f"wrote {args.out}")

    problems = check_claims(rows, n_elements)
    if problems:
        print("\nclaim check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(from_cli=True))

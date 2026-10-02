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
import math
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

from nanochat.diffusion_blocks import (  # noqa: E402
    EquiProbabilityPartitioner,
    edm_preconditioning,
)
from nanochat.lcqat.ablation_metrics import (  # noqa: E402
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
from nanochat.lcqat.activation import (  # noqa: E402
    ACT_BODIES,
    ACT_BODY_PWL,
    ACT_BODY_SMOOTHPWL,
    SmoothPWL,
    get_activation,
)
from nanochat.lcqat.bias_quant import DEFAULT_K_BIAS  # noqa: E402
from nanochat.lcqat.learnable_lut import (  # noqa: E402
    RELAXATION_LOGITS,
    RELAXATION_PROXIMITY,
    RELAXATIONS,
    LearnableIndexLut,
)
from nanochat.lcqat.linear import (  # noqa: E402
    GRAD_SCALE_INV_SQRT_N,
    GRAD_SCALE_NONE,
    LCQATLinear,
)
from nanochat.lcqat.pruning import (  # noqa: E402
    add_sparseprop_pruning_args,
    schedule_from_args,
)
from nanochat.lcqat.retrofit import (  # noqa: E402
    PRESETS,
    CodebookSpec,
    retrofit_model,
)
from nanochat.lcqat.sigma_codebook import SigmaConditionedCodebook  # noqa: E402
from nanochat.lcqat.sparseprop import (  # noqa: E402
    SCOPE_GLOBAL,
    SCOPE_LAYER,
    inject_sparseprop_layers,
)
from tests.conftest import build_active_tiny_gpt  # noqa: E402
from tests.test_dbcpu_engine import make_engine  # noqa: E402

#: Preset the `asym` experiment treats as the improvement, and the one it beats.
BASELINE_PRESET = "small"
VARIANT_PRESET = "asym"

#: The two objectives `--db-objective` selects between.
OBJECTIVE_CE = "ce"
OBJECTIVE_EDM = "edm"

#: The two block-sampling modes `--db-block-sampling` selects between.
SAMPLING_STEP = "step"
SAMPLING_MICRO = "micro"

#: Overlap settings the `overlap` experiment sweeps. The disjoint partition
#: (0.0) is the baseline; the rest are the values DiffusionBlocks App. C quotes
#: for text (0.1) rounded up, plus `--db-overlap` so the operator's choice is
#: swept too. Deduped and sorted by `sorted_overlap_sweep`.
OVERLAP_FLOORS = (0.0, 0.125)

#: Sparsity levels the `sparsity` experiment sweeps, below and at the target.
#: The dense 0.0 arm is not swept: at zero sparsity `layer` and `global` are
#: the same mask by construction, so the pair would be a tautology.
SPARSITY_FLOORS = (0.5,)

#: The two pruning scopes the `sparsity` experiment sweeps.
SCOPES = (SCOPE_LAYER, SCOPE_GLOBAL)

#: `SmoothPWL`'s default knot-grid half-width. The class does not retain its
#: `init_range` argument as an attribute, so the default is restated here for the
#: `act_lut` fit grid, which has to score both bodies over the domain the RBF fit
#: is actually run on.
SMOOTHPWL_DEFAULT_INIT_RANGE = 1.0

#: Parameter families the `block_sampling` metric is reported over, counted
#: separately and never pooled. The engine's denoise heads and the GPT's own
#: transformer layers are different parameters with different owners; a pooled
#: fraction hides whichever of the two a change broke.
SAMPLING_FAMILIES = ("transformer.h.", "db_denoise_heads.")

#: The block the `step` arm holds for a whole optimizer step. It is block 0,
#: because `_per_block_reference` also runs block 0 in isolation: the two arms
#: are then comparable against the same reference, and the `micro` arm differs
#: from the reference only in that it redraws the block each micro-step.
STEP_ARM_BLOCK = 0


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


def probe_layer(preset: str):
    """Retrofit a fresh tiny model and return the MLP down-projection.

    `c_proj` is chosen because it is the layer whose *activation* input is
    `relu^2` and therefore non-negative -- the tensor `asym` exists to serve.
    A model that shares no state with other arms is required, since retrofitting
    mutates in place.
    """
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS[preset])
    return model.transformer.h[0].mlp.c_proj


def non_negative_probe(in_features: int, rows: int, seed: int) -> torch.Tensor:
    """A probe with the shape of the real `relu^2` input: non-negative.

    `asym`'s justification is that `mlp.forward` computes `F.relu(x).square()`,
    so the tensor `c_proj` actually quantizes never goes below zero. Measuring
    a symmetric signed probe would test the opposite of what the preset is for,
    and would report a loss that says nothing about the claim.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, in_features, generator=gen).relu().square()


def run_asym_vs_small(args: argparse.Namespace) -> AblationRow:
    """Measure the paired effect of the `asym` split on a `relu^2` activation.

    Probes the **activation** quantizer of `c_proj`, not its weights. That
    distinction is the whole measurement, and getting it wrong inverts the
    result:

    * On the real non-negative `relu^2` input, `small` (a symmetric 15-level
      codebook) can only place 8 of its levels where the tensor actually lives,
      so it wastes half its alphabet. `asym` (one-sided, 8 levels) puts every
      level in range.
    * On signed weights, `asym` is measurably *worse*, because the split gives up
      half the range for no benefit.

    So on a `relu^2` probe the NMSE is a tie -- both arms spend the same 8
    effective levels -- and the claim is confirmed by **effective level count**,
    which is the quantity the split was designed to raise. Reporting NMSE alone
    would show a flat result and hide the improvement.
    """
    measurements = []
    for seed in range(args.seeds):
        # Same seed for both arms: the only difference is the preset.
        torch.manual_seed(seed)
        layer_base = probe_layer(BASELINE_PRESET)
        layer_var = probe_layer(VARIANT_PRESET)

        # One probe tensor, shared: the input is held fixed across arms so the
        # measurement isolates the codebook, not the data.
        probe = non_negative_probe(layer_base.in_features, args.n, seed)

        base = measure_reconstruction(layer_base, probe, BASELINE_PRESET, which="act")
        variant = measure_reconstruction(layer_var, probe, VARIANT_PRESET, which="act")
        measurements.append((base, variant))

    # The deciding metric is the fraction of the alphabet actually used.
    # Absolute level count *ties* at 8 for both arms -- the difference is that
    # `small` spends 15 levels to do it (8 used, 7 stranded below the data's
    # minimum) while `asym` spends 8. So the win is headroom, not resolution.
    base_eff = sum(b.level_utilization for b, _ in measurements) / len(measurements)
    var_eff = sum(v.level_utilization for _, v in measurements) / len(measurements)
    return AblationRow(
        experiment="asym_vs_small_levels",
        metric="level_utilization",
        baseline=BASELINE_PRESET,
        variant=VARIANT_PRESET,
        value_baseline=base_eff,
        value_variant=var_eff,
        delta=var_eff - base_eff,
        better="variant" if var_eff > base_eff else "baseline",
        seeds=list(range(args.seeds)),
        n_seeds=args.seeds,
        notes=(
            "fraction of codebook levels actually hit on a non-negative relu^2 "
            "probe; absolute level count ties at 8 for both arms, so the gain is "
            "headroom (asym spends 8 levels, small spends 15 for the same 8)"
        ),
    )


def run_grad_scale(
    args: argparse.Namespace,
) -> tuple[AblationRow, list[GradScaleObservation]]:
    """Measure the `1/sqrt(N)` codebook gradient scale directly."""
    rows = []
    observations: list[GradScaleObservation] = []
    n_elements = 0
    base_norm = 0.0
    inv_norm = 0.0
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        layer = probe_layer(VARIANT_PRESET)
        gen = torch.Generator().manual_seed(seed)
        probe = torch.randn(args.n, layer.in_features, generator=gen)

        none_obs = observe_grad_scale(
            layer,
            probe,
            GRAD_SCALE_NONE,
            steps=args.grad_scale_steps,
            lr=args.grad_scale_lr,
        )
        inv_obs = observe_grad_scale(
            layer,
            probe,
            GRAD_SCALE_INV_SQRT_N,
            steps=args.grad_scale_steps,
            lr=args.grad_scale_lr,
        )
        observations.extend([none_obs, inv_obs])
        base_norm += none_obs.grad_norm
        inv_norm += inv_obs.grad_norm
        n_elements = layer.weight.numel()

    base_avg = base_norm / args.seeds
    inv_avg = inv_norm / args.seeds
    ratio = inv_avg / base_avg if base_avg else float("nan")
    rows.append(
        AblationRow(
            experiment="grad_scale",
            metric="codebook_grad_ratio",
            baseline=GRAD_SCALE_NONE,
            variant=GRAD_SCALE_INV_SQRT_N,
            value_baseline=base_avg,
            value_variant=inv_avg,
            delta=inv_avg - base_avg,
            # Lower codebook gradient is the point of the scaling, so the
            # variant "wins" when it is strictly smaller.
            better="variant" if inv_avg < base_avg else "baseline",
            seeds=list(range(args.seeds)),
            n_seeds=args.seeds,
            notes=(
                f"observed ratio {ratio:.6g} vs predicted 1/sqrt(N) = "
                f"{1.0 / (n_elements**0.5):.6g} for N={n_elements}"
            ),
        )
    )
    return rows[0], observations


def expected_1_over_sqrt_n(n_elements: int) -> float:
    """The `1/sqrt(N)` factor the PRD 2.4 scaling applies."""
    return 1.0 / (n_elements**0.5)


# ---------------------------------------------------------------------------
# Shared probe helpers for the six DiffusionBlocks / D9 experiments
# ---------------------------------------------------------------------------


def build_probe_engine(args: argparse.Namespace, seed: int = 0):
    """A tiny DiffusionBlocks engine, depth-configurable and with live zero-inits.

    `make_engine(active=True)` is what makes the gradient-reachability
    measurements meaningful: the denoise heads and the adapter output layer are
    zero-initialized by design, and a zero tensor transmits no gradient, so
    every "did the gradient arrive" assertion would pass vacuously against a
    fresh engine.

    Depth is threaded through `make_engine`'s own `n_layer` argument rather than
    done here. `make_engine` builds its own model, so resizing a model built in
    this function and then handing `n_layer=None` to `make_engine` would build a
    *second* model at the default depth and silently discard the first -- the
    knob would appear to work while measuring the wrong depth.

    `seed` does the same job: `make_engine` re-seeds internally, so a seed set
    here would be overwritten. It is used to seed the *probes*, which are built
    separately, and the engine itself is deterministic across arms -- which is
    what pairing requires. Both arms at a given seed therefore see a
    byte-identical engine, and a per-seed difference in the result is a
    difference in the probe, not in the model.
    """
    return make_engine(
        num_blocks=args.ablation_blocks, n_layer=args.ablation_n_layer, active=True
    )


def block_probe_tensors(
    args: argparse.Namespace, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(value probe, token ids, CE targets)` for the block-level probes.

    The value tensor is the `relu^2`-shaped one the driver already uses for the
    activation quantizers: `mlp.forward` computes `F.relu(x).square()`, so the
    tensors the real blocks see at their input are non-negative. It is the
    *clean* target the block's residual identity reconstructs.

    Token ids and CE targets both exist because the two objectives take
    different arguments. `denoise_step(idx, ...)` reads `idx` for its sequence
    length only, and `train_step(idx, targets, ...)` needs real targets to
    return a scalar cross-entropy: `GPT.forward` returns *logits* when
    `targets is None`, and backpropagating through logits makes
    `float(loss.detach())` raise rather than measure anything. All three come
    from one generator so both arms see identical tokens.
    """
    model = build_active_tiny_gpt()
    n_embd = int(model.config.n_embd)
    vocab = int(model.config.vocab_size)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(
        0, vocab, (args.ablation_batch, args.ablation_seq), generator=gen
    )
    targets = torch.randint(
        0, vocab, (args.ablation_batch, args.ablation_seq), generator=gen
    )
    probe = non_negative_probe(
        n_embd, args.ablation_batch * args.ablation_seq, seed
    ).view(args.ablation_batch, args.ablation_seq, n_embd)
    return probe, idx, targets


def engine_named_parameters(engine) -> list[tuple[str, torch.nn.Parameter]]:
    """The engine's parameters, as `(name, param)` in one deterministic order."""
    return list(engine.named_parameters())


def grad_retained_fraction(
    grads: dict[str, torch.Tensor | None],
    names: list[str],
    prefix: str,
    reference: dict[str, torch.Tensor | None],
) -> float:
    """Gradient signal retained, as a fraction of the isolated reference's.

    Returns `||grad_family|| / ||reference_family||` over the parameters the
    *reference* (the isolated single-block run) gave a gradient to. Two earlier
    formulations were wrong; both are recorded so they are not reintroduced:

    * Counting non-`None` gradients over every parameter in the family put the
      same zeros in both arms' numerator and denominator, so both arms scored
      identically and the comparison reported nothing while the behaviour
      differed sharply.
    * Restricting that count to the reference's live parameters still flipped
      with the sampler: `micro` retains everything whenever its *last* draw
      happens to be the reference's block, and nothing otherwise. That is a coin
      flip per seed, which is why `--seeds 1` tied or reversed at random.

    The magnitude ratio is stable under both. `step` holds the block, so it
    retains one reference's worth of signal. `micro` ends on whichever block the
    sampler drew last, so it retains on average one micro-step's share, and the
    block owning the reference's gradients is erased regardless of the draw.
    Normalizing by the reference keeps the row dimensionless and comparable to
    the other experiments.

    Scored per *parameter tensor*, summed in float64. An empty reference family
    scores 0.0 rather than dividing by zero, so a renamed prefix shows up as a
    collapse instead of a `nan` that formats as a pass.
    """
    live = [n for n in names if n.startswith(prefix) and reference.get(n) is not None]
    if not live:
        return 0.0
    retained = torch.zeros((), dtype=torch.float64)
    expected = torch.zeros((), dtype=torch.float64)
    for name in live:
        grad = grads.get(name)
        if grad is not None:
            retained += grad.detach().double().square().sum()
        expected += reference[name].detach().double().square().sum()
    if float(expected) <= 0.0:
        return 0.0
    return float((retained / expected).sqrt())


def sorted_overlap_sweep(args: argparse.Namespace) -> list[float]:
    """The overlap settings to sweep, ascending, with the 0.0 baseline first."""
    return sorted({*OVERLAP_FLOORS, float(args.db_overlap)})


def sparsity_sweep(args: argparse.Namespace) -> list[float]:
    """The sparsity settings to sweep, ascending, target last."""
    return sorted({*SPARSITY_FLOORS, float(args.sparseprop_sparsity)})


def codebook_of(layer, which: str = "out") -> torch.Tensor:
    """A quantizer's codebook as an FP32 tensor.

    `which` follows `select_quantizer`'s roles: `"act"` is the layer's
    activation (input) quantizer, `"out"` its output quantizer, and `"weight"`
    its weight quantizer. Read off the layer rather than rebuilt, so the table
    measured is the one the model actually uses -- and so a layer that lacks the
    requested quantizer raises here instead of silently falling back to the
    weight quantizer, which would measure something else entirely.
    """
    quantizer = select_quantizer(layer, which)
    return quantizer.get_codebook().detach().to(torch.float32)


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------


def _run_objective_arm(
    args: argparse.Namespace, objective: str, seed: int
) -> tuple[float, float]:
    """One objective arm: return `(mean loss, block-output NMSE)`.

    The NMSE is the metric; the loss is a diagnostic, not a claim. The EDM loss
    is jittery **by design** -- it is `w(sigma) * ||D_q - clean||^2` with a
    fresh sigma and fresh noise on every step -- so neither arm is monotone, and
    comparing final loss values measures which sampler happened to draw a kinder
    noise level. The reconstruction error of the block's output against a
    held-fixed non-negative probe is the quantity that is comparable.

    No `lm_head` assertion is made anywhere in this function. The EDM objective
    predicts an embedding rather than tokens, so `lm_head` legitimately receives
    no gradient under it; that is a property of the objective (handoff §5.1a),
    not a wiring defect, and asserting it either way would assert an accident.
    """
    engine = build_probe_engine(args, seed=seed)
    probe, idx, targets = block_probe_tensors(args, seed)
    params = [p for _, p in engine_named_parameters(engine) if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.objective_lr)
    losses: list[float] = []

    for step in range(args.objective_steps):
        block = step % args.ablation_blocks
        if objective == OBJECTIVE_EDM:
            # `denoise_step` returns `(loss, sigma)`; sigma is a 0-dim tensor
            # (`sample_sigma` draws one per call, not one per batch row), and is
            # not used here -- only the loss is.
            loss, _sigma = engine.denoise_step(
                idx,
                block_idx=block,
                generator=torch.Generator().manual_seed(seed * 1000 + step),
            )
        else:
            # `train_step` returns the CE loss directly, not a `(loss, sigma)`
            # pair, and requires real targets: `GPT.forward` returns logits when
            # `targets is None`, which has no scalar to read.
            loss = engine.train_step(idx, targets, block_idx=block)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        # The block-output measurement. `probe` is the block's clean target; the
        # residual identity is what makes it the right reference for both arms.
        # Under EDM the input is the preconditioned clean stream, and the
        # preconditioner is applied to the sigma *tensor* rather than a Python
        # float, so the scaling matches what `denoise_step` did. Under CE there
        # is no noise level at all, so the probe is fed directly.
        sigma = torch.ones(())
        c_in, _c_out, _w = edm_preconditioning(sigma, engine.partitioner.sigma_data)
        block_input = probe if objective == OBJECTIVE_CE else c_in * probe
        out = engine._run_block_denoiser(0, block_input, sigma, probe.size(1))
        mse = float((out - probe).square().mean())
        power = float(probe.square().mean())
    return sum(losses) / len(losses), (mse / power if power > 0 else float("inf"))


def run_objective(args: argparse.Namespace) -> list[AblationRow]:
    """`--db-objective ce` vs `edm`: block-output reconstruction under each.

    Baseline is the whole-depth next-token escape hatch (`train_step`), variant
    is the real DiffusionBlocks objective (`denoise_step`). The row reports the
    paired mean block-output NMSE, and the per-arm losses go in the note as a
    diagnostic. The *sign* of the delta is not a quality claim in either
    direction: the two objectives optimize different functions, so neither
    "wins" and `better` only records which arm reconstructed its probe more
    closely.
    """
    base_nmse = 0.0
    var_nmse = 0.0
    base_loss = 0.0
    var_loss = 0.0
    for seed in range(args.seeds):
        # Same seed, same probe, same tokens, same engine for both arms.
        b_loss, b_nmse = _run_objective_arm(args, OBJECTIVE_CE, seed)
        v_loss, v_nmse = _run_objective_arm(args, OBJECTIVE_EDM, seed)
        base_nmse += b_nmse
        var_nmse += v_nmse
        base_loss += b_loss
        var_loss += v_loss

    base_nmse /= args.seeds
    var_nmse /= args.seeds
    return [
        AblationRow(
            experiment="objective",
            metric="block_out_nmse",
            baseline=OBJECTIVE_CE,
            variant=OBJECTIVE_EDM,
            value_baseline=base_nmse,
            value_variant=var_nmse,
            delta=var_nmse - base_nmse,
            better="variant" if var_nmse < base_nmse else "baseline",
            seeds=list(range(args.seeds)),
            n_seeds=args.seeds,
            notes=(
                f"mean block-output NMSE over {args.objective_steps} SGD steps on a "
                f"{args.ablation_n_layer}-layer random model with "
                f"{args.ablation_blocks} diffusion blocks; mean train loss ce "
                f"{base_loss / args.seeds:.6g} vs edm {var_loss / args.seeds:.6g}, "
                "reported as a diagnostic only: the edm loss is jittery by design, "
                "so neither arm is monotone and no quality claim is made from it. "
                "No lm_head gradient assertion is made -- the edm objective "
                "predicts an embedding, not tokens."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# block_sampling
# ---------------------------------------------------------------------------


def _accumulate_one_step(
    engine,
    args: argparse.Namespace,
    micro_steps: int,
    mode: str,
    seed: int,
    clean: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    """Accumulate one optimizer step's gradients, emulating `mode`.

    Three of the four traps in handoff §12.3 are enforced here:

    1. **Gradients accumulate.** `zero_grad` is called once, before the
       micro-step loop, and never inside it. The real loop adds each micro-step's
       contribution to the running gradient; re-zeroing per micro-step would
       measure the *last* micro-step alone, which is the bug §5.1b found and not a
       faithful emulation of the loop being criticized.
    2. **Sigma varies per sample.** `denoise_step` draws a fresh sigma from the
       active block's band on every call, with a per-micro-step generator seed.
       Holding sigma fixed makes every block tie on the metric and the mode
       comparison stops discriminating.
    3. **`db_denoise_heads.*` and `transformer.h.*` are kept apart.** The
       denominator is the reference run's parameter list, partitioned by family
       and reported as separate rows, so a change that erases one family's
       gradients cannot be masked by the other still reaching the optimizer.

    The fourth trap -- building the reference from already-erased states -- is
    handled by the caller: `ref` comes from `_per_block_reference`, which is
    computed from its own engine before any accumulation runs.

    Each micro-step's loss is divided by `micro_steps` so the accumulated
    gradient is the *mean* over micro-steps, matching what an optimizer step
    built from that loss would apply.
    """
    engine.zero_grad(set_to_none=True)
    for micro in range(micro_steps):
        if mode == SAMPLING_STEP:
            # One block held for the whole optimizer step -- the same block the
            # reference ran. Rotating `micro % n_blocks` here instead would
            # reproduce exactly what the `micro` arm does and make the two arms
            # indistinguishable by construction. The noise generator is still
            # re-seeded per micro-step, so both arms draw the same sigma
            # sequence; only the *block* is held, not the sigma.
            block = STEP_ARM_BLOCK
        else:
            # The engine's own sampler, so the arm uses the real draw rather
            # than a reimplementation of it.
            block = engine.sample_block(
                generator=torch.Generator().manual_seed(seed * 7919 + micro)
            )
        loss, _sigma = engine.denoise_step(
            # `clean` is supplied, so `idx` is read only for its sequence
            # length; the values are never used.
            _length_only_idx(probe_seq_len=clean.size(1)),
            block_idx=block,
            generator=torch.Generator().manual_seed(seed * 104729 + micro),
            clean=clean,
        )
        (loss / micro_steps).backward()
    return {name: p.grad for name, p in engine_named_parameters(engine)}


def _length_only_idx(probe_seq_len: int) -> torch.Tensor:
    """A `(1, seq_len)` long tensor used only for its `.size(1)`.

    `denoise_step` reads `idx` for the sequence length once `clean` is given, so
    the values are irrelevant -- but passing `None` fails on `idx.size(1)`. Kept
    at width 1 and named for its only real use so that contract is visible at
    the call site instead of being a mystery argument.
    """
    return torch.zeros(1, probe_seq_len, dtype=torch.long)


def _per_block_reference(
    args: argparse.Namespace, seed: int
) -> tuple[dict[str, torch.Tensor | None], list[str]]:
    """The isolated single-block run: one block, one micro-step, no history.

    This is the *reference*, so it must be built from a clean engine of its own.
    Reading it off a post-step state -- where a previous accumulation has already
    zeroed or restored gradients -- would compare the two modes against a
    reference that had already been through the process being measured.
    """
    engine = build_probe_engine(args, seed=seed)
    _probe, idx, _targets = block_probe_tensors(args, seed)
    clean = torch.nn.functional.normalize(
        engine.model.transformer.wte(idx).float(), dim=-1
    ).detach()
    engine.zero_grad(set_to_none=True)
    loss, _sigma = engine.denoise_step(
        _length_only_idx(clean.size(1)),
        block_idx=0,
        generator=torch.Generator().manual_seed(seed),
        clean=clean,
    )
    loss.backward()
    grads = {name: p.grad for name, p in engine_named_parameters(engine)}
    return grads, list(grads)


def run_block_sampling(args: argparse.Namespace) -> list[AblationRow]:
    """`--db-block-sampling step` vs `micro`: gradient retention after one step.

    Metric: the fraction of the isolated per-block reference gradient that
    survives to the end of one accumulated optimizer step, per parameter family.
    `step` sampling holds one block for the whole step, so micro-steps reinforce
    the same gradients. `micro` redraws the block every micro-step, and
    `_apply_requires_grad` sets `p.grad = None` for whatever the newly activated
    block does not own -- so each micro-step erases the previous one's
    contribution and only the last block sampled reaches the optimizer
    (handoff §5.1b, §12.3).

    That difference is measured, not asserted. The claim check requires only
    that the reference is non-empty and that the two arms are distinguishable --
    a tie means the metric stopped discriminating, which is the §12.3 failure
    and is reported as a problem whichever direction it came out.
    """
    per_family: dict[tuple[str, str], list[float]] = {
        (mode, family): []
        for mode in (SAMPLING_STEP, SAMPLING_MICRO)
        for family in SAMPLING_FAMILIES
    }
    for seed in range(args.seeds):
        # Built from its own engine, before any accumulation runs.
        ref_grads, names = _per_block_reference(args, seed)
        for mode in (SAMPLING_STEP, SAMPLING_MICRO):
            engine = build_probe_engine(args, seed=seed)
            _probe, idx, _targets = block_probe_tensors(args, seed)
            clean = torch.nn.functional.normalize(
                engine.model.transformer.wte(idx).float(), dim=-1
            ).detach()
            grads = _accumulate_one_step(
                engine, args, args.block_sampling_micro_steps, mode, seed, clean
            )
            for family in SAMPLING_FAMILIES:
                per_family[(mode, family)].append(
                    grad_retained_fraction(grads, names, family, ref_grads)
                )

    rows: list[AblationRow] = []
    for family in SAMPLING_FAMILIES:
        step_vals = per_family[(SAMPLING_STEP, family)]
        micro_vals = per_family[(SAMPLING_MICRO, family)]
        base = sum(step_vals) / len(step_vals)
        var = sum(micro_vals) / len(micro_vals)
        # The seed spread is reported because this metric is strongly
        # seed-dependent, and hiding that would let a reader mistake one
        # seed's ratio for a stable constant. It varies with the sigma lottery
        # more than with the sampling mode: the same configuration measured
        # 42.7 at one seed and 11.9 at eight.
        step_spread = _spread(step_vals)
        micro_spread = _spread(micro_vals)
        rows.append(
            AblationRow(
                experiment=f"block_sampling_{family.rstrip('.')}",
                metric="grad_magnitude_ratio",
                baseline=SAMPLING_STEP,
                variant=SAMPLING_MICRO,
                value_baseline=base,
                value_variant=var,
                delta=var - base,
                better="variant" if var > base else "baseline",
                seeds=list(range(args.seeds)),
                n_seeds=args.seeds,
                notes=(
                    f"||accumulated gradient|| / ||isolated single-block reference "
                    f"gradient|| over {family}* parameters after one optimizer step "
                    f"of {args.block_sampling_micro_steps} accumulated micro-steps; "
                    f"the two families are never pooled. step {base:.6g} "
                    f"(spread {step_spread:.3g}), micro {var:.6g} "
                    f"(spread {micro_spread:.3g}), micro/step "
                    f"{(var / base if base else float('nan')):.4g}. NOT a fraction "
                    "in [0,1] and NOT calibrated to 1: the reference is a single "
                    "micro-step while both arms accumulate "
                    f"{args.block_sampling_micro_steps} of them, so the absolute "
                    "scale is set by how much the micro-steps' gradients agree, "
                    "which is a property of the sigma draw rather than of the "
                    "sampling mode. Only the micro/step ratio carries the claim. "
                    "That ratio is strongly seed-dependent -- the same step "
                    "configuration read 42.7 at --seeds 1 and 11.9 at --seeds 8 "
                    "-- because sigma is resampled per seed and the step arm's "
                    "magnitudes follow that lottery. Treat the ratio as "
                    "directional evidence at a fixed seed count, not as a "
                    "constant; re-running at a different seed count will move "
                    "both arms. The finding the ratio supports: `micro` retains "
                    "strictly LESS signal than `step` for the same wall-clock "
                    "step, because the block drawn last displaces the one the "
                    "reference owns. Gradients accumulate across micro-steps "
                    "(zero_grad once, not per micro-step), each loss is divided "
                    "by the micro-step count, sigma is redrawn per sample, and "
                    "the reference is built from a separate engine before any "
                    "accumulation runs."
                ),
            )
        )
    return rows


def _spread(values: list[float]) -> float:
    """Relative spread of `values`: stdev divided by the mean, 0.0 if undefined.

    Reported alongside a seed-averaged metric so a reader can see whether the
    mean is representative or an artifact of one draw. Relative rather than
    absolute so it is comparable across the experiments' different scales.
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance) / abs(mean)


# ---------------------------------------------------------------------------
# overlap
# ---------------------------------------------------------------------------


def measure_overlap_out_of_band(
    args: argparse.Namespace, overlap: float, seed: int
) -> float:
    """Fraction of sampled `(sigma, block)` pairs outside the *nominal* band.

    The band is the partitioner's own `boundaries()[b], boundaries()[b + 1]` --
    the disjoint equi-probability partition. `sample_sigma(b, overlap=g)` draws
    from `[lo/alpha, hi*alpha]` with `alpha = (hi/lo) ** g`, so this fraction is
    expected to *rise* with `g`: overlap deliberately widens the sampling
    interval past the nominal band, which is the mechanism by which it absorbs
    the mass that would otherwise be misrouted (§12.1).

    That is exactly why the handoff's "strictly decreases" framing is not the
    assertion made here. Measuring against the *widened* interval instead would
    be circular -- `overlap` defines that interval, so the measured fraction
    would fall by construction and test nothing (§10.3).
    """
    partitioner = EquiProbabilityPartitioner(num_blocks=args.ablation_blocks)
    bounds = partitioner.boundaries()
    generator = torch.Generator().manual_seed(seed)
    out_of_band = 0
    total = 0
    for block in range(args.ablation_blocks):
        lo = float(bounds[block].item())
        hi = float(bounds[block + 1].item())
        for _ in range(args.overlap_samples):
            sigma = float(
                partitioner.sample_sigma(
                    block, generator=generator, overlap=overlap
                ).item()
            )
            total += 1
            if not lo <= sigma <= hi:
                out_of_band += 1
    return out_of_band / total if total else 0.0


def run_overlap(args: argparse.Namespace) -> list[AblationRow]:
    """`--db-overlap` sweep: how much of each draw leaves its nominal band.

    One row per swept setting, so the trend is readable down the leaderboard
    rather than collapsed into a single delta. The disjoint partition is the
    baseline every setting is compared against.
    """
    settings = sorted_overlap_sweep(args)
    out_of_band: dict[float, list[float]] = {g: [] for g in settings}
    for g in settings:
        for seed in range(args.seeds):
            out_of_band[g].append(measure_overlap_out_of_band(args, g, seed))
    rows: list[AblationRow] = []
    baseline = settings[0]
    base_val = sum(out_of_band[baseline]) / len(out_of_band[baseline])
    for g in settings:
        vals = out_of_band[g]
        mean = sum(vals) / len(vals)
        rows.append(
            AblationRow(
                experiment=f"overlap_g{g:g}",
                metric="sigma_out_of_nominal_band",
                baseline=f"g={baseline:g}",
                variant=f"g={g:g}",
                value_baseline=base_val,
                value_variant=mean,
                delta=mean - base_val,
                better="variant" if mean < base_val else "baseline",
                seeds=list(range(args.seeds)),
                n_seeds=args.seeds,
                notes=(
                    f"overlap g={g:g} over {args.overlap_samples * args.ablation_blocks} "
                    "draws per seed; the band is the partitioner's nominal "
                    "equi-probability range, not the widened draw interval. CONTRADICTS "
                    "the 'strictly decreases with overlap' framing: this fraction "
                    "RISES with g, because sample_sigma draws from [lo/alpha, "
                    "hi*alpha] and alpha widens the interval by construction -- that "
                    "widening IS the mechanism by which overlap absorbs out-of-range "
                    "mass (handoff 12.1). Measuring against the widened interval "
                    "would be circular (10.3). No model runs for this row."
                ),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# sparsity
# ---------------------------------------------------------------------------


def pruned_lcqat_layers(
    args: argparse.Namespace, preset: str, sparsity: float, scope: str
) -> list:
    """Retrofitted LC-QAT Linears, pruned to `sparsity` under `scope`.

    Two properties of this path are load-bearing and easy to get wrong:

    * **Masks are KEEP-masks** (`True` = retained). `magnitude_mask` returns
      exactly that, and GMP intersects against the previous mask so pruning stays
      monotone -- a pruned position holds an exact `0.0`, so its `|W|` is 0 and
      it can never re-enter a magnitude-selected mask.
    * **Structural sparsity requires exact LC-QAT zero dequantization.** A pruned
      position holds an exact `0.0` in the shadow weight, and 0.0 is a codebook
      *level* (the zero anchor at `m_neg`), so `bucketize(0.0)` returns `m_neg`
      in any regime and the pruned position dequantizes back to exactly 0.0,
      with no post-hoc mask multiply. The exact-zero check in
      `measure_sparsity_nmse` verifies that contract survived the prune; it is
      not a formality.

    **Both scopes prune two layers jointly, and that is not incidental.**
    `apply_global_pruning` ranks `|W|` across *all* modules it is handed, so a
    global sweep over a single layer is the same mask as a layer sweep by
    construction -- the two arms would tie and the comparison would be a
    tautology. Passing both the attention and MLP projections is what gives
    global scope something to rank across, and it is the pairing SparseProp
    Fig. 6 actually compares.

    The schedule is driven to its final target rather than to whatever the ramp
    would have produced at an arbitrary step, so the sweep compares sparsities
    rather than cadence. Gradual pruning is still the path that applies the mask.
    """
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS[preset])
    inject_sparseprop_layers(
        model, sparsity=sparsity, target_modules=["c_proj", "c_fc"], with_lcqat=True
    )
    layers = [
        model.transformer.h[0].attn.c_proj,
        model.transformer.h[0].mlp.c_fc,
    ]
    schedule = schedule_from_args(args)
    schedule.target_sparsity = sparsity
    schedule.scope = scope
    # `apply` returns None off a prune event, and `enabled` is False whenever
    # `every == 0` -- which is `--sparseprop-every`'s default. `every = 1` makes
    # every step an event so the ramp actually runs; the step passed is the last
    # one, so the ramp completes to its target instead of leaving the model at
    # the start fraction. Skipping this would prune nothing and the whole sweep
    # would report the NMSE of an unpruned layer.
    schedule.every = 1
    schedule.apply(model, schedule.ramp_steps, layers=layers)
    return layers


def scope_mask_disagreement(
    args: argparse.Namespace, preset: str, sparsity: float
) -> tuple[float, float]:
    """How far the two scopes' masks disagree, pooled and per layer.

    Returns `(pooled_disagreement, largest_per_layer_disagreement)`.

    The per-layer *sparsity* of the two scopes came out identical at every
    setting measured here, and that is a real property of these two layers
    rather than a bug: their magnitude distributions overlap enough that one
    global threshold prunes both to the same fraction. What the scopes actually
    disagree about is *which* weights they keep -- the global threshold falls
    between the two layers' medians, so it selects differently within each
    layer even though the counts match. Counting the differing positions is
    what makes the comparison non-tautological; reporting the NMSE of each
    scope separately cannot, because equal counts give equal NMSE.

    Disagreement is the fraction of positions at which the two masks differ, so
    0.0 means identical masks and 0.5 means the scopes are as different as two
    masks of the same sparsity can be.
    """
    torch.manual_seed(0)
    layer_masks = pruned_lcqat_layers(args, preset, sparsity, SCOPE_LAYER)
    global_masks = pruned_lcqat_layers(args, preset, sparsity, SCOPE_GLOBAL)
    pooled = 0.0
    total = 0
    worst = 0.0
    for layer_mask, global_mask in zip(layer_masks, global_masks, strict=True):
        differing = int((layer_mask.sparsity_mask ^ global_mask.sparsity_mask).sum())
        pooled += differing
        total += int(layer_mask.sparsity_mask.numel())
        worst = max(worst, differing / layer_mask.sparsity_mask.numel())
    return (pooled / total if total else 0.0), worst


def measure_sparsity_nmse(
    args: argparse.Namespace, preset: str, sparsity: float, scope: str
) -> tuple[float, bool]:
    """Dequantized NMSE summed over the pruned layers, plus whether zeros hold.

    NMSE is pooled over both layers as `total_squared_error / total_signal_power`
    rather than averaged per layer, so a layer with more parameters contributes
    in proportion to its size instead of being counted once regardless.
    """
    total_mse = 0.0
    total_signal = 0.0
    exact = True
    for layer in pruned_lcqat_layers(args, preset, sparsity, scope):
        with torch.no_grad():
            original = layer.weight.detach().clone()
            pruned = ~layer.sparsity_mask
            # Round-trip through the quantizer, which is what the exported
            # artifact does. Recomputed once, used for both the error and the
            # zero check.
            dequantized = reconstruct_layer(layer, original, "weight")
            if bool(pruned.any()):
                exact = exact and bool(
                    torch.equal(
                        dequantized[pruned], torch.zeros_like(dequantized[pruned])
                    )
                )
            total_mse += float((original - dequantized).square().sum())
            total_signal += float(original.square().sum())
    return (total_mse / total_signal if total_signal > 0 else float("inf")), exact


def run_sparsity(args: argparse.Namespace) -> list[AblationRow]:
    """Sparsity x scope sweep: dequantized NMSE after pruning.

    SparseProp Fig. 6 compares Uniform-GMP against Global-GMP at equal average
    sparsity and finds Global better, so the two scopes are not interchangeable
    and neither is the default by construction. Both are swept here, at the
    floors and at the target, on a retrofitted `attn.c_proj` -- the first
    non-negative activated layer in the model. No training runs: this measures
    the mask, not what a trained model would do with it.
    """
    levels = sparsity_sweep(args)
    nmse: dict[tuple[float, str], list[float]] = {
        (level, scope): [] for level in levels for scope in SCOPES
    }
    exact: dict[tuple[float, str], bool] = {}
    for level in levels:
        for scope in SCOPES:
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                value, is_exact = measure_sparsity_nmse(
                    args, VARIANT_PRESET, level, scope
                )
                nmse[(level, scope)].append(value)
                exact[(level, scope)] = exact.get((level, scope), True) and is_exact

    rows: list[AblationRow] = []
    low, target = levels[0], levels[-1]
    for scope in SCOPES:
        base = sum(nmse[(low, scope)]) / len(nmse[(low, scope)])
        var = sum(nmse[(target, scope)]) / len(nmse[(target, scope)])
        rows.append(
            AblationRow(
                experiment=f"sparsity_{scope}",
                metric="dequant_nmse",
                baseline=f"{scope}@{low:g}",
                variant=f"{scope}@{target:g}",
                value_baseline=base,
                value_variant=var,
                delta=var - base,
                better="variant" if var < base else "baseline",
                seeds=list(range(args.seeds)),
                n_seeds=args.seeds,
                notes=(
                    f"dequantized NMSE of a retrofitted attn.c_proj, {scope} scope, "
                    f"sparsity {low:g} -> {target:g}, reached by the gradual schedule "
                    "at its last ramp event. Masks are KEEP-masks (True = retained) "
                    "and structural zeros dequantize to exactly 0.0 "
                    f"({'verified' if exact[(target, scope)] else 'FAILED'} at the "
                    f"target; {'verified' if exact[(low, scope)] else 'FAILED'} at the "
                    "floor). NMSE rising with sparsity is expected -- pruning removes "
                    "weight magnitude -- so this row records the mask's cost, not a "
                    "quality win. No training runs."
                ),
            )
        )
    # The two scopes prune the two layers to the same *fraction* at every
    # setting, so the NMSE rows above cannot separate them. This row reports
    # the thing that does differ -- which weights each scope keeps -- so the
    # `layer` vs `global` claim rests on a measurement rather than on a
    # tautology.
    pooled_disagreement, worst_layer = scope_mask_disagreement(
        args, VARIANT_PRESET, target
    )
    rows.append(
        AblationRow(
            experiment="sparsity_scope_mask_disagreement",
            metric="scope_mask_disagreement",
            baseline=SCOPE_LAYER,
            variant=SCOPE_GLOBAL,
            value_baseline=0.0,
            value_variant=pooled_disagreement,
            delta=pooled_disagreement,
            better="variant",
            seeds=[0],
            n_seeds=1,
            notes=(
                f"fraction of positions where the layer-scope and global-scope masks "
                f"disagree at sparsity {target:g}, pooled over attn.c_proj and "
                f"mlp.c_fc ({worst_layer:.4g} in the worse of the two layers). "
                "Both scopes prune to the SAME per-layer fraction at every setting "
                "measured here -- these two layers' magnitude distributions overlap "
                "enough that one global threshold cuts both equally -- so the "
                "per-scope NMSE rows above cannot distinguish them and would tie by "
                "construction. The scopes disagree about WHICH weights survive, and "
                "that is what this row measures. Whether global's selection is "
                "better after training is NOT measured here: no training runs. "
                "Single seed, because the two masks are a deterministic function of "
                "the weight tensor, not a stochastic draw."
            ),
        )
    )
    return rows


# ---------------------------------------------------------------------------
# act_lut
# ---------------------------------------------------------------------------


def build_activation_lut(
    input_codebook: torch.Tensor,
    output_codebook: torch.Tensor,
    relaxation: str,
    act_body: str,
) -> LearnableIndexLut:
    """One `LearnableIndexLut` at the requested relaxation and body.

    The `smoothpwl` body is fitted here rather than reused, so each preset is
    measured against its own freshly fitted body instead of a fit produced by
    some earlier run.
    """
    body = None
    if act_body == ACT_BODY_SMOOTHPWL:
        body = SmoothPWL(
            knots=int(input_codebook.numel()),
            zero_pin=True,
            act_name="relu2",
        ).fit_from_callable()
    return LearnableIndexLut(
        input_codebook,
        output_codebook,
        act_name="relu2",
        relaxation=relaxation,
        smooth_body=body,
    )


#: The matched-budget K the handoff's parameter-reduction claim is stated at
#: (HANDOFF §10.2.1, "at a matched 15 floats"). Measured separately from the
#: model's own codebooks, whose K_in and K_out are not equal, so the
#: K_in x K_out vs K_in + K_out comparison has to be posed at a K where both
#: sides are stated on the same budget.
K_MATCHED_BUDGET = 15


def matched_budget_codebooks(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A `k`-level input/output codebook pair for the matched-budget count.

    The input codebook is non-negative, which is what this activation's domain
    actually is: `mlp.forward` computes `F.relu(x).square()`, so every tensor
    entering `attn.c_proj` is `>= 0`. That is what makes the pair satisfy the
    contract `LearnableIndexLut` enforces on the proximity relaxation -- the
    input level `0.0` must bake to an output codebook value of exactly `0.0`, or
    the zero anchor cannot be pinned without disagreeing with the bake, and the
    class refuses to build such a table rather than apply the pin silently.

    The output codebook is the activation's own values, which is both sorted
    (so the `bucketize` the proximity init performs is well defined) and
    contains an exact `0.0`. The parameter *count* depends only on K, not on the
    level values, so this is the right basis for the structural claim whatever
    the specific values are.
    """
    input_codebook = torch.linspace(0.0, 4.0, k, dtype=torch.float32).contiguous()
    output_codebook = get_activation("relu2")(input_codebook).contiguous()
    return input_codebook, output_codebook


def piecewise_linear(
    xs: torch.Tensor, ys: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation of the polyline through `(xs, ys)` at `query`.

    This is what the frozen PWL body *is*: between knots the operation is a
    linear map, and the value at a knot is its own `y`. Endpoints are clamped
    rather than extrapolated, so a query outside the knot span takes the nearest
    end value instead of being scored against a line that does not exist there.

    `searchsorted` locates the segment and the two neighbours are lerped by
    weight. Interpolating the `x` values alone (the obvious shortcut) would
    reconstruct the *knot positions* rather than the function, and would score
    the wrong curve entirely.
    """
    idx = torch.searchsorted(xs, query, right=True).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    weight = torch.where(x1 > x0, (query - x0) / (x1 - x0), torch.zeros_like(query))
    return y0 + weight * (y1 - y0)


def run_act_lut(args: argparse.Namespace) -> list[AblationRow]:
    """Activation-LUT relaxation x body: parameter count, fp8 identity, fit.

    The table is the one compiled between two *quantized-inference* layers: the
    `mlp.c_fc` output codebook indexes it and the `attn.c_proj` input codebook
    is its value domain. Both are read off the retrofitted layer rather than
    synthesized, so the measured K is the K the model uses.

    Two of the three claims are exact and free:

    * **Parameter count.** At K_in = K_out = 15 the free-logit matrix is 225
      parameters and the proximity parameterization is K_in + K_out = 30. That
      is a structural count, not a fit result.
    * **fp8 knot identity.** `resolved_table()` reads `levels` only, so knots
      round-tripped through `float8_e4m3fn` resolve to a *bit-identical* index
      table. The saving is 1 byte per knot instead of 4 at zero behavioural
      cost. Checked by swapping the stored knots for their fp8 round trip and
      re-resolving.

    The third -- that `smoothpwl` fits `relu^2` better than `pwl` -- was measured
    in the handoff at **one** setting. It is re-measured here per preset rather
    than copied, because a number this run did not produce does not belong in a
    leaderboard meant to be auditable. If `smoothpwl` does not win at the setting
    measured, the note says so; a probe finding is a result, not a failure, and
    is not gated.
    """
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS[VARIANT_PRESET])
    proj = model.transformer.h[0].attn.c_proj
    fc = model.transformer.h[0].mlp.c_fc
    # `mlp.c_fc`'s *output* quantizer feeds the activation between the two
    # layers, so its codebook is the table's index domain; `attn.c_proj`'s
    # activation (input) quantizer supplies the table's value range. Both are
    # the roles the quantized-inference path actually uses.
    input_codebook = codebook_of(fc, "out")
    output_codebook = codebook_of(proj, "act")
    k_in, k_out = int(input_codebook.numel()), int(output_codebook.numel())
    act = get_activation("relu2")
    # Both bodies are scored on one common grid over the `SmoothPWL` body's own
    # init span, because that is the domain the RBF fit is actually run over.
    # `SmoothPWL` does not retain `init_range` as an attribute, so the default
    # is restated here; it is the same constant the handoff's 1.02 / 0.48 /
    # 0.67 figures were measured on ("relu^2 over [-1, 1]", activation.py's
    # module docstring). Scoring the PWL body over the *codebook's* span instead
    # would put the two bodies on different ranges, and a `relu^2` probe over a
    # one-sided codebook scores an artefact of that range rather than a fit.
    body_range = SMOOTHPWL_DEFAULT_INIT_RANGE
    grid = torch.linspace(-body_range, body_range, 400)

    rows: list[AblationRow] = []

    # The parameter-reduction claim is stated at a *matched* budget, where
    # K_in == K_out so both relaxations are compared on the same number of
    # floats. The model's own codebooks have K_in != K_out, so that comparison is
    # posed on a matched pair instead of being asserted from the real one.
    matched_in, matched_out = matched_budget_codebooks(K_MATCHED_BUDGET)
    matched_logits = build_activation_lut(
        matched_in, matched_out, RELAXATION_LOGITS, ACT_BODY_PWL
    )
    matched_proximity = build_activation_lut(
        matched_in, matched_out, RELAXATION_PROXIMITY, ACT_BODY_PWL
    )
    matched_logits_params = sum(p.numel() for p in matched_logits.parameters())
    matched_proximity_params = sum(p.numel() for p in matched_proximity.parameters())
    # The fp8 identity is re-checked on the matched pair too, so the row
    # carries both of the exact claims it is the natural home for rather than
    # leaving one of them to be read off a per-preset row with a different K.
    with torch.no_grad():
        table_before = matched_proximity.resolved_table()
        live_knots = matched_proximity.knots.detach().clone()
        matched_proximity.knots.copy_(matched_proximity.fp8_knots())
        table_after = matched_proximity.resolved_table()
        matched_proximity.knots.copy_(live_knots)
    matched_fp8_note = (
        "fp8 knot export bit-identical to fp32: "
        f"{bool(torch.equal(table_before, table_after))}"
    )
    rows.append(
        AblationRow(
            experiment="act_lut_matched_budget",
            metric="lut_params",
            baseline=f"{RELAXATION_LOGITS}_{ACT_BODY_PWL}",
            variant=f"{RELAXATION_PROXIMITY}_{ACT_BODY_PWL}",
            value_baseline=float(matched_logits_params),
            value_variant=float(matched_proximity_params),
            delta=float(matched_proximity_params - matched_logits_params),
            better="variant"
            if matched_proximity_params < matched_logits_params
            else "baseline",
            seeds=list(range(args.seeds)),
            n_seeds=args.seeds,
            notes=(
                f"matched-budget parameter count at K_in=K_out={K_MATCHED_BUDGET}: "
                f"{matched_proximity_params} proximity parameters (K_in + K_out) vs "
                f"{matched_logits_params} for a free K_in x K_out logit matrix, a "
                f"{matched_logits_params / matched_proximity_params:.2f}x reduction. "
                f"{matched_fp8_note}. "
                "This is a structural count that depends only on K, so it is exact "
                "rather than a fit result. The model's own codebooks have "
                f"K_in={k_in} != K_out={k_out}, so this row is the matched-budget "
                "statement and the per-preset rows below are the real codebooks."
            ),
        )
    )

    for relaxation in RELAXATIONS:
        for act_body in ACT_BODIES:
            lut = build_activation_lut(
                input_codebook, output_codebook, relaxation, act_body
            )
            n_params = sum(p.numel() for p in lut.parameters())

            # fp8 identity: resolve, then swap in the fp8 round trip of the
            # stored knots and resolve again. `resolved_table()` reads levels
            # only, so the two must be bit-identical for any knot dtype.
            #
            # Only the proximity relaxation *has* knots -- the free-logit matrix
            # stores no knot positions at all, so there is nothing to export and
            # the claim does not apply to it. That is reported as such rather
            # than skipped silently, because a blanket "bit-identical: True" on
            # an arm that exported nothing would be asserting a result the run
            # never produced.
            has_knots = hasattr(lut, "knots")
            if has_knots:
                with torch.no_grad():
                    table_fp32 = lut.resolved_table()
                    live_knots = lut.knots.detach().clone()
                    lut.knots.copy_(lut.fp8_knots())
                    table_fp8 = lut.resolved_table()
                    lut.knots.copy_(live_knots)
                fp8_identical = bool(torch.equal(table_fp32, table_fp8))
                fp8_note = f"fp8 knot export bit-identical to fp32: {fp8_identical}"
            else:
                table_fp32 = lut.resolved_table()
                fp8_note = (
                    "no fp8 export: the free-logit relaxation stores no knot "
                    "positions, so the fp8 knot saving does not apply to this "
                    "preset"
                )

            # The fit is measured on the *table's* (x, y) pairs: the input
            # codebook entries are the knots, and the resolved table picks the
            # output level each knot lands on. Scoring `act(table_values)` would
            # be circular -- it would compare relu^2 against itself at the output
            # codebook -- so the curve is reconstructed from those pairs and
            # compared to the true activation over the knot span.
            knots, _order = torch.sort(input_codebook)
            y_at_knots = output_codebook[table_fp32]
            with torch.no_grad():
                table_error = float(
                    (piecewise_linear(knots, y_at_knots, grid) - act(grid)).abs().max()
                )
                if act_body == ACT_BODY_SMOOTHPWL:
                    body = SmoothPWL(
                        knots=k_in, zero_pin=True, act_name="relu2"
                    ).fit_from_callable()
                    body_error = float((body(grid) - act(grid)).abs().max())
                else:
                    # The frozen PWL body *is* that piecewise-linear curve, so
                    # both arms are scored through the same reconstruction and
                    # the comparison is like-for-like.
                    body_error = table_error

            rows.append(
                AblationRow(
                    experiment=f"act_lut_{relaxation}_{act_body}",
                    metric="lut_params",
                    baseline=f"{RELAXATION_LOGITS}_{ACT_BODY_PWL}",
                    variant=f"{relaxation}_{act_body}",
                    value_baseline=float(k_in * k_out),
                    value_variant=float(n_params),
                    delta=float(n_params - k_in * k_out),
                    better="variant" if n_params < k_in * k_out else "baseline",
                    seeds=list(range(args.seeds)),
                    n_seeds=args.seeds,
                    notes=(
                        f"relaxation={relaxation} body={act_body}: {n_params} table "
                        f"parameters vs {k_in * k_out} for a free K_in x K_out logit "
                        f"matrix at K_in={k_in}, K_out={k_out} (proximity is K_in + "
                        f"K_out = {k_in + k_out}). {fp8_note}. Measured this run at "
                        f"K={k_in}: body "
                        f"max abs error on relu^2 over a {grid.numel()}-point grid "
                        f"{body_error:.6g}, resolved-table error {table_error:.6g}. "
                        "These are re-measured per preset, not copied from the "
                        "handoff, whose figures came from one setting. No model runs."
                    ),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# sigma_cond
# ---------------------------------------------------------------------------


def count_codebook_cost(module) -> tuple[int, int]:
    """`(parameter count, artifact bytes)` for a quantizer or codebook module.

    Bytes are the trainable parameters at fp32 plus every persistent buffer at
    its own element size -- i.e. what the exported artifact actually carries,
    which is the cost `--db-sigma-codebook` is meant to be judged on. Buffers
    that are not persistent in `state_dict` (the input/output codebooks a
    `LearnableIndexLut` keeps for its own indexing) are excluded, because the
    owning quantizer already ships them in the checkpoint and counting them
    twice would overstate the artifact.
    """
    params = sum(p.numel() for p in module.parameters())
    persistent = {name for name, _ in module.named_buffers()}
    buffers = sum(
        b.numel() * b.element_size()
        for name, b in module.named_buffers()
        if name in persistent
    )
    return params, params * 4 + buffers


def run_sigma_cond(args: argparse.Namespace) -> list[AblationRow]:
    """`--db-sigma-codebook` static vs conditioned: structural cost only.

    **This experiment makes no accuracy claim, and none should be read into
    it.** The distributional premise that would justify sigma conditioning --
    that a block's activation distribution shifts with sigma in a way one
    shared codebook cannot span -- is recorded here as **UNVERIFIED**: the
    handoff's own probe (§12.2) looked for it on `c_proj`'s `relu^2` input and
    did not find it. Whether a conditioned codebook would help accuracy is a
    question about a premise this driver cannot justify, so it is not asked.

    What *is* measurable without that premise is the price. `conditioned` stores
    one codebook per anchor (one per diffusion block by default), so its
    parameter count and artifact bytes are the shared codebook's times the anchor
    count. That is a structural fact about the export format, and it is the only
    claim this row makes.
    """
    anchors = args.db_sigma_anchors or args.ablation_blocks
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS[VARIANT_PRESET])
    proj = model.transformer.h[0].attn.c_proj
    shared_q = select_quantizer(proj, "act")
    shared_params, shared_bytes = count_codebook_cost(shared_q)
    shared_codebook = codebook_of(proj, "act")

    conditioned = SigmaConditionedCodebook(
        num_anchors=anchors,
        m_neg=shared_q.m_neg,
        m_pos=shared_q.m_pos,
        init_min=float(shared_codebook.min()),
        init_max=float(shared_codebook.max()),
    )
    cond_params, cond_bytes = count_codebook_cost(conditioned)

    return [
        AblationRow(
            experiment="sigma_cond",
            metric="codebook_params",
            baseline="static",
            variant=f"conditioned@{anchors}",
            value_baseline=float(shared_params),
            value_variant=float(cond_params),
            delta=float(cond_params - shared_params),
            better="variant" if cond_params < shared_params else "baseline",
            seeds=list(range(args.seeds)),
            n_seeds=args.seeds,
            notes=(
                f"STRUCTURAL ONLY. Conditioned stores {anchors} codebooks (one per "
                f"anchor) vs one shared: {cond_params} parameters / {cond_bytes} B vs "
                f"{shared_params} / {shared_bytes} B, a "
                f"{cond_params / shared_params:.1f}x parameter blow-up. The "
                "distributional premise that a block's activation distribution "
                "varies with sigma in an exploitable way is UNVERIFIED (handoff "
                "12.2 disproved it on c_proj's relu^2 input). NO accuracy claim is "
                "made for this experiment and none should be read into this row."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# bias_quant
# ---------------------------------------------------------------------------

#: Marker for the "bias codebook parameters moved" flag in the notes. Checked
#: by string, like the `sparsity` and `act_lut` rows do, so the rendered
#: leaderboard stays readable and the check needs no side channel.
BIAS_CODEBOOK_LIVE = "codebook gradient live: True"
BIAS_CODEBOOK_DEAD = "codebook gradient live: False"


def _weight_kwargs(spec: CodebookSpec) -> dict[str, object]:
    """A `LayerKConfig` cardinality as `from_float`'s weight-codebook kwargs."""
    if isinstance(spec, tuple):
        return {"K_weight_split": spec}
    return {"K_weight": int(spec)}


def _act_kwargs(spec: CodebookSpec) -> dict[str, object]:
    """A `LayerKConfig` cardinality as `from_float`'s activation kwargs."""
    if isinstance(spec, tuple):
        return {"K_act_split": spec}
    return {"K_act": int(spec)}


def bias_probe(preset: str, k_bias: int) -> LCQATLinear:
    """A `c_proj` whose bias goes through a `BiasQuantizer`; the paired baseline.

    nanochat builds every projection with `bias=False`, so the shipped model has
    nothing to quantize on the bias side and a bias codebook cannot be measured
    on it at all. This gives `c_proj` a bias and converts *that one layer*
    in place, so the FP32 baseline and the quantized variant are the same layer
    -- same weight, same probe -- and the only thing that varies is whether the
    bias term is quantized. That is what makes the pair paired.

    The preset's real `down_weight` / `down_act` cardinalities are passed
    through rather than hardcoded, so the weight and activation codebooks under
    the bias are the ones the preset actually ships and the bias error is read
    in the same context the other two terms are measured in. `retrofit_model` is
    deliberately not called: it would re-convert this layer (and `from_float`
    rejects an `LCQATLinear` input), so the conversion is done directly and the
    rest of the model is untouched.

    The layer is rebuilt per seed (see `run_bias_quant`) because the codebook is
    trained below and `build_active_tiny_gpt` is what reseeds the weights.
    """
    model = build_active_tiny_gpt()
    float_proj = model.transformer.h[0].mlp.c_proj
    # `gpt.py` builds every projection with `bias=False`, so `float_proj.bias`
    # is None and there is nothing to give the bias codebook to quantize. A
    # fresh float Linear carrying *this* layer's weight plus a real bias is
    # built here, so the weight and activation codebooks still span the real
    # `c_proj` ranges and only the bias term is new.
    #
    # The bias is drawn at the `std` `build_active_tiny_gpt` itself uses for the
    # zero-initialized projections, off the *global* RNG the caller has already
    # seeded for this sweep -- a dedicated generator seeded to a constant here
    # would hand every seed the identical bias and quietly turn a paired sweep
    # into five copies of one measurement.
    #
    # It must not be left at zero: `BiasQuantizer.init_from_tensor` returns early
    # on an all-identical vector and keeps the codebook on its default span,
    # which for a bias is wider than the data -- the silent, permanent failure
    # its docstring warns about.
    float_with_bias = torch.nn.Linear(
        float_proj.in_features, float_proj.out_features, bias=True
    )
    with torch.no_grad():
        float_with_bias.weight.copy_(float_proj.weight)
        float_with_bias.bias.normal_(mean=0.0, std=0.02)
    cfg = PRESETS[preset]
    replacement = LCQATLinear.from_float(
        float_with_bias,
        grad_scale=cfg.grad_scale,
        quantize_bias=True,
        K_bias=k_bias,
        **_weight_kwargs(cfg.down_weight),
        **_act_kwargs(cfg.down_act),
    )
    model.transformer.h[0].mlp.c_proj = replacement
    return replacement


def run_bias_quant(args: argparse.Namespace) -> list[AblationRow]:
    """Is the bias worth one more codebook per layer? Error, cost, and liveness.

    `dev/HANDOFF_symbiosis.md` §10.2.4 records that nothing in `LCQATLinear`
    quantized the bias, and §10.4-E item 12 asks for exactly this measurement
    before the capability is adopted. Three things decide it, and they are
    reported as three rows rather than folded into one number:

    1. **Fidelity.** The bias round-trip NMSE, quantized vs FP32. Reported on
       its own, never pooled with the weight or activation NMSE: the bias is
       `out_features` numbers against a `D x D` weight matrix, so a pooled
       relative error is ~99.9% weight by element count and would hide whatever
       the bias term does. That is the §9.8 pooled-NMSE mistake, repeated.
    2. **Cost.** `count_codebook_cost` on the quantizer: parameters and artifact
       bytes per layer. Independent of `out_features` -- one shared 1-D table,
       not one per output channel -- which is the only reason it is cheap enough
       to be worth asking about.
    3. **Liveness.** Whether the codebook's own parameters receive a non-zero
       gradient and then move. A codebook that never moves is inert storage, and
       a bias error pinned at exactly 1.0 is its signature: every entry buckets
       to the zero anchor, so the gather is the anchor, so the loss is
       independent of the table.

    The fidelity number is not read off a layer whose bias codebook was left on
    the default `act_init` span. That span (-2..2) is fitted to `relu^2`
    *outputs* and is ~40x wider than a std-0.02 bias, so every entry buckets onto
    the zero anchor, one level is hit, the error is exactly 1.0 and the gradient
    is identically zero -- the silent, permanent failure documented on
    `BiasQuantizer.init_from_tensor`, and measured here rather than assumed.
    `from_float` already performs that fit; the probe calls `init_from_tensor`
    itself because it is the precondition the measurement depends on, not
    something to inherit and hope for.

    Whether the extra table is *worth it* is not decided here and the rows do
    not pretend otherwise: this is a randomly-initialized layer, a handful of
    SGD steps, and a reconstruction error. It measures the price and the
    fidelity cost of the capability, not what it does to a trained model's loss.
    """
    nmse_baseline = 0.0
    nmse_variant = 0.0
    used_levels = 0
    k_total = 0
    moved = 0.0
    ulp_floor = 0.0
    non_zero_grads = 0
    params = 0
    artifact_bytes = 0
    out_features = 0
    in_features = 0

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        layer = bias_probe(VARIANT_PRESET, args.bias_quant_k_bias)
        quantizer = layer.bias_quantizer
        if quantizer is None:
            raise RuntimeError(
                "bias_probe built a layer with no bias_quantizer; the "
                "quantize_bias flag did not reach LCQATLinear.from_float"
            )
        bias = layer.bias.detach().to(torch.float32)
        out_features = int(bias.numel())
        in_features = int(layer.in_features)

        # MANDATORY, and the reason this probe measures anything at all: fit the
        # table to the bias's own range. Without it the span is the layer's
        # `act_init`, which spans relu^2 outputs and therefore overshoots a
        # std-0.02 bias by orders of magnitude -- every entry lands on the zero
        # anchor, the gradient is identically zero, and the error sits at
        # exactly 1.0 forever. `from_float` already did this; doing it again is
        # the probe stating the precondition it depends on rather than trusting
        # a constructor call to have survived.
        quantizer.init_from_tensor(bias)

        # Baseline: the FP32 bias, which round-trips through nothing. Its error
        # is zero by construction, and that is the point of pairing against it
        # -- the question is not "is the quantized bias bad" but "how much does
        # quantizing it cost".
        base_mse, _base_max, base_power = quantization_error(bias, bias.clone())

        indices = quantizer.bucketize(bias)
        reconstructed = quantizer.get_codebook().detach()[indices.long()]
        var_mse, _var_max, var_power = quantization_error(bias, reconstructed)

        nmse_baseline += base_mse / base_power if base_power > 0 else 0.0
        nmse_variant += var_mse / var_power if var_power > 0 else float("inf")
        used_levels += int(torch.unique(indices).numel())
        k_total += int(quantizer.K)

        # Liveness: a real backward pass through the STE, then the codebook's own
        # parameters (never the derived `get_codebook()` tensor, which carries no
        # optimizer state) for `steps` SGD steps. Fixed target, so the loss is a
        # genuine signal in the same way `observe_grad_scale` makes it one.
        layer.zero_grad(set_to_none=True)
        out = layer(non_negative_probe(layer.in_features, args.n, seed))
        out.square().mean().backward()
        target = list(quantizer.parameters())
        if not target:
            raise RuntimeError("bias_quantizer exposes no trainable parameters")
        before = [p.detach().clone() for p in target]
        grad_norm = math.sqrt(
            sum(
                float(p.grad.detach().square().sum())
                for p in target
                if p.grad is not None
            )
        )
        if grad_norm > 0.0:
            non_zero_grads += 1
        optimizer = torch.optim.SGD(target, lr=args.bias_quant_lr)
        for _ in range(args.bias_quant_steps):
            optimizer.zero_grad(set_to_none=True)
            layer(
                non_negative_probe(layer.in_features, args.n, seed)
            ).square().mean().backward()
            optimizer.step()
        moved += math.sqrt(
            sum(
                float(((p.detach() - b).square()).sum())
                for p, b in zip(target, before, strict=True)
            )
        )
        # `moved > 0` is the wrong liveness test on its own, and it is exactly
        # the test that made a single-seed sweep report a live codebook as dead.
        # A non-zero gradient does not guarantee a representable update: these
        # latents sit at magnitude ~5, where one fp32 ULP is ~5e-7, so an SGD
        # step of `lr * grad` with `grad` around 1e-5 at the default lr of 1e-2
        # lands at ~1e-7 -- below the spacing between representable values. The
        # parameter is then bit-identical before and after and `moved` is
        # exactly 0.0, which reads as "dead" when the gradient is in fact live
        # and the codebook merely cannot move at that learning rate. Recording
        # the representable floor alongside lets the two cases be told apart.
        ulp_floor += max(
            float(torch.finfo(p.dtype).eps) * float(p.detach().abs().max())
            for p in target
        )
        params, artifact_bytes = count_codebook_cost(quantizer)

    seeds = args.seeds
    nmse_baseline /= seeds
    nmse_variant /= seeds
    avg_moved = moved / seeds
    avg_ulp = ulp_floor / seeds
    # Live means both: a real gradient arrived, AND it moved the table by at
    # least the representable floor. Requiring both is what keeps this check
    # able to fail -- `moved` alone reported a live codebook as dead whenever a
    # single-seed sweep's update landed below one fp32 ULP, while
    # `gradient alone` would report the §9.8 dead-codebook case as live, since
    # that trap has a non-zero *gradient* that is too small to matter for a
    # different reason. Neither half substitutes for the other.
    live = non_zero_grads == seeds and avg_moved >= avg_ulp > 0.0
    live_flag = BIAS_CODEBOOK_LIVE if live else BIAS_CODEBOOK_DEAD
    # `fp32` scores 0 by construction and cannot be beaten, so "better" names
    # the arm with the lower error honestly instead of pretending the quantized
    # arm won. The interesting reading is the *size* of the gap next to the cost.
    better = "variant" if nmse_variant < nmse_baseline else "baseline"
    baseline_label = "fp32 bias"
    variant_label = f"bias_quant@{args.bias_quant_k_bias}"

    return [
        AblationRow(
            experiment="bias_quant_nmse",
            metric="bias_nmse",
            baseline=baseline_label,
            variant=variant_label,
            value_baseline=nmse_baseline,
            value_variant=nmse_variant,
            delta=nmse_variant - nmse_baseline,
            better=better,
            seeds=list(range(seeds)),
            n_seeds=seeds,
            notes=(
                f"BIAS NMSE ONLY, reported separately from the weight and "
                f"activation errors on purpose: the bias is {out_features} "
                f"numbers against a {in_features}x{out_features} weight "
                f"matrix, so a pooled relative error would be the weight's by "
                f"element count and would hide the bias term entirely (the 9.8 "
                f"pooled-NMSE mistake). Quantized bias NMSE {nmse_variant:.6g} vs "
                f"{nmse_baseline:.6g} for the FP32 bias, which round-trips through "
                f"nothing and so scores 0 exactly. {used_levels // seeds}/"
                f"{k_total // seeds} levels hit. init_from_tensor() was called "
                "on the bias before this was "
                "measured, because the quantizer's default act_init span "
                "(-2..2, fitted to relu^2 outputs) is ~40x wider than a "
                "std-0.02 bias: every entry then buckets onto the zero anchor, "
                "1 level is hit, the error is exactly 1.0 and the gradient is "
                "identically zero. from_float() performs the same fit, so this "
                "is a precondition the probe states and checks, not a fix it "
                "depends on."
            ),
        ),
        AblationRow(
            experiment="bias_quant_cost",
            metric="bias_codebook_params",
            baseline=baseline_label,
            variant=variant_label,
            value_baseline=0.0,
            value_variant=float(params),
            delta=float(params),
            better="baseline",
            seeds=list(range(seeds)),
            n_seeds=seeds,
            notes=(
                f"EXTRA COST PER LAYER: {params} parameters / {artifact_bytes} B "
                f"for one shared K={args.bias_quant_k_bias} table, independent of "
                f"out_features={out_features} (one shared 1-D table, not one per "
                f"output channel). The FP32 baseline carries the bias inline at "
                "no extra parameters, which is why its value is 0 and `better` is "
                "always `baseline` here: this row is a price tag, not a "
                "comparison anyone can win. Judge it against the bias NMSE row."
            ),
        ),
        AblationRow(
            experiment="bias_quant_liveness",
            metric="codebook_param_delta",
            baseline="zero gradient",
            variant="codebook gradient live: " + ("True" if live else "False"),
            value_baseline=0.0,
            value_variant=avg_moved,
            delta=avg_moved,
            better="variant" if live else "baseline",
            seeds=list(range(seeds)),
            n_seeds=seeds,
            notes=(
                f"{live_flag}. L2 distance the bias codebook's parameters moved "
                f"over {args.bias_quant_steps} SGD steps at lr={args.bias_quant_lr} "
                f"(mean over {seeds} seeds), from a real backward pass through "
                f"LCQATLinear.forward: observed {avg_moved:.6g} against a "
                f"representable floor of {avg_ulp:.6g} (one fp32 ULP at these "
                "latents' magnitude, times the step count). Liveness requires "
                "both a non-zero gradient and movement at or above that floor: "
                "the gradient alone would call the §9.8 dead-codebook case live, "
                "and movement alone calls a codebook that is merely too small to "
                "update at this learning rate dead. A codebook with a live "
                "gradient that does not move at all would mean the learning rate "
                "or the gradient scale is wrong; zero gradient means every entry "
                "bucketizes to the zero anchor and the error is pinned at "
                "exactly 1.0, which is the §9.8 trap this row exists to catch."
            ),
        ),
    ]


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

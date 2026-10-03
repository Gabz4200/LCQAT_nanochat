"""Shared CLI surface and setup for the W6 DiffusionBlocks features.

Three opt-in features, all defined against the DiffusionBlocks engine rather
than against any one entry point, and all three of them are easy to add to one
script and forget in another:

* **Sigma-conditioned activation codebooks** (PRD 3.2). A block's quantizer
  otherwise has to span every noise level in the run, so most of its levels are
  spent on values that never occur.
* **EfQAT per-block latching** (PRD 3.2). A DiffusionBlocks block trains on one
  disjoint noise range, so its quantization parameters converge early and must
  then stop moving *permanently*.
* **Denoiser-distillation KD** (PRD 3.1, EDM form). The logit-KL anchor needs
  logits, which the denoising objective never produces, so it is its own path.

Registered through one helper so `base_train` / `chat_sft` / `chat_rl` cannot
drift apart on the same hardware knob -- the same argument
`lcqat.pruning.add_sparseprop_pruning_args` makes for the SparseProp flags.

Importing this module must not drag in the training scripts, so the setup
helpers take plain objects rather than reaching for module-level script state.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from nanochat.models.quant.activation import ACT_BODIES, ACT_BODY_PWL
from nanochat.models.quant.learnable_lut import RELAXATION_LOGITS, RELAXATIONS


def add_w6_args(parser: argparse.ArgumentParser) -> None:
    """Register the W6 DiffusionBlocks flags plus `--lcqat-channel-center`.

    `--lcqat-channel-center` is PRD 3.4 rather than W6, but it is orthogonal to
    these three -- it changes a weight table's cardinality, not the objective or
    the freeze schedule -- and it applies identically wherever DiffusionBlocks
    and the LC-QAT retrofit meet, so it belongs on the same shared surface.

    This helper owns *only* flag registration. It deliberately does not touch
    `lcqat_config_from_args`, which is what actually consumes
    `--lcqat-channel-center`.
    """
    parser.add_argument(
        "--kd-denoiser-alpha",
        type=float,
        default=0.0,
        help=(
            "weight of the float-twin denoiser anchor for --db-objective edm: "
            "L = (1-a)*w*||D_q-clean||^2 + a*w*||D_q-D_fp||^2. 0.0 disables it. "
            "Independent of --kd-alpha (the logit-KL form), which is CE-only."
        ),
    )
    parser.add_argument(
        "--efqat-latch-blocks",
        type=str,
        default="",
        help=(
            "EfQAT per-block permanent freeze (PRD 3.2): comma-separated diffusion "
            "block indices to latch permanently once their noise range is "
            "specialized. Empty (default) = disabled. A latched block's "
            "quantization parameters never receive gradients again, for the rest of "
            "training, however many later steps sample it."
        ),
    )
    parser.add_argument(
        "--efqat-latch-after",
        type=int,
        default=-1,
        help=(
            "step at which --efqat-latch-blocks fires, -1 = latch immediately at "
            "step 0 (the default when block indices are given)"
        ),
    )
    parser.add_argument(
        "--db-sigma-codebook",
        type=str,
        default="",
        choices=["", "conditioned", "modulated"],
        help=(
            "sigma-conditioned LC-QAT activation codebooks (PRD 3.2). "
            "'conditioned' = one codebook per noise-level anchor, exact but "
            "num_anchors x the codebook parameters and inference LUT. "
            "'modulated' = one codebook scaled by a learned gain on log(sigma), "
            "same artifact size as the unconditional codebook. "
            "Empty (default) = static codebook, unchanged from the existing recipe."
        ),
    )
    parser.add_argument(
        "--db-sigma-anchors",
        type=int,
        default=0,
        help=(
            "number of noise-level anchors for --db-sigma-codebook=conditioned. "
            "0 = match --db-blocks, so each diffusion block owns one codebook. "
            "Raising it buys resolution at a linear parameter cost."
        ),
    )
    parser.add_argument(
        "--lcqat-channel-center",
        action="store_true",
        help=(
            "per-output-channel value-centered weight quantization (PRD 3.4). "
            "Replaces each layer's single shared weight codebook with one table "
            "per output channel, so channels with very different scales are not "
            "forced onto a compromise grid. Costs out_features x K levels instead "
            "of K, and makes the layer un-exportable, so it is opt-in and cannot "
            "be combined with the fused inference path."
        ),
    )
    # D9: the activation-LUT relaxation and the activation body. Registered here
    # rather than in the scripts so all three entry points cannot drift apart,
    # the same reason `--lcqat-channel-center` lives in this helper.
    parser.add_argument(
        "--lcqat-lut-relaxation",
        type=str,
        default=RELAXATION_LOGITS,
        choices=list(RELAXATIONS),
        help=(
            "how the trained activation LUT selects its output level. "
            "'logits' (default) = a free (K_in x K_out) logit matrix with a "
            "straight-through round; unchanged from the shipped recipe. "
            "'proximity' = a learnable knot grid plus output levels, selected by "
            "inverse-square distance: K_in + K_in parameters instead of "
            "K_in x K_out, and no softmax-saturation cliff, at the cost of a "
            "table that must resolve to integer indices for the fused kernel. "
            "Init is bit-identical to the bake either way."
        ),
    )
    parser.add_argument(
        "--lcqat-act-body",
        type=str,
        default=ACT_BODY_PWL,
        choices=list(ACT_BODIES),
        help=(
            "the learnable body approximating the elementwise activation between "
            "two quantized layers. 'pwl' (default) = the shipped frozen "
            "K_in -> K_out index table, exact at the knots and linear between. "
            "'smoothpwl' = a radial-basis map with free knots, slopes and "
            "intercepts, which fits relu^2 far better at matched parameter count "
            "but has to be re-baked into an integer table for inference."
        ),
    )


def install_sigma_codebooks(
    engine, mode: str, num_anchors: int, num_db_blocks: int
) -> int:
    """Replace every activation quantizer under `engine` with a sigma-conditioned one.

    Opt-in. A DiffusionBlocks engine's blocks train on disjoint noise ranges, so
    one activation codebook has to span all of them and most of its levels are
    spent on values that never occur. Two mechanisms with very different costs,
    so both are opt-in and the default recipe is untouched.

    Must run *before* SparseProp is applied and before the float twin is taken,
    for the same reason those two have that ordering: the codebooks are the
    student's, not the float model's.

    Args:
        engine: the `DiffusionBlockEngine` whose quantizers are replaced.
        mode: `"conditioned"` or `"modulated"`. Empty means "do nothing"; this
            function is only called when a mode was requested.
        num_anchors: anchor count for `"conditioned"`. 0 means "match the block
            count", so each diffusion block owns one codebook.
        num_db_blocks: the engine's block count, used to resolve that 0.

    Returns:
        number of layers conditioned.
    """
    from nanochat.models.quant.sigma_codebook import (
        SigmaConditionedCodebook,
        SigmaModulatedCodebook,
    )

    anchors = num_anchors or num_db_blocks
    if mode == "conditioned" and anchors < 2:
        raise SystemExit(
            f"--db-sigma-anchors must be >= 2 for a conditioned codebook, got "
            f"{anchors}. Use --db-sigma-codebook=modulated for a single "
            "gain-modulated codebook."
        )
    # Only *activation* quantizers are conditioned. A weight matrix feeds every
    # noise level, so a per-sigma weight codebook would need a separate matrix
    # per sigma -- that is a different (and much larger) change than this flag.
    n_conditioned = 0
    for module in engine.modules():
        quantizer = getattr(module, "act_quantizer", None)
        if quantizer is None or not hasattr(quantizer, "m_neg"):
            continue
        m_neg, m_pos = quantizer.m_neg, quantizer.m_pos
        if mode == "conditioned":
            module.act_quantizer = SigmaConditionedCodebook(
                num_anchors=anchors, m_neg=m_neg, m_pos=m_pos
            )
        else:
            module.act_quantizer = SigmaModulatedCodebook(m_neg=m_neg, m_pos=m_pos)
        n_conditioned += 1
    return n_conditioned


def require_edm_objective(feature: str, db_objective: str) -> None:
    """Raise unless `db_objective` is `edm`.

    Both the denoiser-distillation anchor and sigma-conditioned activation
    codebooks need the *denoising* forward's noise level. `GPT.forward` -- the
    whole-depth next-token path that `train_step` and `logprobs` use -- has no
    sigma to give, so a conditioned quantizer under it raises from
    `LCQATLinear.forward` on the first micro-step. That is a late, opaque
    failure for something known at flag-parse time, so it is checked at startup
    instead.
    """
    if db_objective != "edm":
        raise SystemExit(
            f"{feature} needs --db-objective edm, but --db-objective is "
            f"{db_objective!r}. The denoising forward is the only path that "
            "carries a per-batch noise level to the activation quantizer."
        )


def describe_sigma_codebooks(args) -> dict:
    """Checkpoint-metadata record of the codebook structure in use.

    Recorded so a resumed run reports the model it is actually training. The
    codebook tensors are already in the state_dict; this is the human-readable
    description of them.
    """
    return {
        "mode": args.db_sigma_codebook or "static",
        "anchors": args.db_sigma_anchors or 0,
    }


def parse_latch_targets(spec: str, num_db_blocks: int) -> list[int]:
    """Parse and range-check a `--efqat-latch-blocks` spec.

    Raises `SystemExit` on an unparseable or out-of-range index: a block index
    that silently latches nothing looks exactly like a block that has not
    specialized yet, which is the failure the latch exists to make visible.
    """
    targets = [int(tok) for tok in spec.split(",") if tok.strip()]
    bad = [b for b in targets if not 0 <= b < num_db_blocks]
    if bad:
        raise SystemExit(
            f"--efqat-latch-blocks {bad} out of range for {num_db_blocks} "
            f"diffusion blocks (valid 0..{num_db_blocks - 1})"
        )
    return targets


def make_latch_freezer(engine, latch_spec: str, num_db_blocks: int):
    """Build a `BlockLatchFreezer` for `latch_spec`, or None if latching is off.

    The middle-band `SelectiveFreezer` is global and one-shot; this one is per
    diffusion block and, crucially, *permanent*: once a block's noise range is
    specialized its quantization parameters must never move again, however many
    later steps resample it. That only holds if the veto runs inside
    `_requires_grad_for`, before `_activate_block` re-enables the block -- which
    `engine.set_freezer` arranges.

    Returns:
        `(freezer, targets)`, either of which may be None/empty.
    """
    from nanochat.models.quant.efqat import BlockLatchFreezer

    if not latch_spec.strip():
        return None, []
    targets = parse_latch_targets(latch_spec, num_db_blocks)
    return BlockLatchFreezer(engine, engine.block_layers()), targets


def strip_lcqat(root: nn.Module) -> int:
    """Replace every LC-QAT layer under `root` with a plain float `nn.Linear`.

    Only needed on the resume path, where `prepare_lcqat_before_load` retrofits
    the base model before the engine is constructed, so the twin's deepcopy can
    already carry quantized layers. On a fresh run this finds nothing, because
    the twin is taken before `apply_lcqat` / `apply_sparseprop`.

    The replacement materializes the raw float weights, which is exactly what a
    pre-retrofit model holds -- the LC-QAT codebooks are the student's, not the
    float model's. Returning a *new* `nn.Linear` rather than class-swapping in
    place is what drops the quantizer submodules: a live-but-unused codebook
    would otherwise stay in `parameters()`, invisible to `verify_partition`,
    which only walks the student.

    KNOWN GAP: this matches `LCQATLinear` only. `SparsePropLinearLCQAT`
    subclasses `SparsePropLinear`, not `LCQATLinear`, so a sparse LC-QAT
    layer is NOT stripped and survives into the twin. Widening the isinstance
    would change what this function returns, so it is recorded here rather
    than changed inside a behaviour-preserving pass -- the resume path is
    where the twin is built after the retrofit, and that is where it bites.
    """
    from nanochat.models.quant.linear import LCQATLinear

    replaced = 0
    for parent in list(root.modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, LCQATLinear):
                continue
            plain = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                plain.weight.copy_(child.weight)
                if plain.bias is not None and child.bias is not None:
                    plain.bias.copy_(child.bias)
            setattr(parent, child_name, plain)
            replaced += 1
    return replaced


def build_float_twin(engine):
    """Deep-copy `engine` into a frozen, float-only, un-quantized twin.

    The twin is what `--kd-denoiser-alpha` distills *from*. It must be the same
    architecture on the same block partition (a denoiser output is only
    comparable across twins that own the same layers and the same per-block
    heads), but it must be genuinely float: LC-QAT codebooks and SparseProp
    sparsity are exactly the two things the anchor measures, so a twin carrying
    either would make `--kd-denoiser-alpha` self-distillation and its loss
    identically zero.

    Call this immediately after constructing `engine`, before `apply_lcqat` /
    `apply_sparseprop`, so on a fresh run the copy is already float. It runs
    after the resume load, so the twin inherits the checkpoint's trained weights
    rather than the fresh init -- anchoring against random weights would be worse
    than no anchor at all.

    Memory: an independent model, so this doubles parameter memory. The twin's
    forward runs only the active block (L/B layers) and never a backward.

    Returns:
        `(twin, n_lcqat_layers_stripped)`.
    """
    import copy

    twin = copy.deepcopy(engine)
    stripped = (
        strip_lcqat(twin.model)
        + sum(strip_lcqat(a) for a in twin.adapters)
        + strip_lcqat(twin.denoise_heads)
    )
    twin.freezer = None
    twin.distiller = None
    return twin, stripped


def build_denoiser_teacher(engine, kd_denoiser_alpha: float, db_objective: str):
    """Build the float twin for `--kd-denoiser-alpha`, or None when disabled.

    Raises `SystemExit` if the flag is set under a non-EDM objective. The anchor
    is defined against the EDM denoising objective's own output space; there is
    no logits term to regress, so under any other objective it cannot be applied
    at all. Silently skipping would look like the anchor "not helping" rather
    than "never applied".
    """
    if kd_denoiser_alpha <= 0.0:
        return None
    if db_objective != "edm":
        raise SystemExit(
            "--kd-denoiser-alpha is the EDM-native KD anchor and needs the EDM "
            f"objective, but --db-objective is {db_objective!r}. Use "
            "--db-objective edm with --kd-denoiser-alpha, or --kd-alpha with "
            "the CE objective."
        )
    return build_float_twin(engine)

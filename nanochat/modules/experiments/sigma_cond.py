"""The `sigma_cond` experiment: the structural cost of sigma conditioning.

Counts codebook parameters and artifact bytes for the shared codebook against
the conditioned one. Structural only -- the distributional premise is recorded
in the row as UNVERIFIED.
"""

from __future__ import annotations

import argparse

from nanochat.models.quant.ablation_metrics import AblationRow, select_quantizer
from nanochat.models.quant.retrofit import PRESETS, retrofit_model
from nanochat.models.quant.sigma_codebook import SigmaConditionedCodebook
from nanochat.modules.experiments.common import (
    VARIANT_PRESET,
    codebook_of,
    count_codebook_cost,
)
from tests.conftest import build_active_tiny_gpt


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

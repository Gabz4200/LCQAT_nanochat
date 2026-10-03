"""The `objective` experiment: `ce` vs `edm` block-output reconstruction.

Runs both objectives for a fixed number of SGD steps and reports the paired
mean block-output NMSE, with the per-arm losses as a diagnostic only.
"""

from __future__ import annotations

import argparse

import torch

from nanochat.models.quant.ablation_metrics import AblationRow
from nanochat.modules.experiments.common import (
    OBJECTIVE_CE,
    OBJECTIVE_EDM,
    block_probe_tensors,
    build_probe_engine,
    engine_named_parameters,
)
from nanochat.training.diffusion_blocks import edm_preconditioning


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
    engine = build_probe_engine(args)
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

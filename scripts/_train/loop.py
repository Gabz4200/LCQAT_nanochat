"""
The `scripts.base_train` training loop, plus the LR schedule helper and the
gradient-accumulation bookkeeping.

Moved out of `scripts/base_train.py` verbatim. `get_lr_multiplier` and
`compute_grad_accum_steps` are lifted out unchanged; `train_loop` is the same
`while True:` body calling them in the same order, with everything it reads
arriving on `LoopContext` instead of out of the entry point's module globals.
Evaluation blocks, checkpoint saving and the end-of-run summary stay in `eval`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.callbacks.training import (
    log_step_metrics,
    manage_gc,
    print_run_summary,
)
from nanochat.models.dard.loss import dard_state, load_dard_state
from nanochat.models.quant.w6 import describe_sigma_codebooks
from nanochat.modules.checkpoint_manager import save_checkpoint
from nanochat.utils.common import is_ddp_initialized, print0
from scripts._train.eval import (
    EvalContext,
    run_core_metric,
    run_samples,
    run_val_bpb,
)


@dataclass
class LoopContext:
    """The per-run objects the loop reads out of the entry point."""

    args: object
    model: object
    orig_model: object
    engine: object
    optimizer: object
    scaler: object
    trainable_root: object
    tokenizer: object
    train_loader: object
    x: object
    y: object
    dataloader_state_dict: dict
    wandb_run: object
    device: object
    ddp_rank: int
    ddp_world_size: int
    synchronize: object
    get_max_memory: object
    gpu_peak_flops: float
    token_bytes: object
    build_val_loader: object
    sparse_schedule: object
    kd_loss_fn: object
    kd_denoiser: object
    efqat_freezer: object
    block_latch_freezer: object
    efqat_latch_targets: list
    #: The DARD auxiliary loss and its EMA balancer; None under `--no-dard`.
    dard_loss_fn: object | None
    dard_balancer: object | None
    use_diffusion_blocks: bool
    num_db_blocks: int
    checkpoint_dir: str
    model_config_kwargs: dict
    lcqat_active: object | None
    user_config: dict
    resuming: bool
    meta_data: dict | None
    total_batch_size: int
    num_iterations: int
    num_flops_per_token: float
    num_scaling_params: int
    master_process: bool
    eval_ctx: EvalContext


def get_lr_multiplier(it, args, num_iterations):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        frac = it / warmup_iters
        return frac * 1.0 + (1 - frac) * args.init_lr_frac
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


def compute_grad_accum_steps(ctx):
    """Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step."""
    args = ctx.args
    tokens_per_fwdbwd = (
        args.device_batch_size * args.max_seq_len
    )  # tokens per iteration for a single rank
    world_tokens_per_fwdbwd = (
        tokens_per_fwdbwd * ctx.ddp_world_size
    )  # total tokens per iteration for all ranks
    assert ctx.total_batch_size % world_tokens_per_fwdbwd == 0, (
        f"total_batch_size ({ctx.total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
    )
    grad_accum_steps = ctx.total_batch_size // world_tokens_per_fwdbwd
    print0(
        f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}"
    )
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(
        f"Total batch size {ctx.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
    )
    if args.db_block_sampling == "micro" and grad_accum_steps > 1:
        print0(
            "WARNING: --db-block-sampling micro with gradient accumulation is LOSSY, "
            "not a different sampling strategy. `_apply_requires_grad` clears "
            "p.grad for parameters the active block does not own, so each micro-step "
            "erases the previous one's gradients and only the last block sampled "
            "contributes to the optimizer step. Use the default 'step' unless you "
            "are reproducing the ablation."
        )
    return grad_accum_steps


def save_step_checkpoint(
    ctx, step, val_bpb, min_val_bpb, smooth_train_loss, total_training_time
):
    """Save the checkpoint: at the end of the run, or every save_every steps,
    except at the first step or the resume step."""
    args = ctx.args
    save_checkpoint(
        ctx.checkpoint_dir,
        step,
        # engine.state_dict(), not orig_model.state_dict(): the engine owns
        # db_adapters.* / db_denoise_heads.* on top of the bare GPT, and meta
        # below declares meta["db"]. Saving the bare model writes a
        # checkpoint that claims a diffusion engine but carries none of its
        # parameters, so every resume silently reloads zero adapters/heads.
        # engine.state_dict() when DiffusionBlocks is on, because the engine
        # owns db_adapters.* / db_denoise_heads.* on top of the bare GPT.
        # In LM mode there is no engine and the bare model IS the complete
        # trainable tree.
        (ctx.engine if ctx.use_diffusion_blocks else ctx.model).state_dict(),
        ctx.optimizer.state_dict(),  # optimizer state
        {  # metadata saved as json
            "step": step,
            "val_bpb": val_bpb,  # loss at last step
            "model_config": ctx.model_config_kwargs,
            "user_config": ctx.user_config,  # inputs to the training script
            "lcqat": ctx.lcqat_active.as_dict()
            if ctx.lcqat_active is not None
            else None,
            # KD anchoring state. Two independent, incompatible-in-practice
            # anchors: `alpha` is the logit KL (CE objective), `denoiser_alpha`
            # is the float-twin denoiser anchor (EDM objective). Both default
            # to 0.0 = off, and both are recorded so a resume can see that a
            # run's objective differs from the default rather than silently
            # changing it.
            "kd": {
                "alpha": args.kd_alpha,
                "tau": args.kd_temperature,
                "denoiser_alpha": args.kd_denoiser_alpha,
            },
            # None, not a zeroed dict: `build_model` gates the whole
            # DiffusionBlocks reconstruction on meta["db"] being present, so
            # an LM-mode checkpoint loads as a bare GPT with a strict
            # load and no engine adapters to be silently zero.
            "db": (
                ctx.engine.partitioner.to_meta() if ctx.use_diffusion_blocks else None
            ),
            # EfQAT per-block latch: which blocks have been permanently
            # retired. Metadata only -- the tensors themselves are already
            # in engine.state_dict() -- but the freeze decision is not
            # recoverable from tensor values, so it has to be written out
            # or a resume silently restarts training a converged block.
            "efqat_latch": (
                ctx.engine.freezer_metadata() if ctx.use_diffusion_blocks else None
            ),
            "sigma_codebook": describe_sigma_codebooks(args),
            "sparseprop": ctx.sparse_schedule.to_meta(
                args.sparseprop, args.sparseprop_sparsity
            ),
            # DARD provenance. The tensors (rule CSR matrices + balancer EMA)
            # go to a sidecar `.pt`, because this dict is `json.dump`ed and
            # cannot hold them. What lands in the JSON is only what a resume
            # needs to decide whether the sidecar applies.
            "dard": (
                {
                    "has_sidecar": True,
                    "num_rules": int(ctx.dard_loss_fn.premise.shape[0]),
                    "vocab_size": int(ctx.dard_loss_fn.vocab_size),
                    "gamma": float(ctx.dard_loss_fn.gamma),
                    "lambda_global": float(ctx.dard_balancer.lambda_global),
                    "ema_decay": float(ctx.dard_balancer.ema_decay),
                    "steps": int(ctx.dard_balancer.steps),
                }
                if ctx.dard_loss_fn is not None and ctx.dard_balancer is not None
                else None
            ),
            "device_batch_size": args.device_batch_size,
            "max_seq_len": args.max_seq_len,
            "total_batch_size": ctx.total_batch_size,
            "dataloader_state_dict": ctx.dataloader_state_dict,
            "loop_state": {  # all loop state (other than step) so that we can resume training
                "min_val_bpb": min_val_bpb,
                "smooth_train_loss": smooth_train_loss,
                "total_training_time": total_training_time,
            },
        },
        rank=ctx.ddp_rank,
    )
    # DARD's tensors go in a sidecar next to the checkpoint. They cannot ride
    # in the meta dict (that is `json.dump`ed) and they are not part of
    # engine.state_dict() (the DARD modules are optimizer-side attachments).
    # Without this file a resume re-mines from whichever batch it sees next and
    # silently enforces different rules than the loss curve was measured under.
    if ctx.dard_loss_fn is not None and ctx.dard_balancer is not None:
        save_dard_sidecar(
            ctx.checkpoint_dir, step, ctx.dard_loss_fn, ctx.dard_balancer, ctx.ddp_rank
        )


def save_dard_sidecar(checkpoint_dir, step, loss_fn, balancer, rank=0) -> None:
    """Write the DARD rule set + balancer EMA beside the step's checkpoint.

    A separate file rather than a meta field because `save_checkpoint` writes
    meta as JSON and these are sparse tensors. `weights_only=True` on load is
    deliberate: only tensors and primitives, never pickled code.
    """
    if rank != 0:
        return
    path = os.path.join(checkpoint_dir, f"dard_{step:06d}.pt")
    torch.save(dard_state(loss_fn, balancer), path)
    print0(f"Saved DARD rule state to: {path}")


def load_dard_sidecar(checkpoint_dir, step, device):
    """Load a DARD sidecar written by `save_dard_sidecar`, or None if absent.

    None is a legitimate answer -- a checkpoint predating DARD, or one saved
    with `--no-dard`, has no sidecar -- and the caller then mines fresh rather
    than resuming a rule set that does not exist.
    """
    path = os.path.join(checkpoint_dir, f"dard_{step:06d}.pt")
    if not os.path.exists(path):
        return None
    return load_dard_state(torch.load(path, map_location=device, weights_only=True))


def train_loop(ctx):
    """Go! The training loop itself."""
    args = ctx.args
    num_iterations = ctx.num_iterations

    grad_accum_steps = compute_grad_accum_steps(ctx)

    # Assigned inside the grad-accumulation loop, which runs every
    # iteration; initialized so the read after it is not unbound.
    train_loss = torch.tensor(float("nan"))

    # Loop state (variables updated by the training loop)
    if not ctx.resuming:
        step = 0
        val_bpb = float("nan")  # real value once eval_every > 0
        min_val_bpb = float("inf")
        smooth_train_loss = 0  # EMA of training loss
        total_training_time = 0  # total wall-clock time of training
    else:
        step = ctx.meta_data["step"]
        loop_state = ctx.meta_data["loop_state"]
        val_bpb = ctx.meta_data["val_bpb"]
        min_val_bpb = loop_state["min_val_bpb"]
        smooth_train_loss = loop_state["smooth_train_loss"]
        total_training_time = loop_state["total_training_time"]

    x, y = ctx.x, ctx.y

    # Packed batches arrive as document boundaries; `block_diagonal_mask` turns them
    # into a causal + same-document mask so a document cannot attend across a
    # neighbour's tokens. `denoise_step` calls the transformer blocks directly, so
    # the mask has to be handed to it explicitly (the CE path threads it through
    # GPT.forward). None means "no packing": the model's attention is already
    # causal, so a full causal mask would be a no-op.
    train_attn_mask: torch.Tensor | None = None

    while True:
        last_step = (
            step == num_iterations
        )  # loop runs num_iterations+1 times so that we can eval/save at the end
        flops_so_far = ctx.num_flops_per_token * ctx.total_batch_size * step

        # once in a while: evaluate the val bpb (all ranks participate)
        if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            val_bpb, min_val_bpb = run_val_bpb(
                ctx.eval_ctx,
                step,
                flops_so_far,
                total_training_time,
                min_val_bpb,
            )

        # once in a while: estimate the CORE metric (all ranks participate)
        # use the original uncompiled model because the inputs keep changing shape
        # disable FP8 for evaluation to use BF16 for more consistent/accurate results
        if args.core_metric_every > 0 and (
            last_step or (step > 0 and step % args.core_metric_every == 0)
        ):
            run_core_metric(ctx.eval_ctx, step, flops_so_far)

        # once in a while: sample from the model (only on master process)
        # use the original uncompiled model because the inputs keep changing shape
        if (
            args.sample_every > 0
            and ctx.master_process
            and (last_step or (step > 0 and step % args.sample_every == 0))
        ):
            run_samples(ctx.eval_ctx)

        # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
        if last_step or (
            step > 0
            and step != args.resume_from_step
            and args.save_every > 0
            and step % args.save_every == 0
        ):
            save_step_checkpoint(
                ctx,
                step,
                val_bpb,
                min_val_bpb,
                smooth_train_loss,
                total_training_time,
            )

        # termination conditions (TODO: possibly also add loss explosions etc.)
        if last_step:
            break

        # ---------------------------------------------------------------------
        # single training step
        # evaluate the gradient
        ctx.synchronize()
        t0 = time.time()
        # EfQAT: drive the selective freezer from the global step (PRD 3.2), then
        # hand it to the engine as the single `requires_grad` arbiter. Without the
        # handoff the engine re-enables every transformer parameter on the next
        # block activation, silently undoing the freeze.
        if ctx.efqat_freezer is not None:
            if ctx.efqat_freezer.update(step):
                n_frozen = sum(
                    1 for p in ctx.trainable_root.parameters() if not p.requires_grad
                )
                print0(f"EfQAT froze {n_frozen} parameters at step {step}")
        # EfQAT per-block latch (PRD 3.2). Fires before the forward, so the retired
        # block is already frozen in the very step the latch lands; latching after
        # the forward would let one more update through. `>=` rather than `==` so a
        # resumed run (which starts above the threshold) still latches, and
        # `latch_blocks` is idempotent so the re-fire is free.
        if ctx.block_latch_freezer is not None and step >= max(
            args.efqat_latch_after, 0
        ):
            n_new = ctx.block_latch_freezer.latch_blocks(ctx.efqat_latch_targets)
            if n_new:
                print0(
                    f"EfQAT latched blocks {ctx.efqat_latch_targets} permanently at step "
                    f"{step} ({n_new} parameters frozen; total latched: "
                    f"{ctx.block_latch_freezer.latched_blocks()})"
                )
        # Gradual magnitude pruning. Runs BEFORE the forward so the mask and the
        # forward agree in the same step, and so the weights it prunes are the ones
        # the optimizer is about to write into. Mask changes are monotone (a pruned
        # weight is exactly 0.0, so |W| == 0 and it can never be re-selected), which
        # is what makes a resumed ramp safe.
        if args.sparseprop:
            pruned = ctx.sparse_schedule.apply(ctx.trainable_root, step)
            if pruned is not None:
                # `ctx.trainable_root`, matching the `apply` above: in LM mode
                # (`--db-blocks=0`) `ctx.engine` is None, and collecting over it
                # raises instead of reporting a count.
                n_sparse_above = len(
                    ctx.sparse_schedule.layers_above_threshold(ctx.trainable_root)
                )
                print0(
                    f"Step {step:05d} | SparseProp pruned to {pruned:.4f} sparsity; "
                    f"{n_sparse_above} layers past the "
                    f"{ctx.sparse_schedule.dense_threshold:.0%} sparse-kernel threshold"
                )
        # The block is drawn ONCE per optimizer step, not once per micro-step.
        # Accumulating gradients from several blocks into one update would preserve
        # the memory saving but destroy the noise-range specialization that
        # DiffusionBlocks depends on.
        block_idx = (
            ctx.engine.sample_block()
            if ctx.use_diffusion_blocks
            and args.db_objective == "edm"
            and args.db_block_sampling == "step"
            else None
        )
        # Hoisted out of the micro-step loop: denoise_step reuses the same (x, y)
        # across grad_accum_steps, so the embedding lookup + L2 normalization would
        # otherwise be redone once per micro-step.
        clean = None
        if ctx.use_diffusion_blocks and args.db_objective == "edm":
            with torch.no_grad():
                clean = F.normalize(ctx.engine.model.transformer.wte(x).float(), dim=-1)

        # Reported alongside the total so an enabled anchor is observable in the
        # log: a KD term that stays at 0.0 for a whole run means the twin is not
        # actually being compared against, not that the gap vanished.
        step_kd_logged = 0.0
        # Reported alongside the total for the same reason as the KD term: the
        # balanced loss is normalized to ~1 by construction, so the *raw* rule
        # penalty is the only thing in the log that shows whether DARD is still
        # doing anything. A run that silently mined zero rules logs 0.0 here
        # forever and its balanced loss looks perfectly healthy.
        step_dard_logged = 0.0
        # Auxiliary per-block next-token loss (EDM): the only observable of
        # whether `lm_head` is actually training under the EDM objective, since
        # the EDM term itself never touches the head. 0.0 when
        # --db-aux-lm-weight is 0.
        step_aux_lm_logged = 0.0

        for micro_step in range(grad_accum_steps):
            # DARD needs a vocabulary distribution. The CE and KD branches
            # already hold `logits`; the two branches below do not, so
            # `d_logits` is filled by the branch that can produce them and the
            # penalty is evaluated once, after the branch, rather than inside
            # each of the four.
            d_logits = None
            d_block_logits = None
            if not ctx.use_diffusion_blocks:
                # Plain autoregressive LM training. `model` here is the LC-QAT
                # (and possibly SparseProp) retrofitted GPT, so this is the
                # conventional next-token objective over the same quantized model
                # the DiffusionBlocks arms train, differing only in the objective.
                if ctx.dard_loss_fn is not None:
                    d_logits = ctx.model(x, targets=None, attn_mask=train_attn_mask)
                    loss = F.cross_entropy(
                        d_logits.reshape(-1, d_logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=-1,
                    )
                else:
                    loss = ctx.model(x, targets=y, attn_mask=train_attn_mask)
            elif args.db_objective == "edm":
                # `block_idx=None` under --db-block-sampling micro, which redraws
                # per micro-step (the ablation arm).
                loss, sigma = ctx.engine.denoise_step(
                    x,
                    block_idx=block_idx,
                    overlap=args.db_overlap,
                    attn_mask=train_attn_mask,
                    clean=clean,
                    backend=args.db_denoise_backend,
                    aux_lm_weight=args.db_aux_lm_weight,
                    targets=y,
                )
                if ctx.engine.distiller is not None:
                    step_kd_logged = step_kd_logged + float(ctx.engine.last_kd_loss)
                if args.db_aux_lm_weight > 0:
                    step_aux_lm_logged = step_aux_lm_logged + float(
                        ctx.engine.last_aux_lm_loss
                    )
                if ctx.dard_loss_fn is not None:
                    # The denoiser's own logits, from the same lm_head
                    # projection the Euler sampler uses. Read out of the engine
                    # rather than recomputed, so the penalty sees exactly the
                    # distribution the denoiser step was trained on.
                    d_logits = ctx.engine.last_dard_logits
                    # Sum the penalty over EVERY block's distribution, not just
                    # the final one: each is that block's own hidden state
                    # through the shared head, so the rules shape all blocks.
                    d_block_logits = ctx.engine.last_block_logits
            elif ctx.kd_loss_fn is not None:
                # KD anchoring needs logits, so the CE objective asks for them
                # explicitly (targets=None) and computes the CE term here.
                # `model` is the bare GPT in both modes -- the engine wraps it, and in
                # LM mode there is no engine at all.
                student_logits = ctx.model(x, targets=None, attn_mask=train_attn_mask)
                # Mirror GPT.forward's loss exactly: the dataloader already shifted
                # y, so no extra shift here.
                loss_ce = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-1,
                )
                loss_kd = ctx.kd_loss_fn(student_logits, x)
                loss = (
                    1.0 - ctx.kd_loss_fn.alpha
                ) * loss_ce + ctx.kd_loss_fn.alpha * loss_kd
                if ctx.dard_loss_fn is not None:
                    d_logits = student_logits
            else:
                if ctx.dard_loss_fn is not None:
                    d_logits = ctx.model(x, targets=None, attn_mask=train_attn_mask)
                    loss = F.cross_entropy(
                        d_logits.reshape(-1, d_logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=-1,
                    )
                else:
                    loss = ctx.engine.train_step(
                        x, y, block_idx=block_idx, attn_mask=train_attn_mask
                    )
            if d_logits is not None and ctx.dard_loss_fn is not None:
                # `d_logits` carries the graph; so does `loss`, so the two terms
                # are balanced *before* the single backward below rather than
                # summed. Both stay differentiable through the same forward.
                # When the engine published per-block distributions, the penalty
                # is summed over them so every block is reached; otherwise it is
                # the single distribution.
                if d_block_logits:
                    # `sum()` over a 1-element generator returns the element
                    # itself rather than a tensor, so build the stack
                    # explicitly and mean over it.
                    aux = torch.stack(
                        [ctx.dard_loss_fn(v) for v in d_block_logits.values()]
                    ).mean()
                else:
                    aux = ctx.dard_loss_fn(d_logits)
                step_dard_logged = step_dard_logged + float(aux.detach())
                if ctx.dard_balancer is not None:
                    loss, _ = ctx.dard_balancer(loss, aux)
                else:
                    loss = loss + aux
            train_loss = loss.detach()  # for logging
            loss = (
                loss / grad_accum_steps
            )  # each .backward() is a grad sum => normalize loss here
            if ctx.scaler is not None:
                ctx.scaler.scale(loss).backward()
            else:
                loss.backward()
            x, y, ctx.dataloader_state_dict = next(
                ctx.train_loader
            )  # prefetch the next batch while the GPU is busy with forward/backward
        # step the optimizer
        lrm = get_lr_multiplier(step, args, num_iterations)
        for group in ctx.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        if ctx.scaler is not None:
            ctx.scaler.unscale_(ctx.optimizer)
            # In distributed training, all ranks must agree on whether to skip the step.
            # Each rank may independently encounter inf/nan gradients, so we all-reduce
            # the found_inf flag (MAX = if any rank found inf, all ranks skip).
            if is_ddp_initialized():
                for v in ctx.scaler._found_inf_per_device(ctx.optimizer).values():
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            ctx.scaler.step(ctx.optimizer)
            ctx.scaler.update()
        else:
            ctx.optimizer.step()
        # MinExpLSTM joint online co-training: every K optimizer
        # steps, re-evaluate the loss on the *updated* parameters
        # (rebuilt differentiably through the cell), backprop to the
        # cell, and step its AdamW meta-optimizer. The closure
        # rebuilds the loss from the updated params via
        # functional_call, so the graph runs cell -> (gamma, delta)
        # -> theta_new -> loss. Only the MinExpLSTM optimizer
        # implements meta_step; AdamW baselines skip it.
        meta_every = getattr(args, "minexp_meta_every", 50)
        meta_lr = getattr(args, "minexp_meta_lr", 1e-3)
        if (
            meta_every > 0
            and hasattr(ctx.optimizer, "meta_step")
            and step % meta_every == 0
            and step > 0
        ):
            trainable = ctx.trainable_root

            import gc

            gc.collect()

            def _meta_closure(params):
                # Rebuild the same objective the primary step
                # optimized, on the updated parameters, and
                # return a differentiable loss. functional_call
                # reparametrizes an nn.Module tree. The DiffusionBlocks
                # engine is NOT an nn.Module (it owns three: model,
                # adapters, denoise_heads), so the engine is reparametrized
                # by swapping the *model* subtree's parameters in place
                # for this closure -- the model is where the learned
                # optimizer's residual applies and where the meta-gradient
                # belongs. Adapters and denoise heads (small conditioning
                # modules) keep their current values.
                if ctx.use_diffusion_blocks:
                    engine = trainable
                    model_params = {
                        n: p for n, p in params.items() if not n.startswith("db_")
                    }
                    # functional_call(engine.model, model_params) returns
                    # a stateless view; the engine's train_step reads the
                    # real model, so we swap the model's parameters in
                    # place, run the block objective, and restore. The
                    # updated tensors carry the cell's graph, so the loss
                    # backprops to the cell.
                    saved = {n: p.data for n, p in engine.model.named_parameters()}
                    try:
                        for n, p in engine.model.named_parameters():
                            if n in model_params:
                                p.data = model_params[n]
                        return engine.train_step(x, y)
                    finally:
                        for n, p in engine.model.named_parameters():
                            if n in saved:
                                p.data = saved[n]
                return torch.func.functional_call(trainable, params, (x, y))

            try:
                meta_loss = ctx.optimizer.meta_step(
                    trainable, _meta_closure, meta_lr=meta_lr
                )
                if step % 100 == 0:
                    print0(f"  [minexp] meta_loss: {float(meta_loss.detach()):.6f}")
            except (RuntimeError, TypeError) as exc:
                # A meta step that cannot close its graph (e.g. an
                # engine whose forward does not accept functional_call
                # args) degrades to skipping the co-training step
                # rather than crashing the run.
                print0(f"  [minexp] meta_step skipped: {exc}")
        # `trainable_root`, not `model`: in DiffusionBlocks mode the optimizer
        # owns the engine, and its `db_adapters.*` / `db_denoise_heads.*` are
        # separate subtrees that `ctx.model` cannot reach. Zeroing the bare GPT
        # left those gradients alive, and `_apply_requires_grad` only clears
        # `p.grad` for parameters the *active* block does not own -- so when a
        # block is drawn twice inside one optimizer step, the second draw's
        # adapter and denoise-head gradients accumulate onto the first.
        ctx.trainable_root.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()  # .item() is a CPU-GPU sync point
        ctx.synchronize()
        t1 = time.time()
        dt = t1 - t0
        # ---------------------------------------------------------------------

        # logging (CPU action only)
        ema_beta = 0.9  # EMA decay factor for smoothing, for nicer logging
        smooth_train_loss = (
            ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        )  # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (
            1 - ema_beta ** (step + 1)
        )  # debias the EMA
        pct_done = 100 * step / num_iterations
        tok_per_sec = int(ctx.total_batch_size / dt)
        flops_per_sec = ctx.num_flops_per_token * ctx.total_batch_size / dt
        mfu = 100 * flops_per_sec / (ctx.gpu_peak_flops * ctx.ddp_world_size)
        if step > 10:
            total_training_time += dt  # only count the time after the first 10 steps
        # Calculate ETA based on average time per step (excluding first 10 steps)
        steps_done = step - 10
        if steps_done > 0:
            avg_time_per_step = total_training_time / steps_done
            remaining_steps = num_iterations - step
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = f" | eta: {eta_seconds / 60:.1f}m"
        else:
            eta_str = ""
        epoch = f"{ctx.dataloader_state_dict['epoch']} pq: {ctx.dataloader_state_dict['pq_idx']} rg: {ctx.dataloader_state_dict['rg_idx']}"
        kd_mean = step_kd_logged / max(grad_accum_steps, 1)
        kd_str = f" | kd: {kd_mean:.6f}" if ctx.kd_denoiser else ""
        dard_mean = step_dard_logged / max(grad_accum_steps, 1)
        dard_str = f" | dard: {dard_mean:.6f}" if ctx.dard_loss_fn is not None else ""
        aux_lm_mean = step_aux_lm_logged / max(grad_accum_steps, 1)
        aux_lm_str = (
            f" | aux_lm: {aux_lm_mean:.6f}" if args.db_aux_lm_weight > 0 else ""
        )
        print0(
            f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f}{kd_str}{dard_str}{aux_lm_str} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time / 60:.2f}m{eta_str}"
        )
        if step % 100 == 0:
            log_data = {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/epoch": epoch,
            }
            # The float-vs-quantized denoiser gap this step's anchor actually saw.
            # Logged separately from `train/loss` because the returned loss mixes it
            # with the data term, so a silently-dead anchor is otherwise invisible.
            if ctx.kd_denoiser is not None:
                log_data["train/kd_denoiser"] = kd_mean
            if ctx.dard_loss_fn is not None:
                # The raw (pre-normalization) penalty. Logged because the
                # balanced total is normalized to ~1 by construction, so a
                # rule set that quietly stopped enforcing anything would be
                # invisible in `train/loss` alone.
                log_data["train/dard_rule_penalty"] = dard_mean
            # EfQAT latch progress: how many diffusion blocks have been permanently
            # retired. Plateaus at `num_db_blocks` once all of them have finished
            # specializing to their noise range.
            if ctx.block_latch_freezer is not None:
                log_data["train/efqat_latched_blocks"] = float(
                    len(ctx.block_latch_freezer.latched_blocks())
                )
            log_step_metrics(ctx.wandb_run, log_data, 100, step)

        # state update
        first_step_of_run = (step == 0) or (
            ctx.resuming and step == args.resume_from_step
        )
        step += 1

        manage_gc(step, first_step_of_run)

    print_run_summary(ctx.get_max_memory, total_training_time, val_bpb, min_val_bpb)

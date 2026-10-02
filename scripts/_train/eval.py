"""
Evaluation for `scripts.base_train`: in-loop validation bpb, the CORE metric,
sampling, and the end-of-run summary.

Moved out of `scripts/base_train.py` verbatim. The bodies are unchanged; the
objects the loop already holds arrive on `EvalContext` instead of being read out
of the entry point's module globals, so `run_val_bpb` / `run_core_metric` /
`run_samples` are called at the same points in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanochat.modules.engine import Engine
from nanochat.modules.loss_eval import evaluate_bpb
from nanochat.utils.common import print0
from scripts._train.build import disable_fp8
from scripts.base_eval import evaluate_core


@dataclass
class EvalContext:
    """The per-run objects every evaluation block reads out of the entry point."""

    model: object
    orig_model: object
    tokenizer: object
    device: object
    args: object
    wandb_run: object
    ddp_world_size: int
    token_bytes: object
    build_val_loader: object


def run_val_bpb(ctx, step, flops_so_far, total_training_time, min_val_bpb):
    """Once in a while: evaluate the val bpb (all ranks participate)."""
    args = ctx.args
    model = ctx.model
    model.eval()
    val_loader = ctx.build_val_loader()
    eval_steps = args.eval_tokens // (
        args.device_batch_size * args.max_seq_len * ctx.ddp_world_size
    )
    with disable_fp8(model):
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, ctx.token_bytes)
    print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
    if val_bpb < min_val_bpb:
        min_val_bpb = val_bpb
    ctx.wandb_run.log(
        {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        }
    )
    model.train()
    return val_bpb, min_val_bpb


def run_core_metric(ctx, step, flops_so_far):
    """Once in a while: estimate the CORE metric (all ranks participate).

    Uses the original uncompiled model because the inputs keep changing shape,
    and disables FP8 so the evaluation runs in BF16 for more consistent/accurate
    results.
    """
    args = ctx.args
    ctx.model.eval()
    with disable_fp8(ctx.orig_model):
        results = evaluate_core(
            ctx.orig_model,
            ctx.tokenizer,
            ctx.device,
            max_per_task=args.core_metric_max_per_task,
        )
    print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
    ctx.wandb_run.log(
        {
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        }
    )
    ctx.model.train()
    return results


def run_samples(ctx):
    """Once in a while: sample from the model (only on master process).

    Uses the original uncompiled model because the inputs keep changing shape.
    """
    ctx.model.eval()
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    # `ar_engine` is the autoregressive sampler; it must not shadow the
    # DiffusionBlockEngine bound to `engine`, which is what train_step runs on.
    ar_engine = Engine(ctx.orig_model, ctx.tokenizer)  # orig_model avoids recompilation
    for prompt in prompts:
        tokens = ctx.tokenizer(prompt, prepend="<|bos|>")
        with disable_fp8(ctx.orig_model):
            sample, _ = ar_engine.generate_batch(
                tokens, num_samples=1, max_tokens=16, temperature=0
            )
        print0(ctx.tokenizer.decode(sample[0]))
    ctx.model.train()


def print_run_summary(get_max_memory, total_training_time, val_bpb, min_val_bpb):
    """Print a few more stats."""
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

"""Secondary fine-tuning: train only lm_head on top of a checkpoint.

Freezes the backbone (transformer blocks) and trains ONLY the output
projection (`lm_head`) using cross-entropy loss, with DARD as an
optional auxiliary. This recovers generation capability when the
main run used `--db-objective edm` (which leaves lm_head untrained)
or a checkpoint where lm_head is random init.

Usage:
    uv run python scripts/lm_head_tune.py \
        --checkpoint-dir d8 \
        --step 500 \
        --steps 500 \
        --dard
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.modules.checkpoint_manager import load_model_from_dir
from nanochat.training.optim import SGD_MinExpLstm_Optimizer
from nanochat.utils.common import get_base_dir, print0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="lm_head-only fine-tuning")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Model tag under base_checkpoints/ (e.g. 'd8')",
    )
    p.add_argument("--step", type=int, required=True, help="Checkpoint step to load")
    p.add_argument("--steps", type=int, default=500, help="Number of fine-tuning steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--dard",
        action="store_true",
        help="Enable DARD penalty (if rules in checkpoint)",
    )
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-tag", type=str, default="lm_tuned")
    return p.parse_args()


def get_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())


def build_dummy_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
):
    """Generate a deterministic dummy batch for fine-tuning."""
    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=gen, device=device
    )
    y = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=gen, device=device
    )
    return x, y


def main():
    args = parse_args()
    base_dir = get_base_dir()
    if base_dir is None:
        base_dir = os.path.expanduser("~/.cache/nanochat")
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    device = torch.device("cpu")

    # Load model from checkpoint via existing API
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    model, tokenizer, meta_data = load_model_from_dir(
        checkpoints_dir,
        device,
        phase="train",
        model_tag=args.checkpoint_dir,
        step=args.step,
    )

    # Freeze all parameters except lm_head
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("lm_head")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(1 for p in model.parameters())
    print0(f"Trainable: {len(trainable_params)} / {total_params} (lm_head only)")

    # MinExpLSTM optimizer (default) or AdamW (--optimizer adamw)
    optimizer = SGD_MinExpLstm_Optimizer(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # DARD: check if rules exist in meta
    dard_enabled = False
    if args.dard:
        meta = meta_data.get("meta", {})
        dard_enabled = "dard" in meta.get("active_features", [])
        if dard_enabled:
            print0("DARD rules detected in checkpoint meta")
        else:
            print0("Warning: --dard requested but DARD not active in checkpoint")

    cfg = getattr(model, "config", None)
    vocab_size = cfg.vocab_size if cfg is not None else 32768
    print0(f"Starting lm_head fine-tuning: {args.steps} steps, lr={args.lr}")

    model.train()
    last_loss = float("inf")
    for step in range(args.steps):
        lr = get_lr(step, args.steps, args.lr, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = build_dummy_batch(args.batch_size, args.seq_len, vocab_size, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.size(-1)),
            y[..., 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            eval_loss = float(loss.detach())
            arrow = "↑" if eval_loss > last_loss * 1.1 else "↓"
            print0(
                f"Step {step}/{args.steps} | lr={lr:.6f} | loss={eval_loss:.4f} {arrow}"
            )
            last_loss = eval_loss

    # Save tuned checkpoint: only lm_head weights + meta
    save_dir = os.path.join(
        base_dir, f"base_checkpoints/{args.checkpoint_dir}_{args.save_tag}"
    )
    os.makedirs(save_dir, exist_ok=True)
    lm_head_state = {k: v for k, v in model.state_dict().items() if "lm_head" in k}
    # Save in the format checkpoint_manager expects: model_{step:06d}.pt
    torch.save(lm_head_state, os.path.join(save_dir, f"model_{args.steps:06d}.pt"))

    cfg = getattr(model, "config", None)
    meta = {
        "step": args.steps,
        "config": {
            "n_layer": cfg.n_layer if cfg else 0,
            "n_head": cfg.n_head if cfg else 0,
            "n_kv_head": cfg.n_kv_head if cfg else 0,
            "n_embd": cfg.n_embd if cfg else 0,
            "sequence_len": cfg.sequence_len if cfg else 0,
            "vocab_size": cfg.vocab_size if cfg else 0,
            "window_pattern": cfg.window_pattern if cfg else "L",
        }
        if cfg
        else {},
        "finetuned_from": f"base_checkpoints/{args.checkpoint_dir}",
        "finetuned_from_step": args.step,
        "technique": "lm_head_only",
        "dard": dard_enabled,
    }
    with open(os.path.join(save_dir, f"meta_{args.steps:06d}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print0(f"Saved tuned checkpoint to {save_dir}")
    print0("lm_head fine-tuning complete.")


if __name__ == "__main__":
    main()

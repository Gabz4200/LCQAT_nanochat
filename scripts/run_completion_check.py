"""Run completion prompts on a trained checkpoint.

Loads a model from checkpoint, freezes nothing, and runs basic
text generation with prompts to verify output quality.

Usage:
    uv run python scripts/run_completion_check.py \
        --checkpoint-dir d8_lm_tuned \
        --step 500 \
        --max-tokens 64
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.modules.checkpoint_manager import load_model_from_dir
from nanochat.utils.common import get_base_dir, print0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Completion check on a checkpoint")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Model tag under base_checkpoints/ (e.g. 'd8_lm_tuned')",
    )
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt to use; if empty, uses default test prompts",
    )
    p.add_argument(
        "--prompts-file", type=str, default="", help="JSON file with a list of prompts"
    )
    return p.parse_args()


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "def fibonacci(n):",
    "In a world where",
]


def load_checkpoint(
    ckpt_tag: str, step: int, base_tag: str | None = None, base_step: int | None = None
):
    """Load model + tokenizer from a checkpoint directory.

    If the tag ends with ``_lm_tuned``, loads the base model first
    and overlays the lm_head-only weights.
    """
    base_dir = get_base_dir()
    if base_dir is None:
        base_dir = os.path.expanduser("~/.cache/nanochat")
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")

    lm_head_only = "_lm_tuned" in ckpt_tag

    if lm_head_only:
        base_tag_full = base_tag or ckpt_tag.replace("_lm_tuned", "")
        base_step_full = base_step or 500
        model, tokenizer, base_meta = load_model_from_dir(
            checkpoints_dir,
            torch.device("cpu"),
            phase="eval",
            model_tag=base_tag_full,
            step=base_step_full,
        )
        lm_head_path = os.path.join(checkpoints_dir, ckpt_tag, f"model_{step:06d}.pt")
        lm_head_state = torch.load(lm_head_path, map_location="cpu", weights_only=True)
        model.load_state_dict(lm_head_state, strict=False)
        print0(
            f"Loaded base from {base_tag_full} step {base_step_full}, overlaid lm_head from step {step}"
        )
        return model, tokenizer, base_meta

    model, tokenizer, meta = load_model_from_dir(
        checkpoints_dir,
        torch.device("cpu"),
        phase="eval",
        model_tag=ckpt_tag,
        step=step,
    )
    return model, tokenizer, meta


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate text from a prompt using the model."""
    model.eval()

    # Encode the prompt
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(prompt)
    elif hasattr(tokenizer, "tokenize"):
        tokens = tokenizer.tokenize(prompt)
    else:
        tokens = [tokenizer.bos_id]

    # Handle both token lists and token id lists
    if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], str):
        # Tokenizer returned tokens, convert to ids
        token_ids = [tokenizer.piece_to_id(t) for t in tokens]
    else:
        token_ids = tokens

    context = torch.tensor([token_ids], dtype=torch.long, device="cpu")
    original_len = context.shape[1]

    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            logits = model(context)
            if isinstance(logits, tuple):
                logits = logits[0]  # Some models return (logits, kv_cache)

            next_logits = logits[0, -1, :]

            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0 and top_k < next_logits.shape[0]:
                top_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_vals[-1]
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            # Sample from the filtered distribution
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to context
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

            # Stop on EOS token if available
            eos_id = getattr(tokenizer, "eos_id", None)
            if eos_id is not None and next_token.item() == eos_id:
                break

    # Decode
    generated_ids = context[0].tolist()
    decoded = ""
    if hasattr(tokenizer, "decode"):
        try:
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        except TypeError:
            decoded = tokenizer.decode(generated_ids)
    elif hasattr(tokenizer, "detokenize"):
        decoded = tokenizer.detokenize(generated_ids)
    else:
        # Fallback: decode token ids to strings
        try:
            decoded = "".join(tokenizer.id_to_piece(i) for i in generated_ids)
        except Exception:
            decoded = f"[generated {len(generated_ids) - original_len} tokens]"

    # Strip the prompt from the generated text (show only the completion)
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded.strip()


def evaluate_coherence(text: str) -> str:
    """Simple heuristic evaluation of generation coherence."""
    generated_part = text
    # Check for repetition
    words = generated_part.split()
    if len(words) > 10:
        unique_words = set(words)
        repetition_ratio = 1 - len(unique_words) / len(words)
        if repetition_ratio > 0.7:
            return f"UNCOHERENT (high repetition: {repetition_ratio:.1%})"

    # Check for length
    if len(words) < 5:
        return f"INCOMPLETE (only {len(words)} words generated)"

    # Check for garbage tokens
    token_indicators = sum(1 for w in words if ">" in w and "<" in w)
    if token_indicators > len(words) * 0.3:
        return f"UNCOHERENT (too many raw tokens: {token_indicators}/{len(words)})"

    return "COHERENT"


def main():
    args = parse_args()

    # Load prompts
    if args.prompts_file:
        import json

        with open(args.prompts_file, "r") as f:
            prompts = json.load(f)
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = DEFAULT_PROMPTS

    model, tokenizer, meta = load_checkpoint(args.checkpoint_dir, args.step)
    print0(f"Loaded checkpoint: {args.checkpoint_dir} step {args.step}")

    for prompt in prompts:
        print0(f"\nPrompt: {prompt}")
        try:
            generated = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            coherence = evaluate_coherence(generated)
            print0(f"Generated: {generated}")
            print0(f"Coherence: {coherence}")
        except Exception as e:
            print0(f"Error generating from prompt '{prompt}': {e}")


if __name__ == "__main__":
    main()

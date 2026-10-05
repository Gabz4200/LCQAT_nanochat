"""Generate from the night4 base checkpoint and report coherence signals.

Run after `runs/night4.sh` finishes:

    python -m scripts.dard_report --model-tag night4

Reports the things that actually distinguish "a language model" from "a
gradient that went down": whether the text is locally well-formed English, how
much it repeats, and its perplexity. Repetition rate is the one that matters
most at this token budget -- a model that has seen 3M tokens learns the local
statistics of English well before it learns to say anything, so low repetition
with rising loss is the expected and healthy outcome here.

Deliberately *not* a pass/fail gate. At ~3M tokens (0.1 tokens/param) no model
is coherent in the way a trained one is, and a threshold that says otherwise
would be measuring the wrong thing.
"""

from __future__ import annotations

import argparse
import os

import torch

from nanochat.data.dataset import list_parquet_files
from nanochat.data.tokenizer import get_tokenizer
from nanochat.models.backbone import GPT, GPTConfig
from nanochat.modules.checkpoint_manager import load_checkpoint
from nanochat.utils.common import get_base_dir

PROMPTS = [
    "The capital of France is",
    "The meaning of life is",
    "Once upon a time, there was a",
    "Water boils at a temperature of",
    "In the beginning, the universe",
]


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams in `text` that have appeared before.

    The single most useful coherence signal at a small token budget: a model
    that has learned nothing emits its unigram marginal, which repeats heavily;
    one that has learned local structure repeats far less.
    """
    words = text.split()
    if len(words) < n + 1:
        return 1.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def load_base_model(model_tag: str, step: int | None) -> tuple[GPT, int]:
    base = get_base_dir() or os.path.expanduser("~/.cache/nanochat")
    ckpt_dir = os.path.join(base, "base_checkpoints", model_tag)
    if step is None:
        steps = sorted(
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(ckpt_dir)
            if f.startswith("model_") and f.endswith(".pt")
        )
        if not steps:
            raise SystemExit(f"no checkpoints in {ckpt_dir}")
        step = steps[-1]
    sd, _, meta = load_checkpoint(ckpt_dir, step, device="cpu")
    cfg = GPTConfig(**meta["model_config"])
    model = GPT(cfg)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, step


def main() -> None:
    parser = argparse.ArgumentParser(description="Report on a trained base model")
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model, step = load_base_model(args.model_tag, args.step)
    tokenizer = get_tokenizer()
    print(
        f"\n=== {args.model_tag} step {step} "
        f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params) ===\n"
    )

    for prompt in PROMPTS:
        # `GPT.generate` streams: it takes a list of ids and yields one id at a
        # time, so the full sequence is the prompt plus everything yielded.
        ids = tokenizer.encode(prompt)
        generated = list(
            model.generate(
                ids,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=40,
            )
        )
        text = tokenizer.decode(list(ids) + generated)
        rep = repetition_rate(text)
        print(f"[{prompt!r}]  rep4={rep:.2f}")
        print(f"  {text!r}\n")

    # Perplexity on real held-out text, so "coherent" has a number attached.
    try:
        import pyarrow.parquet as pq

        paths = list_parquet_files(warn_on_legacy=False)
        text = "".join(pq.read_table(paths[-1]).column("text").to_pylist()[:200])
    except Exception as e:  # noqa: BLE001 - a missing corpus is not fatal here
        print(f"(skipping val bpb: {e})")
        return

    # Respect the model's trained context: asking for more tokens than the
    # rotary cache holds trips an assertion inside `GPT.forward`, which is a
    # confusing failure for what is meant to be a read-only report.
    ctx = model.config.sequence_len
    n = min(4096, ctx)
    ids = torch.tensor(tokenizer.encode(text)[:n], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        per_token = model(ids, targets=ids, loss_reduction="none")
    # Same normalisation the trainer uses (`evaluate_bpb`): sum nats over sum
    # BYTES of the target tokens, not over token count. Dividing by tokens
    # instead overstates the number by ~4x on this tokenizer, which is how an
    # earlier version of this script reported 15.2 where the trainer said 3.7.
    token_bytes = token_bytes_table()
    target = ids.view(-1)
    n_bytes = int(token_bytes[target].sum())
    nats_per_byte = float(per_token.view(-1).sum()) / max(n_bytes, 1)
    print(f"=== nats/byte on real held-out text: {nats_per_byte:.4f} ===")
    print(f"=== bits/char (x ln2)                  : {nats_per_byte * 0.6931:.4f} ===")
    print(
        "\nReference: English text is ~1.0-1.5 bits/char; a uniform model over "
        "this\nvocabulary is ~11.5. The trainer's own 'Validation bpb' uses the "
        "same\nnats/byte normalisation despite the name."
    )


def token_bytes_table() -> torch.Tensor:
    """Per-token byte length, matching `nanochat.modules.loss_eval`."""
    from nanochat.data.tokenizer import get_token_bytes

    return get_token_bytes(device="cpu").to(torch.int64).cpu()


if __name__ == "__main__":
    main()

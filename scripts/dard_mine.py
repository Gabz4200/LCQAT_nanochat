"""Mine a DARD rule set offline and save it as a reusable artifact.

DARD mines algebraic rules from a corpus once, offline; the training loop only
ever reads the compiled result. This script is that offline step, so a run can
be reproduced, inspected, or re-used across several training runs instead of
re-mining at every startup.

    python -m scripts.dard_mine --out ~/.cache/nanochat/dard_rules.pt
    python -m scripts.base_train --dard-rules ~/.cache/nanochat/dard_rules.pt

It reads the same parquet shards the dataloader reads, so the rules describe the
corpus the model will actually train on. Without `--out` it mines and reports
without saving, which is the quick way to see what a corpus yields.
"""

from __future__ import annotations

import argparse
import os

import torch

from nanochat.data.dataset import list_parquet_files
from nanochat.data.tokenizer import get_tokenizer
from nanochat.models.dard.atomize import PREMISE_WIDTH
from nanochat.models.dard.cli import add_dard_args, shard_size_for, validate_dard_args
from nanochat.models.dard.compile import mine_rules, save_rule_set
from nanochat.models.dard.shard import shard_corpus
from nanochat.utils.common import print0


def read_corpus(max_tokens: int, split: str = "train") -> torch.Tensor:
    """Read up to `max_tokens` token ids from the corpus parquet shards.

    Uses the same shard layout as the dataloader: all but the last parquet file
    is train, the last is val. Tokens are decoded to ids with the project's own
    tokenizer, so a rule set mined here indexes exactly the vocabulary a model
    trained on this data will use.
    """
    import pyarrow.parquet as pq

    tokenizer = get_tokenizer()
    paths = list_parquet_files(warn_on_legacy=False)
    if not paths:
        raise SystemExit(
            "no dataset parquet files found; run `python -m nanochat.data.dataset` first"
        )
    paths = paths[:-1] if split == "train" else paths[-1:]

    chunks: list[torch.Tensor] = []
    total = 0
    for path in paths:
        table = pq.read_table(path)
        column = table.column("text").to_pylist()
        if not column:
            continue
        ids = tokenizer.encode("".join(column))
        chunks.append(torch.tensor(ids, dtype=torch.long))
        total += len(ids)
        print0(f"  {os.path.basename(path)}: {len(ids):,} tokens (total {total:,})")
        if total >= max_tokens:
            break
    if not chunks:
        raise SystemExit(f"no text found in the {split} split")
    return torch.cat(chunks)[:max_tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine a DARD rule set")
    add_dard_args(parser)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="path to write the compiled rule set to; empty = mine and report only",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=2_000_000,
        help="how many corpus tokens to mine from (default: 2000000)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="which corpus split to mine from (default: train)",
    )
    args = parser.parse_args()
    validate_dard_args(args)

    print0(f"Reading up to {args.tokens:,} tokens from the {args.split} split...")
    tokens = read_corpus(args.tokens, args.split)
    print0(f"Read {tokens.numel():,} tokens")

    shard_size = shard_size_for(args, tokens)
    shards = shard_corpus(tokens, max_shard_size=shard_size)
    print0(
        f"Sharded into K={len(shards)} micro-shards of {shard_size} tokens "
        f"(PRD cap 999); premise width {PREMISE_WIDTH}"
    )

    vocab_size = get_tokenizer().get_vocab_size()
    rule_set = mine_rules(
        shards,
        vocab_size=vocab_size,
        workers=args.dard_workers,
        max_rules_per_shard=args.dard_max_rules,
        min_shard_frequency=args.dard_min_shard_frequency,
        min_compression_threshold=args.dard_min_compression,
        alpha=args.dard_alpha,
        beta=args.dard_beta,
        weight_temperature=args.dard_weight_temperature,
    )
    nnz = rule_set.premise.values().numel() + rule_set.conclusion.values().numel()
    print0(
        f"Mined {rule_set.num_rules} rules over V={vocab_size} "
        f"({nnz} stored entries, {nnz * 4 / 1e6:.1f} MB as int32)"
    )
    if rule_set.num_rules:
        w = rule_set.weights
        print0(
            f"  weights: min={w.min():.3e} max={w.max():.3e} "
            f"(floor {1e-6:.0e}, sum {w.sum():.6f})"
        )
        print0(
            f"  cross-shard frequency f_r: min={int(rule_set.frequency.min())} "
            f"max={int(rule_set.frequency.max())}"
        )
    else:
        print0(
            "WARNING: no rules survived. The corpus may be too small or too "
            "unstructured; try more tokens, or relax --dard-min-shard-frequency."
        )

    if args.out:
        path = args.out
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        save_rule_set(
            path,
            rule_set,
            {
                "num_shards": len(shards),
                "shard_size": shard_size,
                "tokens": tokens.numel(),
                "split": args.split,
                "alpha": args.dard_alpha,
                "beta": args.dard_beta,
                "weight_temperature": args.dard_weight_temperature,
            },
        )
        print0(f"Saved rule set to {path}")
    else:
        print0("No --out given; nothing written.")


if __name__ == "__main__":
    main()

"""DARD CLI flags and their training-loop wiring helpers.

Registered through one shared `add_dard_args` so `base_train`, `chat_sft` and
`chat_rl` cannot drift apart, for the reason `add_sparseprop_pruning_args` and
`add_w6_args` exist.

**DARD is always-on by default**, like LC-QAT and SparseProp: the auxiliary loss
is off only with `--no-dard`. It is opt-*out* rather than opt-in because the
whole premise is sample-efficiency on structured corpora -- a flag nobody
turns on measures nothing.

**The one hard constraint: DARD needs a vocabulary distribution.** The penalty
is a function of `softmax(logits)`. The DiffusionBlocks EDM objective predicts
*denoised embeddings*, and `denoise_step` returns embeddings, not logits, so the
two cannot be added in the naive way. Rather than reject DARD under EDM -- which
would make it unusable on this repo's default `--db-objective edm` -- `ENGINE`
projects the denoised suffix through `lm_head` to obtain the token distribution
the penalty is defined on. That is the same projection `generate()` already
uses to sample from the denoiser, so no new inference concept is introduced.
"""

from __future__ import annotations

import argparse
import os

import torch

from nanochat.models.dard.atomize import PREMISE_WIDTH
from nanochat.models.dard.compile import load_rule_set, mine_rules
from nanochat.models.dard.loss import DardLossBalancer, DardRuleLoss
from nanochat.models.dard.shard import MAX_SHARD_SIZE, shard_corpus


def add_dard_args(parser: argparse.ArgumentParser) -> None:
    """Register the DARD flags on `parser`.

    Registers the always-on toggle here rather than in each entry point, the
    same argument `add_sparseprop_pruning_args` makes: a helper named for a
    surface that leaves its own toggle in the caller is a helper that drifts.
    """
    parser.add_argument(
        "--no-dard",
        action="store_false",
        dest="dard",
        default=True,
        help="disable the DARD algebraic rule-distillation loss (default: on)",
    )
    parser.add_argument(
        "--dard-rules",
        type=str,
        default="",
        help=(
            "path to a compiled DARD rule set (.pt from scripts/dard_mine). "
            "Empty (default) mines one from the training shard at startup, "
            "which costs a few seconds and needs no separate command."
        ),
    )
    parser.add_argument(
        "--dard-shards",
        type=int,
        default=8,
        help=(
            "how many AML micro-shards to split the mining tokens into. 0 "
            "means 'as many as the tokens allow, never exceeding the PRD's "
            "N<1000 cap per shard'. K matters more than it looks: S_r = f_r / K "
            "is the Global Invariance term, and at K=1 every rule has f_r = K so "
            "S_r = 1 for all of them and the term carries no signal at all."
        ),
    )
    parser.add_argument(
        "--dard-mine-tokens",
        type=int,
        default=0,
        help=(
            "how many tokens of the training batch to mine rules from; "
            "0 (default) mines the whole first batch. The cap exists because "
            "mining is O(candidates) and a full corpus would be minutes."
        ),
    )
    parser.add_argument(
        "--dard-lambda",
        type=float,
        default=0.20,
        help=(
            "lambda_global, the proportion of gradient norm assigned to rule "
            "enforcement, in (0, 1] (default: 0.20, the PRD value)"
        ),
    )
    parser.add_argument(
        "--dard-gamma",
        type=float,
        default=10.0,
        help="softplus hinge temperature for the order-violation penalty (default: 10.0)",
    )
    parser.add_argument(
        "--dard-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay rho for the dynamic loss balancer (default: 0.99)",
    )
    parser.add_argument(
        "--dard-weight-temperature",
        type=float,
        default=2.0,
        help=(
            "T_w, the rule-weight temperature. Larger flattens the rule-weight "
            "distribution and keeps tail rules from being starved (default: 2.0)"
        ),
    )
    parser.add_argument(
        "--dard-alpha",
        type=float,
        default=1.0,
        help="exponent alpha on cross-shard frequency in the rule weight (default: 1.0)",
    )
    parser.add_argument(
        "--dard-beta",
        type=float,
        default=1.0,
        help=(
            "exponent beta on algebraic compression in the rule weight (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--dard-min-shard-frequency",
        type=int,
        default=1,
        help=(
            "prune rules seen in fewer than this many shards; 1 (default) keeps "
            "every rule, which is the PRD's default of no hard pruning"
        ),
    )
    parser.add_argument(
        "--dard-min-compression",
        type=float,
        default=0.0,
        help=(
            "prune rules whose algebraic compression kappa falls below this; "
            "0.0 (default) keeps every rule"
        ),
    )
    parser.add_argument(
        "--dard-max-rules",
        type=int,
        default=2048,
        help="cap on rules mined per micro-shard, bounding crossing cost (default: 2048)",
    )
    parser.add_argument(
        "--dard-workers",
        type=int,
        default=0,
        help=(
            "CPU worker processes for AML mining; 0 (default) mines serially, "
            "which is faster at the PRD's N<1000 shard size because the "
            "process spawn costs more than the work it offloads"
        ),
    )


def validate_dard_args(args) -> None:
    """Reject a DARD configuration that cannot work, before training starts.

    Raising here rather than at the first micro-step: a flag the user typed
    that silently does nothing is the failure mode the `--db-blocks<=0` guard
    tuple exists to prevent, and DARD adds three such flags.
    """
    if args.dard_shards < 0:
        raise SystemExit(f"--dard-shards must be >= 0, got {args.dard_shards}")
    if args.dard_mine_tokens < 0:
        raise SystemExit(
            f"--dard-mine-tokens must be >= 0, got {args.dard_mine_tokens}"
        )
    if args.dard_max_rules < 1:
        raise SystemExit(f"--dard-max-rules must be >= 1, got {args.dard_max_rules}")
    if args.dard_workers < 0:
        raise SystemExit(f"--dard-workers must be >= 0, got {args.dard_workers}")
    if args.dard_alpha < 0.0 or args.dard_beta < 0.0:
        raise SystemExit(
            f"--dard-alpha and --dard-beta must be >= 0, got {args.dard_alpha} "
            f"and {args.dard_beta}"
        )
    if args.dard_weight_temperature <= 0.0:
        raise SystemExit(
            f"--dard-weight-temperature must be > 0, got {args.dard_weight_temperature}"
        )
    if not 0.0 < args.dard_lambda <= 1.0:
        raise SystemExit(f"--dard-lambda must be in (0, 1], got {args.dard_lambda}")
    if args.dard_gamma <= 0.0:
        raise SystemExit(f"--dard-gamma must be > 0, got {args.dard_gamma}")
    if args.dard_rules and not os.path.exists(args.dard_rules):
        raise SystemExit(f"--dard-rules path does not exist: {args.dard_rules}")


def build_dard(
    args,
    vocab_size: int,
    device: torch.device,
    mine_tokens: torch.Tensor | None,
):
    """Load or mine a rule set.

    Returns `(DardRuleLoss, DardLossBalancer, list[str])`. The third element is
    the log lines the caller should print. They are *returned*, not printed
    here, for two reasons: this module is in the functional core, which must not
    `print` (a leftover-debug smell) and must not import
    `nanochat.utils.common.print0` (the shell -- enforced for the whole package by
    `tests/test_architecture_boundary.py`); and `print0` is rank-gated, so the
    caller that owns DDP is the one that should decide whether this rank speaks.

    Args:
        args: the parsed namespace, carrying the `--dard-*` flags.
        vocab_size: the model's output vocabulary. Every compiled rule indexes
            into this, so a mismatch is rejected rather than tolerated.
        device: where the rule matrices are placed. They are small (PRD: 15-40
            MB at R=1e5) and belong in system RAM next to the weights.
        mine_tokens: a 1-D token tensor to mine from when no artifact was
            given. `None` means "mining was requested but no tokens are
            available", which is an error rather than an empty rule set: an
            empty rule set makes `--no-dard` and "DARD with zero rules"
            indistinguishable in the loss curve.
    """
    messages: list[str] = []
    if args.dard_rules:
        rule_set, _ = load_rule_set(args.dard_rules)
        messages.append(
            f"DARD: loaded {rule_set.num_rules} compiled rules from {args.dard_rules}"
        )
    else:
        if mine_tokens is None or mine_tokens.numel() == 0:
            raise SystemExit(
                "DARD is enabled but there are no tokens to mine rules from. "
                "Pass --no-dard, or --dard-rules <artifact>, or run a training "
                "step with data before the rule set is needed."
            )
        tokens = mine_tokens.reshape(-1)
        if args.dard_mine_tokens > 0:
            tokens = tokens[: args.dard_mine_tokens]
        shards = shard_corpus(tokens, max_shard_size=shard_size_for(args, tokens))
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
        messages.append(
            f"DARD: mined {rule_set.num_rules} rules over {len(shards)} shards "
            f"({rule_set.premise.values().numel()} premise nnz)"
        )

    if rule_set.vocab_size != vocab_size:
        raise SystemExit(
            f"the DARD rule set was compiled for vocabulary {rule_set.vocab_size} "
            f"but this model outputs {vocab_size}. Recompile the rules against "
            "this tokenizer."
        )
    if rule_set.num_rules == 0:
        messages.append(
            "WARNING: DARD enabled but the rule set is empty; the auxiliary loss "
            "will contribute exactly 0.0. Check --dard-min-shard-frequency / "
            "--dard-min-compression, or mine from more data."
        )

    loss_fn = DardRuleLoss(rule_set, gamma=args.dard_gamma).to(device)
    balancer = DardLossBalancer(
        lambda_global=args.dard_lambda, ema_decay=args.dard_ema_decay
    ).to(device)
    loss_fn.train()
    balancer.train()
    return loss_fn, balancer, messages


def shard_size_for(args, tokens: torch.Tensor) -> int:
    """Tokens per micro-shard: enough to yield `--dard-shards` of them, capped.

    The tension this resolves: the PRD caps a shard at `MAX_SHARD_SIZE` for
    crossing-cost reasons, but a small corpus would otherwise land in one shard,
    where `S_r = f_r / K` is identically 1 and the weighting's Global Invariance
    term does no work. So the shard *size* is derived from the requested shard
    *count* rather than fixed at the cap, and the cap is the upper bound.

    A shard shorter than `PREMISE_WIDTH + 1` tokens mines nothing, so the size is
    also floored there: below it we would manufacture empty shards.
    """
    if args.dard_shards <= 0:
        return MAX_SHARD_SIZE
    want = -(-tokens.numel() // args.dard_shards)  # ceil
    return max(min(want, MAX_SHARD_SIZE), MIN_SHARD_TOKENS)


#: Below this a shard cannot form a single premise/conclusion pair, so mining it yields nothing. Sizing shards under it would only manufacture empty ones.
MIN_SHARD_TOKENS = PREMISE_WIDTH + 1


__all__ = [
    "add_dard_args",
    "build_dard",
    "shard_size_for",
    "validate_dard_args",
]

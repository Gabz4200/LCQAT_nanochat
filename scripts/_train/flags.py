"""Shared argparse flags for the training entry points."""

from __future__ import annotations

import argparse


def add_minexp_flags(parser: argparse.ArgumentParser) -> None:
    """Register the ``--minexp-*`` MinExpLSTM hyperparameter block."""
    parser.add_argument(
        "--minexp-alpha",
        type=float,
        default=0.1,
        help="Deprecated: non-zero values are accepted and ignored (the FiLM gain "
        "replaces the residual coefficient); 0 freezes the FiLM head (pure SGD)",
    )
    parser.add_argument(
        "--minexp-beta",
        type=float,
        default=0.9,
        help="MinExpLSTM: fixed EMA decay target (used when --minexp-learnable-beta "
        "is off; the lambda-coupled predictor is frozen to emit it at g_norm=0)",
    )
    parser.add_argument(
        "--minexp-lam",
        type=float,
        default=0.10,
        help="MinExpLSTM: fast/slow EMA horizon spacing in (0, 0.4] "
        "(beta_slow - beta_fast == lam)",
    )
    parser.add_argument(
        "--minexp-clamp",
        type=float,
        default=1.0,
        help="MinExpLSTM: clamp bound on the FiLM directional shift delta",
    )
    parser.add_argument(
        "--minexp-learnable-beta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MinExpLSTM: lambda-coupled betas learned via the cell's predictor "
        "(default: on); off freezes the predictor at --minexp-beta",
    )
    parser.add_argument(
        "--minexp-meta-every",
        type=int,
        default=50,
        help="MinExpLSTM: joint online co-training interval (meta_step every K optimizer steps; 0 disables)",
    )
    parser.add_argument(
        "--minexp-meta-lr",
        type=float,
        default=1e-3,
        help="MinExpLSTM: meta-gradient LR for the cell co-training step (AdamW)",
    )


def add_optimizer_flags(parser: argparse.ArgumentParser) -> None:
    """Register ``--optimizer`` plus the ``--minexp-*`` block."""
    parser.add_argument(
        "--optimizer",
        type=str,
        default="minexplstm",
        choices=["minexplstm", "adamw", "mezo"],
        help="optimizer: MinExpLSTM joint online learned optimizer (default) "
        "or AdamW baseline; MeZO is zeroth-order and needs the dual-forward "
        "step in the base-train loop, so it is opt-in here",
    )
    add_minexp_flags(parser)


__all__ = ["add_minexp_flags", "add_optimizer_flags"]

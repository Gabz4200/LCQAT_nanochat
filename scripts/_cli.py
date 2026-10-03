"""CLI flags the training entry points share.

`base_train`, `chat_sft` and `chat_rl` each open with the same four
`add_argument` calls. The values were already identical; only the help prose
differed, because each script described the flag from its own stage's point of
view. Unified here, so a flag added to one entry point cannot be missing from
another.

`--model-step` is conditional: only the chat stages *load* a model, so only they
select a step to resume from. `base_train` trains from scratch and has no such
flag -- adding one there would advertise a resume path it does not implement.
"""

from __future__ import annotations

import argparse


def add_common_cli_args(
    parser: argparse.ArgumentParser, *, model_step: bool = False
) -> None:
    """Register the run/device/model-identity flags shared by the entry points.

    `model_step` adds `--model-step`, which selects a checkpoint step to load.
    Only the stages that load a pretrained model pass it.
    """
    parser.add_argument(
        "--run",
        type=str,
        default="dummy",
        help="wandb run name ('dummy' disables wandb logging)",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        help="cuda|cpu|mps (empty = autodetect)",
    )
    # One wording for both directions: pretraining reads it as the tag to save
    # under, the chat stages as the tag to load from, and the flag is the same
    # either way.
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="model tag (checkpoint directory name)",
    )
    if model_step:
        parser.add_argument(
            "--model-step",
            type=int,
            default=None,
            help="model step to load from",
        )

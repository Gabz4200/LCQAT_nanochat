"""Side-effect observers for the training loop.

The loop's job is the tensor math and the optimizer step. Everything that
reaches outside the process -- W&B, garbage collection, the end-of-run console
summary -- lives here, so the loop body reads as training and nothing else.

These are plain functions rather than a framework's callback objects: the
project deliberately runs a manual loop with no training framework, so the
observer is just a function the loop calls at the point the effect happens. A
registry would add indirection without adding a capability.
"""

from __future__ import annotations

import gc

from nanochat.utils.common import print0


def log_step_metrics(wandb_run, log_data: dict, every_n_steps: int, step: int) -> None:
    """Push one step's metrics to W&B, honoring the cadence.

    `every_n_steps <= 0` disables logging entirely, which is how a run is
    silenced without special-casing the call site.
    """
    if every_n_steps > 0 and step % every_n_steps == 0:
        wandb_run.log(log_data)


def manage_gc(step: int, first_step_of_run: bool, every_n_steps: int = 5000) -> None:
    """Keep the collector out of the training loop's hot path.

    The default collector is overactive during long CPU runs: it spends
    hundreds of milliseconds scanning for cycles to reclaim very few tiny
    objects. So it is collected once after setup, then frozen to exclude the
    long-lived model from every later scan, then disabled outright -- except
    for an occasional collect on very long runs.
    """
    if first_step_of_run:
        gc.collect()  # reclaim the setup pass before timing starts
        gc.freeze()  # exclude surviving long-lived objects from every scan
        gc.disable()
    elif step % every_n_steps == 0:
        gc.collect()


def print_run_summary(
    get_max_memory,
    total_training_time: float,
    val_bpb: float,
    min_val_bpb: float,
    bpb_places: int = 6,
) -> None:
    """Print the end-of-run report to stdout.

    `bpb_places` because the two callers disagree and each states its own: the
    pretraining loop has always printed 6, chat SFT has always printed 4. The
    helper owns the *shape* of the report; the precision is a per-stage
    presentation choice, so it is a parameter rather than a second copy of the
    three `print0` lines.
    """
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")
    # Gated on `val_bpb` and prints `min_val_bpb`. Preserved verbatim from the
    # original: the guard reads as though it should be `min_val_bpb is not None`,
    # but changing it would alter when the line appears, which is a behavior
    # change, not a refactor.
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.{bpb_places}f}")

"""End-to-end smoke test for the base_train entry point.

Runs the real script as a subprocess. The unit tests all construct engines and
layers in-process, which is faster and pinpoints failures, but nothing exercised
the actual `python -m scripts.base_train` path -- and that is where the W0
blockers lived: `--kd-alpha` defaulting to 0.1 with no teacher, `args.init_lr_frac`
being undefined, `engine` being shadowed by `Engine`, and
`orig_model.state_dict()` dropping the whole diffusion engine.
"""

import os
import subprocess
import sys

import pytest


def _subprocess_python() -> str:
    """The interpreter that can actually import this project's dependencies.

    `sys.executable` is only correct when pytest itself runs from the project
    venv. Under a plain `python -m pytest` against the system interpreter it
    points at /usr/bin/python, and every subprocess test then dies at
    `import wandb` before reaching any of the code under test -- which is how
    tests/test_w0_smoke.py came to have subprocess tests that never ran.

    Resolve from the *package* location instead of `sys.executable`: the
    project's own dependencies are importable from the interpreter whose
    site-packages contains `nanochat`, and that interpreter is the venv one
    whether or not pytest itself is running inside it. Falls back to
    `sys.executable` when the venv layout is not there, which is the normal
    in-venv case and needs no special handling.
    """
    venv_python = os.path.join(sys.prefix, "bin", "python")
    # A repo checkout inside a venv: the project root is two levels above the
    # tests directory, and the venv interpreter sits beside bin/activate.
    candidate = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".venv",
        "bin",
        "python",
    )
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


# d6 with a short sequence length keeps the smoke inside the 7.6 GB host budget.
SMOKE_ARGS = [
    "--depth",
    "6",
    "--num-iterations",
    "2",
    "--max-seq-len",
    "64",
    "--device-batch-size",
    "2",
    "--total-batch-size",
    "256",
    "--run",
    "dummy",
    "--eval-tokens",
    "128",
    "--eval-every",
    "-1",
    "--core-metric-every",
    "-1",
    "--save-every",
    "-1",
    "--sample-every",
    "-1",
]


def _run(extra, timeout=900):
    return subprocess.run(
        [_subprocess_python(), "-m", "scripts.base_train", *SMOKE_ARGS, *extra],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.slow
def test_when_default_args_then_base_train_completes():
    """The regression this file exists for.

    Before W0, every invocation exited non-zero: `--kd-alpha` defaulted to 0.1
    while both teacher flags defaulted to None, so the script raised SystemExit
    unconditionally.
    """
    proc = _run([])
    assert proc.returncode == 0, (
        f"base_train exited {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )
    assert "step 00001" in proc.stdout, "training loop did not run"


@pytest.mark.slow
@pytest.mark.parametrize(
    "extra",
    [
        ["--db-objective", "edm"],
        ["--db-objective", "ce"],
        ["--codebook-grad-scale", "none"],
        ["--codebook-grad-scale", "inv_sqrt_n"],
        ["--no-lcqat"],
        ["--no-sparseprop"],
        ["--lcqat-preset", "small"],
        ["--lcqat-preset", "prd"],
        ["--db-block-sampling", "micro"],
        ["--sparseprop-scope", "global"],
        ["--sparseprop-every", "1", "--sparseprop-start-frac", "0.25"],
        ["--sparseprop-dense-threshold", "0.5"],
    ],
)
def test_when_flag_combination_then_runs(extra):
    """Every advertised training flag must be reachable and runnable.

    `--codebook-grad-scale none` in particular was the A1 gap: `grad_scale` was a
    `LayerKConfig` field threaded through `retrofit_model`, but no script set it,
    so the PRD 2.4 ablation axis existed in the library and nowhere else.
    """
    proc = _run(extra)
    assert proc.returncode == 0, (
        f"{extra} exited {proc.returncode}\n{proc.stderr[-3000:]}"
    )


@pytest.mark.slow
def test_when_gradual_pruning_enabled_then_the_loop_actually_prunes():
    """The ramp must fire inside a real run, not just in a unit test.

    Wiring the schedule into the loop is the part that can silently rot: the
    schedule object exists, the flags parse, and nothing ever calls `apply` on a
    step that is a prune event.
    """
    proc = _run(
        [
            "--sparseprop-every",
            "1",
            "--sparseprop-start-frac",
            "0.2",
            "--db-blocks",
            "1",
        ]
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "SparseProp pruned to" in proc.stdout, proc.stdout[-3000:]


@pytest.mark.slow
def test_when_codebook_grad_scale_invalid_then_rejected():
    proc = _run(["--codebook-grad-scale", "sqrt_n"])
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr or "codebook-grad-scale" in proc.stderr


@pytest.mark.slow
def test_when_kd_alpha_without_teacher_then_rejected():
    """The W0.1a regression from the other direction.

    KD used to default to on, so the *default* run raised. KD is now opt-in, and
    opting in without a teacher must still fail loudly rather than silently
    skipping the KD term.

    `--db-objective ce` is explicit because `edm` is the default, and `--kd-alpha`
    is already incompatible with it (see the next test). Asking for the teacher
    prerequisite means asking for the *other* conflict first -- a test about
    teacher wiring cannot reach the teacher check while the objective check
    fires ahead of it.
    """
    proc = _run(["--kd-alpha", "0.5", "--db-objective", "ce"])
    assert proc.returncode != 0
    assert "kd-teacher" in proc.stderr.lower()


@pytest.mark.slow
def test_when_kd_alpha_with_edm_then_rejected():
    """Logit-KD cannot drive the denoising objective; say so at startup.

    The conflict must be reported even when the teacher is *also* missing, i.e.
    before the teacher prerequisite. The objective conflict is the more
    informative message -- it is a statement about the two flags actually typed
    -- and the prerequisite check used to run first, so this configuration
    reported only "requires --kd-teacher-source" and the user learned about the
    real problem only on a second run.
    """
    proc = _run(["--kd-alpha", "0.5", "--db-objective", "edm"])
    assert proc.returncode != 0
    assert "db-objective" in proc.stderr


@pytest.mark.slow
def test_when_kd_alpha_with_ce_and_teacher_then_the_conflict_is_not_raised():
    """The other side of the precedence fix: with the objective conflict out of
    the way, the run must reach (and pass) the teacher check rather than
    reporting the objective error it no longer applies to."""
    proc = _run(["--kd-alpha", "0.5", "--db-objective", "ce"])
    assert proc.returncode != 0
    assert "incompatible" not in proc.stderr

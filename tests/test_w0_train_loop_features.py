"""End-to-end verification of the W6 features inside the real `base_train` loop.

`tests/test_dbcpu_efqat_latch.py` and `tests/test_dbcpu_kd_denoiser.py` pin the
`BlockLatchFreezer` and `DenoiserDistiller` objects in isolation, and
`tests/test_w0_smoke.py` proves the CLI starts at all. Neither of those answers
the question that actually matters for W6: does the latch *fire* inside a real
training loop, and does the denoiser anchor survive the whole
arg-parse -> retrofit -> engine -> step path? Both are wiring claims, and wiring
is exactly what breaks when an intermediate step is renamed.

Every probe here is a real `python -m scripts.base_train` subprocess on the
d6/`--run dummy` recipe from `test_w0_smoke.SMOKE_ARGS` (reused rather than
re-derived, so the two files cannot drift on what "a smoke run" means). Handoff
§12.4 measured that recipe at 171 s wall / 68 s CPU, so the three probes cost
roughly 3 minutes each and every test is marked `slow` (the marker is registered
in `pyproject.toml`, so `pytest tests/` deselects them with `-m "not slow"`).
"""

import hashlib
import subprocess
import sys

import pytest

from tests.test_w0_smoke import SMOKE_ARGS

# ~171 s measured per probe (§12.4) with headroom for the float twin, which adds
# a second full model to the KD probe. Comfortably above the 600 s floor.
RUN_TIMEOUT = 1200


def _model_tag(tmp_path) -> str:
    """A unique, filesystem-safe checkpoint tag for one probe run.

    Handoff §9.7: a probe that omitted `--model-tag` wrote into
    `~/.cache/nanochat/base_checkpoints/d6` and clobbered a real checkpoint. The
    tag therefore has to be unique per invocation, not merely per file.

    It is deliberately NOT a redirect of `NANOCHAT_BASE_DIR` into `tmp_path`:
    the tokenizer lives under the base dir, so pointing the base dir at an
    empty temp dir would fail the run at dataloading instead of exercising the
    feature. A unique subdirectory of the real base dir is the safe isolation.

    Hashed rather than taken from `tmp_path.name` so the tag cannot inherit
    pytest's parametrized id characters (`[`, `]`, `/`).
    """
    digest = hashlib.sha1(str(tmp_path).encode()).hexdigest()[:12]
    return f"w6-probe-{digest}"


def _run(extra, tmp_path, timeout=RUN_TIMEOUT):
    """Run `base_train` with the shared smoke recipe, tagged for this probe.

    Same subprocess shape as `test_w0_smoke._run` (inherited environment, text
    pipes, explicit timeout); only the model tag is added.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.base_train",
            *SMOKE_ARGS,
            "--model-tag",
            _model_tag(tmp_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_ran(proc) -> None:
    """Assert the run exited clean and actually executed a training step.

    Without the step-line check, "exit code 0" would be satisfied by a script
    that parsed the flags and then did no work.
    """
    assert proc.returncode == 0, (
        f"base_train exited {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )
    assert "step 00001" in proc.stdout, (
        f"training loop did not run\n--- stdout tail ---\n{proc.stdout[-3000:]}"
    )


@pytest.mark.slow
def test_when_latch_blocks_are_set_then_the_run_completes_and_reports_latched_blocks(
    tmp_path,
) -> None:
    """The per-block latch must actually fire inside a real training loop.

    Two distinct print0 calls carry the claim, and both are asserted: the setup
    line proves the freezer was built and handed to the engine as the
    `requires_grad` arbiter, the step line proves `latch_blocks` was *called*
    during the loop. The setup line alone would pass with a freezer nobody ever
    consults, which is the exact rot the harness exists to catch -- the object
    exists, the flags parse, and nothing ever calls it on a latch step.

    Block 0 of 4 is latched (`num_db_blocks = min(--db-blocks", --depth) = 4`).
    `--efqat-latch-after` is left at its default of -1, i.e. fire at step 0, so
    the latch lands inside the two-iteration budget. The engine's sampler
    excludes latched blocks, so 3 of 4 remain live and the run is not starved.
    """
    proc = _run(["--efqat-latch-blocks", "0"], tmp_path)

    _assert_ran(proc)
    assert "EfQAT per-block latch enabled" in proc.stdout, proc.stdout[-3000:]

    latch_lines = [
        ln for ln in proc.stdout.splitlines() if "EfQAT latched blocks" in ln
    ]
    assert latch_lines, (
        "the latch never fired during the run\n--- stdout tail ---\n"
        f"{proc.stdout[-3000:]}"
    )
    # The line reports the latch step and the accumulated latched set. Reading
    # the accumulated set (not just the requested targets) is what proves block
    # 0 was actually retired rather than the freezer being constructed and left
    # unconsulted.
    assert "total latched: [0]" in latch_lines[0], latch_lines[0]
    assert "permanently at step 0" in latch_lines[0], latch_lines[0]


@pytest.mark.slow
def test_when_kd_denoiser_alpha_is_set_then_the_run_completes(tmp_path) -> None:
    """The EDM denoiser anchor must survive the whole CLI path.

    This asserts WIRING only, and deliberately says nothing about the anchor's
    value. Handoff §9.6: `DiffusionBlockEngine.__init__` zero-inits every
    denoise head and `base_train` never randomizes them, so on a fresh run the
    student and the FP32 twin are bit-identical at step 0 and
    `w(sigma)*||D_quant - D_float||^2` is exactly 0. `kd: 0.000000` in the step
    line is therefore the expected, correct output of a healthy anchor -- there
    is simply no quantization gap yet to measure. Do NOT assert `kd > 0`; that
    assertion would be pinning a coincidence of the zero-init, not a property of
    the code.

    Also per §9.8, `denoise_step` returns `(loss, sigma)` and the loss is jittery
    by design, so no loss value or monotonicity is asserted here either.

    `--db-objective edm` is explicit even though it is the default: the anchor
    is defined only against the EDM denoising output space, and being explicit
    keeps the run honest about which objective it needs.
    """
    proc = _run(["--db-objective", "edm", "--kd-denoiser-alpha", "0.5"], tmp_path)

    _assert_ran(proc)
    assert "KD denoiser distillation enabled" in proc.stdout, proc.stdout[-3000:]
    # The anchor is installed on the engine, so the per-step line carries the
    # kd field. That the field reads 0.0 is expected -- see the docstring.
    assert "kd:" in proc.stdout, (
        f"the per-step kd field is missing, so the anchor is not installed\n"
        f"{proc.stdout[-3000:]}"
    )


@pytest.mark.slow
def test_when_the_denominator_is_overridden_then_it_fails_loudly(tmp_path) -> None:
    """A wrong-objective denoiser anchor must be rejected, not ignored.

    Handoff §9.6 and §3 bug 9 are the same trap: a flag parsed but not applied
    is invisible, because the run looks exactly like a run on which the feature
    "did not help". `--kd-denoiser-alpha` regresses against the CE objective,
    where the denoiser emits no embedding to anchor and the float twin has
    nothing to predict, so `build_denoiser_teacher` raises `SystemExit`. This
    test pins that it still does: a silent skip here would make every later
    denoiser-KD measurement meaningless.

    The guard fires during setup, before the training loop, so the assertion is
    on the exit code and the message naming both flags -- not on any step
    output.
    """
    proc = _run(["--db-objective", "ce", "--kd-denoiser-alpha", "0.5"], tmp_path)

    assert proc.returncode != 0, (
        "the denoiser anchor was silently skipped under --db-objective ce\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}"
    )
    assert "kd-denoiser-alpha" in proc.stderr, (
        f"the error does not name the rejected flag\n--- stderr tail ---\n"
        f"{proc.stderr[-3000:]}"
    )
    assert "db-objective" in proc.stderr, (
        f"the error does not name the conflicting flag\n--- stderr tail ---\n"
        f"{proc.stderr[-3000:]}"
    )

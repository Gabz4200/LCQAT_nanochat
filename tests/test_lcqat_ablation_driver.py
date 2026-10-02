"""Tests for `scripts/lcqat_ablation.py` (the ablation driver).

The driver's job is to produce a *trustworthy* number, so these tests are about
its protocol and its failure behaviour rather than about it printing a table:

* Argument validation must fail fast. A sweep run with `--seeds 0` that silently
  reports an average over nothing is the worst possible outcome for a tool whose
  whole job is evidence.
* The driver must be runnable end to end and exit 0 on a clean measurement.
* It must exit non-zero when a claim the write-up advertises comes out wrong.
  A driver that cannot fail is decoration.
* The leaderboard writer must preserve hand-written prose outside its generated
  section, or re-running it silently deletes the analysis around the numbers.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_driver():
    """Import `scripts/lcqat_ablation.py` as a module, by path."""
    path = REPO_ROOT / "scripts" / "lcqat_ablation.py"
    spec = importlib.util.spec_from_file_location("lcqat_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lcqat_ablation"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return load_driver()


class TestArgumentValidation:
    @pytest.mark.parametrize(
        ("flags", "message"),
        [
            (["--seeds", "0"], "--seeds must be >= 1"),
            (["--n", "0"], "--n must be >= 1"),
            (["--grad-scale-steps", "0"], "--grad-scale-steps must be >= 1"),
            (["--grad-scale-lr", "0"], "--grad-scale-lr must be positive"),
            (["--grad-scale-lr", "-1"], "--grad-scale-lr must be positive"),
        ],
    )
    def test_when_an_argument_is_invalid_then_it_fails_fast(
        self, driver, flags: list[str], message: str
    ) -> None:
        """A sweep over zero seeds must not quietly report an average of nothing."""
        parser = driver.argparse.ArgumentParser()
        driver.register_args(parser)
        args = parser.parse_args(flags)
        with pytest.raises(ValueError, match=message):
            driver.validate_args(args)

    def test_when_the_experiment_is_unknown_then_argparse_rejects_it(
        self, driver
    ) -> None:
        parser = driver.argparse.ArgumentParser()
        driver.register_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--experiment", "not_an_experiment"])

    def test_when_args_are_valid_then_validation_passes(self, driver) -> None:
        parser = driver.argparse.ArgumentParser()
        driver.register_args(parser)
        args = parser.parse_args(["--seeds", "2", "--n", "64"])
        driver.validate_args(args)  # must not raise


class TestProbeConstruction:
    def test_when_the_probe_is_built_then_it_is_non_negative(self, driver) -> None:
        """`asym` exists because `mlp.forward` computes `F.relu(x).square()`.

        A signed probe would test the opposite of what the preset is for, and
        measured that way the preset looks worse than the baseline.
        """
        probe = driver.non_negative_probe(16, rows=128, seed=0)
        assert probe.shape == (128, 16)
        assert float(probe.min()) >= 0.0

    def test_when_the_probe_is_built_then_the_seed_determines_it(self, driver) -> None:
        a = driver.non_negative_probe(16, rows=32, seed=0)
        b = driver.non_negative_probe(16, rows=32, seed=0)
        c = driver.non_negative_probe(16, rows=32, seed=1)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_when_the_layer_is_probed_then_each_arm_is_an_independent_model(
        self, driver
    ) -> None:
        """Retrofitting mutates in place, so arms must not share a model.

        If they did, the second arm would retrofit an already-retrofitted model
        and the comparison would be meaningless. The two presets give `c_proj` the
        same *weight* K, so identity of the objects -- not of their codebook
        shapes -- is what has to be checked here.
        """
        first = driver.probe_layer("small")
        second = driver.probe_layer("asym")
        assert first is not second
        assert first.weight_quantizer is not second.weight_quantizer
        # Independent parameters: mutating one arm cannot move the other.
        with torch.no_grad():
            first.weight_quantizer.raw_pos_deltas.add_(1.0)
        assert not torch.equal(
            first.weight_quantizer.get_codebook(),
            second.weight_quantizer.get_codebook(),
        )

    def test_when_the_presets_differ_then_the_activation_codebook_differs(
        self, driver
    ) -> None:
        """The `asym` split is on `c_proj`'s *activation* quantizer.

        Its weight quantizer is symmetric under both presets, so a harness that
        measured weights would find no difference at all.
        """
        small = driver.probe_layer("small")
        asym = driver.probe_layer("asym")
        assert small.act_quantizer.K != asym.act_quantizer.K
        assert not torch.equal(
            small.act_quantizer.get_codebook().detach(),
            asym.act_quantizer.get_codebook().detach(),
        )


class TestDriverRuns:
    def test_when_run_as_dry_run_then_it_prints_and_exits_zero(
        self, driver, capsys
    ) -> None:
        """One seed is enough to prove the pipeline is wired end to end."""
        code = driver.main(
            ["--experiment", "all", "--seeds", "1", "--n", "64", "--dry-run"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "grad_scale" in out
        assert "asym_vs_small" in out

    def test_when_run_then_the_grad_scale_row_matches_one_over_sqrt_n(
        self, driver, capsys
    ) -> None:
        """The PRD 2.4 claim is checked by the driver itself, every run."""
        code = driver.main(
            ["--experiment", "grad_scale", "--seeds", "1", "--n", "64", "--dry-run"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "1/sqrt(N)" in out

    def test_when_run_with_json_output_then_it_is_written(
        self, driver, tmp_path
    ) -> None:
        import json

        target = tmp_path / "nested" / "out.json"
        code = driver.main(
            [
                "--experiment",
                "grad_scale",
                "--seeds",
                "1",
                "--n",
                "64",
                "--json-out",
                str(target),
                # `--out` defaults to dev/LEADERBOARD.md, so omitting it here
                # made every full-suite run overwrite the published table with
                # this 1-seed row. Scope both outputs to tmp_path.
                "--out",
                str(tmp_path / "LEADERBOARD.md"),
            ]
        )
        assert code == 0
        rows = json.loads(target.read_text())
        assert isinstance(rows, list)
        assert rows and rows[0]["experiment"] == "grad_scale"

    def test_when_a_programmatic_call_omits_out_then_the_published_file_is_untouched(
        self, driver
    ) -> None:
        """A library caller must not be able to overwrite dev/LEADERBOARD.md.

        `main()` defaults `--out` to the published leaderboard, which is right
        for the documented command line. A test that called `main()` with its
        own argv but no `--out` therefore replaced the full multi-experiment
        table with its own 1-seed row, silently, on every full-suite run. The
        published file is an artifact a reader trusts, so a programmatic caller
        now has to name its destination.
        """
        published = driver.PUBLISHED_LEADERBOARD
        before = published.read_bytes() if published.exists() else None

        with pytest.raises(SystemExit, match="refusing to write the published"):
            driver.main(["--experiment", "grad_scale", "--seeds", "1", "--n", "32"])

        after = published.read_bytes() if published.exists() else None
        assert after == before

    def test_when_a_programmatic_call_passes_out_then_it_writes_there(
        self, driver, tmp_path
    ) -> None:
        """The guard must not block a caller that names its own destination."""
        target = tmp_path / "LEADERBOARD.md"
        code = driver.main(
            [
                "--experiment",
                "grad_scale",
                "--seeds",
                "1",
                "--n",
                "32",
                "--out",
                str(target),
            ]
        )
        assert code == 0
        assert "BEGIN GENERATED" in target.read_text()


class TestClaimChecking:
    def test_when_the_grad_scale_ratio_wrongly_agrees_then_it_is_flagged(
        self, driver
    ) -> None:
        """The driver must be able to fail, or its exit code means nothing.

        A ratio of 1.0 means the `1/sqrt(N)` scaling stopped being applied, which
        is the single most important thing this tool exists to catch.
        """
        from nanochat.models.quant.ablation_metrics import AblationRow

        row = AblationRow(
            experiment="grad_scale",
            metric="codebook_grad_ratio",
            baseline="none",
            variant="inv_sqrt_n",
            value_baseline=1.0,
            value_variant=1.0,
            delta=0.0,
            better="baseline",
            n_seeds=1,
            notes="observed ratio 1.0 vs predicted 1/sqrt(N) = 0.00195312 for N=262144",
        )
        problems = driver.check_claims([row], 262144)
        assert problems
        assert "1/sqrt(N)" in problems[0]

    def test_when_the_grad_scale_ratio_agrees_then_nothing_is_flagged(
        self, driver
    ) -> None:
        from nanochat.models.quant.ablation_metrics import AblationRow

        row = AblationRow(
            experiment="grad_scale",
            metric="codebook_grad_ratio",
            baseline="none",
            variant="inv_sqrt_n",
            value_baseline=1.0,
            value_variant=0.001953,
            delta=-0.998,
            better="variant",
            n_seeds=1,
            notes="observed ratio 0.001953 vs predicted 1/sqrt(N) = 0.00195312 for N=262144",
        )
        assert driver.check_claims([row], 262144) == []

    def test_when_asym_wastes_more_levels_than_small_then_it_is_flagged(
        self, driver
    ) -> None:
        """The split must strictly reduce wasted levels on relu^2 data."""
        from nanochat.models.quant.ablation_metrics import AblationRow

        row = AblationRow(
            experiment="asym_vs_small_levels",
            metric="level_utilization",
            baseline="small",
            variant="asym",
            value_baseline=1.0,
            value_variant=0.5,
            delta=-0.5,
            better="baseline",
            n_seeds=3,
        )
        problems = driver.check_claims([row], 262144)
        assert problems
        assert "asym" in problems[0]


class TestLeaderboardWriting:
    def test_when_the_file_is_new_then_it_is_created_with_markers(
        self, driver, tmp_path
    ) -> None:

        target = tmp_path / "LEADERBOARD.md"
        driver.write_leaderboard(target, "| table |\n")
        text = target.read_text()
        assert "<!-- BEGIN GENERATED: lcqat_ablation.py -->" in text
        assert "| table |" in text

    def test_when_the_file_exists_then_hand_written_prose_survives(
        self, driver, tmp_path
    ) -> None:
        """Re-running must not delete the analysis around the numbers."""

        target = tmp_path / "LEADERBOARD.md"
        target.write_text("# My notes\n\nHand-written analysis.\n")
        driver.write_leaderboard(target, "| table |\n")
        first = target.read_text()
        assert "Hand-written analysis." in first

        # A second run must replace only the generated section.
        driver.write_leaderboard(target, "| new table |\n")
        second = target.read_text()
        assert "Hand-written analysis." in second
        assert "| new table |" in second
        assert "| table |" not in second

    def test_when_written_twice_then_the_section_is_not_duplicated(
        self, driver, tmp_path
    ) -> None:
        """Re-running must be idempotent, not cumulative.

        An earlier signature took both the rows and a pre-rendered table, so it
        rendered twice and the leaderboard carried two copies of every result.
        The single most useful check on a generated section is that running the
        generator again leaves exactly one copy.
        """
        target = tmp_path / "LEADERBOARD.md"
        table = "| experiment | metric |\n|---|---|\n| zz0 | nmse |\n"
        for _ in range(3):
            driver.write_leaderboard(target, table)
        text = target.read_text()
        assert text.count("<!-- BEGIN GENERATED: lcqat_ablation.py -->") == 1
        assert text.count("<!-- END GENERATED: lcqat_ablation.py -->") == 1
        assert text.count("| zz0 | nmse |") == 1

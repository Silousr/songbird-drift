"""Tests for the command line interface.

The CLI is what makes this usable by someone who will not read the source. Each command is
tested end to end on generated data, and the failure modes are tested too -- a tool for
planning experiments has to fail loudly and legibly, not silently produce a number.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from songbird.cli import main
from tests.test_pipeline import make_lab_dataset


class TestManifestCommand:
    def test_builds_a_manifest_from_filenames(self, tmp_path, capsys):
        root = tmp_path / "raw"
        root.mkdir()
        for day in ("2024-05-01", "2024-05-02"):
            (root / f"gr41_{day}T0830.wav").write_bytes(b"")
            (root / f"gr41_{day}T0830.wav.csv").write_text("onset_s,offset_s,label\n")
        out = tmp_path / "manifest.csv"
        code = main(["manifest", "--root", str(root), "--out", str(out),
                     "--pattern", r"(?P<bird>\w+)_(?P<timestamp>[\d\-T]+)\.wav$",
                     "--timestamp-format", "%Y-%m-%dT%H%M"])
        assert code == 0
        assert len(pd.read_csv(out)) == 2

    def test_warns_about_unmatched_files(self, tmp_path, capsys):
        root = tmp_path / "raw"
        root.mkdir()
        (root / "unexpected.wav").write_bytes(b"")
        main(["manifest", "--root", str(root), "--out", str(tmp_path / "m.csv"),
              "--pattern", r"(?P<bird>\w+)_(?P<timestamp>[\d\-T]+)\.wav$",
              "--timestamp-format", "%Y-%m-%dT%H%M"])
        assert "unmatched" in capsys.readouterr().out.lower()


class TestAnalyseCommand:
    def test_writes_results_and_prints_a_summary(self, tmp_path, capsys):
        manifest = make_lab_dataset(tmp_path / "lab")
        out = tmp_path / "results.json"
        code = main(["analyse", "--manifest", str(manifest), "--out", str(out),
                     "--n-pca", "8", "--min-renditions", "10", "--n-boot", "60",
                     "--n-null", "60", "--n-freq-bins", "16"])
        assert code == 0
        payload = json.loads(out.read_text())
        assert "b1" in payload["birds"]
        assert "noise floor" in capsys.readouterr().out.lower()

    def test_reports_a_missing_manifest_clearly(self, tmp_path, capsys):
        code = main(["analyse", "--manifest", str(tmp_path / "nope.csv"),
                     "--out", str(tmp_path / "r.json")])
        assert code != 0
        assert "not found" in capsys.readouterr().err.lower()

    def test_refuses_a_single_day_dataset_with_an_explanation(self, tmp_path, capsys):
        manifest = make_lab_dataset(tmp_path / "lab", days=("2024-05-01",))
        code = main(["analyse", "--manifest", str(manifest),
                     "--out", str(tmp_path / "r.json"), "--n-pca", "8",
                     "--min-renditions", "10", "--n-freq-bins", "16"])
        assert code != 0
        assert "two recording days" in capsys.readouterr().err.lower()


class TestCompareCommand:
    def test_runs_a_group_comparison_from_a_csv(self, tmp_path, capsys):
        path = tmp_path / "drift.csv"
        pd.DataFrame({
            "bird": [f"t{i}" for i in range(6)] + [f"c{i}" for i in range(6)],
            "group": ["treated"] * 6 + ["control"] * 6,
            "drift": [0.30, 0.28, 0.35, 0.31, 0.33, 0.29,
                      0.05, 0.04, 0.06, 0.05, 0.03, 0.05],
        }).to_csv(path, index=False)
        out = tmp_path / "comparison.json"
        code = main(["compare", "--drift", str(path), "--out", str(out)])
        assert code == 0
        assert json.loads(out.read_text())["p_value"] < 0.01

    def test_rejects_a_csv_missing_the_group_column(self, tmp_path, capsys):
        path = tmp_path / "drift.csv"
        pd.DataFrame({"bird": ["a", "b"], "drift": [0.1, 0.2]}).to_csv(path, index=False)
        code = main(["compare", "--drift", str(path), "--out", str(tmp_path / "c.json")])
        assert code != 0
        assert "group" in capsys.readouterr().err.lower()


class TestPlanCommand:
    def test_reports_birds_needed_for_an_effect(self, tmp_path, capsys):
        code = main(["plan", "--effect", "0.2", "--between-bird-sd", "0.05",
                     "--n-simulations", "60", "--n-permutations", "300"])
        assert code == 0
        assert "birds per group" in capsys.readouterr().out.lower()


class TestTopLevel:
    def test_no_command_prints_help_and_fails(self, capsys):
        assert main([]) != 0

    def test_version_is_reported(self, capsys):
        with pytest.raises(SystemExit):
            main(["--version"])

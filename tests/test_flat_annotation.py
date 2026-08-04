"""Tests for the 'one annotation file describes many recordings' pattern.

Common in practice: a lab exports a single CSV (or Raven selection table) whose rows carry
an `audio_file` column, rather than one annotation file per recording. The TweetyNet canary
deposit is shaped this way, and so is anything exported from Raven.

Timestamps come from the audio filename via an explicit regex and strptime format, for the
same reason the manifest loader exists: guessing a convention is how a time axis gets
silently corrupted.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from songbird.ingest.generic import load_flat_annotations

PATTERN = r"(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.wav$"
FORMAT = "%Y_%m_%d_%H_%M_%S"


def make_flat(tmp_path, rows=None):
    root = tmp_path / "audio"
    root.mkdir(parents=True, exist_ok=True)
    rows = rows or [
        ("llb3_0002_2018_04_23_14_18_03.wav", 0.10, 0.20, "1"),
        ("llb3_0002_2018_04_23_14_18_03.wav", 0.30, 0.42, "2"),
        ("llb3_0003_2018_04_24_09_01_10.wav", 0.15, 0.25, "1"),
    ]
    for name in {r[0] for r in rows}:
        (root / name).write_bytes(b"")
    csv = tmp_path / "annotations.csv"
    pd.DataFrame(rows, columns=["audio_file", "onset_s", "offset_s", "label"]).to_csv(
        csv, index=False)
    return csv, root


class TestLoadFlatAnnotations:
    def test_builds_one_row_per_syllable(self, tmp_path):
        csv, root = make_flat(tmp_path)
        table = load_flat_annotations(csv, root, bird="llb3",
                                      timestamp_pattern=PATTERN,
                                      timestamp_format=FORMAT)
        assert len(table) == 3

    def test_parses_timestamps_from_the_audio_filename(self, tmp_path):
        csv, root = make_flat(tmp_path)
        table = load_flat_annotations(csv, root, bird="llb3",
                                      timestamp_pattern=PATTERN, timestamp_format=FORMAT)
        assert dt.date(2018, 4, 23) in set(table["day"])
        assert dt.date(2018, 4, 24) in set(table["day"])

    def test_uses_the_audio_file_as_the_bout(self, tmp_path):
        csv, root = make_flat(tmp_path)
        table = load_flat_annotations(csv, root, bird="llb3",
                                      timestamp_pattern=PATTERN, timestamp_format=FORMAT)
        assert table["audio_file"].nunique() == 2

    def test_resolves_audio_paths_under_the_root(self, tmp_path):
        csv, root = make_flat(tmp_path)
        table = load_flat_annotations(csv, root, bird="llb3",
                                      timestamp_pattern=PATTERN, timestamp_format=FORMAT)
        assert all(p.startswith(str(root)) for p in table["audio_path"])

    def test_finds_audio_in_nested_subdirectories(self, tmp_path):
        csv, root = make_flat(tmp_path)
        nested = root / "annotated"
        nested.mkdir()
        for wav in list(root.glob("*.wav")):
            wav.rename(nested / wav.name)
        table = load_flat_annotations(csv, root, bird="llb3",
                                      timestamp_pattern=PATTERN, timestamp_format=FORMAT)
        assert len(table) == 3

    def test_counts_rows_whose_audio_is_absent(self, tmp_path):
        csv, root = make_flat(tmp_path)
        for wav in root.glob("llb3_0003*"):
            wav.unlink()
        table = load_flat_annotations(csv, root, bird="llb3",
                                      timestamp_pattern=PATTERN, timestamp_format=FORMAT,
                                      on_missing="skip")
        assert len(table) == 2
        assert table.attrs["n_missing_audio"] == 1

    def test_missing_audio_raises_by_default(self, tmp_path):
        csv, root = make_flat(tmp_path)
        for wav in root.glob("llb3_0003*"):
            wav.unlink()
        with pytest.raises(FileNotFoundError):
            load_flat_annotations(csv, root, bird="llb3",
                                  timestamp_pattern=PATTERN, timestamp_format=FORMAT)

    def test_rejects_a_filename_the_pattern_cannot_parse(self, tmp_path):
        csv, root = make_flat(tmp_path, rows=[("no_timestamp_here.wav", 0.1, 0.2, "a")])
        with pytest.raises(ValueError, match="timestamp"):
            load_flat_annotations(csv, root, bird="llb3",
                                  timestamp_pattern=PATTERN, timestamp_format=FORMAT)

    def test_rejects_a_csv_missing_required_columns(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"onset_s": [0.1]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="column"):
            load_flat_annotations(csv, tmp_path, bird="b",
                                  timestamp_pattern=PATTERN, timestamp_format=FORMAT)

    def test_accepts_alternative_column_names(self, tmp_path):
        csv = tmp_path / "alt.csv"
        root = tmp_path / "audio"
        root.mkdir()
        (root / "llb3_0002_2018_04_23_14_18_03.wav").write_bytes(b"")
        pd.DataFrame({"file": ["llb3_0002_2018_04_23_14_18_03.wav"],
                      "start": [0.1], "stop": [0.2], "syllable": ["1"]}).to_csv(
            csv, index=False)
        table = load_flat_annotations(
            csv, root, bird="llb3", timestamp_pattern=PATTERN, timestamp_format=FORMAT,
            columns={"audio_file": "file", "onset_s": "start", "offset_s": "stop",
                     "label": "syllable"})
        assert len(table) == 1

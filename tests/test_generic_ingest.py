"""Tests for ingesting a lab's own recordings.

Design choice under test: ingestion is driven by an explicit **manifest** (a CSV listing
audio file, annotation file, bird, timestamp) rather than by inferring conventions from
filenames. Every lab names files differently, and this project has already been bitten
twice by filename conventions in a single public dataset — a reversed date encoding and a
signed counter that wrapped negative. A manifest makes the mapping explicit and checkable
instead of guessed, and `build_manifest` exists to generate one from filenames when a
convention does hold.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from songbird.ingest.generic import build_manifest, load_from_manifest, load_manifest

CSV = "onset_s,offset_s,label\n0.10,0.20,a\n0.35,0.48,b\n0.60,0.71,a\n"


def make_dataset(root, bird="bird1", stamps=("2024-05-01T08:30:00",)):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, stamp in enumerate(stamps):
        audio = root / f"{bird}_{index}.wav"
        annot = root / f"{bird}_{index}.wav.csv"
        audio.write_bytes(b"")
        annot.write_text(CSV)
        rows.append({"bird": bird, "timestamp": stamp,
                     "audio_path": str(audio), "annot_path": str(annot)})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


class TestLoadManifest:
    def test_reads_required_columns(self, tmp_path):
        manifest = load_manifest(make_dataset(tmp_path / "d"))
        assert {"bird", "timestamp", "audio_path", "annot_path"} <= set(manifest.columns)

    def test_parses_timestamps(self, tmp_path):
        manifest = load_manifest(make_dataset(tmp_path / "d"))
        assert manifest["timestamp"].iloc[0] == pd.Timestamp("2024-05-01 08:30:00")

    def test_rejects_a_manifest_missing_a_required_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame([{"bird": "b", "audio_path": "x"}]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing"):
            load_manifest(path)

    def test_rejects_unparseable_timestamps(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame([{"bird": "b", "timestamp": "not-a-date",
                       "audio_path": "a", "annot_path": "c"}]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="timestamp"):
            load_manifest(path)


class TestLoadFromManifest:
    def test_builds_one_row_per_syllable(self, tmp_path):
        table = load_from_manifest(load_manifest(make_dataset(tmp_path / "d")))
        assert len(table) == 3

    def test_carries_the_canonical_columns(self, tmp_path):
        table = load_from_manifest(load_manifest(make_dataset(tmp_path / "d")))
        assert {"bird", "day", "timestamp", "audio_file", "onset_s", "offset_s",
                "duration_s", "label", "source"} <= set(table.columns)

    def test_derives_day_from_timestamp(self, tmp_path):
        table = load_from_manifest(load_manifest(make_dataset(tmp_path / "d")))
        assert set(table["day"]) == {dt.date(2024, 5, 1)}

    def test_uses_the_audio_file_as_the_bout_identifier(self, tmp_path):
        manifest = make_dataset(tmp_path / "d",
                                stamps=("2024-05-01T08:00:00", "2024-05-01T09:00:00"))
        table = load_from_manifest(load_manifest(manifest))
        assert table["audio_file"].nunique() == 2

    def test_spans_multiple_days_and_birds(self, tmp_path):
        first = make_dataset(tmp_path / "a", bird="b1",
                             stamps=("2024-05-01T08:00:00", "2024-05-02T08:00:00"))
        second = make_dataset(tmp_path / "b", bird="b2", stamps=("2024-05-01T08:00:00",))
        combined = pd.concat([load_manifest(first), load_manifest(second)])
        table = load_from_manifest(combined)
        assert set(table["bird"]) == {"b1", "b2"}
        assert table["day"].nunique() == 2

    def test_carries_an_optional_group_column_through(self, tmp_path):
        # Needed for the treated-vs-control comparison downstream.
        path = make_dataset(tmp_path / "d")
        manifest = load_manifest(path)
        manifest["group"] = "treated"
        table = load_from_manifest(manifest)
        assert set(table["group"]) == {"treated"}

    def test_counts_missing_annotation_files_rather_than_failing(self, tmp_path):
        path = make_dataset(tmp_path / "d")
        manifest = load_manifest(path)
        manifest.loc[0, "annot_path"] = str(tmp_path / "does_not_exist.csv")
        table = load_from_manifest(manifest, on_missing="skip")
        assert len(table) == 0
        assert table.attrs["n_missing_annotations"] == 1

    def test_missing_annotations_raise_by_default(self, tmp_path):
        manifest = load_manifest(make_dataset(tmp_path / "d"))
        manifest.loc[0, "annot_path"] = str(tmp_path / "nope.csv")
        with pytest.raises(FileNotFoundError):
            load_from_manifest(manifest)

    def test_rejects_an_unsupported_annotation_format(self, tmp_path):
        manifest = load_manifest(make_dataset(tmp_path / "d"))
        with pytest.raises(ValueError, match="format"):
            load_from_manifest(manifest, annot_format="not-a-real-format")


class TestBuildManifest:
    def test_extracts_fields_from_filenames(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        for name in ("gr41_2024-05-01T0830.wav", "gr41_2024-05-02T0915.wav"):
            (root / name).write_bytes(b"")
            (root / f"{name}.csv").write_text(CSV)
        manifest = build_manifest(
            root,
            pattern=r"(?P<bird>\w+)_(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{4})\.wav$",
            timestamp_format="%Y-%m-%dT%H%M",
        )
        assert len(manifest) == 2
        assert set(manifest["bird"]) == {"gr41"}
        assert manifest["timestamp"].iloc[0] == pd.Timestamp("2024-05-01 08:30:00")

    def test_reports_files_the_pattern_did_not_match(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        (root / "good_2024-05-01T0830.wav").write_bytes(b"")
        (root / "good_2024-05-01T0830.wav.csv").write_text(CSV)
        (root / "unexpected_name.wav").write_bytes(b"")
        manifest = build_manifest(
            root,
            pattern=r"(?P<bird>\w+)_(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{4})\.wav$",
            timestamp_format="%Y-%m-%dT%H%M",
        )
        assert manifest.attrs["n_unmatched"] == 1

    def test_requires_the_pattern_to_capture_bird_and_timestamp(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        with pytest.raises(ValueError, match="named group"):
            build_manifest(root, pattern=r"(\w+)\.wav$", timestamp_format="%Y")

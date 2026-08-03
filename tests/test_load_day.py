"""Tests for building a canonical syllable table from a day of recordings.

Annotation coverage in this dataset is partial and uneven -- the authors note that 882
``gr41rd51`` audio files carry no annotation. Unannotated audio must therefore be
*counted and reported*, never silently dropped: a loader that quietly ignores it would
make a partially-annotated day look identical to a fully-annotated one.
"""

from __future__ import annotations

import datetime as dt

import pytest

from songbird.ingest.bfsongrepo import DateMismatchError, load_day, load_dataset

CSV_BODY = "onset_s,offset_s,label\n0.7935,0.87703125,i\n1.97978125,2.02009375,a\n"


def make_day(root, bird="gy6or6", day_dir="032212", stems=("gy6or6_baseline_220312_0836.3",)):
    """Create a {bird}/{MMDDYY}/ tree with paired wav + wav.csv files."""
    day = root / bird / day_dir
    day.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (day / f"{stem}.wav").write_bytes(b"")
        (day / f"{stem}.wav.csv").write_text(CSV_BODY)
    return day


class TestLoadDay:
    def test_returns_one_row_per_syllable(self, tmp_path):
        day = make_day(tmp_path)
        table = load_day(day)
        assert len(table) == 2

    def test_attaches_bird_day_and_timestamp_to_every_row(self, tmp_path):
        day = make_day(tmp_path)
        table = load_day(day)
        assert set(table["bird"]) == {"gy6or6"}
        assert set(table["day"]) == {dt.date(2012, 3, 22)}
        assert set(table["timestamp"]) == {dt.datetime(2012, 3, 22, 8, 36)}

    def test_carries_onsets_offsets_and_labels(self, tmp_path):
        day = make_day(tmp_path)
        table = load_day(day).sort_values("onset_s").reset_index(drop=True)
        assert table.loc[0, "label"] == "i"
        assert table.loc[0, "onset_s"] == pytest.approx(0.7935)
        assert table.loc[1, "label"] == "a"

    def test_computes_syllable_duration(self, tmp_path):
        day = make_day(tmp_path)
        table = load_day(day).sort_values("onset_s").reset_index(drop=True)
        assert table.loc[0, "duration_s"] == pytest.approx(0.08353125)

    def test_sorts_by_timestamp_then_onset(self, tmp_path):
        day = make_day(
            tmp_path,
            stems=(
                "gy6or6_baseline_220312_0900.9",
                "gy6or6_baseline_220312_0836.3",
            ),
        )
        table = load_day(day)
        assert list(table["timestamp"]) == sorted(table["timestamp"])

    def test_raises_when_filename_date_contradicts_directory(self, tmp_path):
        day = tmp_path / "gy6or6" / "032312"
        day.mkdir(parents=True)
        (day / "gy6or6_baseline_220312_0836.3.wav.csv").write_text(CSV_BODY)
        with pytest.raises(DateMismatchError):
            load_day(day)

    def test_counts_unannotated_audio_rather_than_ignoring_it(self, tmp_path):
        day = make_day(tmp_path)
        (day / "gy6or6_baseline_220312_0901.7.wav").write_bytes(b"")
        table = load_day(day)
        assert table.attrs["n_audio_files"] == 2
        assert table.attrs["n_annotated_files"] == 1
        assert table.attrs["n_unannotated_files"] == 1

    def test_reports_full_coverage_when_every_file_is_annotated(self, tmp_path):
        day = make_day(tmp_path)
        assert load_day(day).attrs["n_unannotated_files"] == 0

    def test_empty_day_yields_empty_table_with_expected_columns(self, tmp_path):
        day = tmp_path / "gy6or6" / "032212"
        day.mkdir(parents=True)
        table = load_day(day)
        assert len(table) == 0
        assert {"bird", "day", "timestamp", "onset_s", "offset_s", "label"} <= set(
            table.columns
        )


class TestLoadDataset:
    def test_combines_multiple_days_and_birds(self, tmp_path):
        make_day(tmp_path, "gy6or6", "032212", ("gy6or6_baseline_220312_0836.3",))
        make_day(tmp_path, "gy6or6", "032312", ("gy6or6_baseline_230312_0836.4",))
        make_day(tmp_path, "or60yw70", "092712", ("or60yw70_270912_0726.1179",))
        table = load_dataset(tmp_path)
        assert set(table["bird"]) == {"gy6or6", "or60yw70"}
        assert table["day"].nunique() == 3
        assert len(table) == 6

    def test_aggregates_coverage_across_days(self, tmp_path):
        day = make_day(tmp_path, "gy6or6", "032212")
        (day / "gy6or6_baseline_220312_0901.7.wav").write_bytes(b"")
        make_day(tmp_path, "gy6or6", "032312", ("gy6or6_baseline_230312_0836.4",))
        table = load_dataset(tmp_path)
        assert table.attrs["n_audio_files"] == 3
        assert table.attrs["n_unannotated_files"] == 1

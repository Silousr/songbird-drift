"""Tests for Bengalese Finch Song Repository ingestion.

Filenames in this dataset are a known trap: the *directory* date is MMDDYY while the
*filename* date is DDMMYY. They encode the same day, reversed. Every example string in
this file is a real path from figshare article 4805749.
"""

from __future__ import annotations

import datetime as dt

import pytest

from songbird.ingest.bfsongrepo import (
    DateMismatchError,
    parse_day_dir,
    parse_recording_filename,
    read_annotation_csv,
)


class TestParseRecordingFilename:
    def test_extracts_bird_and_timestamp_from_baseline_filename(self):
        rec = parse_recording_filename("gy6or6_baseline_220312_0836.3.wav")
        assert rec.bird == "gy6or6"
        assert rec.timestamp == dt.datetime(2012, 3, 22, 8, 36)
        assert rec.serial == 3

    def test_handles_filename_with_no_template_field(self):
        rec = parse_recording_filename("or60yw70_270912_0726.1179.wav")
        assert rec.bird == "or60yw70"
        assert rec.timestamp == dt.datetime(2012, 9, 27, 7, 26)
        assert rec.serial == 1179

    def test_handles_double_underscore_template_field(self):
        # gr41rd51 carries an evTAF template name containing underscores.
        rec = parse_recording_filename(
            "gr41rd51__3part_SYLc_th4191_belowhits_220612_0712.13495.wav"
        )
        assert rec.bird == "gr41rd51"
        assert rec.timestamp == dt.datetime(2012, 6, 22, 7, 12)
        assert rec.template == "_3part_SYLc_th4191_belowhits"

    def test_records_template_as_none_when_absent(self):
        rec = parse_recording_filename("bl26lb16_190412_0721.20144.wav")
        assert rec.template is None

    def test_parses_ddmmyy_not_mmddyy(self):
        # 27/09/12 is 27 September. Read as MMDDYY it would be an invalid month.
        rec = parse_recording_filename("or60yw70_270912_0726.1179.wav")
        assert (rec.timestamp.month, rec.timestamp.day) == (9, 27)

    def test_accepts_csv_suffix(self):
        rec = parse_recording_filename("gy6or6_baseline_260312_0810.3440.wav.csv")
        assert rec.timestamp == dt.datetime(2012, 3, 26, 8, 10)

    def test_rejects_unparseable_name(self):
        with pytest.raises(ValueError):
            parse_recording_filename("not_a_song_file.txt")


class TestParseDayDir:
    def test_parses_mmddyy_directory_name(self):
        assert parse_day_dir("032212") == dt.date(2012, 3, 22)

    def test_parses_october_directory(self):
        assert parse_day_dir("100112") == dt.date(2012, 10, 1)

    def test_rejects_invalid_month(self):
        with pytest.raises(ValueError):
            parse_day_dir("991212")


class TestDateCrossCheck:
    def test_directory_and_filename_dates_agree(self):
        rec = parse_recording_filename(
            "gy6or6_baseline_220312_0836.3.wav", day_dir="032212"
        )
        assert rec.timestamp.date() == dt.date(2012, 3, 22)

    def test_raises_when_directory_contradicts_filename(self):
        # Silently accepting this is how a time axis gets corrupted.
        with pytest.raises(DateMismatchError):
            parse_recording_filename(
                "gy6or6_baseline_220312_0836.3.wav", day_dir="032312"
            )


class TestReadAnnotationCsv:
    def test_reads_onsets_offsets_and_labels(self, tmp_path):
        csv = tmp_path / "song.wav.csv"
        csv.write_text(
            "onset_s,offset_s,label\n"
            "0.7935,0.87703125,i\n"
            "0.9989687500000006,1.08428125,i\n"
            "1.97978125,2.02009375,a\n"
        )
        syllables = read_annotation_csv(csv)
        assert len(syllables) == 3
        assert syllables[0].onset_s == pytest.approx(0.7935)
        assert syllables[0].offset_s == pytest.approx(0.87703125)
        assert [s.label for s in syllables] == ["i", "i", "a"]

    def test_preserves_label_as_string_not_number(self, tmp_path):
        # Labels are arbitrary characters; some datasets use digits. Coercing to int
        # would collide with genuinely numeric label schemes elsewhere.
        csv = tmp_path / "song.wav.csv"
        csv.write_text("onset_s,offset_s,label\n0.1,0.2,0\n")
        assert read_annotation_csv(csv)[0].label == "0"

    def test_rejects_offset_before_onset(self, tmp_path):
        csv = tmp_path / "bad.wav.csv"
        csv.write_text("onset_s,offset_s,label\n0.5,0.4,a\n")
        with pytest.raises(ValueError):
            read_annotation_csv(csv)

    def test_returns_empty_list_for_header_only_file(self, tmp_path):
        csv = tmp_path / "empty.wav.csv"
        csv.write_text("onset_s,offset_s,label\n")
        assert read_annotation_csv(csv) == []

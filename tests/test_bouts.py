"""Tests for deriving bouts from rendition timestamps.

Some datasets give a timestamp per rendition rather than one file per bout. Since every
statistic in this toolkit treats the bout as its sampling unit, those recordings need bouts
reconstructed before anything else can run. A song bout is a run of renditions seconds
apart, separated from the next by a much longer silence.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.ingest.bouts import bouts_from_timestamps, suggest_gap_seconds


class TestBoutsFromTimestamps:
    def test_splits_on_a_long_gap(self):
        # three renditions, a five-minute silence, then three more
        t = np.array([0.0, 2.0, 4.0, 304.0, 306.0, 308.0]) / 86400
        assert list(bouts_from_timestamps(t, gap_s=60)) == [0, 0, 0, 1, 1, 1]

    def test_keeps_a_short_gap_within_one_bout(self):
        t = np.array([0.0, 2.0, 40.0, 42.0]) / 86400
        assert len(np.unique(bouts_from_timestamps(t, gap_s=60))) == 1

    def test_a_single_rendition_is_its_own_bout(self):
        assert list(bouts_from_timestamps(np.array([0.0]), gap_s=60)) == [0]

    def test_handles_unsorted_input_without_reordering_the_output(self):
        t = np.array([304.0, 0.0, 2.0, 306.0]) / 86400
        labels = bouts_from_timestamps(t, gap_s=60)
        # positions 1,2 are the early pair; 0,3 the late pair
        assert labels[1] == labels[2] and labels[0] == labels[3]
        assert labels[1] != labels[0]

    def test_larger_gap_threshold_yields_fewer_bouts(self):
        rng = np.random.default_rng(0)
        t = np.cumsum(rng.exponential(20, 500)) / 86400
        assert (len(np.unique(bouts_from_timestamps(t, gap_s=120)))
                <= len(np.unique(bouts_from_timestamps(t, gap_s=30))))

    def test_rejects_a_nonpositive_gap(self):
        with pytest.raises(ValueError):
            bouts_from_timestamps(np.array([0.0, 1.0]), gap_s=0)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            bouts_from_timestamps(np.array([]), gap_s=60)

    def test_labels_are_contiguous_from_zero(self):
        t = np.array([0.0, 2.0, 304.0, 700.0]) / 86400
        assert set(bouts_from_timestamps(t, gap_s=60)) == {0, 1, 2}


class TestSuggestGapSeconds:
    def test_finds_the_silence_between_bouts(self):
        # renditions 2 s apart in runs of 20, bouts 10 min apart
        rng = np.random.default_rng(0)
        times = []
        clock = 0.0
        for _ in range(30):
            for _ in range(20):
                clock += 2.0 + rng.normal(0, 0.2)
                times.append(clock)
            clock += 600.0
        gap = suggest_gap_seconds(np.array(times) / 86400)
        assert 10 < gap < 600

    def test_returns_a_positive_value_for_irregular_data(self):
        rng = np.random.default_rng(1)
        t = np.cumsum(rng.exponential(30, 2000)) / 86400
        assert suggest_gap_seconds(t) > 0

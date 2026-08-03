"""Tests for scoring recovered segmentation against hand-labelled ground truth.

Matching is strictly one-to-one. Without that constraint a segmenter that fires twice
on every syllable would score perfect recall, which is exactly the failure mode these
metrics exist to catch.
"""

from __future__ import annotations

import pytest

from songbird.metrics.segmentation import (
    boundary_scores,
    match_segments,
    segment_scores,
)

TRUE = [(0.0, 0.1), (0.5, 0.7), (1.0, 1.4)]


class TestMatchSegments:
    def test_matches_identical_segments_pairwise(self):
        assert match_segments(TRUE, TRUE, tolerance_s=0.01) == [(0, 0), (1, 1), (2, 2)]

    def test_matches_within_tolerance(self):
        pred = [(0.005, 0.105), (0.495, 0.695), (1.004, 1.396)]
        assert len(match_segments(TRUE, pred, tolerance_s=0.01)) == 3

    def test_rejects_when_one_boundary_exceeds_tolerance(self):
        # Onset is fine, offset is 20 ms out with a 10 ms tolerance.
        assert match_segments([(0.0, 0.1)], [(0.0, 0.12)], tolerance_s=0.01) == []

    def test_is_one_to_one(self):
        # Two predictions both plausibly match the single true segment.
        pred = [(0.0, 0.1), (0.001, 0.101)]
        assert len(match_segments([(0.0, 0.1)], pred, tolerance_s=0.01)) == 1

    def test_returns_empty_when_nothing_predicted(self):
        assert match_segments(TRUE, [], tolerance_s=0.01) == []

    def test_matches_in_time_order_not_index_order(self):
        pred = [(1.0, 1.4), (0.0, 0.1)]
        assert sorted(match_segments(TRUE, pred, tolerance_s=0.01)) == [(0, 1), (2, 0)]


class TestSegmentScores:
    def test_perfect_segmentation_scores_one(self):
        scores = segment_scores(TRUE, TRUE, tolerance_s=0.01)
        assert scores.precision == 1.0
        assert scores.recall == 1.0
        assert scores.f1 == 1.0

    def test_missed_syllables_reduce_recall_only(self):
        scores = segment_scores(TRUE, TRUE[:2], tolerance_s=0.01)
        assert scores.precision == 1.0
        assert scores.recall == pytest.approx(2 / 3)

    def test_spurious_segments_reduce_precision_only(self):
        scores = segment_scores(TRUE, TRUE + [(2.0, 2.2)], tolerance_s=0.01)
        assert scores.recall == 1.0
        assert scores.precision == pytest.approx(3 / 4)

    def test_no_predictions_scores_zero(self):
        scores = segment_scores(TRUE, [], tolerance_s=0.01)
        assert scores.recall == 0.0
        assert scores.precision == 0.0
        assert scores.f1 == 0.0

    def test_both_empty_is_vacuously_perfect(self):
        scores = segment_scores([], [], tolerance_s=0.01)
        assert scores.f1 == 1.0

    def test_reports_counts(self):
        scores = segment_scores(TRUE, TRUE[:2] + [(5.0, 5.1)], tolerance_s=0.01)
        assert (scores.n_true, scores.n_pred, scores.n_matched) == (3, 3, 2)


class TestBoundaryScores:
    def test_scores_onsets_and_offsets_separately_from_segments(self):
        # Every onset is right but every offset is 20 ms long: onset recall stays
        # perfect while whole-segment recall collapses.
        pred = [(0.0, 0.12), (0.5, 0.72), (1.0, 1.42)]
        assert boundary_scores(TRUE, pred, tolerance_s=0.01).onset.recall == 1.0
        assert segment_scores(TRUE, pred, tolerance_s=0.01).recall == 0.0

    def test_offset_scores_degrade_when_offsets_drift(self):
        pred = [(0.0, 0.12), (0.5, 0.72), (1.0, 1.42)]
        assert boundary_scores(TRUE, pred, tolerance_s=0.01).offset.recall == 0.0

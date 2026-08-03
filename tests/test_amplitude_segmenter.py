"""Tests for amplitude-threshold segmentation.

Synthetic tone bursts with exactly known onsets/offsets, so a failure points at the
segmenter rather than at ambiguous real audio.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.segment.amplitude import AmplitudeSegmenter

SR = 32_000


def burst_signal(bursts, duration_s=2.0, sr=SR, freq=3000.0, amplitude=1.0):
    """Build a signal that is silent except for tone bursts at the given intervals."""
    t = np.arange(int(duration_s * sr)) / sr
    y = np.zeros_like(t)
    for onset, offset in bursts:
        mask = (t >= onset) & (t < offset)
        y[mask] = amplitude * np.sin(2 * np.pi * freq * t[mask])
    return y


@pytest.fixture
def segmenter():
    return AmplitudeSegmenter(
        threshold=0.05, min_syllable_s=0.02, min_silence_s=0.02, smooth_ms=2.0
    )


class TestAmplitudeSegmenter:
    def test_finds_single_burst(self, segmenter):
        y = burst_signal([(0.5, 0.7)])
        assert len(segmenter.segment(y, SR)) == 1

    def test_recovers_burst_boundaries_within_10ms(self, segmenter):
        y = burst_signal([(0.5, 0.7)])
        (onset, offset), = segmenter.segment(y, SR)
        assert onset == pytest.approx(0.5, abs=0.01)
        assert offset == pytest.approx(0.7, abs=0.01)

    def test_separates_bursts_split_by_a_long_silence(self, segmenter):
        y = burst_signal([(0.2, 0.4), (0.8, 1.0)])
        assert len(segmenter.segment(y, SR)) == 2

    def test_merges_bursts_split_by_less_than_min_silence(self, segmenter):
        # 5 ms gap, below the 20 ms min_silence_s.
        y = burst_signal([(0.2, 0.4), (0.405, 0.6)])
        assert len(segmenter.segment(y, SR)) == 1

    def test_drops_bursts_shorter_than_min_syllable(self, segmenter):
        y = burst_signal([(0.5, 0.505)])  # 5 ms, below the 20 ms floor
        assert segmenter.segment(y, SR) == []

    def test_returns_nothing_for_silence(self, segmenter):
        assert segmenter.segment(np.zeros(SR), SR) == []

    def test_ignores_low_amplitude_noise(self, segmenter):
        rng = np.random.default_rng(0)
        y = burst_signal([(0.5, 0.7)]) + 0.005 * rng.standard_normal(int(2.0 * SR))
        assert len(segmenter.segment(y, SR)) == 1

    def test_segments_are_ordered_and_non_overlapping(self, segmenter):
        y = burst_signal([(0.1, 0.2), (0.4, 0.5), (0.9, 1.1)])
        segments = segmenter.segment(y, SR)
        assert segments == sorted(segments)
        assert all(a[1] <= b[0] for a, b in zip(segments, segments[1:]))

    def test_handles_burst_running_to_end_of_signal(self, segmenter):
        y = burst_signal([(1.5, 2.0)], duration_s=2.0)
        segments = segmenter.segment(y, SR)
        assert len(segments) == 1
        assert segments[0][1] == pytest.approx(2.0, abs=0.01)

    def test_threshold_is_relative_to_signal_scale(self, segmenter):
        # Halving the signal amplitude must not change what counts as a syllable;
        # otherwise recording gain silently determines the segmentation.
        loud = segmenter.segment(burst_signal([(0.5, 0.7)], amplitude=1.0), SR)
        quiet = segmenter.segment(burst_signal([(0.5, 0.7)], amplitude=0.5), SR)
        assert len(loud) == len(quiet) == 1
        assert quiet[0][0] == pytest.approx(loud[0][0], abs=0.005)

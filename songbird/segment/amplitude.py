"""Amplitude-threshold syllable segmentation.

This is the classical approach (evsonganaly, Sound Analysis Pro, AVA): bandpass, rectify,
smooth, threshold, then merge across short silences and drop short blips. It is the
baseline the learned segmenter must beat, and it is what the hand annotations in the
Bengalese Finch repository were originally drawn with.

The threshold is applied to a **peak-normalised** envelope so that recording gain does not
silently determine the segmentation -- two recordings of the same bird at different input
levels must segment the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

__all__ = ["AmplitudeSegmenter", "DEFAULT_THRESHOLD_GRID", "tune_threshold"]

Segment = tuple[float, float]

#: Threshold grid spanning the range that fits real birds. On the Bengalese finch
#: repository the per-bird optimum runs from 0.004 to 0.010.
DEFAULT_THRESHOLD_GRID = (0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05)


@dataclass(frozen=True)
class AmplitudeSegmenter:
    """Segment syllables by thresholding a smoothed amplitude envelope.

    Parameters
    ----------
    threshold:
        Fraction of the peak envelope above which audio counts as vocalisation.
    min_syllable_s:
        Segments shorter than this are discarded as blips.
    min_silence_s:
        Gaps shorter than this are bridged, merging the segments either side.
    smooth_ms:
        Width of the moving-average window applied to the rectified signal.
    bandpass_hz:
        Band to restrict to before enveloping, or ``None`` to skip filtering.
    """

    threshold: float = 0.05
    min_syllable_s: float = 0.02
    min_silence_s: float = 0.02
    smooth_ms: float = 2.0
    bandpass_hz: tuple[float, float] | None = (500.0, 10_000.0)

    def envelope(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Return the peak-normalised amplitude envelope."""
        y = np.asarray(y, dtype=float)
        if y.ndim > 1:
            y = y.mean(axis=tuple(range(1, y.ndim)))

        if self.bandpass_hz is not None:
            low, high = self.bandpass_hz
            nyquist = sr / 2
            high = min(high, nyquist * 0.999)
            if 0 < low < high:
                sos = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
                y = sosfiltfilt(sos, y)

        width = max(int(round(self.smooth_ms * 1e-3 * sr)), 1)
        env = uniform_filter1d(np.abs(y), size=width, mode="nearest")

        peak = float(env.max()) if env.size else 0.0
        return env / peak if peak > 0 else env

    def segment(self, y: np.ndarray, sr: int) -> list[Segment]:
        """Return ordered, non-overlapping ``(onset_s, offset_s)`` pairs."""
        env = self.envelope(y, sr)
        if env.size == 0 or not np.any(env >= self.threshold):
            return []

        above = (env >= self.threshold).astype(np.int8)
        edges = np.diff(np.concatenate(([0], above, [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)  # exclusive

        segments = [(start / sr, end / sr) for start, end in zip(starts, ends)]

        # Bridge short silences first, then drop what is still too brief -- the reverse
        # order would delete blips that are really one syllable split by a dip.
        merged: list[Segment] = []
        for onset, offset in segments:
            if merged and onset - merged[-1][1] < self.min_silence_s:
                merged[-1] = (merged[-1][0], offset)
            else:
                merged.append((onset, offset))

        return [seg for seg in merged if seg[1] - seg[0] >= self.min_syllable_s]


def tune_threshold(
    examples: list[tuple[np.ndarray, int, list[Segment]]],
    candidates=DEFAULT_THRESHOLD_GRID,
    tolerance_s: float = 0.01,
    **segmenter_params,
) -> tuple[float, float]:
    """Pick the amplitude threshold that best reproduces annotated segments.

    ``examples`` is a list of ``(audio, sample_rate, true_segments)``. Returns the
    ``(threshold, f1)`` of the best candidate.

    Fit this **per bird**, on a tuning subset held apart from whatever you go on to
    report. Recording gain, microphone placement and cage noise all differ between
    birds, and on the Bengalese finch repository a single global threshold costs the
    worst-fitting bird ~0.18 F1 relative to its own optimum.

    A caution for longitudinal work: if recording conditions shift across days, a
    threshold fitted once will segment later days differently, and that drift in
    segmentation quality is indistinguishable from drift in the song itself unless it is
    measured. Re-fit per day, or report segmentation quality per day alongside any
    drift estimate.
    """
    from songbird.metrics.segmentation import segment_scores

    if not examples:
        raise ValueError("need at least one (audio, sample_rate, segments) example")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("need at least one candidate threshold")

    best_threshold, best_f1 = candidates[0], -1.0
    for threshold in candidates:
        segmenter = AmplitudeSegmenter(threshold=threshold, **segmenter_params)
        n_true = n_pred = n_matched = 0
        for audio, sample_rate, true_segments in examples:
            predicted = segmenter.segment(audio, sample_rate)
            scores = segment_scores(true_segments, predicted, tolerance_s)
            n_true += scores.n_true
            n_pred += scores.n_pred
            n_matched += scores.n_matched
        precision = n_matched / n_pred if n_pred else 0.0
        recall = n_matched / n_true if n_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold, best_f1

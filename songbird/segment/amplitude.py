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

__all__ = ["AmplitudeSegmenter"]

Segment = tuple[float, float]


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

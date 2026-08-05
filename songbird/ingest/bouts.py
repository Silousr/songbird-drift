"""Reconstruct song bouts from per-rendition timestamps.

Some datasets store one timestamp per rendition rather than one file per bout — the Zai
et al. deafening deposit is one, and any lab recording continuously rather than in
triggered files will be another. Every statistic here treats the **bout** as its sampling
unit, so bouts have to be recovered before anything else can run.

A song bout is a run of renditions a few seconds apart, separated from the next run by a
much longer silence. The split is therefore a threshold on the inter-rendition interval.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bouts_from_timestamps", "suggest_gap_seconds"]


def bouts_from_timestamps(timestamps: np.ndarray, gap_s: float = 60.0) -> np.ndarray:
    """Label each rendition with a bout index, splitting on gaps longer than ``gap_s``.

    ``timestamps`` are in days (the MATLAB/serial-date convention these deposits use).
    Input need not be sorted; labels are returned in the caller's original order.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.size == 0:
        raise ValueError("no timestamps given")
    if gap_s <= 0:
        raise ValueError(f"gap_s must be positive, got {gap_s}")

    order = np.argsort(timestamps, kind="stable")
    gaps = np.diff(timestamps[order]) * 86_400.0
    sorted_labels = np.concatenate([[0], np.cumsum(gaps > gap_s)])

    labels = np.empty(len(timestamps), dtype=int)
    labels[order] = sorted_labels
    return labels


def suggest_gap_seconds(
    timestamps: np.ndarray, low_percentile: float = 50.0, high_percentile: float = 99.0
) -> float:
    """A gap threshold sitting between within-bout and between-bout intervals.

    Uses the geometric mean of two percentiles of the interval distribution, which lands
    in the trough between the two scales without assuming either is known. Report the
    value used and check bout sizes look sane; this is a heuristic, not a measurement.
    """
    timestamps = np.sort(np.asarray(timestamps, dtype=float))
    if timestamps.size < 3:
        raise ValueError("need at least 3 timestamps to suggest a gap")
    intervals = np.diff(timestamps) * 86_400.0
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        raise ValueError("all timestamps are identical")
    low = np.percentile(intervals, low_percentile)
    high = np.percentile(intervals, high_percentile)
    return float(np.sqrt(max(low, 1e-6) * max(high, 1e-6)))

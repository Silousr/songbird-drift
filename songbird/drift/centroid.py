"""Distance between the acoustic centroids of a syllable type on two days.

The naive quantity -- squared distance between two sample means -- is a biased estimate of
the squared distance between the underlying distribution means:

    E[||x̄a - x̄b||²] = ||μa - μb||² + tr(Σa)/na + tr(Σb)/nb

Both correction terms are strictly positive, so the naive estimate reports drift between
two samples drawn from **the same** distribution, and reports more of it when samples are
small. For this project that failure is not academic: recording volume varies several-fold
between days (39 to 248 songs/day for one bird in this dataset), so a naive metric would
manufacture the most "drift" on exactly the quietest days.

Subtracting the two variance terms removes the bias.
"""

from __future__ import annotations

import numpy as np

__all__ = ["naive_squared_centroid_distance", "unbiased_squared_centroid_distance"]


def _check(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected 2-D arrays of shape (n_renditions, n_dimensions)")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"dimension mismatch: {a.shape[1]} vs {b.shape[1]}"
        )
    return a, b


def naive_squared_centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Squared distance between two sample means. **Biased upward** -- see module docstring.

    Provided so the bias can be measured and reported rather than merely asserted; it is
    not the estimator to build a drift metric on.
    """
    a, b = _check(a, b)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both samples must be non-empty")
    return float(np.sum((a.mean(axis=0) - b.mean(axis=0)) ** 2))


def _bout_means(points: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Collapse renditions to one row per bout.

    Each bout counts once regardless of how many syllables it contains, so a single long
    bout cannot dominate the day's estimate.
    """
    if len(points) != len(groups):
        raise ValueError(
            f"group labels must match sample length: {len(points)} vs {len(groups)}"
        )
    keys = np.unique(groups)
    return np.stack([points[groups == key].mean(axis=0) for key in keys])


def unbiased_squared_centroid_distance(
    a: np.ndarray,
    b: np.ndarray,
    groups_a: np.ndarray | None = None,
    groups_b: np.ndarray | None = None,
) -> float:
    """Unbiased estimate of the squared distance between the two distribution means.

    **Pass ``groups_a``/``groups_b`` whenever renditions are clustered in song bouts.**
    The correction assumes the sampling unit is independent. Applied per rendition to
    bout-clustered data it under-corrects by the design effect -- measured at ~10x on the
    test fixture -- leaving a large positive residual that never goes negative. That is a
    drift signal manufactured entirely out of within-bout correlation, and songbird data
    is always bout-clustered. Supplying bout labels makes the bout the sampling unit and
    restores unbiasedness.

    **May return a negative value, and must not be clipped.** Under the null of no drift
    the estimate is negative roughly half the time; clipping at zero would restore exactly
    the upward bias this correction removes, and would turn a symmetric null distribution
    into a one-sided pile-up that no longer supports a calibrated test.
    """
    a, b = _check(a, b)

    if groups_a is not None or groups_b is not None:
        if groups_a is None or groups_b is None:
            raise ValueError("supply bout labels for both samples, or for neither")
        a, b = _bout_means(a, np.asarray(groups_a)), _bout_means(b, np.asarray(groups_b))
        unit = "bouts"
    else:
        unit = "renditions"

    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            f"need at least 2 {unit} per sample to estimate variance; "
            f"got {len(a)} and {len(b)}"
        )

    naive = np.sum((a.mean(axis=0) - b.mean(axis=0)) ** 2)
    # ddof=1: the sample variance must itself be unbiased, or the correction is wrong.
    correction = a.var(axis=0, ddof=1).sum() / len(a) + b.var(axis=0, ddof=1).sum() / len(b)
    return float(naive - correction)

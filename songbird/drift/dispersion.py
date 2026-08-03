"""Dispersion drift — change in rendition-to-rendition variability.

The centroid metric asks whether a syllable type *moved*. This asks whether it got
*sloppier*. Both are forms of song change and they are not redundant: a manipulation could
destabilise renditions without shifting their average, and a centroid distance would read
approximately zero throughout. That blind spot is why this exists.

The statistic is the log ratio of total variance::

    log( tr Var(b) / tr Var(a) )

Positive means day ``b`` is more variable. It is scale-free (so it does not inherit the
embedding's arbitrary units), antisymmetric under swapping the days, and unaffected by a
pure translation — the last property being what makes it genuinely complementary to the
centroid metric rather than a re-expression of it.

Unlike the centroid distance there is no closed-form bias correction here, because the
quantity of interest is total rendition-to-rendition variance, which includes the
within-bout component that a bout-mean collapse would discard. Uncertainty therefore comes
entirely from a **bout-level bootstrap**, and the null from splitting a single day's bouts
in half. Both keep the bout as the sampling unit, for the same reason as everywhere else:
renditions inside a bout are correlated, and resampling them independently gives intervals
far too narrow.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bootstrap_dispersion_ci",
    "log_variance_ratio",
    "split_half_dispersion_null",
]


def _total_variance(points: np.ndarray) -> float:
    if len(points) < 2:
        raise ValueError(
            f"need at least 2 renditions to estimate variance; got {len(points)}"
        )
    return float(points.var(axis=0, ddof=1).sum())


def log_variance_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """``log(tr Var(b) / tr Var(a))``. Positive when ``b`` is the more variable day."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected 2-D arrays of shape (n_renditions, n_dimensions)")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: {a.shape[1]} vs {b.shape[1]}")

    variance_a, variance_b = _total_variance(a), _total_variance(b)
    if variance_a <= 0 or variance_b <= 0:
        raise ValueError("total variance must be positive in both samples")
    return float(np.log(variance_b / variance_a))


def _grouped(groups: np.ndarray) -> dict:
    index = {}
    for position, group in enumerate(groups):
        index.setdefault(group, []).append(position)
    return {key: np.asarray(value) for key, value in index.items()}


def bootstrap_dispersion_ci(
    a: np.ndarray,
    groups_a: np.ndarray,
    b: np.ndarray,
    groups_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(estimate, low, high)`` for the log variance ratio, resampling bouts."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    groups_a, groups_b = np.asarray(groups_a), np.asarray(groups_b)
    if len(a) != len(groups_a) or len(b) != len(groups_b):
        raise ValueError(
            f"group labels must match sample lengths: {len(a)}/{len(groups_a)} and "
            f"{len(b)}/{len(groups_b)}"
        )

    estimate = log_variance_ratio(a, b)
    by_a, by_b = _grouped(groups_a), _grouped(groups_b)
    keys_a, keys_b = np.array(list(by_a)), np.array(list(by_b))
    rng = np.random.default_rng(seed)

    draws = []
    for _ in range(n_boot):
        index_a = np.concatenate(
            [by_a[k] for k in rng.choice(keys_a, len(keys_a), replace=True)]
        )
        index_b = np.concatenate(
            [by_b[k] for k in rng.choice(keys_b, len(keys_b), replace=True)]
        )
        if len(index_a) < 2 or len(index_b) < 2:
            continue
        try:
            draws.append(log_variance_ratio(a[index_a], b[index_b]))
        except ValueError:
            continue

    if not draws:
        raise ValueError("no valid bootstrap resamples")
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(estimate), float(low), float(high)


def split_half_dispersion_null(
    points: np.ndarray,
    groups: np.ndarray,
    n_draws: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Null distribution of the log variance ratio, splitting one day's bouts in half."""
    points = np.asarray(points, dtype=float)
    groups = np.asarray(groups)
    if len(points) != len(groups):
        raise ValueError(
            f"group labels must match sample length: {len(points)} vs {len(groups)}"
        )

    by_group = _grouped(groups)
    keys = np.array(list(by_group))
    if len(keys) < 4:
        raise ValueError(
            f"need at least 4 bouts to split a day into two estimable halves; "
            f"got {len(keys)}"
        )

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_draws):
        shuffled = rng.permutation(keys)
        half = len(shuffled) // 2
        left = np.concatenate([by_group[k] for k in shuffled[:half]])
        right = np.concatenate([by_group[k] for k in shuffled[half:]])
        if len(left) < 2 or len(right) < 2:
            continue
        try:
            values.append(log_variance_ratio(points[left], points[right]))
        except ValueError:
            continue
    return np.asarray(values)

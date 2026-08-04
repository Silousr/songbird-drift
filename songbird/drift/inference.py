"""Uncertainty for drift estimates, and the within-day null they are judged against.

Two design commitments, both of which change the numbers materially:

**The resampling unit is the song bout, not the syllable.** Syllables produced within one
bout share the bird's state and the bout's recording conditions, so they are not
independent draws. Resampling syllables would treat a few hundred correlated renditions as
a few hundred independent observations and produce confidence intervals far too narrow --
in the test suite, roughly a third the width they should be.

**The null is built by splitting a single day in half.** The noise floor is defined as what
the drift statistic reads when there is genuinely no drift, computed with the *same*
estimator on the *same* kind of data. Halves are split by bout, never by syllable: sharing
a bout across both halves would leak correlated renditions into both sides and bias the
null toward zero, making real drift look more significant than it is.
"""

from __future__ import annotations

import numpy as np

from songbird.drift.centroid import unbiased_squared_centroid_distance

__all__ = ["bootstrap_drift_ci", "split_half_null"]


def _grouped_indices(groups: np.ndarray) -> dict:
    order = {}
    for index, group in enumerate(groups):
        order.setdefault(group, []).append(index)
    return {key: np.asarray(value) for key, value in order.items()}


def _resample(by_group: dict, keys: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    drawn = rng.choice(keys, size=len(keys), replace=True)
    return np.concatenate([by_group[key] for key in drawn])


def bootstrap_drift_ci(
    a: np.ndarray,
    groups_a: np.ndarray,
    b: np.ndarray,
    groups_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(estimate, low, high)`` for the unbiased squared centroid distance.

    Bouts are resampled with replacement independently on each side. The interval is the
    percentile interval of the bootstrap distribution.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    groups_a, groups_b = np.asarray(groups_a), np.asarray(groups_b)
    if len(a) != len(groups_a) or len(b) != len(groups_b):
        raise ValueError(
            f"group labels must match sample lengths: {len(a)}/{len(groups_a)} and "
            f"{len(b)}/{len(groups_b)}"
        )

    estimate = unbiased_squared_centroid_distance(a, b, groups_a, groups_b)

    by_a, by_b = _grouped_indices(groups_a), _grouped_indices(groups_b)
    keys_a, keys_b = np.array(list(by_a)), np.array(list(by_b))
    rng = np.random.default_rng(seed)

    draws = []
    for _ in range(n_boot):
        index_a, index_b = _resample(by_a, keys_a, rng), _resample(by_b, keys_b, rng)
        # Distinct bouts, not renditions: the bout-level estimator needs two of them to
        # form a variance, and resampling with replacement can draw one bout repeatedly.
        if (len(np.unique(groups_a[index_a])) < 2
                or len(np.unique(groups_b[index_b])) < 2):
            continue
        draws.append(unbiased_squared_centroid_distance(
            a[index_a], b[index_b], groups_a[index_a], groups_b[index_b]
        ))

    if not draws:
        raise ValueError("no valid bootstrap resamples; too few renditions")
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(estimate), float(low), float(high)


def split_half_null(
    points: np.ndarray,
    groups: np.ndarray,
    n_draws: int = 200,
    seed: int = 0,
    return_splits: bool = False,
):
    """Distribution of the drift statistic under no drift, by splitting bouts in half.

    Each draw partitions the bouts of one day into two disjoint halves and applies the
    same unbiased estimator. The resulting spread is the noise floor: drift between two
    real days is only evidence of change if it lies outside this.

    With ``return_splits=True`` returns the ``(left_bouts, right_bouts)`` pairs instead,
    for verifying disjointness.
    """
    points = np.asarray(points, dtype=float)
    groups = np.asarray(groups)
    if len(points) != len(groups):
        raise ValueError(
            f"group labels must match sample length: {len(points)} vs {len(groups)}"
        )

    by_group = _grouped_indices(groups)
    keys = np.array(list(by_group))
    if len(keys) < 4:
        raise ValueError(
            f"need at least 4 bouts to split a day into two estimable halves; "
            f"got {len(keys)}"
        )

    rng = np.random.default_rng(seed)
    values, splits = [], []
    for _ in range(n_draws):
        shuffled = rng.permutation(keys)
        half = len(shuffled) // 2
        left_keys, right_keys = shuffled[:half], shuffled[half:]
        left = np.concatenate([by_group[key] for key in left_keys])
        right = np.concatenate([by_group[key] for key in right_keys])
        if (len(np.unique(groups[left])) < 2 or len(np.unique(groups[right])) < 2):
            continue
        splits.append((list(left_keys), list(right_keys)))
        values.append(unbiased_squared_centroid_distance(
            points[left], points[right], groups[left], groups[right]
        ))

    if return_splits:
        return splits
    return np.asarray(values)

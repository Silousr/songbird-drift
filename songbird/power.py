"""Sensitivity analysis: the smallest drift detectable at a given recording volume.

Power is estimated by **injection into real data** rather than from a formula. Each draw
takes a real day, samples bouts, splits them into two disjoint halves, displaces one half
by a known amount, and asks whether the drift statistic clears its critical value. The
critical value comes from the same procedure with zero displacement.

Doing it this way preserves the real within-bout correlation structure. A parametric power
calculation assuming independent syllables would be wrong by the design effect, which
Phase 3 measured at roughly 10x on this data -- it would promise far more sensitivity than
the experiment can deliver.

**Recording volume is counted in bouts.** Not syllables, not minutes. The drift estimator
treats the bout as the sampling unit, so its precision scales with bout count; collecting
more syllables inside the same bouts buys very little.

Effect sizes are *standardised*: ``scale`` is the pooled within-type variance, so an effect
of 0.04 means the centroid moved by 0.04 units of natural rendition-to-rendition variance —
the same units the Phase 3 noise floor is reported in.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_FOLD_CHANGE_GRID",
    "dispersion_detection_power",
    "minimum_detectable_fold_change",
    "DEFAULT_EFFECT_GRID",
    "critical_value",
    "detection_power",
    "minimum_detectable_effect",
]

#: Standardised effect sizes spanning well below to well above the measured noise floor
#: (~0.04 standardised units at the volumes in the Bengalese finch repository).
DEFAULT_EFFECT_GRID = np.array([
    0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0,
])


def _bout_index(groups: np.ndarray) -> dict:
    index = {}
    for position, group in enumerate(groups):
        index.setdefault(group, []).append(position)
    return {key: np.asarray(value) for key, value in index.items()}


def _draw_statistics(
    points: np.ndarray,
    groups: np.ndarray,
    n_bouts: int,
    effect_size: float,
    scale: float,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """Statistic values from repeated split-half draws with a known injected effect."""
    from songbird.drift import unbiased_squared_centroid_distance

    points = np.asarray(points, dtype=float)
    groups = np.asarray(groups)
    if len(points) != len(groups):
        raise ValueError(
            f"group labels must match sample length: {len(points)} vs {len(groups)}"
        )

    by_bout = _bout_index(groups)
    keys = np.array(list(by_bout))
    if n_bouts < 2:
        raise ValueError(f"need at least 2 bouts per side; got {n_bouts}")
    if 2 * n_bouts > len(keys):
        raise ValueError(
            f"need {2 * n_bouts} bouts to form two disjoint halves of {n_bouts}; "
            f"only {len(keys)} available"
        )

    rng = np.random.default_rng(seed)
    # Displacement magnitude such that the *standardised* squared distance equals
    # effect_size: ||delta||^2 = effect_size * scale.
    magnitude = np.sqrt(max(effect_size, 0.0) * scale)

    values = []
    for _ in range(n_draws):
        chosen = rng.choice(keys, size=2 * n_bouts, replace=False)
        left = np.concatenate([by_bout[key] for key in chosen[:n_bouts]])
        right = np.concatenate([by_bout[key] for key in chosen[n_bouts:]])
        if len(left) < 2 or len(right) < 2:
            continue

        b = points[right]
        if magnitude > 0:
            direction = rng.standard_normal(points.shape[1])
            direction /= np.linalg.norm(direction)
            b = b + direction * magnitude

        values.append(
            unbiased_squared_centroid_distance(
                points[left], b, groups[left], groups[right]
            )
        )
    return np.asarray(values)


def critical_value(
    points: np.ndarray,
    groups: np.ndarray,
    n_bouts: int,
    alpha: float = 0.05,
    n_draws: int = 500,
    seed: int = 0,
) -> float:
    """The statistic's ``1 - alpha`` quantile under no drift, at this bout count."""
    null = _draw_statistics(points, groups, n_bouts, 0.0, 1.0, n_draws, seed)
    if len(null) == 0:
        raise ValueError("no valid null draws")
    return float(np.percentile(null, 100 * (1 - alpha)))


def detection_power(
    points: np.ndarray,
    groups: np.ndarray,
    n_bouts: int,
    effect_size: float,
    scale: float,
    alpha: float = 0.05,
    n_draws: int = 500,
    seed: int = 0,
) -> float:
    """Probability of detecting a standardised drift of ``effect_size``.

    The critical value is recomputed at the same bout count from an independent seed, so
    the threshold is not tuned to the alternative draws it is being applied to.
    """
    threshold = critical_value(points, groups, n_bouts, alpha, n_draws, seed + 10_007)
    alternative = _draw_statistics(
        points, groups, n_bouts, effect_size, scale, n_draws, seed
    )
    if len(alternative) == 0:
        raise ValueError("no valid alternative draws")
    return float(np.mean(alternative > threshold))


def minimum_detectable_effect(
    points: np.ndarray,
    groups: np.ndarray,
    n_bouts: int,
    scale: float,
    alpha: float = 0.05,
    power: float = 0.8,
    grid: np.ndarray = DEFAULT_EFFECT_GRID,
    n_draws: int = 500,
    seed: int = 0,
) -> float:
    """Smallest standardised drift on ``grid`` reaching ``power``.

    Returns ``nan`` when no effect on the grid reaches it -- reported as such rather than
    silently returning the grid maximum, which would understate what the experiment needs.
    """
    for effect in np.sort(np.asarray(grid, dtype=float)):
        achieved = detection_power(
            points, groups, n_bouts, float(effect), scale, alpha, n_draws, seed
        )
        if achieved >= power:
            return float(effect)
    return float("nan")


#: Fold-changes in rendition variance spanning tightening and loosening.
DEFAULT_FOLD_CHANGE_GRID = np.array([
    1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0,
])


def _scale_dispersion(points: np.ndarray, fold_change: float) -> np.ndarray:
    """Scale deviations from the mean so variance changes by ``fold_change`` exactly.

    The centroid is left untouched, which is the point: this injects the kind of change
    the centroid metric is structurally blind to.
    """
    centre = points.mean(axis=0)
    return centre + (points - centre) * np.sqrt(fold_change)


def _dispersion_draws(
    points: np.ndarray,
    groups: np.ndarray,
    n_bouts: int,
    fold_change: float,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    from songbird.drift import log_variance_ratio

    points = np.asarray(points, dtype=float)
    groups = np.asarray(groups)
    if len(points) != len(groups):
        raise ValueError(
            f"group labels must match sample length: {len(points)} vs {len(groups)}"
        )

    by_bout = _bout_index(groups)
    keys = np.array(list(by_bout))
    if n_bouts < 2:
        raise ValueError(f"need at least 2 bouts per side; got {n_bouts}")
    if 2 * n_bouts > len(keys):
        raise ValueError(
            f"need {2 * n_bouts} bouts to form two disjoint halves of {n_bouts}; "
            f"only {len(keys)} available"
        )

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_draws):
        chosen = rng.choice(keys, size=2 * n_bouts, replace=False)
        left = np.concatenate([by_bout[k] for k in chosen[:n_bouts]])
        right = np.concatenate([by_bout[k] for k in chosen[n_bouts:]])
        if len(left) < 2 or len(right) < 2:
            continue
        try:
            values.append(
                log_variance_ratio(points[left], _scale_dispersion(points[right],
                                                                   fold_change))
            )
        except ValueError:
            continue
    return np.asarray(values)


def dispersion_detection_power(
    points: np.ndarray,
    groups: np.ndarray,
    n_bouts: int,
    fold_change: float,
    alpha: float = 0.05,
    n_draws: int = 500,
    seed: int = 0,
) -> float:
    """Probability of detecting a ``fold_change`` in rendition variance.

    Two-sided: a manipulation could tighten renditions as easily as loosen them, so the
    threshold is a quantile of ``|log ratio|`` under no change.
    """
    null = _dispersion_draws(points, groups, n_bouts, 1.0, n_draws, seed + 10_007)
    if len(null) == 0:
        raise ValueError("no valid null draws")
    threshold = float(np.percentile(np.abs(null), 100 * (1 - alpha)))

    alternative = _dispersion_draws(points, groups, n_bouts, fold_change, n_draws, seed)
    if len(alternative) == 0:
        raise ValueError("no valid alternative draws")
    return float(np.mean(np.abs(alternative) > threshold))


def minimum_detectable_fold_change(
    points: np.ndarray,
    groups: np.ndarray,
    n_bouts: int,
    alpha: float = 0.05,
    power: float = 0.8,
    grid: np.ndarray = DEFAULT_FOLD_CHANGE_GRID,
    n_draws: int = 500,
    seed: int = 0,
) -> float:
    """Smallest fold-change in variance on ``grid`` reaching ``power``.

    Returns ``nan`` when nothing on the grid reaches it, rather than the grid maximum --
    which would understate what the experiment needs.
    """
    for fold in np.sort(np.asarray(grid, dtype=float)):
        achieved = dispersion_detection_power(
            points, groups, n_bouts, float(fold), alpha, n_draws, seed
        )
        if achieved >= power:
            return float(fold)
    return float("nan")

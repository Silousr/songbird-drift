"""Tests for dispersion drift — change in rendition-to-rendition variability.

The centroid metric (Phase 3) asks whether a syllable type *moved*. This asks whether it
got *sloppier*. A manipulation that reopened a critical period might well destabilise
renditions without shifting their average, and that change is completely invisible to a
centroid distance — the gap recorded in the Phase 4 report.

The statistic is the log ratio of total variance, ``log(tr Var(b) / tr Var(a))``:

* scale-free, so it does not inherit the arbitrary units of the embedding;
* symmetric under swapping the two days, up to sign;
* naturally centred on zero under the null, since splitting one day in half gives a ratio
  distributed symmetrically about 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.drift import (
    bootstrap_dispersion_ci,
    log_variance_ratio,
    split_half_dispersion_null,
)

DIMS = 8


def sample(n=400, spread=1.0, shift=0.0, seed=0, dims=DIMS):
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, dims)) * spread
    points[:, 0] += shift
    return points


def bouted(n_bouts=60, per_bout=10, spread=1.0, dims=DIMS, seed=0):
    rng = np.random.default_rng(seed)
    points, groups = [], []
    for bout in range(n_bouts):
        centre = rng.standard_normal(dims) * 0.8
        points.append(centre + rng.standard_normal((per_bout, dims)) * spread)
        groups += [f"bout{bout}"] * per_bout
    return np.vstack(points), np.array(groups)


class TestLogVarianceRatio:
    def test_identical_distributions_give_about_zero(self):
        values = [log_variance_ratio(sample(seed=2 * i), sample(seed=2 * i + 1))
                  for i in range(100)]
        assert abs(np.mean(values)) < 0.02

    def test_positive_when_the_second_day_is_more_variable(self):
        assert log_variance_ratio(sample(spread=1.0), sample(spread=2.0, seed=1)) > 0

    def test_negative_when_the_second_day_is_tighter(self):
        assert log_variance_ratio(sample(spread=2.0), sample(spread=1.0, seed=1)) < 0

    def test_recovers_a_known_ratio(self):
        # Variance scales with the square of spread, so a 2x spread is log(4) in trace.
        value = log_variance_ratio(sample(spread=1.0, n=4000),
                                   sample(spread=2.0, n=4000, seed=1))
        assert value == pytest.approx(np.log(4.0), rel=0.05)

    def test_is_invariant_to_scaling_both_sides(self):
        a, b = sample(seed=0), sample(spread=1.7, seed=1)
        assert log_variance_ratio(a, b) == pytest.approx(
            log_variance_ratio(a * 5.0, b * 5.0), rel=1e-9
        )

    def test_is_antisymmetric(self):
        a, b = sample(seed=0), sample(spread=1.7, seed=1)
        assert log_variance_ratio(a, b) == pytest.approx(-log_variance_ratio(b, a))

    def test_ignores_a_pure_shift(self):
        # A centroid move with no change in spread must not register as dispersion drift.
        a = sample(seed=0)
        assert log_variance_ratio(a, a + 10.0) == pytest.approx(0.0, abs=1e-9)

    def test_rejects_samples_too_small_for_a_variance(self):
        with pytest.raises(ValueError):
            log_variance_ratio(sample(n=1), sample())

    def test_rejects_mismatched_dimensions(self):
        with pytest.raises(ValueError):
            log_variance_ratio(sample(dims=4), sample(dims=5))


class TestInference:
    def test_interval_brackets_the_estimate(self):
        a, ga = bouted(seed=0)
        b, gb = bouted(seed=1)
        estimate, low, high = bootstrap_dispersion_ci(a, ga, b, gb, n_boot=200, seed=0)
        assert low <= estimate <= high

    def test_interval_covers_zero_under_the_null(self):
        a, ga = bouted(seed=0)
        b, gb = bouted(seed=1)
        _, low, high = bootstrap_dispersion_ci(a, ga, b, gb, n_boot=300, seed=0)
        assert low <= 0.0 <= high

    def test_interval_excludes_zero_for_a_real_change(self):
        a, ga = bouted(spread=1.0, seed=0)
        b, gb = bouted(spread=2.5, seed=1)
        _, low, _ = bootstrap_dispersion_ci(a, ga, b, gb, n_boot=300, seed=0)
        assert low > 0.0

    def test_resamples_bouts_not_renditions(self):
        # Same argument as the centroid metric: renditions within a bout are correlated,
        # so treating them as independent gives intervals that are far too narrow.
        a, ga = bouted(seed=0)
        b, gb = bouted(seed=1)
        _, low_c, high_c = bootstrap_dispersion_ci(a, ga, b, gb, n_boot=300, seed=0)
        independent = np.arange(len(a)).astype(str)
        _, low_i, high_i = bootstrap_dispersion_ci(
            a, independent, b, np.arange(len(b)).astype(str), n_boot=300, seed=0
        )
        assert (high_c - low_c) > 1.3 * (high_i - low_i)


class TestSplitHalfNull:
    def test_null_is_centred_on_zero(self):
        points, groups = bouted(n_bouts=80, seed=0)
        null = split_half_dispersion_null(points, groups, n_draws=200, seed=0)
        assert abs(np.median(null)) < 0.1

    def test_null_has_both_signs(self):
        points, groups = bouted(n_bouts=80, seed=0)
        null = split_half_dispersion_null(points, groups, n_draws=200, seed=0)
        assert (null < 0).any() and (null > 0).any()

    def test_spread_grows_with_fewer_bouts(self):
        many, gm = bouted(n_bouts=100, seed=0)
        few, gf = bouted(n_bouts=8, seed=0)
        wide = np.std(split_half_dispersion_null(few, gf, n_draws=200, seed=0))
        narrow = np.std(split_half_dispersion_null(many, gm, n_draws=200, seed=0))
        assert wide > narrow

    def test_rejects_too_few_bouts(self):
        points, groups = bouted(n_bouts=2, seed=0)
        with pytest.raises(ValueError):
            split_half_dispersion_null(points, groups, n_draws=10, seed=0)

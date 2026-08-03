"""Tests for drift uncertainty and the within-day null.

Two statistical points these enforce.

**Resample bouts, not syllables.** Syllables produced inside one song bout are not
independent -- they share the bird's state, posture, and the bout's recording conditions.
Bootstrapping individual syllables treats a few hundred correlated renditions as a few
hundred independent observations and returns confidence intervals that are far too narrow.
The resampling unit must be the bout.

**The null must be built with the same estimator and the same sample sizes.** The noise
floor is what the drift statistic reads when there is no drift, so it is obtained by
splitting a *single* day into two halves and running the identical computation. Comparing
a between-day estimate against a null computed at different sample sizes would compare two
different quantities.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.drift import bootstrap_drift_ci, split_half_null


def clustered_sample(n_bouts=40, per_bout=10, dims=8, bout_spread=2.0, seed=0):
    """Renditions grouped into bouts, with real between-bout structure."""
    rng = np.random.default_rng(seed)
    points, groups = [], []
    for bout in range(n_bouts):
        centre = rng.standard_normal(dims) * bout_spread
        points.append(centre + rng.standard_normal((per_bout, dims)) * 0.3)
        groups += [f"bout{bout}"] * per_bout
    return np.vstack(points), np.array(groups)


class TestBootstrapDriftCi:
    def test_interval_brackets_the_point_estimate(self):
        a, groups_a = clustered_sample(seed=0)
        b, groups_b = clustered_sample(seed=1)
        estimate, low, high = bootstrap_drift_ci(a, groups_a, b, groups_b, n_boot=200, seed=0)
        assert low <= estimate <= high

    def test_interval_covers_zero_under_the_null(self):
        a, groups_a = clustered_sample(seed=0)
        b, groups_b = clustered_sample(seed=1)
        _, low, high = bootstrap_drift_ci(a, groups_a, b, groups_b, n_boot=300, seed=0)
        assert low <= 0.0 <= high

    def test_clustered_resampling_is_wider_than_ignoring_clusters(self):
        # The whole reason the bout is the resampling unit.
        a, groups_a = clustered_sample(seed=0)
        b, groups_b = clustered_sample(seed=1)
        _, low_c, high_c = bootstrap_drift_ci(a, groups_a, b, groups_b, n_boot=300, seed=0)
        independent_a = np.arange(len(a)).astype(str)
        independent_b = np.arange(len(b)).astype(str)
        _, low_i, high_i = bootstrap_drift_ci(
            a, independent_a, b, independent_b, n_boot=300, seed=0
        )
        assert (high_c - low_c) > 1.5 * (high_i - low_i)

    def test_is_reproducible_for_a_fixed_seed(self):
        a, groups_a = clustered_sample(seed=0)
        b, groups_b = clustered_sample(seed=1)
        first = bootstrap_drift_ci(a, groups_a, b, groups_b, n_boot=100, seed=7)
        second = bootstrap_drift_ci(a, groups_a, b, groups_b, n_boot=100, seed=7)
        assert first == second

    def test_detects_a_real_shift(self):
        a, groups_a = clustered_sample(seed=0)
        b, groups_b = clustered_sample(seed=1)
        b = b + 5.0
        _, low, _ = bootstrap_drift_ci(a, groups_a, b, groups_b, n_boot=300, seed=0)
        assert low > 0.0

    def test_rejects_mismatched_group_lengths(self):
        a, groups_a = clustered_sample(seed=0)
        b, groups_b = clustered_sample(seed=1)
        with pytest.raises(ValueError):
            bootstrap_drift_ci(a, groups_a[:-1], b, groups_b, n_boot=10, seed=0)


class TestSplitHalfNull:
    def test_null_is_centred_on_zero(self):
        points, groups = clustered_sample(n_bouts=60, seed=0)
        null = split_half_null(points, groups, n_draws=200, seed=0)
        assert abs(np.mean(null)) < 0.5 * np.std(null) + 0.05

    def test_null_has_both_signs(self):
        points, groups = clustered_sample(n_bouts=60, seed=0)
        null = split_half_null(points, groups, n_draws=200, seed=0)
        assert (null < 0).any() and (null > 0).any()

    def test_returns_the_requested_number_of_draws(self):
        points, groups = clustered_sample(n_bouts=40, seed=0)
        assert len(split_half_null(points, groups, n_draws=37, seed=0)) == 37

    def test_spread_grows_when_fewer_BOUTS_are_available(self):
        # Precision is governed by the number of bouts, because the bout is the sampling
        # unit. This is what sets the recording-volume axis of the Phase 4 power analysis.
        many, groups_many = clustered_sample(n_bouts=80, per_bout=8, seed=0)
        few, groups_few = clustered_sample(n_bouts=10, per_bout=8, seed=0)
        wide = np.std(split_half_null(few, groups_few, n_draws=200, seed=0))
        narrow = np.std(split_half_null(many, groups_many, n_draws=200, seed=0))
        assert wide > 1.5 * narrow

    def test_more_syllables_per_bout_barely_helps(self):
        # The counterpart, and a warning for planning: collecting more syllables inside
        # the same number of bouts buys far less precision than collecting more bouts.
        # "Minutes of song" is therefore the wrong unit for a power calculation.
        thin, groups_thin = clustered_sample(n_bouts=40, per_bout=3, seed=0)
        thick, groups_thick = clustered_sample(n_bouts=40, per_bout=30, seed=0)
        spread_thin = np.std(split_half_null(thin, groups_thin, n_draws=200, seed=0))
        spread_thick = np.std(split_half_null(thick, groups_thick, n_draws=200, seed=0))
        assert 0.6 < spread_thick / spread_thin < 1.6

    def test_halves_are_disjoint_in_bouts(self):
        points, groups = clustered_sample(n_bouts=20, seed=0)
        # A bout appearing in both halves would leak the same renditions into both sides
        # and bias the null toward zero.
        seen = split_half_null(points, groups, n_draws=50, seed=0, return_splits=True)
        for left, right in seen:
            assert not (set(left) & set(right))

    def test_rejects_too_few_bouts_to_split(self):
        points, groups = clustered_sample(n_bouts=1, per_bout=10, seed=0)
        with pytest.raises(ValueError):
            split_half_null(points, groups, n_draws=10, seed=0)

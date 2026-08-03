"""Tests for the drift estimator.

The central problem: the distance between two sample means is a **biased** estimate of the
distance between the underlying distribution means. Draw two samples from one identical
distribution and their centroids still sit apart, by an amount that grows as sample size
shrinks. A drift metric built on the naive distance therefore reports drift where there is
none, and reports *more* of it on days when the bird sang less.

``unbiased_squared_centroid_distance`` subtracts the expected sampling contribution:

    E[||x̄a - x̄b||²] = ||μa - μb||² + tr(Σa)/na + tr(Σb)/nb

Consequence worth stating: the corrected estimate is **negative about half the time when
true drift is zero**. That is correct behaviour, not a bug, and it must never be clipped to
zero -- clipping would reintroduce exactly the upward bias the correction removes.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.drift import (
    naive_squared_centroid_distance,
    unbiased_squared_centroid_distance,
)

DIMS = 16


def sample(n, shift=0.0, scale=1.0, seed=0, dims=DIMS):
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, dims)) * scale
    points[:, 0] += shift
    return points


class TestNaiveEstimatorIsBiased:
    def test_naive_distance_is_positive_for_identical_distributions(self):
        # Documents the failure mode the correction exists to fix.
        values = [
            naive_squared_centroid_distance(sample(40, seed=2 * i), sample(40, seed=2 * i + 1))
            for i in range(60)
        ]
        assert np.mean(values) > 0.5

    def test_naive_bias_grows_as_samples_shrink(self):
        big = np.mean([
            naive_squared_centroid_distance(sample(400, seed=2 * i), sample(400, seed=2 * i + 1))
            for i in range(40)
        ])
        small = np.mean([
            naive_squared_centroid_distance(sample(25, seed=2 * i), sample(25, seed=2 * i + 1))
            for i in range(40)
        ])
        assert small > 4 * big


class TestUnbiasedEstimator:
    def test_is_unbiased_for_identical_distributions(self):
        values = [
            unbiased_squared_centroid_distance(sample(40, seed=2 * i), sample(40, seed=2 * i + 1))
            for i in range(200)
        ]
        assert abs(np.mean(values)) < 0.1

    def test_goes_negative_roughly_half_the_time_under_the_null(self):
        values = [
            unbiased_squared_centroid_distance(sample(40, seed=2 * i), sample(40, seed=2 * i + 1))
            for i in range(200)
        ]
        assert 0.3 < np.mean(np.array(values) < 0) < 0.7

    def test_recovers_a_known_separation(self):
        values = [
            unbiased_squared_centroid_distance(
                sample(200, shift=0.0, seed=2 * i), sample(200, shift=2.0, seed=2 * i + 1)
            )
            for i in range(40)
        ]
        assert np.mean(values) == pytest.approx(4.0, rel=0.15)

    def test_stays_unbiased_when_sample_sizes_differ(self):
        values = [
            unbiased_squared_centroid_distance(sample(200, seed=2 * i), sample(20, seed=2 * i + 1))
            for i in range(200)
        ]
        assert abs(np.mean(values)) < 0.2

    def test_stays_unbiased_when_variances_differ(self):
        values = [
            unbiased_squared_centroid_distance(
                sample(60, scale=1.0, seed=2 * i), sample(60, scale=3.0, seed=2 * i + 1)
            )
            for i in range(200)
        ]
        assert abs(np.mean(values)) < 0.4

    def test_rejects_mismatched_dimensions(self):
        with pytest.raises(ValueError):
            unbiased_squared_centroid_distance(sample(20, dims=4), sample(20, dims=5))

    def test_rejects_samples_too_small_to_estimate_variance(self):
        with pytest.raises(ValueError):
            unbiased_squared_centroid_distance(sample(1), sample(20))


class TestClusteredCorrection:
    """Regression tests for a bias that would have manufactured drift on real data.

    Song syllables arrive in bouts, and renditions within a bout are correlated. The
    variance correction assumes the sampling unit is independent, so applying it per
    *rendition* under-corrects by the design effect -- measured at ~10x on the fixture
    below, leaving a large positive residual bias that never goes negative. Passing bout
    labels makes the bout the sampling unit and restores unbiasedness.
    """

    @staticmethod
    def clustered(n_bouts=40, per_bout=10, dims=8, bout_spread=2.0, seed=0):
        rng = np.random.default_rng(seed)
        points, groups = [], []
        for bout in range(n_bouts):
            centre = rng.standard_normal(dims) * bout_spread
            points.append(centre + rng.standard_normal((per_bout, dims)) * 0.3)
            groups += [f"bout{bout}"] * per_bout
        return np.vstack(points), np.array(groups)

    def test_ignoring_bouts_leaves_a_large_positive_bias(self):
        values = [
            unbiased_squared_centroid_distance(
                self.clustered(seed=2 * i)[0], self.clustered(seed=2 * i + 1)[0]
            )
            for i in range(120)
        ]
        assert np.mean(values) > 0.5
        assert np.mean(np.array(values) < 0) < 0.05

    def test_supplying_bouts_restores_unbiasedness(self):
        values = []
        for i in range(200):
            a, groups_a = self.clustered(seed=2 * i)
            b, groups_b = self.clustered(seed=2 * i + 1)
            values.append(
                unbiased_squared_centroid_distance(a, b, groups_a=groups_a, groups_b=groups_b)
            )
        assert abs(np.mean(values)) < 0.35 * np.std(values)
        assert 0.3 < np.mean(np.array(values) < 0) < 0.7

    def test_clustered_path_still_recovers_a_real_shift(self):
        values = []
        for i in range(40):
            a, groups_a = self.clustered(seed=2 * i)
            b, groups_b = self.clustered(seed=2 * i + 1)
            values.append(
                unbiased_squared_centroid_distance(
                    a, b + 3.0, groups_a=groups_a, groups_b=groups_b
                )
            )
        assert np.mean(values) == pytest.approx(8 * 9.0, rel=0.25)

    def test_rejects_fewer_than_two_bouts(self):
        a, groups_a = self.clustered(n_bouts=1, seed=0)
        b, groups_b = self.clustered(seed=1)
        with pytest.raises(ValueError):
            unbiased_squared_centroid_distance(a, b, groups_a=groups_a, groups_b=groups_b)

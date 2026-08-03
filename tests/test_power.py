"""Tests for the sensitivity / power analysis.

Power is estimated by **injection into real data**: take a day, split its bouts into two
halves, displace one half by a known amount, and see how often the drift statistic clears
its critical value. That preserves the real within-bout correlation structure, which a
parametric power formula assuming independent syllables would get badly wrong -- Phase 3
measured a ~10x design effect from that correlation.

The unit of recording volume is the **bout**, not the syllable and not the minute, because
that is what the estimator's precision actually scales with.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.power import (
    critical_value,
    detection_power,
    minimum_detectable_effect,
)

DIMS = 8


def bouted(n_bouts=80, per_bout=8, dims=DIMS, bout_spread=1.0, seed=0):
    rng = np.random.default_rng(seed)
    points, groups = [], []
    for bout in range(n_bouts):
        centre = rng.standard_normal(dims) * bout_spread
        points.append(centre + rng.standard_normal((per_bout, dims)) * 0.5)
        groups += [f"bout{bout}"] * per_bout
    return np.vstack(points), np.array(groups)


@pytest.fixture
def data():
    return bouted()


class TestCriticalValue:
    def test_is_positive(self, data):
        points, groups = data
        assert critical_value(points, groups, n_bouts=20, n_draws=200, seed=0) > 0

    def test_falls_as_bouts_increase(self, data):
        points, groups = data
        few = critical_value(points, groups, n_bouts=6, n_draws=300, seed=0)
        many = critical_value(points, groups, n_bouts=30, n_draws=300, seed=0)
        assert many < few

    def test_rejects_more_bouts_than_available(self, data):
        points, groups = data
        with pytest.raises(ValueError):
            critical_value(points, groups, n_bouts=500, n_draws=10, seed=0)


class TestDetectionPower:
    def test_power_at_zero_effect_is_about_alpha(self, data):
        points, groups = data
        power = detection_power(points, groups, n_bouts=20, effect_size=0.0,
                                scale=1.0, alpha=0.05, n_draws=400, seed=0)
        assert 0.0 <= power <= 0.15

    def test_power_rises_with_effect_size(self, data):
        points, groups = data
        small = detection_power(points, groups, n_bouts=20, effect_size=0.05,
                                scale=1.0, n_draws=300, seed=0)
        large = detection_power(points, groups, n_bouts=20, effect_size=1.0,
                                scale=1.0, n_draws=300, seed=0)
        assert large > small

    def test_power_rises_with_more_bouts(self, data):
        points, groups = data
        few = detection_power(points, groups, n_bouts=5, effect_size=0.2,
                              scale=1.0, n_draws=300, seed=0)
        many = detection_power(points, groups, n_bouts=35, effect_size=0.2,
                               scale=1.0, n_draws=300, seed=0)
        assert many > few

    def test_large_effect_is_almost_always_detected(self, data):
        points, groups = data
        assert detection_power(points, groups, n_bouts=30, effect_size=5.0,
                               scale=1.0, n_draws=300, seed=0) > 0.9

    def test_is_reproducible(self, data):
        points, groups = data
        first = detection_power(points, groups, n_bouts=15, effect_size=0.3,
                                scale=1.0, n_draws=200, seed=3)
        second = detection_power(points, groups, n_bouts=15, effect_size=0.3,
                                 scale=1.0, n_draws=200, seed=3)
        assert first == second


class TestMinimumDetectableEffect:
    def test_returns_an_effect_reaching_the_requested_power(self, data):
        points, groups = data
        mde = minimum_detectable_effect(points, groups, n_bouts=25, scale=1.0,
                                        power=0.8, n_draws=300, seed=0)
        achieved = detection_power(points, groups, n_bouts=25, effect_size=mde,
                                   scale=1.0, n_draws=400, seed=1)
        assert achieved >= 0.7

    def test_shrinks_as_recording_volume_grows(self, data):
        # nan means "not reachable at any effect on the grid", i.e. worse than any
        # finite MDE -- so it sorts as +inf, not as a missing value.
        points, groups = data
        as_inf = lambda v: np.inf if np.isnan(v) else v
        curve = [
            as_inf(minimum_detectable_effect(points, groups, n_bouts=n, scale=1.0,
                                             n_draws=300, seed=0))
            for n in (6, 20, 35)
        ]
        assert curve == sorted(curve, reverse=True)
        assert curve[-1] < curve[0]

    def test_too_little_recording_volume_is_reported_as_unreachable(self, data):
        # A volume that cannot reach the target power must say so rather than return the
        # grid maximum, which would understate what the experiment needs.
        points, groups = data
        assert np.isnan(minimum_detectable_effect(points, groups, n_bouts=6, scale=1.0,
                                                  n_draws=300, seed=0))

    def test_scale_divides_the_reported_effect(self, data):
        # `scale` converts raw squared distance into standardised units, so doubling it
        # must halve the reported MDE.
        points, groups = data
        one = minimum_detectable_effect(points, groups, n_bouts=20, scale=1.0,
                                        n_draws=300, seed=0)
        two = minimum_detectable_effect(points, groups, n_bouts=20, scale=2.0,
                                        n_draws=300, seed=0)
        assert two == pytest.approx(one / 2, rel=0.2)

    def test_returns_nan_when_no_effect_in_grid_reaches_the_power(self, data):
        points, groups = data
        mde = minimum_detectable_effect(points, groups, n_bouts=4, scale=1.0,
                                        power=0.999, grid=np.array([0.001, 0.002]),
                                        n_draws=100, seed=0)
        assert np.isnan(mde)

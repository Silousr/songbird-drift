"""Tests for dispersion sensitivity: how many bouts to detect a change in variability.

Effects are injected by scaling each half's deviations from its own mean, which multiplies
its variance by exactly the requested fold change while leaving the centroid untouched --
so this measures sensitivity to the thing the centroid metric cannot see.

The test is two-sided: a manipulation could tighten renditions as easily as loosen them.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.power import (
    dispersion_detection_power,
    minimum_detectable_fold_change,
)


def bouted(n_bouts=80, per_bout=10, dims=8, seed=0):
    rng = np.random.default_rng(seed)
    points, groups = [], []
    for bout in range(n_bouts):
        centre = rng.standard_normal(dims) * 0.8
        points.append(centre + rng.standard_normal((per_bout, dims)))
        groups += [f"bout{bout}"] * per_bout
    return np.vstack(points), np.array(groups)


@pytest.fixture
def data():
    return bouted()


class TestDispersionDetectionPower:
    def test_no_change_gives_about_alpha(self, data):
        points, groups = data
        power = dispersion_detection_power(points, groups, n_bouts=20, fold_change=1.0,
                                           alpha=0.05, n_draws=400, seed=0)
        assert 0.0 <= power <= 0.15

    def test_power_rises_with_fold_change(self, data):
        points, groups = data
        small = dispersion_detection_power(points, groups, 20, 1.1, n_draws=300, seed=0)
        large = dispersion_detection_power(points, groups, 20, 2.5, n_draws=300, seed=0)
        assert large > small

    def test_power_rises_with_more_bouts(self, data):
        points, groups = data
        few = dispersion_detection_power(points, groups, 5, 1.5, n_draws=300, seed=0)
        many = dispersion_detection_power(points, groups, 35, 1.5, n_draws=300, seed=0)
        assert many > few

    def test_detects_a_large_change_almost_always(self, data):
        points, groups = data
        assert dispersion_detection_power(points, groups, 30, 4.0,
                                          n_draws=300, seed=0) > 0.9

    def test_is_two_sided(self, data):
        # A halving of variance must be as detectable as a doubling.
        points, groups = data
        up = dispersion_detection_power(points, groups, 30, 2.0, n_draws=400, seed=0)
        down = dispersion_detection_power(points, groups, 30, 0.5, n_draws=400, seed=0)
        assert up > 0.5 and down > 0.5

    def test_injection_leaves_the_centroid_alone(self):
        # Guards the claim that this measures something the centroid metric cannot.
        from songbird.power import _scale_dispersion

        rng = np.random.default_rng(0)
        points = rng.standard_normal((500, 6)) + 3.0
        scaled = _scale_dispersion(points, 4.0)
        assert np.allclose(scaled.mean(axis=0), points.mean(axis=0), atol=1e-9)
        assert scaled.var(axis=0, ddof=1).sum() == pytest.approx(
            4.0 * points.var(axis=0, ddof=1).sum(), rel=1e-9
        )

    def test_is_reproducible(self, data):
        points, groups = data
        first = dispersion_detection_power(points, groups, 15, 1.8, n_draws=200, seed=3)
        second = dispersion_detection_power(points, groups, 15, 1.8, n_draws=200, seed=3)
        assert first == second


class TestMinimumDetectableFoldChange:
    def test_returns_a_fold_change_reaching_the_power(self, data):
        points, groups = data
        mde = minimum_detectable_fold_change(points, groups, n_bouts=30, power=0.8,
                                             n_draws=300, seed=0)
        achieved = dispersion_detection_power(points, groups, 30, mde,
                                              n_draws=400, seed=1)
        assert achieved >= 0.7

    def test_shrinks_towards_one_as_volume_grows(self, data):
        points, groups = data
        as_inf = lambda v: np.inf if np.isnan(v) else v
        curve = [as_inf(minimum_detectable_fold_change(points, groups, n_bouts=n,
                                                       n_draws=300, seed=0))
                 for n in (10, 30, 39)]
        assert curve == sorted(curve, reverse=True)

    def test_reports_unreachable_rather_than_the_grid_maximum(self, data):
        points, groups = data
        assert np.isnan(minimum_detectable_fold_change(
            points, groups, n_bouts=5, power=0.999,
            grid=np.array([1.01, 1.02]), n_draws=100, seed=0))

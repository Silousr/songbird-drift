"""Tests for the group-level comparison: treated birds versus controls.

Everything before this answers "did *this* bird change". The experiment asks "did the
treated birds change more than the controls", and that is a different statistical object.

Two commitments enforced here.

**The bird is the unit of replication.** Not the syllable, not the day, not the bout.
Pooling syllables across birds and testing on that pool is pseudo-replication -- it treats
one bird's thousands of renditions as thousands of independent observations and returns a
p-value that can be orders of magnitude too small. The API only accepts one value per bird.

**Permutation, not a t-test.** Songbird experiments run 5-15 birds per group and the drift
statistic is skewed, so a test resting on normality of the mean is not safe at that n.
Shuffling group labels makes no distributional assumption at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.group import GroupComparison, birds_needed, compare_groups


class TestCompareGroups:
    def test_identical_groups_usually_give_a_large_p_value(self):
        # Averaged over draws, not a single one: two samples from the same distribution
        # separate by chance often enough that any single seed proves nothing.
        rng = np.random.default_rng(0)
        values = [
            compare_groups(rng.normal(0.05, 0.01, 8), rng.normal(0.05, 0.01, 8),
                           n_permutations=2000, seed=i).p_value
            for i in range(40)
        ]
        assert np.median(values) > 0.2

    def test_clearly_separated_groups_give_a_small_p_value(self):
        treated = np.array([0.30, 0.28, 0.35, 0.31, 0.33, 0.29])
        control = np.array([0.05, 0.04, 0.06, 0.05, 0.03, 0.05])
        assert compare_groups(treated, control, seed=0).p_value < 0.01

    def test_p_value_is_calibrated_under_the_null(self):
        # p should be roughly uniform when the groups truly do not differ.
        rng = np.random.default_rng(1)
        values = [
            compare_groups(rng.normal(0, 1, 7), rng.normal(0, 1, 7),
                           n_permutations=500, seed=i).p_value
            for i in range(120)
        ]
        assert 0.02 < np.mean(np.array(values) < 0.05) < 0.15

    def test_reports_the_observed_difference(self):
        treated = np.array([0.2, 0.3, 0.25, 0.28])
        control = np.array([0.1, 0.1, 0.12, 0.08])
        result = compare_groups(treated, control, seed=0)
        assert result.observed_difference == pytest.approx(
            treated.mean() - control.mean()
        )

    def test_one_sided_is_directional(self):
        treated = np.array([0.30, 0.28, 0.35, 0.31, 0.33, 0.29])
        control = np.array([0.05, 0.04, 0.06, 0.05, 0.03, 0.05])
        greater = compare_groups(treated, control, alternative="greater", seed=0)
        less = compare_groups(treated, control, alternative="less", seed=0)
        assert greater.p_value < 0.01 and less.p_value > 0.9

    def test_returns_a_confidence_interval_for_the_difference(self):
        rng = np.random.default_rng(0)
        result = compare_groups(rng.normal(0.3, 0.05, 10), rng.normal(0.1, 0.05, 10),
                                seed=0)
        assert result.ci_low < result.observed_difference < result.ci_high
        assert result.ci_low > 0

    def test_rejects_groups_too_small_to_permute(self):
        with pytest.raises(ValueError):
            compare_groups(np.array([0.1]), np.array([0.2]), seed=0)

    def test_rejects_an_unknown_alternative(self):
        with pytest.raises(ValueError):
            compare_groups(np.arange(5.0), np.arange(5.0), alternative="sideways")

    def test_is_reproducible(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(0.3, 0.1, 6), rng.normal(0.1, 0.1, 6)
        assert compare_groups(a, b, seed=5) == compare_groups(a, b, seed=5)


class TestVarianceDiagnostic:
    def test_flags_when_within_bird_noise_dominates(self):
        # Recording more per bird helps; adding birds does not. The lab needs to know
        # which of those two it is limited by.
        treated = np.array([0.20, 0.21, 0.19, 0.20])
        control = np.array([0.10, 0.11, 0.09, 0.10])
        result = compare_groups(treated, control, within_bird_sd=0.20, seed=0)
        assert result.limiting_factor == "within-bird"

    def test_flags_when_between_bird_variation_dominates(self):
        treated = np.array([0.05, 0.40, 0.15, 0.35])
        control = np.array([0.02, 0.30, 0.08, 0.25])
        result = compare_groups(treated, control, within_bird_sd=0.001, seed=0)
        assert result.limiting_factor == "between-bird"

    def test_diagnostic_is_absent_without_within_bird_uncertainty(self):
        result = compare_groups(np.arange(5.0), np.arange(5.0) + 1, seed=0)
        assert result.limiting_factor is None


class TestBirdsNeeded:
    def test_more_birds_are_needed_for_a_smaller_effect(self):
        big = birds_needed(effect=0.20, between_bird_sd=0.05, seed=0)
        small = birds_needed(effect=0.05, between_bird_sd=0.05, seed=0)
        assert small > big

    def test_more_birds_are_needed_when_birds_vary_more(self):
        tight = birds_needed(effect=0.10, between_bird_sd=0.02, seed=0)
        loose = birds_needed(effect=0.10, between_bird_sd=0.10, seed=0)
        assert loose > tight

    def test_returns_a_group_size_that_achieves_the_target_power(self):
        n = birds_needed(effect=0.15, between_bird_sd=0.05, power=0.8, seed=0)
        assert 3 <= n <= 40

    def test_reports_unreachable_rather_than_capping_silently(self):
        assert np.isnan(birds_needed(effect=0.001, between_bird_sd=1.0,
                                     max_n=8, seed=0))

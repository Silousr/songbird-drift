"""Tests for scoring recovered syllable labels against hand labels.

Segment-level accuracy alone is not enough: a labeller that drops or invents syllables
changes the *sequence*, and syntax is part of what song drift means. Syllable error rate
(Levenshtein over the label sequence, normalised by the true length) catches that;
per-segment accuracy does not.
"""

from __future__ import annotations

import pytest

from songbird.metrics.labels import (
    confusion_counts,
    edit_distance,
    label_accuracy,
    syllable_error_rate,
)


class TestEditDistance:
    def test_identical_sequences_have_zero_distance(self):
        assert edit_distance("abcde", "abcde") == 0

    def test_single_substitution(self):
        assert edit_distance("abcde", "abXde") == 1

    def test_single_deletion(self):
        assert edit_distance("abcde", "abde") == 1

    def test_single_insertion(self):
        assert edit_distance("abde", "abcde") == 1

    def test_empty_against_sequence_is_its_length(self):
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_both_empty(self):
        assert edit_distance("", "") == 0

    def test_is_symmetric(self):
        assert edit_distance("kitten", "sitting") == edit_distance("sitting", "kitten")

    def test_known_value(self):
        assert edit_distance("kitten", "sitting") == 3

    def test_accepts_lists_of_multicharacter_labels(self):
        # Canary labels are numeric strings; treating them as characters would be wrong.
        assert edit_distance(["1", "12", "3"], ["1", "12", "3"]) == 0
        assert edit_distance(["1", "12", "3"], ["1", "2", "3"]) == 1


class TestSyllableErrorRate:
    def test_perfect_prediction_scores_zero(self):
        assert syllable_error_rate(list("abcde"), list("abcde")) == 0.0

    def test_one_error_in_five(self):
        assert syllable_error_rate(list("abcde"), list("abXde")) == pytest.approx(0.2)

    def test_normalises_by_true_length_not_predicted(self):
        # Two spurious insertions against a 4-symbol truth is a rate of 0.5.
        assert syllable_error_rate(list("abcd"), list("abcdXY")) == pytest.approx(0.5)

    def test_can_exceed_one(self):
        assert syllable_error_rate(list("ab"), list("XYZWV")) > 1.0

    def test_empty_truth_with_predictions_is_undefined_not_zero(self):
        with pytest.raises(ValueError):
            syllable_error_rate([], list("abc"))


class TestLabelAccuracy:
    def test_counts_matching_positions(self):
        assert label_accuracy(list("abcd"), list("abXd")) == pytest.approx(0.75)

    def test_requires_equal_lengths(self):
        with pytest.raises(ValueError):
            label_accuracy(list("abc"), list("ab"))

    def test_perfect(self):
        assert label_accuracy(list("abc"), list("abc")) == 1.0


class TestConfusionCounts:
    def test_counts_true_predicted_pairs(self):
        counts = confusion_counts(list("aab"), list("aXb"))
        assert counts[("a", "a")] == 1
        assert counts[("a", "X")] == 1
        assert counts[("b", "b")] == 1

    def test_empty_input_gives_empty_counts(self):
        assert confusion_counts([], []) == {}

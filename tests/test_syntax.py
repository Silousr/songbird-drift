"""Tests for syntax drift — change in the *order* syllables are produced in.

Birdsong is a learned motor sequence with probabilistic syntax. A manipulation could
reorder syllables without altering any single syllable's acoustics, and both the centroid
and dispersion metrics would read zero throughout.

The statistic is the Jensen-Shannon divergence between the two days' bigram (syllable
transition) distributions. Unlike the acoustic metrics it is **non-negative by
construction**, so it is never centred on zero under the null and there is no bias
correction that could make it so: two finite samples from the same syntax still differ.
The split-half null is therefore not optional here — it is the only thing that says how
much divergence is normal at a given sample size.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.drift import (
    bout_sequences,
    split_half_syntax_null,
    syntax_divergence,
    transition_counts,
)


def sequences(pattern, n_bouts=40, seed=0, noise=0.0):
    """Repeat a motif per bout, optionally corrupting a fraction of transitions."""
    rng = np.random.default_rng(seed)
    out = {}
    for bout in range(n_bouts):
        seq = []
        for _ in range(6):
            for label in pattern:
                if noise and rng.random() < noise:
                    seq.append(rng.choice(list("abc")))
                else:
                    seq.append(label)
        out[f"bout{bout}"] = seq
    return out


class TestTransitionCounts:
    def test_counts_adjacent_pairs(self):
        counts = transition_counts({"b0": list("abab")}, types=list("ab"))
        assert counts[0, 1] == 2  # a -> b
        assert counts[1, 0] == 1  # b -> a

    def test_does_not_join_across_bouts(self):
        # The last syllable of one bout does not precede the first of the next.
        joined = transition_counts({"b0": list("ab"), "b1": list("ba")}, list("ab"))
        assert joined.sum() == 2

    def test_ignores_labels_outside_the_type_set(self):
        counts = transition_counts({"b0": list("axb")}, types=list("ab"))
        assert counts.sum() == 0

    def test_handles_a_single_syllable_bout(self):
        assert transition_counts({"b0": ["a"]}, list("ab")).sum() == 0


class TestSyntaxDivergence:
    def test_identical_syntax_gives_zero(self):
        seqs = sequences("abc")
        assert syntax_divergence(seqs, seqs, list("abc")) == pytest.approx(0.0, abs=1e-12)

    def test_different_syntax_gives_a_positive_value(self):
        assert syntax_divergence(sequences("abc"), sequences("acb", seed=1),
                                 list("abc")) > 0.1

    def test_is_symmetric(self):
        a, b = sequences("abc"), sequences("acb", seed=1)
        assert syntax_divergence(a, b, list("abc")) == pytest.approx(
            syntax_divergence(b, a, list("abc"))
        )

    def test_is_bounded_by_one(self):
        assert syntax_divergence(sequences("ab"), sequences("cc", seed=1),
                                 list("abc")) <= 1.0

    def test_grows_with_the_amount_of_corruption(self):
        base = sequences("abc")
        mild = syntax_divergence(base, sequences("abc", seed=1, noise=0.1), list("abc"))
        severe = syntax_divergence(base, sequences("abc", seed=1, noise=0.6), list("abc"))
        assert severe > mild

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            syntax_divergence({}, sequences("abc"), list("abc"))


class TestSplitHalfNull:
    def test_null_is_positive_when_syntax_is_probabilistic(self):
        # The property that makes the floor mandatory rather than a nicety: with real,
        # variable syntax two halves of the SAME day already diverge.
        null = split_half_syntax_null(sequences("abc", n_bouts=60, noise=0.3),
                                      list("abc"), n_draws=100, seed=0)
        assert (null >= 0).all() and np.median(null) > 0

    def test_null_is_exactly_zero_for_perfectly_deterministic_syntax(self):
        # A bird that always sings the identical motif has no syntactic variability, so
        # there is nothing for the floor to absorb.
        null = split_half_syntax_null(sequences("abc", n_bouts=60), list("abc"),
                                      n_draws=50, seed=0)
        assert np.allclose(null, 0.0)

    def test_null_shrinks_with_more_bouts(self):
        few = split_half_syntax_null(sequences("abc", n_bouts=8, noise=0.3),
                                     list("abc"), n_draws=100, seed=0)
        many = split_half_syntax_null(sequences("abc", n_bouts=200, noise=0.3),
                                      list("abc"), n_draws=100, seed=0)
        assert np.median(many) < np.median(few)

    def test_a_real_syntax_change_exceeds_the_null(self):
        base = sequences("abc", n_bouts=60)
        floor = np.percentile(split_half_syntax_null(base, list("abc"), n_draws=200,
                                                     seed=0), 95)
        observed = syntax_divergence(base, sequences("acb", n_bouts=60, seed=1),
                                     list("abc"))
        assert observed > floor

    def test_stable_syntax_stays_within_the_null(self):
        base = sequences("abc", n_bouts=60, noise=0.2)
        floor = np.percentile(split_half_syntax_null(base, list("abc"), n_draws=200,
                                                     seed=0), 95)
        observed = syntax_divergence(base, sequences("abc", n_bouts=60, seed=99,
                                                     noise=0.2), list("abc"))
        assert observed < floor * 3

    def test_rejects_too_few_bouts(self):
        with pytest.raises(ValueError):
            split_half_syntax_null(sequences("abc", n_bouts=2), list("abc"),
                                   n_draws=10, seed=0)


class TestBoutSequences:
    def test_orders_syllables_within_a_bout_by_onset(self):
        import pandas as pd

        table = pd.DataFrame({
            "audio_file": ["b0"] * 3,
            "onset_s": [0.5, 0.1, 0.3],
            "label": ["c", "a", "b"],
        })
        assert bout_sequences(table)["b0"] == ["a", "b", "c"]

    def test_separates_bouts(self):
        import pandas as pd

        table = pd.DataFrame({
            "audio_file": ["b0", "b0", "b1"],
            "onset_s": [0.1, 0.2, 0.1],
            "label": ["a", "b", "c"],
        })
        assert set(bout_sequences(table)) == {"b0", "b1"}

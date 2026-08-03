"""Syntax drift — change in the order syllables are produced in.

Birdsong is a learned motor sequence with probabilistic syntax, and a manipulation could
reorder syllables while leaving every individual syllable acoustically untouched. Both the
centroid and dispersion metrics would read zero throughout such a change. This is the third
axis: not *where* a syllable sits or *how tightly*, but *what follows what*.

The statistic is the Jensen–Shannon divergence between the two days' bigram (transition)
distributions, in bits, so it is bounded on [0, 1] and symmetric.

**One property changes how it must be used.** Unlike the acoustic metrics, this is
non-negative by construction, so its null is not centred on zero and no bias correction
could make it so: two finite samples drawn from the *same* syntax still differ, and the
smaller the samples the more they differ. The split-half null is therefore not a refinement
here, it is the only thing that says how much divergence is normal at a given number of
bouts. A raw divergence value means nothing on its own.

Transitions never span bouts: the last syllable of one bout does not precede the first of
the next, and counting it that way would invent transitions the bird never produced.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bout_sequences",
    "split_half_syntax_null",
    "syntax_divergence",
    "transition_counts",
]


def bout_sequences(table) -> dict[str, list[str]]:
    """Extract the ordered syllable sequence of each bout from a syllable table."""
    ordered = table.sort_values(["audio_file", "onset_s"], kind="stable")
    return {
        str(name): [str(label) for label in group["label"]]
        for name, group in ordered.groupby("audio_file", sort=False)
    }


def transition_counts(sequences: dict, types) -> np.ndarray:
    """Count syllable-to-syllable transitions, within bouts only."""
    index = {label: position for position, label in enumerate(types)}
    counts = np.zeros((len(types), len(types)), dtype=float)
    for sequence in sequences.values():
        for first, second in zip(sequence, sequence[1:]):
            if first in index and second in index:
                counts[index[first], index[second]] += 1
    return counts


def _bigram_distribution(sequences: dict, types) -> np.ndarray:
    counts = transition_counts(sequences, types).ravel()
    total = counts.sum()
    if total == 0:
        raise ValueError(
            "no within-bout syllable transitions found; syntax needs bouts with at "
            "least two labelled syllables"
        )
    return counts / total


def syntax_divergence(sequences_a: dict, sequences_b: dict, types) -> float:
    """Jensen–Shannon divergence (bits) between two days' transition distributions."""
    if not sequences_a or not sequences_b:
        raise ValueError("both days need at least one bout")

    p = _bigram_distribution(sequences_a, types)
    q = _bigram_distribution(sequences_b, types)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)


def split_half_syntax_null(
    sequences: dict,
    types,
    n_draws: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Divergence expected when nothing changed, by splitting one day's bouts in half.

    Bouts are the split unit, as everywhere else. The resulting distribution is strictly
    positive; its 95th percentile is the threshold a real between-day divergence has to
    clear.
    """
    keys = np.array(list(sequences))
    if len(keys) < 4:
        raise ValueError(
            f"need at least 4 bouts to split a day into two estimable halves; "
            f"got {len(keys)}"
        )

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_draws):
        shuffled = rng.permutation(keys)
        half = len(shuffled) // 2
        left = {k: sequences[k] for k in shuffled[:half]}
        right = {k: sequences[k] for k in shuffled[half:]}
        try:
            values.append(syntax_divergence(left, right, types))
        except ValueError:
            continue
    return np.asarray(values)

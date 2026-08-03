"""Score recovered syllable labels against hand labels.

Two views, because they fail differently:

* :func:`label_accuracy` -- per-position agreement, for sequences already aligned
  one-to-one (e.g. labels attached to matched segments).
* :func:`syllable_error_rate` -- Levenshtein distance over the label *sequence*,
  normalised by the true length. This is the birdsong analogue of word error rate, and
  unlike per-segment accuracy it charges for dropped and invented syllables. Since song
  syntax is part of what drift means, a metric blind to insertions and deletions would
  miss exactly the changes worth detecting.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

__all__ = [
    "confusion_counts",
    "edit_distance",
    "label_accuracy",
    "syllable_error_rate",
]


def edit_distance(true: Sequence, predicted: Sequence) -> int:
    """Levenshtein distance with unit substitution, insertion and deletion costs.

    Accepts strings or sequences of multi-character labels -- canary labels are numeric
    strings, where treating the input as characters would silently mis-score.
    """
    if len(true) < len(predicted):
        true, predicted = predicted, true

    previous = list(range(len(predicted) + 1))
    for i, true_item in enumerate(true, start=1):
        current = [i]
        for j, predicted_item in enumerate(predicted, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (true_item != predicted_item),  # substitution
                )
            )
        previous = current
    return previous[-1]


def syllable_error_rate(true: Sequence, predicted: Sequence) -> float:
    """Edit distance normalised by the length of the true sequence.

    May exceed 1.0 when the prediction is longer than the truth. Raises if the truth is
    empty, where the rate is undefined rather than zero.
    """
    if len(true) == 0:
        raise ValueError("syllable error rate is undefined for an empty true sequence")
    return edit_distance(true, predicted) / len(true)


def label_accuracy(true: Sequence, predicted: Sequence) -> float:
    """Fraction of positions that agree, for already-aligned sequences."""
    if len(true) != len(predicted):
        raise ValueError(
            f"aligned sequences required: got {len(true)} true and "
            f"{len(predicted)} predicted labels"
        )
    if not true:
        return 1.0
    return sum(t == p for t, p in zip(true, predicted)) / len(true)


def confusion_counts(true: Sequence, predicted: Sequence) -> dict[tuple, int]:
    """Count ``(true_label, predicted_label)`` pairs for aligned sequences."""
    if len(true) != len(predicted):
        raise ValueError(
            f"aligned sequences required: got {len(true)} true and "
            f"{len(predicted)} predicted labels"
        )
    return dict(Counter(zip(true, predicted)))

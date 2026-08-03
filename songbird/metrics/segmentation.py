"""Score a recovered segmentation against hand-labelled ground truth.

Matching is one-to-one and greedy in time order. For non-overlapping intervals sorted by
onset, the greedy left-to-right sweep is optimal, and it stays linear -- which matters at
the ~10^5--10^6 syllable scale these datasets reach, where an assignment solver would not.

One-to-one is the point: a segmenter that fires twice per syllable would otherwise score
perfect recall while producing unusable output.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Scores", "BoundaryScores", "match_segments", "segment_scores", "boundary_scores"]

Segment = tuple[float, float]


@dataclass(frozen=True)
class Scores:
    """Precision/recall/F1 with the counts they were computed from."""

    precision: float
    recall: float
    f1: float
    n_true: int
    n_pred: int
    n_matched: int


@dataclass(frozen=True)
class BoundaryScores:
    """Onset and offset scored independently of whole-segment agreement."""

    onset: Scores
    offset: Scores


def _score(n_true: int, n_pred: int, n_matched: int) -> Scores:
    # Both empty is vacuously perfect: nothing to find, nothing wrongly found.
    if n_true == 0 and n_pred == 0:
        return Scores(1.0, 1.0, 1.0, 0, 0, 0)
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_true if n_true else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return Scores(precision, recall, f1, n_true, n_pred, n_matched)


def _match_points(
    true_points: list[float], pred_points: list[float], tolerance_s: float
) -> int:
    """Count one-to-one matches between two sorted point sets within a tolerance."""
    true_order = sorted(range(len(true_points)), key=lambda i: true_points[i])
    pred_order = sorted(range(len(pred_points)), key=lambda i: pred_points[i])
    matched = 0
    p = 0
    for t in true_order:
        while p < len(pred_order) and pred_points[pred_order[p]] < true_points[t] - tolerance_s:
            p += 1
        if p < len(pred_order) and abs(pred_points[pred_order[p]] - true_points[t]) <= tolerance_s:
            matched += 1
            p += 1
    return matched


def match_segments(
    true: list[Segment], pred: list[Segment], tolerance_s: float
) -> list[tuple[int, int]]:
    """Return ``(true_index, pred_index)`` pairs whose *both* boundaries agree.

    A pair matches only when ``|onset_pred - onset_true| <= tolerance_s`` **and**
    ``|offset_pred - offset_true| <= tolerance_s``.
    """
    true_order = sorted(range(len(true)), key=lambda i: true[i][0])
    pred_order = sorted(range(len(pred)), key=lambda i: pred[i][0])

    matches: list[tuple[int, int]] = []
    p = 0
    for t in true_order:
        t_on, t_off = true[t]
        while p < len(pred_order) and pred[pred_order[p]][0] < t_on - tolerance_s:
            p += 1
        if p >= len(pred_order):
            break
        candidate = pred_order[p]
        p_on, p_off = pred[candidate]
        if abs(p_on - t_on) <= tolerance_s and abs(p_off - t_off) <= tolerance_s:
            matches.append((t, candidate))
            p += 1
    return matches


def segment_scores(
    true: list[Segment], pred: list[Segment], tolerance_s: float
) -> Scores:
    """Score whole-segment agreement: both boundaries must land within tolerance."""
    return _score(len(true), len(pred), len(match_segments(true, pred, tolerance_s)))


def boundary_scores(
    true: list[Segment], pred: list[Segment], tolerance_s: float
) -> BoundaryScores:
    """Score onsets and offsets independently.

    Useful for diagnosis: an amplitude segmenter typically nails onsets and runs long on
    offsets, which whole-segment scoring reports only as a flat failure.
    """
    onset = _score(
        len(true),
        len(pred),
        _match_points([s[0] for s in true], [s[0] for s in pred], tolerance_s),
    )
    offset = _score(
        len(true),
        len(pred),
        _match_points([s[1] for s in true], [s[1] for s in pred], tolerance_s),
    )
    return BoundaryScores(onset=onset, offset=offset)

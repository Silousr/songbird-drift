"""Does the embedding's geometry mean anything?

The Phase 2 gate. A drift metric is a distance in some embedding space, so before any
drift number is trusted the space itself has to be shown faithful: syllables that humans
gave the same label must sit near each other, and syllables given different labels must
sit apart. If they do not, movement in that space is noise wearing the costume of signal.

Three complementary views:

* :func:`knn_label_recovery` -- can a held-out syllable's human label be predicted from
  its neighbours? Directly tests "near in this space" == "same syllable type".
* :func:`nearest_neighbour_purity` -- the k=1, leave-one-out version, with the point
  itself excluded.
* :func:`label_silhouette` -- are within-type distances smaller than between-type
  distances, in units of the distances themselves?

None of these prove the embedding captures *acoustic* differences beyond type identity;
they establish that it has not destroyed type identity, which is the necessary condition.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

__all__ = [
    "knn_label_recovery",
    "label_silhouette",
    "nearest_neighbour_purity",
    "within_type_distance_correlation",
]


def _check(x: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError(
            f"{name}: {len(x)} points but {len(y)} labels"
        )
    return x, y


def knn_label_recovery(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    k: int = 5,
    per_label: bool = False,
):
    """Accuracy of predicting held-out human labels from embedding neighbours.

    Set ``per_label=True`` to also get a per-label breakdown, which is what reveals
    whether a high overall score is carried by one abundant syllable type.
    """
    train_x, train_y = _check(train_x, train_y, "train")
    test_x, test_y = _check(test_x, test_y, "test")
    if len(train_x) == 0:
        raise ValueError("empty training set")
    if k > len(train_x):
        raise ValueError(f"k={k} exceeds training set size {len(train_x)}")

    model = KNeighborsClassifier(n_neighbors=k).fit(train_x, train_y)
    predicted = model.predict(test_x)
    accuracy = float((predicted == test_y).mean())

    if not per_label:
        return accuracy

    breakdown = {
        str(label): float((predicted[test_y == label] == label).mean())
        for label in np.unique(test_y)
    }
    return accuracy, breakdown


def nearest_neighbour_purity(x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of points whose nearest *other* point shares their label."""
    x, y = _check(x, y, "points")
    if len(x) < 2:
        raise ValueError("need at least two points")
    # Two neighbours, then drop the first: it is the point itself, and including it would
    # make purity 1.0 for any embedding whatsoever.
    finder = NearestNeighbors(n_neighbors=2).fit(x)
    _, indices = finder.kneighbors(x)
    return float((y[indices[:, 1]] == y).mean())


def label_silhouette(x: np.ndarray, y: np.ndarray) -> float:
    """Mean silhouette coefficient using human labels as the clustering."""
    x, y = _check(x, y, "points")
    if len(np.unique(y)) < 2:
        raise ValueError("silhouette needs at least two distinct labels")
    return float(silhouette_score(x, y))


def within_type_distance_correlation(
    embedding: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    max_pairs_per_label: int = 20_000,
    seed: int = 0,
) -> float:
    """Do embedding distances track reference distances *within* each syllable type?

    Returns the Spearman correlation between pairwise distances in ``embedding`` and in
    ``reference``, computed separately within each label and averaged over labels
    weighted by the number of pairs. Between-type pairs are never compared, so a
    representation that merely spreads the types apart scores no better.

    **This is the half of the fidelity gate that label recovery cannot cover.** Drift in
    crystallised song is a within-type phenomenon: renditions of one syllable slowly
    change shape while staying recognisably that syllable. An embedding that maps every
    rendition onto its type centroid classifies perfectly and yet carries no within-type
    information at all, so a drift metric computed in it would read ~0 regardless of what
    the bird did. A low value here means distances in this space do not correspond to
    acoustic differences, and any drift measured in it is noise.

    Labels with fewer than two renditions contribute no pairs and are skipped.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    embedding = np.asarray(embedding)
    reference = np.asarray(reference)
    labels = np.asarray(labels)
    if not (len(embedding) == len(reference) == len(labels)):
        raise ValueError(
            f"length mismatch: {len(embedding)} embedding, {len(reference)} reference, "
            f"{len(labels)} labels"
        )

    rng = np.random.default_rng(seed)
    correlations, weights = [], []
    for label in np.unique(labels):
        index = np.flatnonzero(labels == label)
        if len(index) < 3:
            continue
        # Cap the pair count: pdist is O(n^2) and abundant syllables would dominate.
        max_n = int((1 + np.sqrt(1 + 8 * max_pairs_per_label)) / 2)
        if len(index) > max_n:
            index = rng.choice(index, max_n, replace=False)

        embedding_distances = pdist(embedding[index])
        reference_distances = pdist(reference[index])
        if np.allclose(embedding_distances, embedding_distances[0]):
            correlations.append(0.0)
        else:
            rho = spearmanr(embedding_distances, reference_distances).statistic
            correlations.append(0.0 if np.isnan(rho) else float(rho))
        weights.append(len(embedding_distances))

    if not correlations:
        raise ValueError("no label had enough renditions to form within-type pairs")
    return float(np.average(correlations, weights=weights))

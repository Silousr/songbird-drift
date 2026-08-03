"""Tests for embedding fidelity metrics.

The Phase 2 gate asks one question: do distances in the embedding correspond to real
syllable-type differences? If they do not, any drift measured as movement in that space is
measuring noise. These metrics answer it two ways -- whether a held-out syllable's
neighbours share its human label, and whether types form separated clusters.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.metrics.fidelity import (
    knn_label_recovery,
    label_silhouette,
    nearest_neighbour_purity,
)


def clustered(n_per_class=30, n_classes=3, spread=0.05, seed=0):
    """Well-separated Gaussian blobs, one per label."""
    rng = np.random.default_rng(seed)
    centres = np.eye(n_classes) * 5.0
    points, labels = [], []
    for index in range(n_classes):
        points.append(centres[index] + spread * rng.standard_normal((n_per_class, n_classes)))
        labels += [f"syl{index}"] * n_per_class
    return np.vstack(points), np.array(labels)


def overlapping(n_per_class=30, n_classes=3, seed=0):
    """Labels assigned at random to points from a single blob -- no structure at all."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n_per_class * n_classes, n_classes))
    labels = np.array([f"syl{i % n_classes}" for i in range(len(points))])
    return points, labels


class TestKnnLabelRecovery:
    def test_separated_clusters_are_fully_recovered(self):
        train_x, train_y = clustered(seed=0)
        test_x, test_y = clustered(seed=1)
        assert knn_label_recovery(train_x, train_y, test_x, test_y, k=5) == 1.0

    def test_unstructured_embedding_scores_near_chance(self):
        train_x, train_y = overlapping(seed=0)
        test_x, test_y = overlapping(seed=1)
        accuracy = knn_label_recovery(train_x, train_y, test_x, test_y, k=5)
        assert accuracy < 0.6  # chance is 1/3

    def test_rejects_mismatched_lengths(self):
        x, y = clustered()
        with pytest.raises(ValueError):
            knn_label_recovery(x, y[:-1], x, y, k=5)

    def test_rejects_k_larger_than_training_set(self):
        x, y = clustered(n_per_class=2, n_classes=2)
        with pytest.raises(ValueError):
            knn_label_recovery(x, y, x, y, k=99)

    def test_rejects_empty_training_set(self):
        x, y = clustered()
        with pytest.raises(ValueError):
            knn_label_recovery(np.empty((0, 3)), np.array([]), x, y, k=1)

    def test_returns_per_label_breakdown_when_requested(self):
        train_x, train_y = clustered(seed=0)
        test_x, test_y = clustered(seed=1)
        accuracy, per_label = knn_label_recovery(
            train_x, train_y, test_x, test_y, k=5, per_label=True
        )
        assert set(per_label) == {"syl0", "syl1", "syl2"}
        assert all(value == 1.0 for value in per_label.values())


class TestLabelSilhouette:
    def test_separated_clusters_score_high(self):
        x, y = clustered()
        assert label_silhouette(x, y) > 0.8

    def test_unstructured_data_scores_near_zero(self):
        x, y = overlapping()
        assert abs(label_silhouette(x, y)) < 0.2

    def test_requires_at_least_two_labels(self):
        x, _ = clustered(n_classes=1)
        with pytest.raises(ValueError):
            label_silhouette(x, np.array(["only"] * len(x)))


class TestNearestNeighbourPurity:
    def test_separated_clusters_are_pure(self):
        x, y = clustered()
        assert nearest_neighbour_purity(x, y) == 1.0

    def test_excludes_the_point_itself(self):
        # Without self-exclusion every point is trivially its own nearest neighbour and
        # purity would be 1.0 regardless of structure.
        x, y = overlapping()
        assert nearest_neighbour_purity(x, y) < 0.9

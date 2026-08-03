"""Tests for whether an embedding preserves variation *within* a syllable type.

This is the half of the Phase 2 gate that a label-recovery score cannot cover. Drift in
crystallised song is a within-type phenomenon: renditions of syllable 'b' gradually change
shape while remaining recognisably 'b'. An embedding can therefore score ~99% on type
classification while having collapsed every rendition of a type onto its centroid -- and a
drift metric computed in that space would read approximately zero no matter what the bird
did.

So: within each label, do distances in the embedding track distances in the raw acoustic
representation?
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.metrics.fidelity import within_type_distance_correlation


def reference_and_labels(n_per_class=40, n_classes=3, dims=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_classes, dims)) * 10
    points, labels = [], []
    for index in range(n_classes):
        points.append(centres[index] + rng.standard_normal((n_per_class, dims)))
        labels += [f"syl{index}"] * n_per_class
    return np.vstack(points), np.array(labels)


class TestWithinTypeDistanceCorrelation:
    def test_identity_embedding_correlates_perfectly(self):
        reference, labels = reference_and_labels()
        rho = within_type_distance_correlation(reference, reference, labels)
        assert rho == pytest.approx(1.0, abs=1e-6)

    def test_rotation_and_scaling_preserve_correlation(self):
        reference, labels = reference_and_labels()
        rng = np.random.default_rng(1)
        rotation, _ = np.linalg.qr(rng.standard_normal((reference.shape[1],) * 2))
        rho = within_type_distance_correlation(reference @ rotation * 3.0, reference, labels)
        assert rho > 0.99

    def test_collapsing_within_type_structure_destroys_correlation(self):
        # Every rendition mapped to its type centroid: perfect classification, zero
        # within-type information. This is the failure mode the metric exists to catch.
        reference, labels = reference_and_labels()
        collapsed = np.zeros_like(reference)
        for label in np.unique(labels):
            mask = labels == label
            collapsed[mask] = reference[mask].mean(axis=0)
        rho = within_type_distance_correlation(collapsed, reference, labels)
        assert abs(rho) < 0.2

    def test_random_embedding_has_no_correlation(self):
        reference, labels = reference_and_labels()
        rng = np.random.default_rng(2)
        rho = within_type_distance_correlation(
            rng.standard_normal(reference.shape), reference, labels
        )
        assert abs(rho) < 0.2

    def test_ignores_between_type_distances(self):
        # Push the type centroids far apart without touching within-type geometry;
        # a metric that leaked between-type distances would change its answer.
        reference, labels = reference_and_labels()
        shifted = reference.copy()
        for index, label in enumerate(np.unique(labels)):
            shifted[labels == label] += index * 500.0
        assert within_type_distance_correlation(shifted, reference, labels) > 0.99

    def test_rejects_mismatched_lengths(self):
        reference, labels = reference_and_labels()
        with pytest.raises(ValueError):
            within_type_distance_correlation(reference, reference, labels[:-1])

    def test_skips_labels_with_too_few_renditions(self):
        reference, labels = reference_and_labels(n_per_class=40, n_classes=2)
        labels = labels.copy()
        labels[0] = "singleton"  # only one rendition; no within-type pairs exist
        rho = within_type_distance_correlation(reference, reference, labels)
        assert rho == pytest.approx(1.0, abs=1e-6)

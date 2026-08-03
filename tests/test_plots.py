"""Tests for the plotting helpers.

Figures are how a result gets read, so they are part of the deliverable. These check the
things that make a drift plot honest rather than decorative: the noise floor is always
drawn, and negative drift is shown rather than clipped -- clipping would hide exactly the
half of the null distribution that shows the estimator is unbiased.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from songbird.plots import plot_drift_vs_separation, plot_power_curve
from songbird.pipeline import AnalysisConfig, analyse
from songbird.ingest.generic import load_from_manifest, load_manifest
from tests.test_pipeline import make_lab_dataset


@pytest.fixture
def result(tmp_path):
    manifest = make_lab_dataset(tmp_path / "lab",
                                days=("2024-05-01", "2024-05-02", "2024-05-04"))
    table = load_from_manifest(load_manifest(manifest))
    return analyse(table, AnalysisConfig(n_pca=8, min_renditions=10, n_boot=60,
                                         n_null=60, n_freq_bins=16))


class TestPlotDriftVsSeparation:
    def test_returns_a_figure(self, result):
        figure = plot_drift_vs_separation(result)
        assert figure.axes

    def test_draws_the_noise_floor(self, result):
        axes = plot_drift_vs_separation(result).axes[0]
        labels = [line.get_label() for line in axes.get_lines()]
        assert any("floor" in str(label).lower() for label in labels)

    def test_does_not_clip_negative_drift(self, result):
        # Negative estimates are information: they say the observed separation is smaller
        # than sampling noise alone would produce.
        axes = plot_drift_vs_separation(result).axes[0]
        assert axes.get_ylim()[0] < 0 or min(
            min(line.get_ydata(), default=0) for line in axes.get_lines()
        ) >= 0

    def test_labels_both_axes(self, result):
        axes = plot_drift_vs_separation(result).axes[0]
        assert axes.get_xlabel() and axes.get_ylabel()


class TestPlotPowerCurve:
    def test_returns_a_figure(self):
        figure = plot_power_curve({5: 0.5, 20: 0.075, 80: 0.02})
        assert figure.axes

    def test_uses_a_log_y_axis_for_a_curve_spanning_orders_of_magnitude(self):
        axes = plot_power_curve({5: 0.5, 20: 0.075, 80: 0.02}).axes[0]
        assert axes.get_yscale() == "log"

    def test_rejects_an_empty_curve(self):
        with pytest.raises(ValueError):
            plot_power_curve({})

"""Tests for the end-to-end pipeline a lab actually runs.

Exercised on real generated audio, not mocks: syllables are written as tone bursts at the
annotated positions, so feature extraction, embedding, drift and the noise floor all run
on genuine waveforms. A pipeline test that mocked the audio would not catch the class of
bug this project kept hitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from songbird.ingest.generic import load_from_manifest, load_manifest
from songbird.pipeline import AnalysisConfig, analyse, extract_features

SR = 32_000


def write_bout(path, syllables, freqs, duration_s=4.0):
    t = np.arange(int(duration_s * SR)) / SR
    audio = np.zeros_like(t)
    for (onset, offset), freq in zip(syllables, freqs):
        mask = (t >= onset) & (t < offset)
        audio[mask] = np.sin(2 * np.pi * freq * t[mask])
    sf.write(path, audio.astype(np.float32), SR)


def make_lab_dataset(root, birds=("b1",), days=("2024-05-01", "2024-05-02"),
                     bouts_per_day=6, jitter=0.0, seed=0):
    """Two syllable types per bout; `jitter` shifts one type's pitch on later days."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for bird in birds:
        for day_index, day in enumerate(days):
            for bout in range(bouts_per_day):
                name = f"{bird}_{day}_{bout}"
                audio_path = root / f"{name}.wav"
                annot_path = root / f"{name}.wav.csv"
                syllables, freqs, lines = [], [], ["onset_s,offset_s,label"]
                for repeat in range(6):
                    base = 0.2 + repeat * 0.55
                    for label, freq in (("a", 3000), ("b", 5000)):
                        onset = base + (0.0 if label == "a" else 0.20)
                        offset = onset + 0.12
                        shift = jitter * day_index if label == "a" else 0.0
                        syllables.append((onset, offset))
                        freqs.append(freq + shift + rng.normal(0, 20))
                        lines.append(f"{onset},{offset},{label}")
                write_bout(audio_path, syllables, freqs)
                annot_path.write_text("\n".join(lines) + "\n")
                rows.append({"bird": bird, "timestamp": f"{day}T08:{bout:02d}:00",
                             "audio_path": str(audio_path),
                             "annot_path": str(annot_path)})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


@pytest.fixture
def table(tmp_path):
    return load_from_manifest(load_manifest(make_lab_dataset(tmp_path / "lab")))


@pytest.fixture
def config():
    return AnalysisConfig(n_pca=8, min_renditions=10, min_bouts=4,
                          n_boot=60, n_null=60, n_freq_bins=16)


class TestExtractFeatures:
    def test_returns_one_row_per_syllable(self, table, config):
        features = extract_features(table, config)
        assert len(features.values) == len(table)

    def test_metadata_arrays_stay_aligned(self, table, config):
        features = extract_features(table, config)
        n = len(features.values)
        assert len(features.labels) == len(features.days) == len(features.bouts) == n

    def test_excludes_requested_labels(self, table):
        features = extract_features(table, AnalysisConfig(exclude_labels=("b",),
                                                          n_freq_bins=16))
        assert set(features.labels) == {"a"}

    def test_caps_syllables_per_day(self, table):
        features = extract_features(table, AnalysisConfig(max_per_day=10, n_freq_bins=16,
                                                          seed=0))
        per_day = pd.Series(features.days).value_counts()
        assert per_day.max() <= 10

    def test_reports_a_clear_error_for_missing_audio(self, table, config):
        broken = table.copy()
        broken["audio_path"] = "/does/not/exist.wav"
        with pytest.raises(FileNotFoundError, match="audio"):
            extract_features(broken, config)


class TestAnalyse:
    def test_produces_a_result_per_bird(self, tmp_path, config):
        manifest = make_lab_dataset(tmp_path / "lab", birds=("b1", "b2"))
        table = load_from_manifest(load_manifest(manifest))
        result = analyse(table, config)
        assert set(result.birds) == {"b1", "b2"}

    def test_reports_a_noise_floor_for_each_metric(self, table, config):
        bird = analyse(table, config).birds["b1"]
        assert bird.centroid_floor > 0
        assert bird.dispersion_floor > 0

    def test_reports_drift_for_each_day_pair(self, table, config):
        bird = analyse(table, config).birds["b1"]
        assert len(bird.day_pairs) == 1
        pair = bird.day_pairs[0]
        assert pair["separation_days"] == 1
        assert "centroid_drift" in pair and "dispersion_drift" in pair

    def test_stable_song_stays_within_the_noise_floor(self, table, config):
        # No injected change between days: the metric must not invent one.
        bird = analyse(table, config).birds["b1"]
        assert bird.day_pairs[0]["n_types_exceeding_centroid_floor"] == 0

    def test_detects_an_injected_acoustic_change(self, tmp_path, config):
        manifest = make_lab_dataset(tmp_path / "shift", jitter=900.0)
        bird = analyse(load_from_manifest(load_manifest(manifest)), config).birds["b1"]
        assert bird.day_pairs[0]["n_types_exceeding_centroid_floor"] >= 1

    def test_summary_is_human_readable(self, table, config):
        text = analyse(table, config).summary()
        assert "b1" in text and "noise floor" in text.lower()

    def test_serialises_to_json_safe_types(self, table, config):
        import json
        json.dumps(analyse(table, config).to_dict())


class TestAnnotationBoundaryDefects:
    """Real annotation files have syllables that overhang the edges of their recording.

    Measured in the TweetyNet canary deposit: 24 syllables in `llb3` start at a negative
    time (down to -9.5 ms), one has an offset at or before its onset, and offsets running
    past the end of the file occur in roughly a tenth of recordings. These are annotation
    boundary artefacts, not corruption, and a pipeline that crashes on them is unusable on
    real data -- while one that silently drops them hides how much data it discarded.
    """

    def test_clamps_a_negative_onset_and_counts_it(self, tmp_path, config):
        manifest = make_lab_dataset(tmp_path / "lab")
        table = load_from_manifest(load_manifest(manifest))
        table.loc[0, "onset_s"] = -0.004
        features = extract_features(table, config)
        assert features.n_clamped_onsets == 1
        assert len(features.values) == len(table)

    def test_clamps_an_offset_past_the_end_of_the_recording(self, tmp_path, config):
        manifest = make_lab_dataset(tmp_path / "lab")
        table = load_from_manifest(load_manifest(manifest))
        table.loc[0, "offset_s"] = 999.0
        features = extract_features(table, config)
        assert features.n_clamped_offsets == 1
        assert len(features.values) == len(table)

    def test_drops_a_syllable_left_empty_after_clamping_and_counts_it(self, tmp_path,
                                                                     config):
        manifest = make_lab_dataset(tmp_path / "lab")
        table = load_from_manifest(load_manifest(manifest))
        table.loc[0, "onset_s"] = 500.0
        table.loc[0, "offset_s"] = 501.0
        features = extract_features(table, config)
        assert features.n_dropped_empty == 1
        assert len(features.values) == len(table) - 1

    def test_clean_data_reports_no_defects(self, table, config):
        features = extract_features(table, config)
        assert features.n_clamped_onsets == 0
        assert features.n_clamped_offsets == 0
        assert features.n_dropped_empty == 0

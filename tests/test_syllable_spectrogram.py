"""Tests for turning an annotated syllable into a fixed-size spectrogram.

Two representation choices are asserted here because both are load-bearing for drift:

* Syllables are **time-padded**, not time-stretched. Stretching to a fixed length would
  discard duration, and syllable duration is one of the acoustic properties most likely
  to change when song destabilises.
* Each syllable is normalised over a fixed dB range below its own peak, which discards
  absolute amplitude. Recording gain is not song, and gain that drifts across days would
  otherwise register as drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from songbird.features.spectrogram import SyllableSpectrogram

SR = 32_000


def tone(duration_s, freq=3000.0, sr=SR, amplitude=1.0):
    t = np.arange(int(duration_s * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def embed_in_silence(signal, onset_s, total_s=1.0, sr=SR):
    audio = np.zeros(int(total_s * sr))
    start = int(onset_s * sr)
    audio[start : start + len(signal)] = signal
    return audio


@pytest.fixture
def spec():
    return SyllableSpectrogram(
        n_fft=512, hop_length=64, freq_range_hz=(500, 10_000),
        max_duration_s=0.2, n_freq_bins=64, dynamic_range_db=60.0,
    )


class TestShape:
    def test_output_shape_is_fixed_for_a_short_syllable(self, spec):
        audio = embed_in_silence(tone(0.05), 0.2)
        assert spec.extract(audio, SR, 0.2, 0.25).shape == (64, spec.n_time_bins)

    def test_output_shape_is_identical_for_a_long_syllable(self, spec):
        audio = embed_in_silence(tone(0.15), 0.2)
        assert spec.extract(audio, SR, 0.2, 0.35).shape == (64, spec.n_time_bins)

    def test_syllables_longer_than_max_duration_are_truncated_not_stretched(self, spec):
        audio = embed_in_silence(tone(0.4), 0.1, total_s=1.0)
        assert spec.extract(audio, SR, 0.1, 0.5).shape == (64, spec.n_time_bins)


class TestPaddingNotStretching:
    def test_longer_syllable_occupies_more_columns(self, spec):
        audio = embed_in_silence(tone(0.15), 0.2)
        short = spec.extract(embed_in_silence(tone(0.04), 0.2), SR, 0.2, 0.24)
        long = spec.extract(audio, SR, 0.2, 0.35)
        # Column energy marks where signal sits; padding contributes near-zero columns.
        assert (long.max(axis=0) > 0.5).sum() > (short.max(axis=0) > 0.5).sum()

    def test_same_frequency_lands_in_the_same_row_regardless_of_duration(self, spec):
        short = spec.extract(embed_in_silence(tone(0.04), 0.2), SR, 0.2, 0.24)
        long = spec.extract(embed_in_silence(tone(0.15), 0.2), SR, 0.2, 0.35)
        assert int(short.mean(axis=1).argmax()) == int(long.mean(axis=1).argmax())


class TestValues:
    def test_values_are_normalised_to_unit_range(self, spec):
        out = spec.extract(embed_in_silence(tone(0.08), 0.2), SR, 0.2, 0.28)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_gain_change_does_not_change_the_representation(self, spec):
        loud = spec.extract(embed_in_silence(tone(0.08, amplitude=1.0), 0.2), SR, 0.2, 0.28)
        quiet = spec.extract(embed_in_silence(tone(0.08, amplitude=0.1), 0.2), SR, 0.2, 0.28)
        assert np.allclose(loud, quiet, atol=1e-6)

    def test_higher_tone_lands_in_a_higher_row(self, spec):
        low = spec.extract(embed_in_silence(tone(0.08, freq=1500), 0.2), SR, 0.2, 0.28)
        high = spec.extract(embed_in_silence(tone(0.08, freq=7000), 0.2), SR, 0.2, 0.28)
        assert int(high.mean(axis=1).argmax()) > int(low.mean(axis=1).argmax())

    def test_two_renditions_of_the_same_tone_are_near_identical(self, spec):
        a = spec.extract(embed_in_silence(tone(0.08), 0.2), SR, 0.2, 0.28)
        b = spec.extract(embed_in_silence(tone(0.08), 0.3), SR, 0.3, 0.38)
        assert np.allclose(a, b, atol=1e-6)


class TestValidation:
    def test_rejects_offset_before_onset(self, spec):
        with pytest.raises(ValueError):
            spec.extract(np.zeros(SR), SR, 0.5, 0.4)

    def test_rejects_syllable_beyond_end_of_audio(self, spec):
        with pytest.raises(ValueError):
            spec.extract(np.zeros(SR), SR, 0.9, 1.5)

    def test_rejects_negative_onset(self, spec):
        with pytest.raises(ValueError):
            spec.extract(np.zeros(SR), SR, -0.1, 0.2)


class TestBatch:
    def test_extract_many_returns_stacked_array(self, spec):
        audio = embed_in_silence(tone(0.05), 0.2)
        out = spec.extract_many(audio, SR, [(0.2, 0.25), (0.2, 0.24)])
        assert out.shape == (2, 64, spec.n_time_bins)

    def test_extract_many_on_empty_list_returns_empty_with_right_dims(self, spec):
        out = spec.extract_many(np.zeros(SR), SR, [])
        assert out.shape == (0, 64, spec.n_time_bins)

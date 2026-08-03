"""Fixed-size spectrogram representation of an annotated syllable.

Follows the AVGN preprocessing idea (log-magnitude spectrogram, clipped dynamic range,
normalised to [0, 1]) with two deliberate choices recorded in DECISION_LOG.md:

* **Time-pad, do not time-stretch.** Rescaling every syllable to a fixed length would
  erase duration, and duration is one of the acoustic properties most likely to move when
  song destabilises. Padding keeps it visible to whatever embedding sits downstream.
* **Normalise within a fixed dB range below each syllable's own peak.** This discards
  absolute amplitude on purpose: recording gain is not song, and gain drifting across days
  would otherwise be indistinguishable from the drift being measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

__all__ = ["SyllableSpectrogram"]


@dataclass(frozen=True)
class SyllableSpectrogram:
    """Turn ``(audio, onset, offset)`` into a fixed ``(n_freq_bins, n_time_bins)`` array.

    Parameters
    ----------
    n_fft, hop_length:
        STFT window and hop in samples. The default hop of 64 at 32 kHz gives 2 ms
        resolution, matching the boundary precision established in Phase 1 (+-5-10 ms).
    freq_range_hz:
        Band retained before resampling to ``n_freq_bins`` rows.
    max_duration_s:
        Syllables are padded up to this length and truncated beyond it.
    n_freq_bins:
        Rows in the output, obtained by averaging the retained STFT bins into equal groups.
    dynamic_range_db:
        Range below each syllable's peak that maps onto [0, 1]; quieter content clips to 0.
    """

    n_fft: int = 512
    hop_length: int = 64
    freq_range_hz: tuple[float, float] = (500.0, 10_000.0)
    max_duration_s: float = 0.2
    n_freq_bins: int = 64
    dynamic_range_db: float = 60.0

    @property
    def n_time_bins(self) -> int:
        """Columns in the output, fixed by ``max_duration_s`` and the hop."""
        return int(round(self.max_duration_s * 32_000 / self.hop_length))

    def _time_bins_for(self, sample_rate: int) -> int:
        return self.n_time_bins

    def extract(
        self, audio: np.ndarray, sample_rate: int, onset_s: float, offset_s: float
    ) -> np.ndarray:
        """Return the normalised spectrogram of one syllable."""
        if offset_s <= onset_s:
            raise ValueError(f"offset {offset_s} does not follow onset {onset_s}")
        if onset_s < 0:
            raise ValueError(f"negative onset: {onset_s}")

        audio = np.asarray(audio, dtype=float)
        start, stop = int(onset_s * sample_rate), int(offset_s * sample_rate)
        if stop > len(audio):
            raise ValueError(
                f"syllable [{onset_s}, {offset_s}] runs past the end of "
                f"{len(audio) / sample_rate:.3f} s of audio"
            )

        max_samples = int(self.max_duration_s * sample_rate)
        segment = audio[start : min(stop, start + max_samples)]

        n_columns = self._time_bins_for(sample_rate)
        if len(segment) < self.n_fft:
            segment = np.pad(segment, (0, self.n_fft - len(segment)))

        frequencies, _, spectrum = stft(
            segment,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=None,
            padded=False,
        )
        magnitude = np.abs(spectrum)

        low, high = self.freq_range_hz
        band = (frequencies >= low) & (frequencies <= high)
        magnitude = magnitude[band]

        # Average the retained bins into n_freq_bins equal groups.
        n_available = magnitude.shape[0]
        edges = np.linspace(0, n_available, self.n_freq_bins + 1).astype(int)
        magnitude = np.stack(
            [
                magnitude[a:b].mean(axis=0) if b > a else magnitude[min(a, n_available - 1)]
                for a, b in zip(edges[:-1], edges[1:])
            ]
        )

        peak = magnitude.max()
        if peak <= 0:
            return np.zeros((self.n_freq_bins, n_columns))

        decibels = 20 * np.log10(np.maximum(magnitude, 1e-12) / peak)
        normalised = np.clip(
            (decibels + self.dynamic_range_db) / self.dynamic_range_db, 0.0, 1.0
        )

        if normalised.shape[1] >= n_columns:
            return normalised[:, :n_columns]
        return np.pad(
            normalised, ((0, 0), (0, n_columns - normalised.shape[1])), constant_values=0.0
        )

    def extract_many(
        self, audio: np.ndarray, sample_rate: int, segments
    ) -> np.ndarray:
        """Stack :meth:`extract` over ``(onset_s, offset_s)`` pairs."""
        segments = list(segments)
        if not segments:
            return np.zeros((0, self.n_freq_bins, self._time_bins_for(sample_rate)))
        return np.stack(
            [self.extract(audio, sample_rate, on, off) for on, off in segments]
        )

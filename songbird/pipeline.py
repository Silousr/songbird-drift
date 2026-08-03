"""End-to-end analysis: a syllable table in, drift and noise floors out.

This is the entry point a lab uses. It wires together the pieces that earlier phases
established, with their defaults already set to the values those phases justified:

* syllable spectrograms, **time-padded not stretched** (duration is drift-sensitive) and
  amplitude-normalised (recording gain is not song);
* a **PCA** embedding, because UMAP destroys the within-type geometry drift lives in;
* the embedding fitted **once on a reference day** — refitting per day moves the axes, and
  axis movement is indistinguishable from song movement;
* **both** drift metrics, centroid and dispersion, each against its own noise floor
  measured by splitting single days in half;
* the **bout** as the sampling unit everywhere.

Every one of those defaults is a finding, not a preference. See DECISION_LOG.md.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from songbird.drift import (
    bootstrap_dispersion_ci,
    bootstrap_drift_ci,
    bout_sequences,
    split_half_dispersion_null,
    split_half_null,
    split_half_syntax_null,
    syntax_divergence,
)
from songbird.features.spectrogram import SyllableSpectrogram

__all__ = ["AnalysisConfig", "AnalysisResult", "BirdResult", "FeatureSet",
           "analyse", "extract_features"]


@dataclass(frozen=True)
class AnalysisConfig:
    """Analysis settings. Defaults are the values earlier phases justified."""

    n_pca: int = 64
    n_freq_bins: int = 32
    max_duration_s: float = 0.2
    hop_length: int = 64
    n_fft: int = 512
    freq_range_hz: tuple[float, float] = (500.0, 10_000.0)
    dynamic_range_db: float = 60.0
    max_per_day: int | None = 2000
    exclude_labels: tuple[str, ...] = ()
    min_renditions: int = 20
    min_bouts: int = 4
    n_boot: int = 400
    n_null: int = 300
    seed: int = 0

    def spectrogram(self) -> SyllableSpectrogram:
        return SyllableSpectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length,
            freq_range_hz=self.freq_range_hz, max_duration_s=self.max_duration_s,
            n_freq_bins=self.n_freq_bins, dynamic_range_db=self.dynamic_range_db,
        )


@dataclass
class FeatureSet:
    """Per-syllable features with the metadata needed for every downstream statistic."""

    values: np.ndarray
    labels: np.ndarray
    days: np.ndarray
    bouts: np.ndarray
    birds: np.ndarray


@dataclass
class BirdResult:
    bird: str
    n_syllables: int
    n_bouts: int
    n_days: int
    days: list[str]
    reference_day: str
    types: list[str]
    within_type_variance: float
    centroid_floor: float
    dispersion_floor: float
    syntax_floor: float
    day_pairs: list[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    birds: dict
    config: dict

    def summary(self) -> str:
        lines = []
        for name, bird in self.birds.items():
            lines.append(
                f"{name}: {bird.n_syllables} syllables, {bird.n_bouts} bouts, "
                f"{bird.n_days} days, {len(bird.types)} syllable types"
            )
            lines.append(
                f"  noise floor  centroid {bird.centroid_floor:.4f} (standardised "
                f"{bird.centroid_floor / bird.within_type_variance:.4f})   "
                f"dispersion {bird.dispersion_floor:.4f} "
                f"({np.exp(bird.dispersion_floor):.2f}x variance)   "
                f"syntax {bird.syntax_floor:.4f} bits"
            )
            for pair in bird.day_pairs:
                lines.append(
                    f"  {pair['day_a']} -> {pair['day_b']} "
                    f"(+{pair['separation_days']}d): "
                    f"centroid {pair['centroid_drift_standardised']:+.4f} "
                    f"[{pair['n_types_exceeding_centroid_floor']}/{pair['n_types']} "
                    f"types over floor], "
                    f"dispersion {pair['dispersion_drift']:+.4f} "
                    f"[{pair['n_types_exceeding_dispersion_floor']}/{pair['n_types']}], "
                    f"syntax {pair['syntax_divergence']:.4f}"
                    f"{' OVER FLOOR' if pair['syntax_exceeds_floor'] else ''}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"config": self.config,
                "birds": {name: asdict(bird) for name, bird in self.birds.items()}}

    def per_bird_drift(self, standardised: bool = True) -> dict:
        """Max centroid drift per bird — the one value per bird the group test wants."""
        key = "centroid_drift_standardised" if standardised else "centroid_drift"
        return {
            name: max((pair[key] for pair in bird.day_pairs), default=float("nan"))
            for name, bird in self.birds.items()
        }


def extract_features(table: pd.DataFrame, config: AnalysisConfig) -> FeatureSet:
    """Turn annotated syllables into fixed-size spectrogram features.

    Audio is opened once per bout, not once per syllable.
    """
    if "audio_path" not in table.columns:
        raise ValueError(
            "syllable table has no 'audio_path' column; use one of the loaders in "
            "songbird.ingest, which populate it"
        )

    rng = np.random.default_rng(config.seed)
    spec = config.spectrogram()

    working = table
    if config.exclude_labels:
        working = working[~working["label"].isin(config.exclude_labels)]
    if config.max_per_day:
        keep = []
        for _, group in working.groupby(["bird", "day"], sort=False):
            if len(group) > config.max_per_day:
                group = group.iloc[rng.choice(len(group), config.max_per_day,
                                              replace=False)]
            keep.append(group)
        working = pd.concat(keep) if keep else working
    working = working.sort_values(["bird", "timestamp", "onset_s"], kind="stable")

    values, labels, days, bouts, birds = [], [], [], [], []
    for audio_path, group in working.groupby("audio_path", sort=False):
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"audio file not found: {path}")
        audio, sample_rate = sf.read(path, dtype="float32")
        limit = len(audio) / sample_rate
        for row in group.itertuples():
            offset = min(row.offset_s, limit)
            if offset <= row.onset_s:
                continue
            values.append(
                spec.extract(audio, sample_rate, row.onset_s, offset).astype(np.float32)
            )
            labels.append(row.label)
            days.append(str(row.day))
            bouts.append(row.audio_file)
            birds.append(row.bird)

    stacked = (np.stack(values).reshape(len(values), -1) if values
               else np.zeros((0, config.n_freq_bins * spec.n_time_bins), np.float32))
    return FeatureSet(stacked, np.array(labels), np.array(days),
                      np.array(bouts), np.array(birds))


def _syntax_by_day(table, bird: str, types) -> tuple[dict, float]:
    """Per-day bout sequences and the within-day syntax floor.

    Uses the FULL table, never the per-day subsample: dropping syllables at random would
    splice together transitions the bird never produced, which is precisely the quantity
    being measured.
    """
    rows = table[(table["bird"] == bird) & (table["label"].isin(types))]
    per_day, nulls = {}, []
    for day, group in rows.groupby("day", sort=True):
        sequences = bout_sequences(group)
        per_day[str(day)] = sequences
        if len(sequences) >= 4:
            try:
                nulls.extend(split_half_syntax_null(sequences, types, n_draws=100,
                                                    seed=0).tolist())
            except ValueError:
                continue
    floor = float(np.percentile(nulls, 95)) if nulls else float("nan")
    return per_day, floor


def _analyse_bird(features: FeatureSet, config: AnalysisConfig, bird: str,
                  table=None) -> BirdResult:
    from sklearn.decomposition import PCA

    mask = features.birds == bird
    values, labels = features.values[mask], features.labels[mask]
    days, bouts = features.days[mask], features.bouts[mask]

    unique_days = sorted(np.unique(days))
    reference = unique_days[0]
    n_components = min(config.n_pca, int(mask.sum()), values.shape[1])
    embedding = PCA(n_components=n_components,
                    random_state=config.seed).fit(values[days == reference])
    z = embedding.transform(values)

    types = sorted(np.unique(labels))
    on_reference = days == reference
    variances = [z[on_reference & (labels == t)].var(axis=0, ddof=1).sum()
                 for t in types if (on_reference & (labels == t)).sum() > 1]
    scale = float(np.mean(variances)) if variances else 1.0

    centroid_null, dispersion_null = [], []
    for day in unique_days:
        for syllable in types:
            selection = (days == day) & (labels == syllable)
            if (selection.sum() < config.min_renditions
                    or len(np.unique(bouts[selection])) < config.min_bouts):
                continue
            centroid_null.extend(split_half_null(
                z[selection], bouts[selection], n_draws=config.n_null,
                seed=config.seed).tolist())
            dispersion_null.extend(split_half_dispersion_null(
                z[selection], bouts[selection], n_draws=config.n_null,
                seed=config.seed).tolist())

    centroid_floor = float(np.percentile(centroid_null, 95)) if centroid_null else np.nan
    dispersion_floor = (float(np.percentile(np.abs(dispersion_null), 95))
                        if dispersion_null else np.nan)

    syntax_by_day, syntax_floor = ({}, float("nan"))
    if table is not None:
        syntax_by_day, syntax_floor = _syntax_by_day(table, bird, types)

    pairs = []
    for i, day_a in enumerate(unique_days):
        for day_b in unique_days[i + 1:]:
            centroids, dispersions = [], []
            over_centroid = over_dispersion = 0
            for syllable in types:
                mask_a = (days == day_a) & (labels == syllable)
                mask_b = (days == day_b) & (labels == syllable)
                if min(mask_a.sum(), mask_b.sum()) < config.min_renditions:
                    continue
                if min(len(np.unique(bouts[mask_a])),
                       len(np.unique(bouts[mask_b]))) < 2:
                    continue
                estimate, low, _ = bootstrap_drift_ci(
                    z[mask_a], bouts[mask_a], z[mask_b], bouts[mask_b],
                    n_boot=config.n_boot, seed=config.seed)
                centroids.append(estimate)
                over_centroid += int(low > centroid_floor)

                value, d_low, d_high = bootstrap_dispersion_ci(
                    z[mask_a], bouts[mask_a], z[mask_b], bouts[mask_b],
                    n_boot=config.n_boot, seed=config.seed)
                dispersions.append(value)
                over_dispersion += int(d_low > dispersion_floor
                                       or d_high < -dispersion_floor)
            if not centroids:
                continue
            separation = (dt.date.fromisoformat(day_b)
                          - dt.date.fromisoformat(day_a)).days
            mean_centroid = float(np.mean(centroids))

            divergence = float("nan")
            if day_a in syntax_by_day and day_b in syntax_by_day:
                try:
                    divergence = syntax_divergence(syntax_by_day[day_a],
                                                   syntax_by_day[day_b], types)
                except ValueError:
                    divergence = float("nan")

            pairs.append({
                "syntax_divergence": divergence,
                "syntax_exceeds_floor": bool(
                    not np.isnan(divergence) and not np.isnan(syntax_floor)
                    and divergence > syntax_floor
                ),
                "day_a": day_a, "day_b": day_b, "separation_days": separation,
                "n_types": len(centroids),
                "centroid_drift": mean_centroid,
                "centroid_drift_standardised": mean_centroid / scale,
                "dispersion_drift": float(np.mean(dispersions)),
                "dispersion_drift_abs": float(np.mean(np.abs(dispersions))),
                "n_types_exceeding_centroid_floor": over_centroid,
                "n_types_exceeding_dispersion_floor": over_dispersion,
            })

    return BirdResult(
        bird=bird, n_syllables=int(mask.sum()), n_bouts=int(len(np.unique(bouts))),
        n_days=len(unique_days), days=list(unique_days), reference_day=reference,
        types=[str(t) for t in types], within_type_variance=scale,
        centroid_floor=centroid_floor, dispersion_floor=dispersion_floor,
        syntax_floor=syntax_floor, day_pairs=pairs,
    )


def analyse(table: pd.DataFrame, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Run the full analysis for every bird in a syllable table."""
    config = config or AnalysisConfig()
    features = extract_features(table, config)
    if len(features.values) == 0:
        raise ValueError("no syllables survived feature extraction")

    results = {}
    for bird in sorted(np.unique(features.birds)):
        if len(np.unique(features.days[features.birds == bird])) < 2:
            continue
        results[str(bird)] = _analyse_bird(features, config, bird, table)
    if not results:
        raise ValueError(
            "no bird had at least two recording days; drift needs a time axis"
        )
    return AnalysisResult(birds=results, config=asdict(config))

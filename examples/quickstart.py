"""End-to-end worked example: a two-group experiment, from raw audio to a p-value.

Run this after installing to check the toolkit works and to see the whole workflow in one
place. It takes about a minute and needs no downloads.

It simulates a small experiment -- 4 treated birds, 4 controls, one baseline day and one
post-manipulation day each -- with a known acoustic shift applied only to the treated
birds. Then it runs exactly the pipeline a real experiment would:

    audio + annotations  ->  syllable table  ->  embedding
    ->  per-bird drift vs each bird's own noise floor
    ->  treated vs control, with the bird as the unit of replication

The expected result is printed at the end, so a failure is obvious.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from songbird.group import compare_groups
from songbird.ingest.generic import load_from_manifest, load_manifest
from songbird.pipeline import AnalysisConfig, analyse

SR = 32_000
N_PER_GROUP = 4
BOUTS_PER_DAY = 10
SHIFT_HZ = 700.0  # applied to treated birds on the post day only


def write_bout(path, syllables, freqs, duration_s=5.0):
    t = np.arange(int(duration_s * SR)) / SR
    audio = np.zeros_like(t)
    for (onset, offset), freq in zip(syllables, freqs):
        mask = (t >= onset) & (t < offset)
        audio[mask] = np.sin(2 * np.pi * freq * t[mask])
    sf.write(path, audio.astype(np.float32), SR)


def simulate(root: Path) -> Path:
    """Write audio, annotations and a manifest for a small two-group experiment."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in ("treated", "control"):
        for index in range(N_PER_GROUP):
            bird = f"{group[:4]}{index}"
            for day_index, day in enumerate(("2024-05-01", "2024-05-08")):
                shift = SHIFT_HZ if (group == "treated" and day_index == 1) else 0.0
                for bout in range(BOUTS_PER_DAY):
                    stem = f"{bird}_{day}_{bout}"
                    audio_path = root / f"{stem}.wav"
                    annot_path = root / f"{stem}.wav.csv"
                    syllables, freqs = [], []
                    lines = ["onset_s,offset_s,label"]
                    for repeat in range(8):
                        base = 0.2 + repeat * 0.55
                        for label, centre in (("a", 3000.0), ("b", 5200.0)):
                            onset = base + (0.0 if label == "a" else 0.22)
                            offset = onset + 0.12
                            syllables.append((onset, offset))
                            # Only syllable 'a' is affected, as a real manipulation might be
                            freqs.append(centre + (shift if label == "a" else 0.0)
                                         + rng.normal(0, 40))
                            lines.append(f"{onset},{offset},{label}")
                    write_bout(audio_path, syllables, freqs)
                    annot_path.write_text("\n".join(lines) + "\n")
                    rows.append({"bird": bird, "group": group,
                                 "timestamp": f"{day}T08:{bout:02d}:00",
                                 "audio_path": str(audio_path),
                                 "annot_path": str(annot_path)})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "recordings"
        print("1. simulating a two-group experiment "
              f"({N_PER_GROUP} treated, {N_PER_GROUP} control, "
              f"2 days, {BOUTS_PER_DAY} bouts/day)")
        manifest_path = simulate(root)

        print("2. loading annotations into the canonical syllable table")
        manifest = load_manifest(manifest_path)
        table = load_from_manifest(manifest)
        print(f"   {len(table)} syllables, {table['bird'].nunique()} birds, "
              f"{table['audio_file'].nunique()} bouts")

        print("3. measuring drift against each bird's own noise floor")
        result = analyse(table, AnalysisConfig(n_pca=16, n_freq_bins=24,
                                               min_renditions=15, min_bouts=4,
                                               n_boot=120, n_null=120))
        print(result.summary())

        print("\n4. comparing treated with control "
              "(one drift value per bird -- the bird is the unit)")
        drift = result.per_bird_drift()
        groups = table.groupby("bird")["group"].first().to_dict()
        treated = np.array([v for b, v in drift.items() if groups[b] == "treated"])
        control = np.array([v for b, v in drift.items() if groups[b] == "control"])
        comparison = compare_groups(treated, control, alternative="greater", seed=0)
        print(comparison.summary())

        detected = comparison.p_value < 0.05
        print("\n" + "=" * 72)
        print("EXPECTED: treated birds drift more than controls, p < 0.05.")
        print(f"OBSERVED: p = {comparison.p_value:.4f} -> "
              f"{'effect detected' if detected else 'NOT DETECTED'}")
        print("Smallest achievable p with 4 birds per group is 1/71 = 0.014, so this "
              "design\nhas little margin -- which is the point of running "
              "`songbird plan` first.")
        print("=" * 72)
        return 0 if detected else 1


if __name__ == "__main__":
    raise SystemExit(main())

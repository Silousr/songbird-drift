"""Extract fixed-size spectrograms for hand-annotated syllables of one bird.

Produces an ``.npz`` holding, per syllable: the spectrogram, the human label, the
recording day, the source file, onset and duration. Day is carried through because the
Phase 2 fidelity check must be evaluated *across days*, not only within one.

Usage::

    python scripts/phase2_build_syllable_set.py --root data/bfsongrepo --bird gy6or6 \
        --out results/phase2/gy6or6_syllables.npz --max-per-day 2000

Rare off-schema labels are dropped (``--min-label-count``). In this dataset those are
contact calls and unclear sounds marked ``0``/``x``/``y``/``z``, not song syllables;
mixing them into a syllable-type inventory would corrupt the fidelity measurement it is
supposed to validate. The count dropped is reported and stored.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from songbird.features.spectrogram import SyllableSpectrogram
from songbird.ingest.bfsongrepo import load_day


def build(root: Path, bird: str, spec: SyllableSpectrogram, max_per_day: int | None,
          min_label_count: int, seed: int, exclude_labels: set[str]) -> dict:
    rng = np.random.default_rng(seed)
    day_dirs = sorted(
        p for p in (root / bird).iterdir() if p.is_dir() and re.fullmatch(r"\d{6}", p.name)
    )
    if not day_dirs:
        raise SystemExit(f"no day directories for {bird} under {root}")

    spectrograms, labels, days, files, onsets, durations = [], [], [], [], [], []
    dropped_rare = 0
    skipped_mismatch = 0

    for day_dir in day_dirs:
        table = load_day(day_dir, on_date_mismatch="skip")
        skipped_mismatch += table.attrs.get("n_date_mismatch_files", 0)
        if table.empty:
            continue

        # Exclude by identity first, then by count. Excluding purely on count would
        # silently drop genuinely rare *song* syllables along with the noise classes.
        dropped_rare += int(table["label"].isin(exclude_labels).sum())
        table = table[~table["label"].isin(exclude_labels)]
        counts = table["label"].value_counts()
        keep = set(counts[counts >= min_label_count].index)
        dropped_rare += int((~table["label"].isin(keep)).sum())
        table = table[table["label"].isin(keep)]

        if max_per_day and len(table) > max_per_day:
            table = table.iloc[rng.choice(len(table), max_per_day, replace=False)]
        table = table.sort_values(["audio_file", "onset_s"])

        for audio_file, group in table.groupby("audio_file"):
            audio, sample_rate = sf.read(day_dir / audio_file, dtype="float32")
            duration_s = len(audio) / sample_rate
            for row in group.itertuples():
                # Annotations occasionally run to the very end; clamp rather than drop.
                offset = min(row.offset_s, duration_s)
                if offset <= row.onset_s:
                    continue
                spectrograms.append(
                    spec.extract(audio, sample_rate, row.onset_s, offset).astype(np.float32)
                )
                labels.append(row.label)
                days.append(str(row.day))
                files.append(audio_file)
                onsets.append(row.onset_s)
                durations.append(offset - row.onset_s)

        print(f"  {day_dir.name}: {len(spectrograms)} syllables cumulative")

    return {
        "spectrograms": np.stack(spectrograms),
        "labels": np.array(labels),
        "days": np.array(days),
        "files": np.array(files),
        "onsets": np.array(onsets, dtype=np.float32),
        "durations": np.array(durations, dtype=np.float32),
        "dropped_rare": dropped_rare,
        "skipped_mismatch": skipped_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bird", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-day", type=int, default=2000)
    parser.add_argument("--min-label-count", type=int, default=20)
    parser.add_argument("--exclude-labels", nargs="*", default=["0", "x", "y", "z", "@", "s", "u", "w"],
                        help="off-schema labels (contact calls, unclear sounds), not song syllables")
    parser.add_argument("--n-freq-bins", type=int, default=32)
    parser.add_argument("--max-duration-s", type=float, default=0.2)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = SyllableSpectrogram(
        n_fft=512, hop_length=args.hop_length, freq_range_hz=(500, 10_000),
        max_duration_s=args.max_duration_s, n_freq_bins=args.n_freq_bins,
        dynamic_range_db=60.0,
    )
    print(f"building syllable set for {args.bird}: "
          f"{args.n_freq_bins} x {spec.n_time_bins} spectrograms")

    data = build(args.root, args.bird, spec, args.max_per_day,
                 args.min_label_count, args.seed, set(args.exclude_labels))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        spectrograms=data["spectrograms"], labels=data["labels"], days=data["days"],
        files=data["files"], onsets=data["onsets"], durations=data["durations"],
    )

    unique_labels, counts = np.unique(data["labels"], return_counts=True)
    summary = {
        "bird": args.bird,
        "n_syllables": int(len(data["labels"])),
        "n_days": int(len(np.unique(data["days"]))),
        "days": sorted(np.unique(data["days"]).tolist()),
        "spectrogram_shape": list(data["spectrograms"].shape[1:]),
        "labels": {str(l): int(c) for l, c in zip(unique_labels, counts)},
        "dropped_offschema_or_rare_syllables": int(data["dropped_rare"]),
        "excluded_labels": sorted(args.exclude_labels),
        "skipped_date_mismatch_files": int(data["skipped_mismatch"]),
        "spec_params": {
            "n_fft": spec.n_fft, "hop_length": spec.hop_length,
            "freq_range_hz": list(spec.freq_range_hz),
            "max_duration_s": spec.max_duration_s,
            "n_freq_bins": spec.n_freq_bins,
            "dynamic_range_db": spec.dynamic_range_db,
        },
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Noise floor for canary song over a 9-11 day window.

The Bengalese finch data caps out at 3-day separations, so every floor measured so far
rests on a very short baseline and the reports say plainly that stationarity over longer
windows is untested. The TweetyNet canary deposit (Cohen et al. 2022, CC0) has 9-11
*consecutive* days per bird with dense syllable-level annotation, which roughly triples
the observable window.

This runs the same analysis as `phase3_noise_floor.py` and `phase5_dispersion_floor.py`,
on canary, and reports drift as a function of separation out to ~10 days.

Usage::

    python scripts/canary_noise_floor.py --root data/canary/llb3 --bird llb3 \
        --out results/phase7/canary_llb3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from songbird.ingest.generic import load_flat_annotations
from songbird.pipeline import AnalysisConfig, analyse

# llb3_0002_2018_04_23_14_18_03.wav
TIMESTAMP_PATTERN = r"(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.wav$"
TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"


def find_annotation_csv(root: Path) -> Path:
    candidates = [p for p in root.rglob("*.csv") if p.stat().st_size > 1000]
    if not candidates:
        raise SystemExit(f"no annotation CSV found under {root}")
    return max(candidates, key=lambda p: p.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bird", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--annotation-csv", type=Path)
    parser.add_argument("--n-pca", type=int, default=64)
    parser.add_argument("--max-per-day", type=int, default=1500)
    parser.add_argument("--min-renditions", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    csv = args.annotation_csv or find_annotation_csv(args.root)
    print(f"annotations: {csv} ({csv.stat().st_size / 1e6:.1f} MB)")

    table = load_flat_annotations(
        csv, args.root, bird=args.bird,
        timestamp_pattern=TIMESTAMP_PATTERN, timestamp_format=TIMESTAMP_FORMAT,
        source="canary-tweetynet", on_missing="skip",
    )
    missing = table.attrs.get("n_missing_audio", 0)
    if missing:
        print(f"NOTE: {missing} referenced audio file(s) absent from {args.root}")
    days = sorted({str(d) for d in table["day"]})
    print(f"{len(table)} syllables | {table['label'].nunique()} types | "
          f"{table['audio_file'].nunique()} bouts | {len(days)} days "
          f"({days[0]} .. {days[-1]})\n")

    config = AnalysisConfig(
        n_pca=args.n_pca, max_per_day=args.max_per_day,
        min_renditions=args.min_renditions, n_boot=args.n_boot,
        n_null=args.n_null, seed=args.seed,
    )
    from songbird.pipeline import extract_features
    features = extract_features(table, config)
    repaired = (features.n_clamped_onsets + features.n_clamped_offsets
                + features.n_dropped_empty)
    if repaired:
        print(f"annotation boundary repairs: {features.n_clamped_onsets} negative onsets "
              f"clamped, {features.n_clamped_offsets} offsets clamped to end of file, "
              f"{features.n_dropped_empty} empty after clamping\n")

    result = analyse(table, config)
    bird = result.birds[args.bird]

    print(f"NOISE FLOOR ({args.bird})")
    print(f"  centroid   {bird.centroid_floor:.4f} "
          f"(standardised {bird.centroid_floor / bird.within_type_variance:.4f})")
    print(f"  dispersion {bird.dispersion_floor:.4f} "
          f"({np.exp(bird.dispersion_floor):.2f}x variance)")
    print(f"  syntax     {bird.syntax_floor:.5f} bits\n")

    by_gap: dict[int, dict[str, list]] = {}
    for pair in bird.day_pairs:
        entry = by_gap.setdefault(pair["separation_days"],
                                  {"centroid": [], "dispersion": [], "syntax": [],
                                   "over_c": 0, "over_d": 0, "over_s": 0, "n": 0})
        entry["centroid"].append(pair["centroid_drift_standardised"])
        entry["dispersion"].append(abs(pair["dispersion_drift"]))
        if not np.isnan(pair["syntax_divergence"]):
            entry["syntax"].append(pair["syntax_divergence"])
        entry["over_c"] += pair["n_types_exceeding_centroid_floor"]
        entry["over_d"] += pair["n_types_exceeding_dispersion_floor"]
        entry["over_s"] += int(pair["syntax_exceeds_floor"])
        entry["n"] += pair["n_types"]

    floor_std = bird.centroid_floor / bird.within_type_variance
    print("DRIFT vs DAY SEPARATION")
    print(f"{'gap':>5}{'pairs':>7}{'centroid':>11}{'vs floor':>10}"
          f"{'|disp|':>9}{'syntax':>9}{'types>floor':>13}")
    rows = []
    for gap in sorted(by_gap):
        e = by_gap[gap]
        centroid = float(np.mean(e["centroid"]))
        dispersion = float(np.mean(e["dispersion"]))
        syntax = float(np.mean(e["syntax"])) if e["syntax"] else float("nan")
        print(f"{gap:>5}{len(e['centroid']):>7}{centroid:>11.4f}"
              f"{centroid / floor_std:>10.2f}{dispersion:>9.4f}{syntax:>9.5f}"
              f"{str(e['over_c']) + '/' + str(e['n']):>13}")
        rows.append({"separation_days": gap, "n_pairs": len(e["centroid"]),
                     "centroid_standardised": centroid,
                     "ratio_to_floor": centroid / floor_std,
                     "dispersion_abs": dispersion, "syntax": syntax,
                     "n_types_over_centroid_floor": e["over_c"],
                     "n_type_comparisons": e["n"]})

    print("\n'vs floor' is drift divided by the within-day floor. Below 1.0 means the "
          "change\nis smaller than the bird's own within-day variability.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "bird": args.bird, "source": str(csv), "n_syllables": int(len(table)),
            "n_days": len(days), "days": days,
            "floors": {
                "centroid": bird.centroid_floor,
                "centroid_standardised": floor_std,
                "dispersion": bird.dispersion_floor,
                "dispersion_fold": float(np.exp(bird.dispersion_floor)),
                "syntax_bits": bird.syntax_floor,
            },
            "within_type_variance": bird.within_type_variance,
            "by_separation": rows,
            "day_pairs": bird.day_pairs,
        }, indent=1, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

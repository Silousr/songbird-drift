"""Re-run the developmental validation using the authors' own syllable labels.

Until August 2026 the per-rendition syllable labels in the Duke deposit
(doi:10.7924/r4j38x43h) were locked inside MATLAB MCOS table objects, so the
dispersion validation had to re-derive syllable groupings by k-means clustering
the VAE latents into 12 modes. Sam Brudner then exported the deposited tables to
CSV (README in his export documents the procedure), which exposed the real
labels -- and showed the real inventories are far smaller than the 12 assumed
modes: grn394 and grn397 have TWO song syllable types, the other birds three.

This script repeats the early-versus-crystallised variability comparison with
those real labels. Three things change against the cluster-based run:

* **No selection step at all.** The fraction-matching machinery existed to
  neutralise a distance-cutoff artefact of my own clustering; with authors'
  labels every rendition of a type counts, so the artefact cannot arise.
  (The labels do inherit the authors' retention: only clear song-syllable
  clusters were kept, ~25% of detected sounds, so early sounds too amorphous
  to assign are absent for both windows.)
* **Real bouts.** The `file` column names the source wav per rendition, so the
  split-half floor uses actual bouts instead of blocks of consecutive batch
  files.
* **Rows tagged `Laser On` are excluded** -- they belong to another experiment.

Also recomputed per type: the consecutive-day centroid contrast (the
"consecutive-day comparison finds nothing" claim) and the distance to a fixed
crystallised reference.

Usage::

    python scripts/validate_developmental_labeled.py \
        --csv-dir <dir with {bird}_table.csv> --bird grn394 \
        --out results/validation/labeled_grn394.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from songbird.drift import (
    split_half_dispersion_null,
    unbiased_squared_centroid_distance,
)

LATENTS = [f"latent_{i}" for i in range(1, 33)]

#: Analysis windows in dph, chosen to match the earlier cluster-based run where
#: one existed (grn394, grn397) and by the same logic elsewhere. grn475 and
#: sil469 begin at 72-73 dph, so their "early" window is late sensorimotor
#: rather than truly juvenile -- stated, not hidden.
WINDOWS = {
    "grn394": {"early": (57, 69), "late": (100, 114)},
    "grn395": {"early": (57, 69), "late": (85, 94)},
    "grn397": {"early": (62, 64), "late": (85, 94), "alt_early": (62, 69)},
    "grn475": {"early": (73, 79), "late": (90, 97)},
    "sil469": {"early": (72, 79), "late": (90, 97)},
}

MIN_RENDITIONS_PER_WINDOW = 200
MIN_RENDITIONS_PER_DAY = 30
MIN_BOUTS_PER_DAY = 4


def load_table(csv_dir: Path, bird: str) -> pd.DataFrame:
    usecols = ["type", "file", "dph", "partition"] + LATENTS
    table = pd.read_csv(csv_dir / f"{bird}_table.csv", usecols=usecols,
                        dtype={c: np.float32 for c in LATENTS})
    n_raw = len(table)
    table["dph"] = table["dph"].str.extract(r"(\d+)").astype(int)
    laser = int((table["partition"] == "Laser On").sum())
    table = table[table["partition"] != "Laser On"]
    table = table.dropna(subset=LATENTS)
    print(f"{bird}: {n_raw} rows, {laser} Laser On excluded, "
          f"{n_raw - laser - len(table)} with missing latents dropped")
    return table


def trace_variance(z: np.ndarray) -> float:
    return float(z.var(axis=0, ddof=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--bird", required=True, choices=sorted(WINDOWS))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    windows = WINDOWS[args.bird]
    table = load_table(args.csv_dir, args.bird)
    types = sorted(table["type"].unique())
    days = np.sort(table["dph"].unique())
    print(f"{args.bird}: dph {days.min()}-{days.max()}, "
          f"types {types}, {table['file'].nunique()} bouts\n")

    z_all = table[LATENTS].to_numpy()

    # ---- dispersion: early vs crystallised variance, per real type ----
    def window_ratio(early_lo, early_hi):
        rows = []
        for t in types:
            sel = table["type"] == t
            early = z_all[(sel & table["dph"].between(early_lo, early_hi)).to_numpy()]
            late = z_all[(sel & table["dph"].between(*windows["late"])).to_numpy()]
            if len(early) < MIN_RENDITIONS_PER_WINDOW or len(late) < MIN_RENDITIONS_PER_WINDOW:
                continue
            ve, vl = trace_variance(early), trace_variance(late)
            rows.append({"type": t, "n_early": len(early), "n_late": len(late),
                         "early_variance": ve, "late_variance": vl,
                         "ratio": ve / vl, "log_ratio": float(np.log(ve / vl))})
        return rows

    per_type = window_ratio(*windows["early"])
    median_log = float(np.median([r["log_ratio"] for r in per_type]))

    # ---- floor: split-half within crystallised days, real bouts ----
    null_values = []
    late_days = [d for d in days if windows["late"][0] <= d <= windows["late"][1]]
    for day in late_days:
        for t in types:
            sel = ((table["dph"] == day) & (table["type"] == t)).to_numpy()
            bouts = table.loc[sel, "file"].to_numpy()
            if sel.sum() < MIN_RENDITIONS_PER_DAY or len(np.unique(bouts)) < MIN_BOUTS_PER_DAY:
                continue
            null_values.extend(split_half_dispersion_null(
                z_all[sel], bouts, n_draws=100, seed=args.seed).tolist())
    floor = float(np.percentile(np.abs(null_values), 95)) if null_values else float("nan")

    print(f"{'type':>6}{'n early':>9}{'n late':>9}{'early var':>11}{'late var':>10}"
          f"{'ratio':>8}{'log':>8}")
    for r in per_type:
        print(f"{r['type']:>6}{r['n_early']:>9}{r['n_late']:>9}"
              f"{r['early_variance']:>11.2f}{r['late_variance']:>10.2f}"
              f"{r['ratio']:>8.2f}{r['log_ratio']:>+8.3f}")
    print(f"\nmedian log ratio {median_log:+.3f} ({np.exp(median_log):.2f}x)   "
          f"floor |log| p95 {floor:.3f} ({np.exp(floor):.2f}x)   "
          f"-> {'EXCEEDS' if abs(median_log) > floor else 'within'} floor, sign "
          f"{'as predicted' if median_log > 0 else 'OPPOSITE'}")

    alt = None
    if "alt_early" in windows:
        alt_rows = window_ratio(*windows["alt_early"])
        alt = float(np.median([r["log_ratio"] for r in alt_rows]))
        print(f"sensitivity window {windows['alt_early']}: "
              f"median log {alt:+.3f} ({np.exp(alt):.2f}x)")

    # ---- centroid: consecutive-day vs fixed-reference, per type ----
    bout_means = (table.groupby(["dph", "type", "file"], sort=True)[LATENTS]
                  .mean().reset_index())
    scale = {}
    for t in types:
        per_day_var = [trace_variance(z_all[((table["dph"] == d) & (table["type"] == t)).to_numpy()])
                       for d in late_days
                       if ((table["dph"] == d) & (table["type"] == t)).sum() >= MIN_RENDITIONS_PER_DAY]
        scale[t] = float(np.mean(per_day_var)) if per_day_var else float("nan")

    def day_bouts(day, t):
        m = bout_means[(bout_means["dph"] == day) & (bout_means["type"] == t)]
        return m[LATENTS].to_numpy()

    consecutive = []
    for a, b in zip(days, days[1:]):
        if b - a != 1:
            continue
        vals = []
        for t in types:
            za, zb = day_bouts(a, t), day_bouts(b, t)
            if len(za) < 2 or len(zb) < 2:
                continue
            vals.append(unbiased_squared_centroid_distance(za, zb) / scale[t])
        if vals:
            consecutive.append({"dph_a": int(a), "dph_b": int(b),
                                "drift": float(np.mean(vals))})
    learning = [c["drift"] for c in consecutive if c["dph_b"] < 95]
    crystal = [c["drift"] for c in consecutive if c["dph_a"] >= 95]

    reference = {t: np.vstack([day_bouts(d, t) for d in late_days
                               if len(day_bouts(d, t))]) for t in types}
    to_ref = []
    for d in days:
        if d >= windows["late"][0]:
            continue
        vals = [unbiased_squared_centroid_distance(day_bouts(d, t), reference[t]) / scale[t]
                for t in types if len(day_bouts(d, t)) >= 2]
        if vals:
            to_ref.append({"dph": int(d), "distance": float(np.mean(vals))})
    early_ref = [r["distance"] for r in to_ref
                 if windows["early"][0] <= r["dph"] <= windows["early"][1]]
    preref = [r["distance"] for r in to_ref if r["dph"] >= windows["late"][0] - 10]

    print(f"\ncentroid, per-type, real bouts:")
    if learning:
        print(f"  consecutive-day median: learning {np.median(learning):.4f} "
              f"(n={len(learning)})" + (f", crystallised {np.median(crystal):.4f} "
              f"(n={len(crystal)}), ratio {np.median(learning)/np.median(crystal):.1f}x"
              if crystal else " (no crystallised pairs in record)"))
    if early_ref and preref:
        print(f"  distance to crystallised reference: early {np.mean(early_ref):.3f} "
              f"vs pre-reference {np.mean(preref):.3f} "
              f"-> {np.mean(early_ref)/np.mean(preref):.1f}x")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "bird": args.bird, "source": "Brudner CSV export of deposited tables",
            "windows": {k: list(v) for k, v in windows.items()},
            "types": [str(t) for t in types],
            "per_type": per_type, "median_log_ratio": median_log,
            "alt_early_median_log_ratio": alt,
            "floor_abs_p95": floor,
            "exceeds_floor": bool(abs(median_log) > floor),
            "consecutive": consecutive,
            "consecutive_learning_median": float(np.median(learning)) if learning else None,
            "consecutive_crystallised_median": float(np.median(crystal)) if crystal else None,
            "distance_to_reference": to_ref,
            "reference_early_mean": float(np.mean(early_ref)) if early_ref else None,
            "reference_preref_mean": float(np.mean(preref)) if preref else None,
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

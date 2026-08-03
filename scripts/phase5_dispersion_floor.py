"""Dispersion drift and its noise floor, alongside the centroid metric.

Mirrors `phase3_noise_floor.py` but for the log variance ratio: did the syllable type get
sloppier, rather than did it move? Reports both metrics on the same day pairs so their
independence can be checked — if dispersion drift merely tracked centroid drift it would
add nothing, and there would be no reason to carry a second statistic.

Usage::

    python scripts/phase5_dispersion_floor.py --data results/phase2/gy6or6_syllables.npz \
        --exclude-days 2012-03-22 --out results/phase5/dispersion_gy6or6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from songbird.drift import (
    bootstrap_dispersion_ci,
    bootstrap_drift_ci,
    split_half_dispersion_null,
)

MIN_RENDITIONS = 20
MIN_BOUTS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exclude-days", nargs="*", default=[])
    parser.add_argument("--n-pca", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--n-null", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    blob = np.load(args.data, allow_pickle=False)
    labels = blob["labels"].astype(str)
    days = blob["days"].astype(str)
    bouts = blob["files"].astype(str)
    flat = blob["spectrograms"].reshape(len(labels), -1).astype(np.float32)

    if args.exclude_days:
        keep = ~np.isin(days, args.exclude_days)
        labels, days, bouts, flat = labels[keep], days[keep], bouts[keep], flat[keep]

    unique_days = sorted(np.unique(days))
    reference_day = unique_days[0]
    pca = PCA(n_components=args.n_pca, random_state=args.seed).fit(flat[days == reference_day])
    z = pca.transform(flat)
    types = sorted(np.unique(labels))

    print(f"{len(z)} syllables | {len(types)} types | {len(unique_days)} days\n")

    # ---- noise floor ----
    null_values = []
    for day in unique_days:
        for syllable in types:
            mask = (days == day) & (labels == syllable)
            if mask.sum() < MIN_RENDITIONS or len(np.unique(bouts[mask])) < MIN_BOUTS:
                continue
            null_values.extend(
                split_half_dispersion_null(z[mask], bouts[mask], n_draws=args.n_null,
                                           seed=args.seed).tolist()
            )
    null_values = np.asarray(null_values)
    floor = float(np.percentile(np.abs(null_values), 95))
    print("DISPERSION NOISE FLOOR (split-half within day)")
    print(f"  draws {len(null_values)}   median {np.median(null_values):+.4f}   "
          f"sd {null_values.std():.4f}")
    print(f"  95th percentile of |log ratio|: {floor:.4f}  "
          f"(= a {np.exp(floor):.2f}x change in variance)\n")

    # ---- between-day, both metrics ----
    rows = []
    for i, day_a in enumerate(unique_days):
        for day_b in unique_days[i + 1:]:
            separation = (np.datetime64(day_b) - np.datetime64(day_a)).astype(int)
            dispersions, centroids, exceed = [], [], 0
            for syllable in types:
                mask_a = (days == day_a) & (labels == syllable)
                mask_b = (days == day_b) & (labels == syllable)
                if min(mask_a.sum(), mask_b.sum()) < MIN_RENDITIONS:
                    continue
                if min(len(np.unique(bouts[mask_a])), len(np.unique(bouts[mask_b]))) < 2:
                    continue
                estimate, low, high = bootstrap_dispersion_ci(
                    z[mask_a], bouts[mask_a], z[mask_b], bouts[mask_b],
                    n_boot=args.n_boot, seed=args.seed)
                dispersions.append(estimate)
                exceed += int(low > floor or high < -floor)
                centroid, _, _ = bootstrap_drift_ci(
                    z[mask_a], bouts[mask_a], z[mask_b], bouts[mask_b],
                    n_boot=50, seed=args.seed)
                centroids.append(centroid)
            if not dispersions:
                continue
            rows.append({
                "day_a": day_a, "day_b": day_b, "separation_days": int(separation),
                "n_types": len(dispersions),
                "mean_abs_dispersion": float(np.mean(np.abs(dispersions))),
                "mean_dispersion": float(np.mean(dispersions)),
                "n_types_exceeding_floor": exceed,
                "per_type_dispersion": dispersions,
                "per_type_centroid": centroids,
            })

    print("BETWEEN-DAY DISPERSION DRIFT (log variance ratio)")
    print(f"{'day A':<12}{'day B':<12}{'gap':>5}{'mean|log|':>11}{'signed':>9}{'>floor':>9}")
    for r in rows:
        print(f"{r['day_a']:<12}{r['day_b']:<12}{r['separation_days']:>5}"
              f"{r['mean_abs_dispersion']:>11.4f}{r['mean_dispersion']:>+9.4f}"
              f"{r['n_types_exceeding_floor']:>5}/{r['n_types']}")

    # ---- are the two metrics independent? ----
    all_disp = np.concatenate([np.abs(r["per_type_dispersion"]) for r in rows])
    all_cent = np.concatenate([r["per_type_centroid"] for r in rows])
    rho = spearmanr(all_disp, all_cent)
    print(f"\nINDEPENDENCE: Spearman rho(|dispersion drift|, centroid drift) = "
          f"{rho.statistic:+.3f}  p={rho.pvalue:.2e}  n={len(all_disp)}")
    print("  near zero => the two metrics see different things and both are worth carrying")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "data": str(args.data), "excluded_days": args.exclude_days,
            "n_pca": args.n_pca,
            "noise_floor": {
                "n_draws": int(len(null_values)),
                "median": round(float(np.median(null_values)), 5),
                "sd": round(float(null_values.std()), 5),
                "abs_p95": round(floor, 5),
                "fold_change": round(float(np.exp(floor)), 4),
            },
            "between_day": rows,
            "independence": {"spearman_rho": float(rho.statistic),
                             "p": float(rho.pvalue), "n": int(len(all_disp))},
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

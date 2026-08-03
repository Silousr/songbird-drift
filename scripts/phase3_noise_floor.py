"""Phase 3: within-bird drift and the noise floor it must be judged against.

For each syllable type, drift between two days is the unbiased squared distance between
that type's acoustic centroids, measured in a PCA space fitted **once** on the reference
day. Fitting the space per day would make the days incomparable -- the axes themselves
would move, and that movement would be indistinguishable from song change.

The noise floor is the same statistic computed where there is definitionally no drift:
split one day's *bouts* into two disjoint halves and compare them. Anything a real
between-day comparison reports that lies inside this distribution is not evidence of
change.

Every estimate resamples **bouts**, not syllables. Renditions within a bout are
correlated, and treating them as independent both narrows confidence intervals and (via
the variance correction) manufactures drift outright -- measured at a ~10x under-correction
on clustered test data.

Drift is reported in two units:

* raw squared PCA distance, and
* **standardised**: divided by the pooled within-type variance on the reference day, which
  makes it a dimensionless effect size comparable across birds and syllable types, and is
  the unit Phase 4's power curves need.

Usage::

    python scripts/phase3_noise_floor.py --data results/phase2/gy6or6_syllables.npz \
        --out results/phase3/noise_floor_gy6or6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from songbird.drift import bootstrap_drift_ci, split_half_null

MIN_RENDITIONS = 20
MIN_BOUTS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n-pca", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--n-null", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-days", nargs="*", default=[],
                        help="days to drop, e.g. ones adjacent to an experimental phase")
    args = parser.parse_args()

    blob = np.load(args.data, allow_pickle=False)
    labels = blob["labels"].astype(str)
    days = blob["days"].astype(str)
    bouts = blob["files"].astype(str)
    flat = blob["spectrograms"].reshape(len(labels), -1).astype(np.float32)

    if args.exclude_days:
        keep = ~np.isin(days, args.exclude_days)
        print(f"excluding days {args.exclude_days}: dropping {int((~keep).sum())} syllables")
        labels, days, bouts, flat = labels[keep], days[keep], bouts[keep], flat[keep]

    unique_days = sorted(np.unique(days))
    reference_day = unique_days[0]

    # Fit the measurement space once, on the reference day only.
    pca = PCA(n_components=args.n_pca, random_state=args.seed).fit(flat[days == reference_day])
    z = pca.transform(flat)

    # Scale: pooled within-type variance on the reference day. Drift divided by this is a
    # dimensionless effect size -- "how far the centroid moved, in units of how much a
    # single rendition naturally varies".
    reference = days == reference_day
    within_var = float(np.mean([
        z[reference & (labels == t)].var(axis=0, ddof=1).sum()
        for t in np.unique(labels[reference & np.isin(labels, np.unique(labels))])
        if (reference & (labels == t)).sum() > 1
    ]))

    print(f"{len(z)} syllables | {len(np.unique(labels))} types | {len(unique_days)} days")
    print(f"PCA fitted on {reference_day}: {args.n_pca} components, "
          f"{pca.explained_variance_ratio_.sum() * 100:.1f}% variance")
    print(f"pooled within-type variance (scale): {within_var:.4f}\n")

    types = sorted(np.unique(labels))

    # ---- noise floor: split-half within each day ----
    null_values = []
    per_day_null = {}
    for day in unique_days:
        day_values = []
        for syllable in types:
            mask = (days == day) & (labels == syllable)
            if mask.sum() < MIN_RENDITIONS or len(np.unique(bouts[mask])) < MIN_BOUTS:
                continue
            draws = split_half_null(z[mask], bouts[mask], n_draws=args.n_null,
                                    seed=args.seed)
            day_values.extend(draws.tolist())
        per_day_null[day] = {
            "n": len(day_values),
            "mean": round(float(np.mean(day_values)), 5) if day_values else None,
            "sd": round(float(np.std(day_values)), 5) if day_values else None,
            "p95": round(float(np.percentile(day_values, 95)), 5) if day_values else None,
        }
        null_values.extend(day_values)

    null_values = np.asarray(null_values)
    floor_p95 = float(np.percentile(null_values, 95))
    print("NOISE FLOOR (split-half within day, no manipulation)")
    print(f"  draws {len(null_values)}   mean {null_values.mean():+.5f}   "
          f"sd {null_values.std():.5f}")
    print(f"  95th percentile {floor_p95:.5f}   "
          f"standardised {floor_p95 / within_var:.4f}")
    print(f"  fraction negative {np.mean(null_values < 0):.2f} "
          f"(≈0.5 expected if unbiased)\n")

    # ---- between-day drift ----
    rows = []
    for i, day_a in enumerate(unique_days):
        for day_b in unique_days[i + 1:]:
            separation = (np.datetime64(day_b) - np.datetime64(day_a)).astype(int)
            per_type = []
            for syllable in types:
                mask_a = (days == day_a) & (labels == syllable)
                mask_b = (days == day_b) & (labels == syllable)
                if min(mask_a.sum(), mask_b.sum()) < MIN_RENDITIONS:
                    continue
                if min(len(np.unique(bouts[mask_a])), len(np.unique(bouts[mask_b]))) < 2:
                    continue
                estimate, low, high = bootstrap_drift_ci(
                    z[mask_a], bouts[mask_a], z[mask_b], bouts[mask_b],
                    n_boot=args.n_boot, seed=args.seed,
                )
                per_type.append({"type": syllable, "drift": estimate,
                                 "ci_low": low, "ci_high": high,
                                 "exceeds_floor": low > floor_p95})
            if not per_type:
                continue
            mean_drift = float(np.mean([p["drift"] for p in per_type]))
            rows.append({
                "day_a": day_a, "day_b": day_b, "separation_days": int(separation),
                "n_types": len(per_type),
                "mean_drift": mean_drift,
                "mean_drift_standardised": mean_drift / within_var,
                "n_types_exceeding_floor": sum(p["exceeds_floor"] for p in per_type),
                "per_type": per_type,
            })

    print("BETWEEN-DAY DRIFT (mean over syllable types)")
    print(f"{'day A':<12}{'day B':<12}{'gap':>5}{'types':>7}{'drift':>10}"
          f"{'standardised':>14}{'>floor':>8}")
    for r in rows:
        print(f"{r['day_a']:<12}{r['day_b']:<12}{r['separation_days']:>5}"
              f"{r['n_types']:>7}{r['mean_drift']:>10.4f}"
              f"{r['mean_drift_standardised']:>14.4f}"
              f"{r['n_types_exceeding_floor']:>4}/{r['n_types']}")

    by_gap = {}
    for r in rows:
        by_gap.setdefault(r["separation_days"], []).append(r["mean_drift_standardised"])
    print("\nstandardised drift by day separation")
    for gap in sorted(by_gap):
        values = by_gap[gap]
        print(f"  {gap} day(s): mean {np.mean(values):+.4f}  "
              f"(n={len(values)} day-pairs, range {min(values):+.4f} to {max(values):+.4f})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "data": str(args.data), "reference_day": reference_day,
            "n_pca": args.n_pca,
            "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
            "within_type_variance_scale": round(within_var, 5),
            "noise_floor": {
                "n_draws": int(len(null_values)),
                "mean": round(float(null_values.mean()), 5),
                "sd": round(float(null_values.std()), 5),
                "p95": round(floor_p95, 5),
                "p95_standardised": round(floor_p95 / within_var, 5),
                "fraction_negative": round(float(np.mean(null_values < 0)), 3),
                "per_day": per_day_null,
            },
            "between_day": rows,
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

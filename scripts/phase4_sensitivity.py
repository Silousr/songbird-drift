"""Phase 4: minimum detectable drift versus recording volume.

Answers the planning question: given N bouts of song per timepoint and K syllable types,
what is the smallest drift this pipeline can detect at 80% power?

Method. Power is estimated by **injection into real data**, never from a formula. Each
draw samples bouts from a real day, splits them into two disjoint halves, displaces one
half by a known standardised amount, and asks whether the aggregated drift statistic
clears its critical value. A parametric calculation assuming independent syllables would
overstate sensitivity by the design effect -- ~10x on this data (Phase 3).

**Recording volume is counted in bouts, not syllables and not minutes.** The estimator
treats the bout as its sampling unit, so precision scales with bout count; more syllables
inside the same bouts buys very little. Minutes of song are reported alongside only as a
convenience conversion, using each bird's measured syllable-seconds per bout.

Because the statistic depends on the data only through per-bout means, those are computed
once and the simulation runs on them -- exactly equivalent, and fast enough to sweep the
grid.

Usage::

    python scripts/phase4_sensitivity.py --data results/phase2/gy6or6_syllables.npz \
        --exclude-days 2012-03-22 --out results/phase4/sensitivity_gy6or6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from songbird.drift import unbiased_squared_centroid_distance

EFFECT_GRID = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
                        0.15, 0.2, 0.3, 0.5, 0.75, 1.0])
BOUT_GRID = [5, 10, 20, 40, 80, 160]
MIN_PER_SIDE = 2


def _full_seconds_per_bout(root, data_path, days_kept):
    """Median seconds of annotated song per bout, from the complete annotations."""
    import re
    from songbird.ingest.bfsongrepo import load_day

    bird = Path(data_path).name.split("_")[0]
    totals = []
    for day_dir in sorted((root / bird).iterdir()):
        if not day_dir.is_dir() or not re.fullmatch(r"\d{6}", day_dir.name):
            continue
        table = load_day(day_dir, on_date_mismatch="skip")
        if table.empty or str(table["day"].iloc[0]) not in days_kept:
            continue
        totals.extend(table.groupby("audio_file")["duration_s"].sum().tolist())
    return float(np.median(totals)) if totals else float("nan")


def bout_means_by_type(z, labels, bouts, types, all_bouts):
    """``type -> (row_of_bout, means)`` keyed by integer bout id.

    ``row_of_bout[i]`` is the row of ``means`` for bout ``i``, or -1 if that bout contains
    no rendition of this type. Integer indexing keeps the simulation loop vectorised.
    """
    bout_id = {name: i for i, name in enumerate(all_bouts)}
    table = {}
    for syllable in types:
        mask = labels == syllable
        names = np.unique(bouts[mask])
        means = np.stack([z[mask & (bouts == name)].mean(axis=0) for name in names])
        row_of_bout = np.full(len(all_bouts), -1, dtype=int)
        for row, name in enumerate(names):
            row_of_bout[bout_id[name]] = row
        table[syllable] = (row_of_bout, means)
    return table


def simulate(table, types, all_bouts, n_bouts, n_draws, rng):
    """Return ``(baseline, projection)`` per draw, from which any effect size follows exactly.

    Injecting a displacement ``s`` into one half changes the statistic analytically:

        stat(s) = ||d - s||^2 - correction = baseline - 2 s.d + ||s||^2

    where ``d`` is the difference of bout-mean centroids. Storing ``baseline`` and the
    projection ``d.direction`` per draw therefore lets every effect size be evaluated in
    closed form from a single simulation, instead of re-simulating the whole grid.
    """
    n_total = len(all_bouts)
    baselines, projections = [], []

    for _ in range(n_draws):
        chosen = rng.choice(n_total, size=2 * n_bouts, replace=False)
        left_ids, right_ids = chosen[:n_bouts], chosen[n_bouts:]

        direction = rng.standard_normal(table[types[0]][1].shape[1])
        direction /= np.linalg.norm(direction)

        stats, projs = [], []
        for syllable in types:
            row_of_bout, means = table[syllable]
            left = row_of_bout[left_ids]
            right = row_of_bout[right_ids]
            left, right = left[left >= 0], right[right >= 0]
            if len(left) < MIN_PER_SIDE or len(right) < MIN_PER_SIDE:
                continue
            stats.append(unbiased_squared_centroid_distance(means[left], means[right]))
            difference = means[left].mean(axis=0) - means[right].mean(axis=0)
            projs.append(float(difference @ direction))
        if stats:
            baselines.append(float(np.mean(stats)))
            projections.append(float(np.mean(projs)))
    return np.asarray(baselines), np.asarray(projections)


def statistic_at(baseline, projection, effect_size, scale):
    """Evaluate the statistic for an injected standardised effect, exactly."""
    magnitude = np.sqrt(max(effect_size, 0.0) * scale)
    return baseline - 2 * magnitude * projection + magnitude ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exclude-days", nargs="*", default=[])
    parser.add_argument("--n-pca", type=int, default=64)
    parser.add_argument("--n-draws", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-subsets", type=int, default=5,
                        help="random type subsets averaged per K")
    parser.add_argument("--bfsongrepo-root", type=Path,
                        help="measure true song-seconds per bout from full annotations")
    args = parser.parse_args()

    blob = np.load(args.data, allow_pickle=False)
    labels = blob["labels"].astype(str)
    days = blob["days"].astype(str)
    bouts = blob["files"].astype(str)
    durations = blob["durations"]
    flat = blob["spectrograms"].reshape(len(labels), -1).astype(np.float32)

    if args.exclude_days:
        keep = ~np.isin(days, args.exclude_days)
        labels, days, bouts, durations, flat = (
            labels[keep], days[keep], bouts[keep], durations[keep], flat[keep]
        )

    unique_days = sorted(np.unique(days))
    reference_day = unique_days[0]
    pca = PCA(n_components=args.n_pca, random_state=args.seed).fit(flat[days == reference_day])
    z = pca.transform(flat)

    on_reference = days == reference_day
    types_all = sorted(np.unique(labels))
    scale = float(np.mean([
        z[on_reference & (labels == t)].var(axis=0, ddof=1).sum()
        for t in types_all if (on_reference & (labels == t)).sum() > 1
    ]))

    # Pool all clean days: volume, not day identity, is the variable under study here.
    all_bouts = np.unique(bouts)
    table = bout_means_by_type(z, labels, bouts, types_all, all_bouts)

    # Measured on the FULL annotation set, not the 2,000/day subsample -- the subsample
    # keeps every bout but only a fraction of each bout's syllables, so deriving song
    # time from it would understate the real recording volume several-fold.
    seconds_per_bout = _full_seconds_per_bout(args.bfsongrepo_root, args.data,
                                              unique_days) if args.bfsongrepo_root \
        else float(np.median([durations[bouts == name].sum() for name in all_bouts]))

    print(f"{len(z)} syllables | {len(types_all)} types | {len(all_bouts)} bouts "
          f"| days {', '.join(unique_days)}")
    print(f"scale (within-type variance) {scale:.4f} | "
          f"median song per bout {seconds_per_bout:.2f} s\n")

    rng_master = np.random.default_rng(args.seed)
    type_counts = sorted({1, 3, 5, len(types_all)})
    results = []

    header = (f"{'bouts/side':>11}{'song min':>10}" +
              "".join(f"{'K=' + str(k):>9}" for k in type_counts))
    print("MINIMUM DETECTABLE DRIFT (standardised), 80% power, alpha=0.05")
    print(header)
    print("-" * len(header))

    for n_bouts in BOUT_GRID:
        if 2 * n_bouts > len(all_bouts):
            continue
        row = []
        for k in type_counts:
            # Average over random subsets of syllable types rather than taking the first
            # k alphabetically: types differ in variance and bout coverage, so a fixed
            # subset makes K non-comparable across columns.
            subsets = ([types_all] if k == len(types_all) else
                       [list(rng_master.choice(types_all, k, replace=False))
                        for _ in range(args.n_subsets)])
            mdes = []
            for types in subsets:
                # Independent draws for the threshold and for the power estimate, so the
                # threshold is not tuned to the draws it is applied to.
                rng_null = np.random.default_rng(args.seed + 1000 * k + n_bouts)
                rng_alt = np.random.default_rng(args.seed + 500_009 + 1000 * k + n_bouts)
                null_base, _ = simulate(table, types, all_bouts, n_bouts,
                                        args.n_draws, rng_null)
                alt_base, alt_proj = simulate(table, types, all_bouts, n_bouts,
                                              args.n_draws, rng_alt)
                if len(null_base) == 0 or len(alt_base) == 0:
                    continue
                threshold = float(np.percentile(null_base, 100 * (1 - args.alpha)))
                found = float("nan")
                for effect in EFFECT_GRID:
                    values = statistic_at(alt_base, alt_proj, float(effect), scale)
                    if float(np.mean(values > threshold)) >= args.power:
                        found = float(effect)
                        break
                mdes.append(found)
            mde = (float(np.nanmedian(mdes)) if mdes and not all(np.isnan(mdes))
                   else float("nan"))
            row.append(mde)
            results.append({"n_bouts": n_bouts, "n_types": k, "mde": mde,
                            "song_minutes": n_bouts * seconds_per_bout / 60})

        minutes = n_bouts * seconds_per_bout / 60
        cells = "".join(
            f"{('>' + format(EFFECT_GRID[-1], '.2f')) if np.isnan(v) else format(v, '.3f'):>9}"
            for v in row
        )
        print(f"{n_bouts:>11}{minutes:>10.1f}{cells}")

    print(f"\nK = number of syllable types aggregated. '>{EFFECT_GRID[-1]:.2f}' means no "
          f"effect on the grid reached {args.power:.0%} power.")
    print(f"Phase 3 noise floor for reference: ~0.03-0.06 standardised.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "data": str(args.data), "excluded_days": args.exclude_days,
            "reference_day": reference_day, "n_pca": args.n_pca,
            "scale_within_type_variance": round(scale, 5),
            "median_song_seconds_per_bout": round(seconds_per_bout, 3),
            "n_bouts_available": int(len(all_bouts)),
            "alpha": args.alpha, "power": args.power,
            "effect_grid": EFFECT_GRID.tolist(),
            "cells": results,
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

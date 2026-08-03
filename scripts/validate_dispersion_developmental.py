"""Validation of the dispersion metric: is juvenile song more variable than adult song?

This is the effect the centroid metric could not see. Juvenile renditions of a syllable are
sloppy and converge on a stereotyped adult form, so **per-syllable variability should fall
across development** even where the syllable's average position barely moves.

Method. Syllable-type modes are defined by clustering the crystallised endpoint (the
authors' own documented labelling approach), every sound is assigned to its nearest mode,
and per-mode variance is compared between early and late development.

**Selection intensity is matched across days, and this matters more than it looks.** A
fixed distance cut-off keeps ~12% of sounds early and ~70% late, so it selects a tight core
from a broad early distribution and nearly everything from a narrow late one. That
artefact alone reverses the sign of the result -- it makes juvenile song look *less*
variable than adult song. Keeping the same *fraction* nearest each mode on every day
removes it. The fraction is swept to show the conclusion does not depend on it.

Usage::

    python scripts/validate_dispersion_developmental.py --proj grn394_proj.zip \
        --out results/validation/dispersion_grn394.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from validate_developmental import load_latents  # noqa: E402

from songbird.drift import split_half_dispersion_null  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proj", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--endpoint-from", type=int, default=108)
    parser.add_argument("--early-before", type=int, default=70)
    parser.add_argument("--late-from", type=int, default=100)
    parser.add_argument("--fractions", nargs="*", type=float,
                        default=[0.12, 0.30, 0.60])
    parser.add_argument("--min-syllables", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bird = args.proj.name.split("_")[0]
    data = load_latents(args.proj, args.block_size)
    data = {d: v for d, v in data.items() if len(v[0]) >= args.min_syllables}
    days = sorted(data)

    endpoint = [d for d in days if d >= args.endpoint_from]
    stacked = np.vstack([data[d][0] for d in endpoint])
    rng = np.random.default_rng(args.seed)
    sample = stacked[rng.choice(len(stacked), min(60_000, len(stacked)), replace=False)]
    model = KMeans(n_clusters=args.n_clusters, n_init=10,
                   random_state=args.seed).fit(sample)

    print(f"{bird}: {len(days)} days, dph {min(days)}-{max(days)}")
    print(f"{args.n_clusters} syllable modes from endpoint dph >= {args.endpoint_from}\n")

    results = []
    print(f"{'keep':>7}{'early var':>12}{'crystallised':>14}{'ratio':>8}{'log':>9}")
    for fraction in args.fractions:
        per_day = {}
        for day in days:
            latents, _ = data[day]
            assignment = model.predict(latents)
            distance = np.linalg.norm(
                latents - model.cluster_centers_[assignment], axis=1
            )
            variances = []
            for cluster in range(args.n_clusters):
                members = np.flatnonzero(assignment == cluster)
                if len(members) < 100:
                    continue
                keep = max(int(len(members) * fraction), 50)
                chosen = members[np.argsort(distance[members])[:keep]]
                variances.append(latents[chosen].var(axis=0, ddof=1).sum())
            per_day[day] = float(np.mean(variances)) if variances else np.nan

        early = np.nanmean([per_day[d] for d in days if d < args.early_before])
        late = np.nanmean([per_day[d] for d in days if d >= args.late_from])
        results.append({"fraction": fraction, "early_variance": early,
                        "crystallised_variance": late, "ratio": early / late,
                        "log_ratio": float(np.log(early / late)),
                        "per_day": {str(k): v for k, v in per_day.items()}})
        print(f"{fraction:>7.0%}{early:>12.3f}{late:>14.3f}"
              f"{early / late:>8.2f}{np.log(early / late):>+9.3f}")

    # Noise floor from the final day, same statistic.
    latents, blocks = data[days[-1]]
    assignment = model.predict(latents)
    null = []
    for cluster in range(args.n_clusters):
        mask = assignment == cluster
        if mask.sum() < 200 or len(np.unique(blocks[mask])) < 4:
            continue
        null.extend(split_half_dispersion_null(latents[mask], blocks[mask],
                                               n_draws=200, seed=args.seed).tolist())
    floor = float(np.percentile(np.abs(null), 95))

    median_log = float(np.median([r["log_ratio"] for r in results]))
    print(f"\ndispersion noise floor (|log ratio| p95): {floor:.3f} "
          f"(= {np.exp(floor):.2f}x change in variance)")
    print(f"observed early-vs-crystallised log ratio: {median_log:+.3f} "
          f"({np.exp(median_log):.2f}x)")
    verdict = "EXCEEDS the floor" if abs(median_log) > floor else "within the floor"
    print(f"-> {verdict}; sign is "
          f"{'as predicted (juvenile song more variable)' if median_log > 0 else 'OPPOSITE to prediction'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "bird": bird, "n_clusters": args.n_clusters,
            "endpoint_from_dph": args.endpoint_from,
            "early_before_dph": args.early_before, "late_from_dph": args.late_from,
            "fractions": results, "median_log_ratio": median_log,
            "noise_floor_abs_p95": round(floor, 5),
            "exceeds_floor": bool(abs(median_log) > floor),
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

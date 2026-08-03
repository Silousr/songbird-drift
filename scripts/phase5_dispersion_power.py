"""How many bouts to detect a given change in rendition variability?

The dispersion counterpart to `phase4_sensitivity.py`. Effects are injected by scaling
each half's deviations from its own mean, which changes variance by exactly the requested
fold-change while leaving the centroid untouched.

Reported per syllable type (median across types), because dispersion is a within-type
property: "for a typical syllable, how big a change in variability could this experiment
detect?"

Usage::

    python scripts/phase5_dispersion_power.py --data results/phase2/gy6or6_syllables.npz \
        --exclude-days 2012-03-22 --out results/phase5/dispersion_power_gy6or6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from songbird.power import DEFAULT_FOLD_CHANGE_GRID, minimum_detectable_fold_change

BOUT_GRID = [5, 10, 20, 40, 80]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exclude-days", nargs="*", default=[])
    parser.add_argument("--n-pca", type=int, default=64)
    parser.add_argument("--n-draws", type=int, default=400)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
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

    reference_day = sorted(np.unique(days))[0]
    pca = PCA(n_components=args.n_pca, random_state=args.seed).fit(flat[days == reference_day])
    z = pca.transform(flat)
    types = sorted(np.unique(labels))
    print(f"{len(z)} syllables | {len(types)} types | "
          f"{len(np.unique(bouts))} bouts\n")

    print("MINIMUM DETECTABLE FOLD-CHANGE IN RENDITION VARIANCE")
    print(f"(80% power, alpha={args.alpha}, median across syllable types)")
    print(f"{'bouts/side':>11}{'median':>10}{'range across types':>22}")
    rows = []
    for n_bouts in BOUT_GRID:
        per_type = []
        for syllable in types:
            mask = labels == syllable
            if len(np.unique(bouts[mask])) < 2 * n_bouts:
                continue
            per_type.append(minimum_detectable_fold_change(
                z[mask], bouts[mask], n_bouts, alpha=args.alpha, power=args.power,
                n_draws=args.n_draws, seed=args.seed))
        if not per_type:
            continue
        finite = [v for v in per_type if not np.isnan(v)]
        median = float(np.nanmedian(per_type)) if finite else float("nan")
        span = (f"{min(finite):.2f} - {max(finite):.2f}" if finite else "--")
        unreachable = sum(np.isnan(v) for v in per_type)
        label = f"{median:.2f}x" if finite else f">{DEFAULT_FOLD_CHANGE_GRID[-1]:.0f}x"
        print(f"{n_bouts:>11}{label:>10}{span:>22}"
              + (f"   ({unreachable}/{len(per_type)} types unreachable)" if unreachable else ""))
        rows.append({"n_bouts": n_bouts, "median_fold_change": median,
                     "n_types": len(per_type), "n_unreachable": int(unreachable)})

    print(f"\nPhase 5 dispersion noise floor for reference: ~1.24-2.07x")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "data": str(args.data), "excluded_days": args.exclude_days,
            "alpha": args.alpha, "power": args.power,
            "fold_change_grid": DEFAULT_FOLD_CHANGE_GRID.tolist(),
            "rows": rows}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

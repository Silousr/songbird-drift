"""Phase 2 gate: is the embedding faithful enough to measure drift in?

Protocol, and why it is shaped this way:

    fit the embedding on ONE day  ->  transform every day  ->  train k-NN on that day
    ->  test label recovery within that day, and on every later day

A drift metric compares song on day A with song on day B by measuring distance in an
embedding. That is only meaningful if the embedding's geometry means the same thing on
both days. So fitting on day 1 and testing recovery on days 2-5 is not a robustness
afterthought -- it is the gate. An embedding that scores well within a day but degrades
across days would manufacture drift out of its own instability.

Three representations are compared so the result is interpretable rather than a bare
number:

* ``pixels``   -- flattened spectrogram, no learning. The reference: any learned embedding
                  that cannot beat this is not earning its complexity.
* ``pca``      -- linear, cheap, deterministic.
* ``umap``     -- non-linear, AVGN-style (PCA first, then UMAP), the representation the
                  brief points at.

Usage::

    python scripts/phase2_fidelity.py --data results/phase2/gy6or6_syllables.npz \
        --out results/phase2/fidelity_gy6or6.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from songbird.metrics.fidelity import (
    knn_label_recovery,
    label_silhouette,
    nearest_neighbour_purity,
    within_type_distance_correlation,
)


def build_embedders(n_pca: int, n_umap: int, seed: int) -> dict:
    """Return ``name -> (fit_transform, transform)`` builders."""

    def pixels(train_x):
        return lambda x: x

    def pca(train_x):
        model = PCA(n_components=n_pca, random_state=seed).fit(train_x)
        return model.transform

    def umap_embed(train_x):
        import umap

        pca_model = PCA(n_components=n_pca, random_state=seed).fit(train_x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = umap.UMAP(
                n_components=n_umap, random_state=seed, n_neighbors=15, min_dist=0.1
            ).fit(pca_model.transform(train_x))
        return lambda x: reducer.transform(pca_model.transform(x))

    return {"pixels": pixels, "pca": pca, "umap": umap_embed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n-pca", type=int, default=32)
    parser.add_argument("--n-umap", type=int, default=8)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    blob = np.load(args.data, allow_pickle=False)
    spectrograms = blob["spectrograms"]
    labels = blob["labels"].astype(str)
    days = blob["days"].astype(str)
    flat = spectrograms.reshape(len(spectrograms), -1).astype(np.float32)

    unique_days = sorted(np.unique(days))
    fit_day, later_days = unique_days[0], unique_days[1:]
    print(f"{len(flat)} syllables, {len(np.unique(labels))} labels, "
          f"{len(unique_days)} days")
    print(f"fitting on {fit_day}; testing across {', '.join(later_days)}\n")

    on_fit_day = days == fit_day
    train_index, test_index = train_test_split(
        np.flatnonzero(on_fit_day), test_size=0.3, random_state=args.seed,
        stratify=labels[on_fit_day],
    )

    results = {}
    for name, builder in build_embedders(args.n_pca, args.n_umap, args.seed).items():
        transform = builder(flat[train_index])
        train_z = transform(flat[train_index])

        within_acc, within_per_label = knn_label_recovery(
            train_z, labels[train_index], transform(flat[test_index]),
            labels[test_index], k=args.k, per_label=True,
        )

        cross = {}
        for day in later_days:
            mask = days == day
            cross[day] = knn_label_recovery(
                train_z, labels[train_index], transform(flat[mask]),
                labels[mask], k=args.k,
            )

        held_out_z = transform(flat[test_index])
        # Reference is the raw spectrogram: the best available proxy for acoustic
        # distance. Drift lives inside a syllable type, so this is the number that
        # decides whether drift is measurable in this space at all.
        within_type_rho = within_type_distance_correlation(
            held_out_z, flat[test_index], labels[test_index]
        )
        results[name] = {
            "within_type_distance_rho": round(within_type_rho, 4),
            "dimensions": int(train_z.shape[1]),
            "within_day_accuracy": round(within_acc, 4),
            "cross_day_accuracy": {d: round(a, 4) for d, a in cross.items()},
            "cross_day_mean": round(float(np.mean(list(cross.values()))), 4),
            "cross_day_drop": round(within_acc - float(np.mean(list(cross.values()))), 4),
            "nn_purity_heldout": round(nearest_neighbour_purity(
                held_out_z, labels[test_index]), 4),
            "silhouette_heldout": round(label_silhouette(
                held_out_z, labels[test_index]), 4),
            "per_label_within_day": {k: round(v, 4) for k, v in within_per_label.items()},
        }

        r = results[name]
        print(f"{name:>7} ({r['dimensions']:>4}d)  within-day {r['within_day_accuracy']:.4f}"
              f"   cross-day {r['cross_day_mean']:.4f}"
              f"   drop {r['cross_day_drop']:+.4f}"
              f"   purity {r['nn_purity_heldout']:.4f}"
              f"   silhouette {r['silhouette_heldout']:+.3f}"
              f"   within-type rho {r['within_type_distance_rho']:+.3f}")

    chance = float(np.max(np.bincount(np.unique(labels, return_inverse=True)[1])) / len(labels))
    print(f"\nmajority-class baseline: {chance:.4f}")

    print("\nper-day cross-day accuracy")
    header = f"{'embedding':>10}" + "".join(f"{d[5:]:>9}" for d in later_days)
    print(header)
    for name, r in results.items():
        print(f"{name:>10}" + "".join(
            f"{r['cross_day_accuracy'][d]:>9.4f}" for d in later_days))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "data": str(args.data), "fit_day": fit_day, "later_days": later_days,
            "n_syllables": int(len(flat)), "n_labels": int(len(np.unique(labels))),
            "majority_class_baseline": round(chance, 4),
            "k": args.k, "n_pca": args.n_pca, "n_umap": args.n_umap,
            "embeddings": results,
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

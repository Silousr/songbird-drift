"""Validation against a manipulation: does deafening drive adult song past its own floor?

This is the test the whole toolkit was built toward and had not yet passed. Everything
before it was validated against a **developmental** effect -- juvenile song crystallising --
while the intended application is a **manipulation of adult crystallized song**. Those are
different regimes, and the reports have said so throughout.

Deafening an adult songbird causes its song to deteriorate over weeks. The effect is large,
long documented, and the closest public analogue to "the critical period reopened".

Data: Zai, Stepien, Giret & Hahnloser (2024), eLife 10.7554/eLife.90445, derived dataset
at doi:10.3929/ethz-b-000670443 (MIT licence). Each bird contributes one measured acoustic
quantity -- the pitch of a target syllable, in Hz -- per rendition, timestamped in days, with
the deafening moment marked by a ``cochlea removal`` annotation.

**Scope, stated plainly.** This validates the drift *statistic* -- the bias-corrected,
bout-clustered estimator, its split-half floor and its bootstrap -- on an adult manipulation.
It does **not** validate this toolkit's own audio-to-embedding path, because the deposit
contains derived pitch rather than audio (the raw audio is a separate 49 GB archive). The
measurement is one-dimensional, so "centroid drift" here is a shift in mean pitch and
"dispersion drift" a change in pitch variability.

Design, entirely within bird:

* the **noise floor** comes from that bird's own pre-deafening days, split in half by bout;
* **baseline-to-baseline** drift between pre-deafening days is the negative control;
* **baseline-to-post** drift is the test.

Because floor, control and test all come from the same bird and the same estimator, nothing
is borrowed across animals.

Usage::

    python scripts/validate_deafening.py --root "<companion>/Data Goal-directed vocal planning" \
        --out results/validation/deafening.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from songbird.drift import (
    bootstrap_drift_ci,
    log_variance_ratio,
    split_half_dispersion_null,
    split_half_null,
    unbiased_squared_centroid_distance,
)
from songbird.ingest.bouts import bouts_from_timestamps

DEAFENING_MARKERS = {"cochlea removal", "deafen", "deafening"}
MIN_BOUTS_PER_SIDE = 4
MIN_RENDITIONS = 30


def load_bird(path: str):
    """Return ``(birdname, group, timestamps_days, quantity, deafening_day)``."""
    import scipy.io as sio

    D = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["D"]
    annotations = getattr(D, "annotations", None)
    marked = []
    if annotations is not None and np.size(annotations):
        for entry in np.atleast_1d(annotations):
            text = str(getattr(entry, "text", "")).strip().lower()
            stamp = getattr(entry, "timestamp", None)
            if text in DEAFENING_MARKERS and stamp is not None:
                marked.append(float(stamp))
    return (
        str(D.birdname),
        str(D.group),
        np.asarray(D.timestamps, dtype=float),
        np.asarray(D.quantity, dtype=float),
        min(marked) if marked else None,
    )


def bout_means(values: np.ndarray, bouts: np.ndarray) -> np.ndarray:
    """Collapse renditions to one row per bout.

    The centroid estimator depends on the data only through these, so computing them once
    is exactly equivalent to passing raw renditions with bout labels -- and avoids
    rebuilding index arrays over thousands of bouts inside every bootstrap draw.
    """
    keys, inverse = np.unique(bouts, return_inverse=True)
    sums = np.zeros((len(keys), values.shape[1]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=len(keys)).reshape(-1, 1)
    return sums / counts


def bootstrap_on_bout_means(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int):
    """Percentile interval for the unbiased distance, resampling bout means directly."""
    estimate = unbiased_squared_centroid_distance(a, b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        if len(np.unique(ia)) < 2 or len(np.unique(ib)) < 2:
            continue
        draws.append(unbiased_squared_centroid_distance(a[ia], b[ib]))
    if not draws:
        return estimate, float("nan"), float("nan")
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(estimate), float(low), float(high)


def analyse_bird(path: str, gap_s: float, seed: int) -> dict | None:
    bird, group, timestamps, quantity, deafened_at = load_bird(path)
    if deafened_at is None:
        return None

    bouts = bouts_from_timestamps(timestamps, gap_s=gap_s)
    days = np.floor(timestamps).astype(int)
    values = quantity.reshape(-1, 1)  # the estimators are dimension-agnostic

    baseline_days = sorted({int(d) for d in days[timestamps < deafened_at]})
    post_days = sorted({int(d) for d in days[timestamps > deafened_at]})
    # The deafening day itself straddles the manipulation; drop it.
    post_days = [d for d in post_days if d > np.floor(deafened_at)]
    if len(baseline_days) < 3 or len(post_days) < 3:
        return None

    baseline_mask = np.isin(days, baseline_days) & (timestamps < deafened_at)
    if baseline_mask.sum() < MIN_RENDITIONS:
        return None

    scale = float(values[baseline_mask].var(ddof=1))
    if scale <= 0:
        return None
    baseline_means = bout_means(values[baseline_mask], bouts[baseline_mask])

    # ---- noise floor from baseline days only ----
    centroid_null, dispersion_null = [], []
    for day in baseline_days:
        mask = baseline_mask & (days == day)
        if mask.sum() < MIN_RENDITIONS or len(np.unique(bouts[mask])) < MIN_BOUTS_PER_SIDE:
            continue
        centroid_null.extend(
            split_half_null(values[mask], bouts[mask], n_draws=200, seed=seed).tolist()
        )
        dispersion_null.extend(
            split_half_dispersion_null(values[mask], bouts[mask], n_draws=200,
                                       seed=seed).tolist()
        )
    if not centroid_null:
        return None
    centroid_floor = float(np.percentile(centroid_null, 95)) / scale
    dispersion_floor = float(np.percentile(np.abs(dispersion_null), 95))

    # ---- negative control: baseline day vs baseline day ----
    control = []
    for i, a in enumerate(baseline_days):
        for b in baseline_days[i + 1:]:
            ma = baseline_mask & (days == a)
            mb = baseline_mask & (days == b)
            if min(ma.sum(), mb.sum()) < MIN_RENDITIONS:
                continue
            if min(len(np.unique(bouts[ma])), len(np.unique(bouts[mb]))) < 2:
                continue
            control.append(
                unbiased_squared_centroid_distance(
                    bout_means(values[ma], bouts[ma]), bout_means(values[mb], bouts[mb])
                ) / scale
            )

    # ---- test: pooled baseline vs each post-deafening day ----
    post = []
    for day in post_days:
        mask = days == day
        if mask.sum() < MIN_RENDITIONS or len(np.unique(bouts[mask])) < 2:
            continue
        estimate, low, high = bootstrap_on_bout_means(
            baseline_means, bout_means(values[mask], bouts[mask]),
            n_boot=200, seed=seed,
        )
        try:
            dispersion = log_variance_ratio(values[baseline_mask], values[mask])
        except ValueError:
            dispersion = float("nan")
        post.append({
            "days_post": int(day - np.floor(deafened_at)),
            "centroid": estimate / scale,
            "ci_low": low / scale, "ci_high": high / scale,
            "dispersion": dispersion,
            "exceeds_floor": bool(low / scale > centroid_floor),
        })
    if not post:
        return None

    crossed = [p["days_post"] for p in post if p["exceeds_floor"]]
    return {
        "bird": bird, "group": group, "file": os.path.basename(path),
        "deafened_at_day": deafened_at,
        "n_baseline_days": len(baseline_days), "n_post_days": len(post),
        "n_renditions": int(timestamps.size),
        "baseline_pitch_hz": float(values[baseline_mask].mean()),
        "centroid_floor": centroid_floor,
        "dispersion_floor": dispersion_floor,
        "control_median": float(np.median(control)) if control else float("nan"),
        "control_max": float(np.max(control)) if control else float("nan"),
        "post_max": float(max(p["centroid"] for p in post)),
        "post_final": float(post[-1]["centroid"]),
        "dispersion_max": float(np.nanmax([abs(p["dispersion"]) for p in post])),
        "first_day_over_floor": min(crossed) if crossed else None,
        "n_post_days_over_floor": len(crossed),
        "post": post,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", nargs="*", default=["BLdeaf"],
                        help="subdirectories to analyse")
    parser.add_argument("--gap-s", type=float, default=60.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = []
    for group in args.groups:
        for path in sorted(glob.glob(str(args.root / group / "*.mat"))):
            outcome = analyse_bird(path, args.gap_s, args.seed)
            if outcome:
                outcome["folder"] = group
                results.append(outcome)

    if not results:
        raise SystemExit("no bird had enough baseline and post-deafening data")

    # One value per BIRD. Several files are different syllables of the same animal
    # (Deg1, b10p5, b8k16, o10g10, p3n8, b8o20), and counting them twice would be
    # pseudo-replication of exactly the kind the group test exists to prevent.
    by_bird: dict[str, list] = {}
    for r in results:
        by_bird.setdefault(r["bird"], []).append(r)

    print(f"{len(results)} recordings from {len(by_bird)} distinct birds "
          f"({len(results) - len(by_bird)} extra syllables merged)\n")
    print(f"{'bird':<10}{'bl days':>8}{'floor':>8}{'control':>9}{'post max':>10}"
          f"{'x floor':>9}{'1st day>':>9}")
    per_bird = []
    for bird, entries in sorted(by_bird.items()):
        floor = float(np.mean([e["centroid_floor"] for e in entries]))
        control = float(np.nanmean([e["control_median"] for e in entries]))
        post_max = float(np.mean([e["post_max"] for e in entries]))
        first = [e["first_day_over_floor"] for e in entries
                 if e["first_day_over_floor"] is not None]
        per_bird.append({
            "bird": bird, "n_files": len(entries), "floor": floor,
            "control_median": control, "post_max": post_max,
            "ratio": post_max / floor if floor else float("nan"),
            "first_day_over_floor": min(first) if first else None,
            "dispersion_max": float(np.mean([e["dispersion_max"] for e in entries])),
            "dispersion_floor": float(np.mean([e["dispersion_floor"] for e in entries])),
        })
        print(f"{bird:<10}{entries[0]['n_baseline_days']:>8}{floor:>8.3f}"
              f"{control:>9.3f}{post_max:>10.3f}{post_max / floor:>9.1f}"
              f"{(min(first) if first else '--'):>9}")

    controls = np.array([b["control_median"] for b in per_bird])
    posts = np.array([b["post_max"] for b in per_bird])
    floors = np.array([b["floor"] for b in per_bird])
    crossing = [b["first_day_over_floor"] for b in per_bird
                if b["first_day_over_floor"] is not None]

    print(f"\n{'':-<62}")
    print(f"birds: {len(per_bird)}")
    print(f"  baseline-to-baseline drift (negative control): median "
          f"{np.median(controls):.4f}, {int((controls > floors).sum())}/{len(per_bird)} "
          f"birds above their own floor")
    print(f"  baseline-to-post drift (test):                 median "
          f"{np.median(posts):.4f}, {int((posts > floors).sum())}/{len(per_bird)} "
          f"birds above their own floor")
    print(f"  post / floor ratio: median {np.median(posts / floors):.1f}x")
    if crossing:
        print(f"  days post-deafening before drift clears the floor: "
              f"median {np.median(crossing):.0f} (range {min(crossing)}-{max(crossing)})")

    # Paired within-bird test: same animals, same estimator, control vs test.
    from songbird.group import compare_groups

    paired = compare_groups(posts, controls, alternative="greater", seed=args.seed)
    print(f"\npaired within-bird comparison (post vs baseline-to-baseline):")
    print("  " + paired.summary().replace("\n", "\n  "))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "source_doi": "10.3929/ethz-b-000670443",
            "groups": args.groups, "gap_s": args.gap_s,
            "n_recordings": len(results), "n_birds": len(per_bird),
            "per_bird": per_bird,
            "summary": {
                "control_median": float(np.median(controls)),
                "post_median": float(np.median(posts)),
                "post_over_floor": int((posts > floors).sum()),
                "control_over_floor": int((controls > floors).sum()),
                "median_ratio": float(np.median(posts / floors)),
                "median_days_to_cross": (float(np.median(crossing)) if crossing else None),
                "paired_p": paired.p_value,
                "paired_difference": paired.observed_difference,
            },
            "recordings": results,
        }, indent=1, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

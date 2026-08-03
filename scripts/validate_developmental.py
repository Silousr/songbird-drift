"""Validation: does the drift metric recover a known effect?

The known effect is **song crystallisation**. A juvenile zebra finch's song changes rapidly
through the sensorimotor period and then stabilises around ~90 days post-hatch. A drift
metric that works must show large day-to-day change early and small change late, **in the
same bird, measured the same way**. That within-bird reversal is a far stronger test than
any single large number, because nothing about the recording rig, the pipeline, or the
embedding changes between the two regimes -- only the bird's developmental stage.

Data: Duke deposit 10.7924/r4j38x43h (Brudner, Pearson & Mooney), CC0. Uses the authors'
precomputed 32-D VAE latents (`_proj.zip`), indexed by day post-hatch.

Two honest caveats, both recorded in DECISION_LOG.md:

* This validates the **drift statistic**, not this toolkit's own audio->embedding path.
  The latents are the original authors'. Validating our own embedding would need the raw
  audio (291 GB).
* Drift is computed on the **whole syllable distribution**, not per syllable type. Duke's
  type labels sit in MATLAB tables this pipeline cannot read, and more importantly
  syllable identity is itself ill-defined mid-development -- which is the thing being
  measured. This is a deliberate difference from the Phase 3 per-type metric.

Clustering unit: **blocks of consecutive batch files**, not bouts. The deposit's latents
are batched 20-per-file and the batch->bout mapping cannot be reconstructed exactly (AVA
drops incomplete batches, and two days have further gaps). A block of many consecutive
batches is *coarser* than a bout, which can only widen confidence intervals -- the safe
direction. Block size is swept to confirm conclusions do not depend on it.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from songbird.drift import unbiased_squared_centroid_distance


def load_latents(archive: Path, block_size: int):
    """Return ``dph -> (latents, block_labels)`` from a Duke ``_proj.zip``."""
    zf = zipfile.ZipFile(archive)
    by_day = defaultdict(list)
    for name in zf.namelist():
        if not name.endswith(".hdf5"):
            continue
        day = int(name.split("/")[1])
        order = int(re.search(r"_(\d+)\.hdf5$", name).group(1))
        by_day[day].append((order, name))

    out = {}
    for day, entries in sorted(by_day.items()):
        entries.sort()
        latents, blocks = [], []
        for position, (_, name) in enumerate(entries):
            with h5py.File(io.BytesIO(zf.read(name)), "r") as handle:
                block = handle["latent_means"][:]
            latents.append(block)
            blocks.append(np.full(len(block), position // block_size))
        if latents:
            out[day] = (np.vstack(latents), np.concatenate(blocks))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proj", type=Path, required=True, help="{bird}_proj.zip")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--block-size", type=int, default=10,
                        help="batch files per clustering block (20 syllables each)")
    parser.add_argument("--crystallised-from", type=int, default=95,
                        help="dph from which song is treated as crystallised")
    parser.add_argument("--min-syllables", type=int, default=500)
    parser.add_argument("--endpoint-from", type=int, default=110,
                        help="dph from which days define the crystallised reference")
    args = parser.parse_args()

    bird = args.proj.name.split("_")[0]
    data = load_latents(args.proj, args.block_size)
    data = {d: v for d, v in data.items() if len(v[0]) >= args.min_syllables}
    days = sorted(data)
    print(f"{bird}: {len(days)} days, dph {min(days)}-{max(days)}, "
          f"{sum(len(v[0]) for v in data.values())} syllables, "
          f"block = {args.block_size} batches\n")

    # Scale: pooled within-day variance, so drift is in units of natural variation.
    scale = float(np.mean([data[d][0].var(axis=0, ddof=1).sum() for d in days]))

    # Consecutive-day drift across development.
    consecutive = []
    for a, b in zip(days, days[1:]):
        if b - a != 1:
            continue
        za, ba = data[a]
        zb, bb = data[b]
        value = unbiased_squared_centroid_distance(za, zb, ba, bb) / scale
        consecutive.append({"dph_a": a, "dph_b": b, "drift": value})

    learning = [c for c in consecutive if c["dph_b"] < args.crystallised_from]
    crystal = [c for c in consecutive if c["dph_a"] >= args.crystallised_from]

    print("CONSECUTIVE-DAY DRIFT (standardised), same bird, same pipeline")
    print(f"{'dph':>9}{'drift':>10}   regime")
    for c in consecutive:
        regime = ("learning" if c["dph_b"] < args.crystallised_from
                  else "crystallised" if c["dph_a"] >= args.crystallised_from else "-")
        print(f"{str(c['dph_a']) + '->' + str(c['dph_b']):>9}{c['drift']:>10.4f}   {regime}")

    summary = {}
    if learning and crystal:
        lm = float(np.median([c["drift"] for c in learning]))
        cm = float(np.median([c["drift"] for c in crystal]))
        summary = {"learning_median": lm, "crystallised_median": cm,
                   "ratio": lm / cm if cm else float("inf"),
                   "n_learning": len(learning), "n_crystallised": len(crystal)}
        print(f"\nmedian consecutive-day drift")
        print(f"  learning     (dph < {args.crystallised_from}): "
              f"{lm:.4f}   n={len(learning)}")
        print(f"  crystallised (dph >= {args.crystallised_from}): "
              f"{cm:.4f}   n={len(crystal)}")
        print(f"  RATIO learning / crystallised: {lm / cm:.1f}x")

    # Drift vs separation, within each regime.
    print("\nDRIFT vs DAY SEPARATION")
    print(f"{'gap':>5}{'learning':>12}{'crystallised':>14}")
    by_gap = {}
    for regime, pool in (("learning", [d for d in days if d < args.crystallised_from]),
                         ("crystallised", [d for d in days if d >= args.crystallised_from])):
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                gap = b - a
                if gap > 10:
                    continue
                za, ba = data[a]
                zb, bb = data[b]
                value = unbiased_squared_centroid_distance(za, zb, ba, bb) / scale
                by_gap.setdefault(gap, {}).setdefault(regime, []).append(value)
    for gap in sorted(by_gap):
        row = by_gap[gap]
        cells = []
        for regime in ("learning", "crystallised"):
            values = row.get(regime)
            cells.append(f"{np.median(values):.4f}" if values else "--")
        print(f"{gap:>5}{cells[0]:>12}{cells[1]:>14}")

    # ---- Test 2: distance to the crystallised endpoint ----
    # Comparing every day against a FIXED adult reference, rather than against the
    # previous day. Consecutive-day differences are small and mostly noise (Phase 3 found
    # the same in adults); distance to a fixed reference integrates the change.
    endpoint_days = [d for d in days if d >= args.endpoint_from]
    block_means = {d: np.stack([data[d][0][data[d][1] == g].mean(axis=0)
                                for g in np.unique(data[d][1])]) for d in days}
    reference = np.vstack([block_means[d] for d in endpoint_days])
    rng = np.random.default_rng(0)

    def interval(a, b, n_boot=300):
        estimate = unbiased_squared_centroid_distance(a, b)
        draws = [unbiased_squared_centroid_distance(
            a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
            for _ in range(n_boot)]
        low, high = np.percentile(draws, [2.5, 97.5])
        return estimate / scale, low / scale, high / scale

    print(f"\nDISTANCE TO CRYSTALLISED ENDPOINT (dph >= {args.endpoint_from})")
    print(f"{'dph':>5}{'distance':>11}{'95% CI':>22}")
    to_endpoint = []
    for day in days:
        if day in endpoint_days:
            continue
        estimate, low, high = interval(block_means[day], reference)
        to_endpoint.append({"dph": day, "distance": estimate, "ci_low": low,
                            "ci_high": high})
        print(f"{day:>5}{estimate:>11.4f}   [{low:>8.4f},{high:>8.4f}]")

    x = np.array([r["dph"] for r in to_endpoint], dtype=float)
    y = np.array([r["distance"] for r in to_endpoint])
    from scipy.stats import linregress, spearmanr
    fit = linregress(x, y)
    rho = spearmanr(x, y)
    early, late = y[x < 70], y[x >= args.crystallised_from + 5]
    convergence = {
        "slope_per_day": float(fit.slope), "slope_p": float(fit.pvalue),
        "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
        "mean_early_dph_lt70": float(early.mean()) if len(early) else None,
        "mean_late": float(late.mean()) if len(late) else None,
        "ratio": float(early.mean() / late.mean()) if len(early) and len(late) else None,
    }
    print(f"\n  linear slope {fit.slope:+.5f} per day (p={fit.pvalue:.2e})")
    print(f"  Spearman rho {rho.statistic:+.3f} (p={rho.pvalue:.2e})")
    if convergence["ratio"]:
        print(f"  mean distance dph<70: {early.mean():.4f} vs late: {late.mean():.4f}"
              f"  -> {convergence['ratio']:.1f}x closer to adult song")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "bird": bird, "proj": str(args.proj), "block_size": args.block_size,
            "crystallised_from_dph": args.crystallised_from,
            "scale_within_day_variance": round(scale, 5),
            "days": days, "consecutive": consecutive, "summary": summary,
            "distance_to_endpoint": to_endpoint, "convergence": convergence,
            "endpoint_days": endpoint_days,
            "by_gap": {str(g): {k: float(np.median(v)) for k, v in r.items()}
                       for g, r in by_gap.items()},
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

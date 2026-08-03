"""Score amplitude segmentation against the Bengalese finch hand annotations.

Usage::

    python scripts/phase1_segmentation_eval.py --root data/bfsongrepo
    python scripts/phase1_segmentation_eval.py --root data/bfsongrepo --global-threshold 0.01

The threshold is fitted **per bird** on that bird's first day, then applied to its
remaining days, which are the only ones reported. Fitting and reporting on the same day
would flatter the result. Per-bird fitting is necessary, not cosmetic: on this dataset the
per-bird optimum ranges 0.004-0.010, and one global value costs `bl26lb16` ~0.18 F1.

TWO INTERPRETATION CAVEATS.

1. The reference boundaries were drawn with ``evsonganaly``, itself an amplitude-threshold
   segmenter. High agreement shows this reproduces *that* algorithm's boundaries, not that
   either finds the acoustically true syllable edge. Label recovery is the non-circular
   test.
2. Predictions are restricted to each file's annotated span (first onset to last offset,
   plus 50 ms) so unannotated lead-in and lead-out are not charged as false positives.

Files whose filename date contradicts their directory are skipped and counted:
``gy6or6/032212`` holds 10 files dated 2012-03-13 and templated ``washout``, a different
experimental phase filed under a baseline day.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from songbird.ingest.bfsongrepo import load_day
from songbird.metrics.segmentation import segment_scores
from songbird.segment.amplitude import (
    DEFAULT_THRESHOLD_GRID,
    AmplitudeSegmenter,
    tune_threshold,
)

# Set from the ground-truth distribution, not guessed: the 1st-percentile inter-syllable
# gap is 9.3 ms (a 20 ms min_silence would merge ~25% of adjacent syllable pairs) and the
# 1st-percentile syllable duration is 32.9 ms.
SHAPE_PARAMS = dict(min_syllable_s=0.02, min_silence_s=0.005, smooth_ms=2.0)
SPAN_PAD_S = 0.05


def load_examples(day_dir: Path, limit: int | None) -> tuple[list, dict]:
    """Return ``(audio, sample_rate, true_segments)`` triples plus coverage attrs."""
    table = load_day(day_dir, on_date_mismatch="skip")
    truth = {
        name: sorted(zip(group["onset_s"], group["offset_s"]))
        for name, group in table.groupby("audio_file")
    }
    names = sorted(truth)[:limit] if limit else sorted(truth)
    examples = []
    for name in names:
        audio, sample_rate = sf.read(day_dir / name, dtype="float32")
        examples.append((audio, sample_rate, truth[name]))
    return examples, dict(table.attrs)


def score_examples(examples: list, segmenter: AmplitudeSegmenter,
                   tolerance_s: float) -> tuple[int, int, int]:
    n_true = n_pred = n_matched = 0
    for audio, sample_rate, true_segments in examples:
        predicted = segmenter.segment(audio, sample_rate)
        low, high = true_segments[0][0], true_segments[-1][1]
        predicted = [
            s for s in predicted if s[0] >= low - SPAN_PAD_S and s[1] <= high + SPAN_PAD_S
        ]
        scores = segment_scores(true_segments, predicted, tolerance_s)
        n_true += scores.n_true
        n_pred += scores.n_pred
        n_matched += scores.n_matched
    return n_true, n_pred, n_matched


def prf(n_true: int, n_pred: int, n_matched: int) -> tuple[float, float, float]:
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=25, help="files per day (0 = all)")
    parser.add_argument("--tune-limit", type=int, default=15, help="files for fitting")
    parser.add_argument("--global-threshold", type=float,
                        help="skip per-bird fitting and use this threshold everywhere")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    limit = args.limit or None
    tolerance = args.tolerance_ms / 1000

    by_bird: dict[str, list[Path]] = defaultdict(list)
    for day in sorted(args.root.glob("*/*")):
        if day.is_dir() and re.fullmatch(r"\d{6}", day.name):
            by_bird[day.parent.name].append(day)
    if not by_bird:
        raise SystemExit(f"no {{bird}}/{{MMDDYY}} day directories under {args.root}")

    results, thresholds = [], {}
    for bird, days in sorted(by_bird.items()):
        if args.global_threshold is not None:
            thresholds[bird] = {"threshold": args.global_threshold, "fit_f1": None,
                                "fit_day": None}
            report_days = days
        else:
            fit_examples, _ = load_examples(days[0], args.tune_limit)
            threshold, fit_f1 = tune_threshold(
                fit_examples, DEFAULT_THRESHOLD_GRID, tolerance, **SHAPE_PARAMS
            )
            thresholds[bird] = {"threshold": threshold, "fit_f1": round(fit_f1, 4),
                                "fit_day": days[0].name}
            report_days = days[1:]  # held out from fitting
            if not report_days:
                print(f"  {bird}: only one day, nothing held out -- skipping")
                continue

        segmenter = AmplitudeSegmenter(threshold=thresholds[bird]["threshold"],
                                       **SHAPE_PARAMS)
        for day in report_days:
            examples, attrs = load_examples(day, limit)
            n_true, n_pred, n_matched = score_examples(examples, segmenter, tolerance)
            precision, recall, f1 = prf(n_true, n_pred, n_matched)
            results.append({
                "bird": bird, "day": day.name, "threshold": thresholds[bird]["threshold"],
                "n_files": len(examples), "n_true": n_true, "n_pred": n_pred,
                "n_matched": n_matched, "precision": round(precision, 4),
                "recall": round(recall, 4), "f1": round(f1, 4),
                "n_date_mismatch_files": attrs.get("n_date_mismatch_files", 0),
                "n_unannotated_files": attrs.get("n_unannotated_files", 0),
            })

    mode = ("global threshold %.3f" % args.global_threshold
            if args.global_threshold is not None else "per-bird fitted threshold")
    print(f"segmenter shape params: {SHAPE_PARAMS}")
    print(f"mode: {mode}; tolerance {args.tolerance_ms:.0f} ms; "
          f"files/day {limit or 'all'}\n")
    if args.global_threshold is None:
        print("fitted thresholds (on each bird's first day, held out below):")
        for bird, info in sorted(thresholds.items()):
            print(f"  {bird:<10} threshold={info['threshold']:<6} "
                  f"fit-day={info['fit_day']} fit-F1={info['fit_f1']}")
        print()

    print(f"{'bird':<10}{'day':<9}{'thr':>7}{'files':>6}{'syll':>7}"
          f"{'F1':>7}{'prec':>7}{'rec':>7}{'skip':>6}")
    for r in results:
        print(f"{r['bird']:<10}{r['day']:<9}{r['threshold']:>7.3f}{r['n_files']:>6}"
              f"{r['n_true']:>7}{r['f1']:>7.3f}{r['precision']:>7.3f}"
              f"{r['recall']:>7.3f}{r['n_date_mismatch_files']:>6}")

    totals = [sum(r[k] for r in results) for k in ("n_true", "n_pred", "n_matched")]
    precision, recall, f1 = prf(*totals)
    print(f"\npooled over {len(results)} held-out bird-days: F1={f1:.3f} "
          f"precision={precision:.3f} recall={recall:.3f} (n={totals[0]})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "shape_params": SHAPE_PARAMS, "tolerance_s": tolerance,
            "files_per_day": limit, "mode": mode, "thresholds": thresholds,
            "days": results,
            "pooled": {"f1": round(f1, 4), "precision": round(precision, 4),
                       "recall": round(recall, 4), "n_true": totals[0]},
        }, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

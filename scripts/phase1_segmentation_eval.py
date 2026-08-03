"""Score amplitude segmentation against the Bengalese finch hand annotations.

Usage::

    python scripts/phase1_segmentation_eval.py --root data/bfsongrepo
    python scripts/phase1_segmentation_eval.py --root data/bfsongrepo --sweep

IMPORTANT INTERPRETATION CAVEAT. The ground-truth boundaries in this dataset were drawn
with ``evsonganaly``, which is itself an amplitude-threshold segmenter (its ``.not.mat``
files store ``threshold``, ``min_int``, ``min_dur`` and ``sm_win``). High agreement here
therefore shows that this segmenter reproduces *that* algorithm's boundaries -- it is not
independent evidence that either finds the acoustically "true" syllable edges. The
non-circular test is label recovery, where the targets are human judgements.

Predictions are restricted to the annotated span of each file (first onset to last offset,
plus 50 ms) so that unannotated lead-in and lead-out are not charged as false positives.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import soundfile as sf

from songbird.ingest.bfsongrepo import load_day
from songbird.metrics.segmentation import segment_scores
from songbird.segment.amplitude import AmplitudeSegmenter

# Chosen from the ground-truth distribution rather than by guessing: the 1st percentile
# inter-syllable gap is 9.3 ms (so a 20 ms min_silence would merge ~25% of adjacent
# syllable pairs), and the 1st percentile syllable duration is 32.9 ms.
DEFAULT_SEGMENTER = dict(
    threshold=0.01, min_syllable_s=0.02, min_silence_s=0.005, smooth_ms=2.0
)
SPAN_PAD_S = 0.05


def evaluate_day(day_dir: Path, segmenter: AmplitudeSegmenter, tolerance_s: float,
                 limit: int | None = None) -> dict:
    table = load_day(day_dir)
    truth = {
        name: sorted(zip(group["onset_s"], group["offset_s"]))
        for name, group in table.groupby("audio_file")
    }
    names = sorted(truth)[:limit] if limit else sorted(truth)

    n_true = n_pred = n_matched = 0
    for name in names:
        audio, sample_rate = sf.read(day_dir / name, dtype="float32")
        true_segments = truth[name]
        predicted = segmenter.segment(audio, sample_rate)
        low, high = true_segments[0][0], true_segments[-1][1]
        predicted = [
            s for s in predicted if s[0] >= low - SPAN_PAD_S and s[1] <= high + SPAN_PAD_S
        ]
        scores = segment_scores(true_segments, predicted, tolerance_s)
        n_true += scores.n_true
        n_pred += scores.n_pred
        n_matched += scores.n_matched

    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "day": day_dir.name,
        "bird": day_dir.parent.name,
        "n_files": len(names),
        "n_true": n_true,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def find_days(root: Path) -> list[Path]:
    return sorted(
        p for p in root.glob("*/*") if p.is_dir() and re.fullmatch(r"\d{6}", p.name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=25, help="files per day (0 = all)")
    parser.add_argument("--sweep", action="store_true", help="sweep threshold on day 1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    days = find_days(args.root)
    if not days:
        raise SystemExit(f"no day directories under {args.root}")
    limit = args.limit or None
    tolerance = args.tolerance_ms / 1000

    if args.sweep:
        print(f"threshold sweep on {days[0]} (tolerance {args.tolerance_ms:.0f} ms)")
        print(f"{'threshold':>10}{'F1':>8}{'precision':>11}{'recall':>8}")
        for threshold in (0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05):
            params = {**DEFAULT_SEGMENTER, "threshold": threshold}
            result = evaluate_day(days[0], AmplitudeSegmenter(**params), tolerance, limit)
            print(
                f"{threshold:>10.3f}{result['f1']:>8.3f}"
                f"{result['precision']:>11.3f}{result['recall']:>8.3f}"
            )
        return

    segmenter = AmplitudeSegmenter(**DEFAULT_SEGMENTER)
    results = [evaluate_day(day, segmenter, tolerance, limit) for day in days]

    print(f"segmenter: {DEFAULT_SEGMENTER}")
    print(f"tolerance: {args.tolerance_ms:.0f} ms; files/day: {limit or 'all'}\n")
    print(f"{'bird':<10}{'day':<9}{'files':>6}{'syllables':>11}{'F1':>7}{'prec':>7}{'rec':>7}")
    for r in results:
        print(
            f"{r['bird']:<10}{r['day']:<9}{r['n_files']:>6}{r['n_true']:>11}"
            f"{r['f1']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}"
        )

    total_true = sum(r["n_true"] for r in results)
    total_pred = sum(r["n_pred"] for r in results)
    total_matched = sum(r["n_matched"] for r in results)
    precision = total_matched / total_pred if total_pred else 0.0
    recall = total_matched / total_true if total_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\npooled over {len(results)} bird-days: F1={f1:.3f} "
          f"precision={precision:.3f} recall={recall:.3f} (n={total_true})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"segmenter": DEFAULT_SEGMENTER, "tolerance_s": tolerance,
             "files_per_day": limit, "days": results,
             "pooled": {"f1": round(f1, 4), "precision": round(precision, 4),
                        "recall": round(recall, 4), "n_true": total_true}},
            indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Evaluate the trained TweetyNet model and report label recovery.

This is the non-circular half of the Phase 1 gate. Segmentation agreement is partly
circular -- the reference boundaries were themselves drawn with an amplitude-threshold
segmenter -- but the syllable *labels* are human judgements, so recovering them is real
evidence that the pipeline reproduces expert annotation.

Two evaluations are run and reported separately:

* **within-day**: the held-out test split of the training day.
* **cross-day**: a different day from the same bird.

The cross-day number is the one that matters for this project. A labeller that degrades
on later days injects a time-varying artefact that is indistinguishable from song drift.
Whatever gap appears between within-day and cross-day accuracy is a floor on the drift
that Phase 3 could spuriously detect, so it must be measured before any drift is claimed.

Usage::

    python scripts/phase1_eval_labels.py \
        --train-results results/phase1/vak/train \
        --config configs/tweetynet_eval_gy6or6_032512.toml
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def latest_results_dir(root: Path) -> Path:
    candidates = sorted(p for p in root.glob("results_*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"no results_* directory under {root}")
    return candidates[-1]


def find_checkpoint(results_dir: Path, prefer: str = "max-val-acc-checkpoint.pt") -> Path:
    checkpoints = results_dir / "TweetyNet" / "checkpoints"
    preferred = checkpoints / prefer
    if preferred.exists():
        return preferred
    fallback = checkpoints / "checkpoint.pt"
    if fallback.exists():
        print(f"NOTE: {prefer} absent, falling back to {fallback.name}")
        return fallback
    raise SystemExit(f"no checkpoint found in {checkpoints}")


def patch_config(config_path: Path, results_dir: Path) -> None:
    """Point the eval config at the trained model's artefacts."""
    checkpoint = find_checkpoint(results_dir)
    labelmap = results_dir / "labelmap.json"
    standardizer = results_dir / "FramesStandardizer"

    text = config_path.read_text()
    settings = {
        "checkpoint_path": str(checkpoint),
        "labelmap_path": str(labelmap),
    }
    if standardizer.exists():
        settings["frames_standardizer_path"] = str(standardizer)

    for key, value in settings.items():
        line = f'{key} = "{value}"'
        if re.search(rf"^{key} = .*$", text, re.M):
            text = re.sub(rf"^{key} = .*$", line, text, flags=re.M)
        else:
            text = text.replace("[vak.eval]", f"[vak.eval]\n{line}", 1)
    config_path.write_text(text)
    print(f"patched {config_path} -> {checkpoint.name}")


def run_eval(config_path: Path) -> int:
    result = subprocess.run(
        [sys.executable.replace("python", "vak"), "eval", str(config_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        vak = Path(sys.executable).with_name("vak")
        result = subprocess.run(
            [str(vak), "eval", str(config_path)], capture_output=True, text=True
        )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
    return result.returncode


def collect_metrics(output_dir: Path) -> dict:
    """Read whatever metrics vak wrote into the eval output directory."""
    metrics = {}
    for csv_path in sorted(output_dir.rglob("*.csv")):
        try:
            import pandas as pd

            frame = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"could not read {csv_path}: {exc}")
            continue
        numeric = {
            column: float(frame[column].iloc[0])
            for column in frame.columns
            if frame[column].dtype.kind in "fi" and len(frame)
        }
        if numeric:
            metrics[csv_path.name] = numeric
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results_dir = latest_results_dir(args.train_results)
    print(f"using training run: {results_dir}")
    patch_config(args.config, results_dir)

    code = run_eval(args.config)
    if code != 0:
        raise SystemExit(f"vak eval failed with exit code {code}")

    output_dir = Path(
        re.search(r"output_dir = '([^']+)'", args.config.read_text()).group(1)
    )
    metrics = collect_metrics(output_dir)
    print(json.dumps(metrics, indent=1))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"training_run": str(results_dir), "config": str(args.config),
             "metrics": metrics}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

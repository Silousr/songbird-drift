"""Command line interface.

Four commands, matching the order a lab actually does things:

    songbird manifest   describe where the recordings are
    songbird analyse    drift and noise floors, per bird
    songbird compare    treated versus control
    songbird plan       how many birds, before running anything

Failures print an explanation to stderr and return a non-zero code. A tool whose output
feeds an experimental design must not fail quietly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from songbird import __version__

__all__ = ["main"]


def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-pca", type=int, default=64,
                        help="embedding dimensions (default 64; UMAP is not offered "
                             "because it destroys within-type geometry)")
    parser.add_argument("--n-freq-bins", type=int, default=32)
    parser.add_argument("--max-per-day", type=int, default=2000)
    parser.add_argument("--exclude-labels", nargs="*", default=[],
                        help="label characters that are not song syllables")
    parser.add_argument("--min-renditions", type=int, default=20)
    parser.add_argument("--min-bouts", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--n-null", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)


def _manifest(args) -> int:
    from songbird.ingest.generic import build_manifest

    manifest = build_manifest(args.root, pattern=args.pattern,
                              timestamp_format=args.timestamp_format,
                              audio_suffix=args.audio_suffix,
                              annot_suffix=args.annot_suffix, out=args.out)
    print(f"wrote {args.out}: {len(manifest)} recordings, "
          f"{manifest['bird'].nunique() if len(manifest) else 0} birds")
    unmatched = manifest.attrs.get("n_unmatched", 0)
    if unmatched:
        print(f"WARNING: {unmatched} audio file(s) did not match the pattern and were "
              f"left out. An unmatched file is usually a naming convention you did not "
              f"know you had:")
        for name in manifest.attrs.get("unmatched_files", []):
            print(f"    {name}")
    return 0


def _analyse(args) -> int:
    from songbird.ingest.generic import load_from_manifest, load_manifest
    from songbird.pipeline import AnalysisConfig, analyse

    if not Path(args.manifest).exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    table = load_from_manifest(manifest, annot_format=args.annot_format,
                               on_missing="skip")
    missing = table.attrs.get("n_missing_annotations", 0)
    if missing:
        print(f"WARNING: {missing} recording(s) had no annotation file and were skipped")
    if len(table) == 0:
        print("error: no annotated syllables were loaded", file=sys.stderr)
        return 2

    config = AnalysisConfig(
        n_pca=args.n_pca, n_freq_bins=args.n_freq_bins, max_per_day=args.max_per_day,
        exclude_labels=tuple(args.exclude_labels), min_renditions=args.min_renditions,
        min_bouts=args.min_bouts, n_boot=args.n_boot, n_null=args.n_null, seed=args.seed,
    )
    try:
        result = analyse(table, config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(result.summary())
    print("\nInterpretation: drift is only evidence of change when it exceeds the noise "
          "floor for that bird.\nBoth metrics are reported because a manipulation may "
          "move a syllable, or make it sloppier, or both.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result.to_dict(), indent=1, default=str))
    print(f"\nwrote {args.out}")

    if args.drift_out:
        drift = result.per_bird_drift()
        import pandas as pd

        pd.DataFrame({"bird": list(drift), "drift": list(drift.values())}).to_csv(
            args.drift_out, index=False)
        print(f"wrote {args.drift_out} (one drift value per bird, for `songbird compare`)")
    return 0


def _compare(args) -> int:
    import pandas as pd

    from songbird.group import compare_groups

    path = Path(args.drift)
    if not path.exists():
        print(f"error: drift table not found: {path}", file=sys.stderr)
        return 2
    frame = pd.read_csv(path)
    for column in ("group", args.value_column):
        if column not in frame.columns:
            print(f"error: drift table needs a {column!r} column; found "
                  f"{list(frame.columns)}", file=sys.stderr)
            return 2

    groups = frame["group"].unique()
    if set(groups) != {args.treated, args.control}:
        print(f"error: expected groups {args.treated!r} and {args.control!r}; "
              f"found {sorted(groups)}", file=sys.stderr)
        return 2

    treated = frame.loc[frame["group"] == args.treated, args.value_column].to_numpy()
    control = frame.loc[frame["group"] == args.control, args.value_column].to_numpy()
    try:
        result = compare_groups(treated, control, alternative=args.alternative,
                                within_bird_sd=args.within_bird_sd, seed=args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(result.summary())
    print("\nNote: one value per bird. The bird is the unit of replication; pooling "
          "syllables\nacross birds would be pseudo-replication and would inflate "
          "significance.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(asdict(result), indent=1))
    print(f"\nwrote {args.out}")
    return 0


def _plan(args) -> int:
    from songbird.group import birds_needed

    n = birds_needed(effect=args.effect, between_bird_sd=args.between_bird_sd,
                     alpha=args.alpha, power=args.power, max_n=args.max_n,
                     n_simulations=args.n_simulations,
                     n_permutations=args.n_permutations, seed=args.seed)
    print(f"effect {args.effect}, between-bird SD {args.between_bird_sd}, "
          f"power {args.power:.0%}, alpha {args.alpha}")
    if np.isnan(n):
        print(f"-> more than {args.max_n} birds per group would be needed")
    else:
        print(f"-> {n:.0f} birds per group")
    print("\nEstimated by simulating the permutation test that would actually be run, "
          "not\nfrom a normal-theory formula, which is optimistic at these sample sizes.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="songbird",
        description="Measure within-bird song drift with a calibrated noise floor.",
    )
    parser.add_argument("--version", action="version", version=f"songbird {__version__}")
    sub = parser.add_subparsers(dest="command")

    manifest = sub.add_parser("manifest", help="build a recording manifest from filenames")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--pattern", required=True,
                          help=r"regex with named groups 'bird' and 'timestamp'")
    manifest.add_argument("--timestamp-format", required=True,
                          help="strptime format for the captured timestamp")
    manifest.add_argument("--audio-suffix", default=".wav")
    manifest.add_argument("--annot-suffix", default=".wav.csv")
    manifest.set_defaults(func=_manifest)

    analyse = sub.add_parser("analyse", help="drift and noise floors, per bird")
    analyse.add_argument("--manifest", required=True)
    analyse.add_argument("--out", required=True)
    analyse.add_argument("--drift-out", help="also write one drift value per bird")
    analyse.add_argument("--annot-format", default="simple-seq")
    _add_analysis_options(analyse)
    analyse.set_defaults(func=_analyse)

    compare = sub.add_parser("compare", help="treated versus control")
    compare.add_argument("--drift", required=True,
                         help="CSV with one row per bird, columns: bird, group, drift")
    compare.add_argument("--out", required=True)
    compare.add_argument("--value-column", default="drift")
    compare.add_argument("--treated", default="treated")
    compare.add_argument("--control", default="control")
    compare.add_argument("--alternative", default="two-sided",
                         choices=["two-sided", "greater", "less"])
    compare.add_argument("--within-bird-sd", type=float)
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(func=_compare)

    plan = sub.add_parser("plan", help="birds per group needed for an effect")
    plan.add_argument("--effect", type=float, required=True)
    plan.add_argument("--between-bird-sd", type=float, required=True)
    plan.add_argument("--alpha", type=float, default=0.05)
    plan.add_argument("--power", type=float, default=0.8)
    plan.add_argument("--max-n", type=int, default=40)
    plan.add_argument("--n-simulations", type=int, default=400)
    plan.add_argument("--n-permutations", type=int, default=2000)
    plan.add_argument("--seed", type=int, default=0)
    plan.set_defaults(func=_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

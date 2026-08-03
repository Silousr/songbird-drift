"""Ingest a lab's own recordings into the canonical syllable table.

Ingestion is driven by an explicit **manifest** — a CSV listing, per recording: the audio
file, its annotation file, which bird, and when it was recorded. It is deliberately not
driven by inferring conventions from filenames.

The reason is experience rather than taste. A single curated public dataset in this project
contained a date encoded backwards relative to its own directory (`MMDDYY` versus `DDMMYY`)
and a file counter that silently wrapped negative partway through a day. Both were caught
only because the loader cross-checked instead of trusting. A manifest makes the mapping
explicit and reviewable; :func:`build_manifest` generates one from filenames when a
convention does hold, and reports how many files it could not match rather than dropping
them quietly.

Annotations are read through `crowsetta`, so any format it supports works: ``simple-seq``
(3-column CSV), ``notmat`` (evsonganaly), ``raven``, ``textgrid`` (Praat), ``aud-seq``
(Audacity), ``birdsong-recognition-dataset``, ``yarden``, ``generic-seq``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from songbird.ingest.schema import SYLLABLE_COLUMNS, empty_table, finalise_table

__all__ = ["build_manifest", "load_from_manifest", "load_manifest", "MANIFEST_COLUMNS"]

MANIFEST_COLUMNS = ("bird", "timestamp", "audio_path", "annot_path")
OPTIONAL_COLUMNS = ("group",)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read and validate a recording manifest."""
    manifest = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(
            f"{path}: manifest is missing required column(s) {missing}. "
            f"Required: {list(MANIFEST_COLUMNS)}"
        )
    try:
        manifest["timestamp"] = pd.to_datetime(manifest["timestamp"], format="mixed")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{path}: could not parse the timestamp column: {exc}"
        ) from exc
    if manifest["timestamp"].isna().any():
        bad = manifest.loc[manifest["timestamp"].isna()].index.tolist()[:5]
        raise ValueError(f"{path}: unparseable timestamp in row(s) {bad}")
    return manifest


def _annotation_class(annot_format: str):
    import crowsetta

    try:
        return crowsetta.formats.by_name(annot_format)
    except Exception as exc:
        available = sorted(crowsetta.formats.as_list())
        raise ValueError(
            f"unsupported annotation format {annot_format!r}. "
            f"crowsetta supports: {available}"
        ) from exc


def load_from_manifest(
    manifest: pd.DataFrame,
    annot_format: str = "simple-seq",
    source: str = "lab",
    on_missing: str = "raise",
) -> pd.DataFrame:
    """Turn a manifest into the canonical syllable table.

    ``on_missing='skip'`` tolerates absent annotation files and records how many were
    skipped in ``table.attrs['n_missing_annotations']`` — never silently, because partial
    annotation coverage otherwise makes a half-annotated day look like a complete one.
    """
    if on_missing not in ("raise", "skip"):
        raise ValueError(f"on_missing must be 'raise' or 'skip', got {on_missing!r}")

    annotation_class = _annotation_class(annot_format)
    has_group = "group" in manifest.columns

    rows: list[dict] = []
    n_missing = 0
    for record in manifest.itertuples():
        annot_path = Path(record.annot_path)
        if not annot_path.exists():
            if on_missing == "raise":
                raise FileNotFoundError(
                    f"annotation file not found: {annot_path} "
                    f"(pass on_missing='skip' to tolerate and count these)"
                )
            n_missing += 1
            continue

        annotation = annotation_class.from_file(annot_path).to_annot()
        sequence = annotation.seq
        timestamp = pd.Timestamp(record.timestamp)
        for onset, offset, label in zip(
            sequence.onsets_s, sequence.offsets_s, sequence.labels
        ):
            if offset <= onset:
                raise ValueError(
                    f"{annot_path}: offset {offset} does not follow onset {onset}"
                )
            row = {
                "bird": str(record.bird),
                "day": timestamp.date(),
                "timestamp": timestamp.to_pydatetime(),
                "audio_file": Path(record.audio_path).name,
                "audio_path": str(record.audio_path),
                "template": None,
                "onset_s": float(onset),
                "offset_s": float(offset),
                "duration_s": float(offset - onset),
                "label": str(label),
                "source": source,
            }
            if has_group:
                row["group"] = getattr(record, "group")
            rows.append(row)

    columns = list(SYLLABLE_COLUMNS) + (["group"] if has_group else [])
    table = finalise_table(rows, columns) if rows else empty_table(columns)
    table.attrs["n_recordings"] = int(len(manifest))
    table.attrs["n_missing_annotations"] = n_missing
    table.attrs["n_bouts"] = int(table["audio_file"].nunique()) if len(table) else 0
    return table


def build_manifest(
    root: str | Path,
    pattern: str,
    timestamp_format: str,
    audio_suffix: str = ".wav",
    annot_suffix: str = ".wav.csv",
    out: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a manifest from filenames matching a regex.

    ``pattern`` must contain named groups ``bird`` and ``timestamp``; any further named
    groups are carried through as extra columns. Files that do not match are counted in
    ``manifest.attrs['n_unmatched']`` and listed, rather than dropped silently — an
    unmatched file is usually a convention you did not know you had.
    """
    compiled = re.compile(pattern)
    required = {"bird", "timestamp"}
    if not required <= set(compiled.groupindex):
        raise ValueError(
            f"pattern must contain named group(s) {sorted(required)}; "
            f"got {sorted(compiled.groupindex)}"
        )

    root = Path(root)
    rows, unmatched = [], []
    for audio in sorted(root.rglob(f"*{audio_suffix}")):
        if audio.name.endswith(annot_suffix):
            continue
        match = compiled.search(audio.name)
        if match is None:
            unmatched.append(audio.name)
            continue
        fields = match.groupdict()
        try:
            timestamp = pd.to_datetime(fields.pop("timestamp"), format=timestamp_format)
        except ValueError as exc:
            raise ValueError(f"{audio.name}: timestamp did not match "
                             f"{timestamp_format!r}: {exc}") from exc

        annot = audio.with_name(audio.name.replace(audio_suffix, annot_suffix))
        rows.append({"bird": fields.pop("bird"), "timestamp": timestamp,
                     "audio_path": str(audio), "annot_path": str(annot), **fields})

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)) if not rows \
        else pd.DataFrame(rows)
    manifest.attrs["n_unmatched"] = len(unmatched)
    manifest.attrs["unmatched_files"] = unmatched[:20]
    if out:
        manifest.to_csv(out, index=False)
    return manifest

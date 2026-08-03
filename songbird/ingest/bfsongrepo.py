"""Ingestion for the Bengalese Finch Song Repository (figshare 10.6084/m9.figshare.4805749).

Layout is ``{bird}/{MMDDYY}/{bird}[_{template}]_{DDMMYY}_{HHMM}.{serial}.wav`` with a
sibling ``.wav.csv`` holding ``onset_s,offset_s,label`` per syllable.

The directory date is **MMDDYY** and the filename date is **DDMMYY** -- reversed. They
encode the same day. :func:`parse_recording_filename` accepts the directory name so the
two can be cross-checked; a disagreement raises rather than being silently accepted,
because a corrupted time axis leaves every downstream number plausible but wrong.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from songbird.ingest.schema import SYLLABLE_COLUMNS, empty_table, finalise_table

__all__ = [
    "DateMismatchError",
    "Recording",
    "Syllable",
    "SYLLABLE_COLUMNS",
    "load_dataset",
    "load_day",
    "parse_day_dir",
    "parse_recording_filename",
    "read_annotation_csv",
]

#: Policies for a filename date that contradicts its directory date.
DATE_MISMATCH_POLICIES = ("raise", "skip")

# Matches the invariant tail: DDMMYY_HHMM.serial.wav
#
# The serial may be NEGATIVE: bl26lb16/042012 wraps a signed 16-bit counter from 32745 to
# -32754 partway through the day (84 of its 202 files). Serial is therefore an opaque
# identifier, not an ordering key -- order recordings by timestamp.
_TAIL = re.compile(r"(?P<date>\d{6})_(?P<time>\d{4})\.(?P<serial>-?\d+)\.wav$")


class DateMismatchError(ValueError):
    """Directory date and filename date disagree about the recording day."""


@dataclass(frozen=True)
class Recording:
    """One recorded song file."""

    bird: str
    timestamp: dt.datetime
    serial: int
    template: str | None = None

    @property
    def day(self) -> dt.date:
        return self.timestamp.date()


@dataclass(frozen=True)
class Syllable:
    """One annotated syllable, with times relative to the start of its audio file."""

    onset_s: float
    offset_s: float
    label: str

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s


def parse_day_dir(name: str) -> dt.date:
    """Parse an ``MMDDYY`` directory name into a date."""
    if not re.fullmatch(r"\d{6}", name):
        raise ValueError(f"not an MMDDYY directory name: {name!r}")
    month, day, year = int(name[:2]), int(name[2:4]), int(name[4:])
    try:
        return dt.date(2000 + year, month, day)
    except ValueError as exc:
        raise ValueError(f"invalid MMDDYY directory name {name!r}: {exc}") from exc


def parse_recording_filename(name: str, day_dir: str | None = None) -> Recording:
    """Parse a recording filename into a :class:`Recording`.

    Accepts either ``....wav`` or ``....wav.csv``. If ``day_dir`` is given, the
    filename's DDMMYY date is cross-checked against the directory's MMDDYY date.
    """
    stem = name[: -len(".csv")] if name.endswith(".csv") else name

    tail = _TAIL.search(stem)
    if tail is None or "_" not in stem:
        raise ValueError(f"unparseable recording filename: {name!r}")

    bird, rest = stem.split("_", 1)
    if not bird:
        raise ValueError(f"unparseable recording filename: {name!r}")

    prefix = rest[: tail.start() - len(bird) - 1]
    template = prefix.rstrip("_") or None

    date_field, time_field = tail["date"], tail["time"]
    day, month, year = int(date_field[:2]), int(date_field[2:4]), int(date_field[4:])
    hour, minute = int(time_field[:2]), int(time_field[2:])
    try:
        timestamp = dt.datetime(2000 + year, month, day, hour, minute)
    except ValueError as exc:
        raise ValueError(f"invalid date/time in filename {name!r}: {exc}") from exc

    if day_dir is not None:
        expected = parse_day_dir(day_dir)
        if timestamp.date() != expected:
            raise DateMismatchError(
                f"{name!r} carries date {timestamp.date()} but sits in directory "
                f"{day_dir!r} meaning {expected}. Refusing to guess which is right."
            )

    return Recording(
        bird=bird, timestamp=timestamp, serial=int(tail["serial"]), template=template
    )


def read_annotation_csv(path: str | Path) -> list[Syllable]:
    """Read a ``.wav.csv`` annotation into a list of :class:`Syllable`."""
    syllables: list[Syllable] = []
    with open(path, newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            onset, offset = float(row["onset_s"]), float(row["offset_s"])
            if offset <= onset:
                raise ValueError(
                    f"{path}:{row_number}: offset {offset} does not follow onset {onset}"
                )
            syllables.append(
                Syllable(onset_s=onset, offset_s=offset, label=str(row["label"]))
            )
    return syllables


def _finalise(rows: list[dict], coverage: dict[str, int]) -> pd.DataFrame:
    table = finalise_table(rows) if rows else empty_table()
    table.attrs.update(coverage)
    return table


def load_day(
    day_dir: str | Path,
    source: str = "bfsongrepo",
    on_date_mismatch: str = "raise",
) -> pd.DataFrame:
    """Load one ``{bird}/{MMDDYY}/`` directory into a canonical syllable table.

    Unannotated audio is counted in ``table.attrs`` rather than silently dropped --
    annotation coverage in this dataset is partial and uneven, and a day that is half
    annotated must not look like a day that is fully annotated.

    ``on_date_mismatch`` controls what happens when a filename's date contradicts its
    directory. ``"raise"`` (default) refuses to guess. ``"skip"`` drops those recordings
    and records how many in ``attrs["n_date_mismatch_files"]``. This is not hypothetical:
    ``gy6or6/032212`` contains 10 files dated 2012-03-13 and templated ``washout`` --
    a different experimental phase filed under a baseline day.
    """
    if on_date_mismatch not in DATE_MISMATCH_POLICIES:
        raise ValueError(
            f"on_date_mismatch must be one of {DATE_MISMATCH_POLICIES}, "
            f"got {on_date_mismatch!r}"
        )

    day_dir = Path(day_dir)
    bird = day_dir.parent.name

    audio_files = sorted(p for p in day_dir.glob("*.wav") if not p.name.endswith(".csv"))
    annotations = sorted(day_dir.glob("*.wav.csv"))

    rows: list[dict] = []
    n_mismatch = 0
    for annotation_path in annotations:
        try:
            recording = parse_recording_filename(
                annotation_path.name, day_dir=day_dir.name
            )
        except DateMismatchError:
            if on_date_mismatch == "raise":
                raise
            n_mismatch += 1
            continue

        for syllable in read_annotation_csv(annotation_path):
            rows.append(
                {
                    "bird": recording.bird or bird,
                    "day": recording.day,
                    "timestamp": recording.timestamp,
                    "audio_file": annotation_path.name[: -len(".csv")],
                    "template": recording.template,
                    "onset_s": syllable.onset_s,
                    "offset_s": syllable.offset_s,
                    "duration_s": syllable.duration_s,
                    "label": syllable.label,
                    "source": source,
                }
            )

    coverage = {
        "n_audio_files": len(audio_files),
        "n_annotated_files": len(annotations),
        "n_unannotated_files": max(len(audio_files) - len(annotations), 0),
        "n_date_mismatch_files": n_mismatch,
    }
    return _finalise(rows, coverage)


def load_dataset(
    root: str | Path,
    source: str = "bfsongrepo",
    on_date_mismatch: str = "raise",
) -> pd.DataFrame:
    """Load an entire ``{bird}/{MMDDYY}/`` tree into one canonical syllable table."""
    root = Path(root)
    day_dirs = sorted(
        p for p in root.glob("*/*") if p.is_dir() and re.fullmatch(r"\d{6}", p.name)
    )
    if not day_dirs:
        raise FileNotFoundError(f"no {{bird}}/{{MMDDYY}} day directories under {root}")

    tables = [
        load_day(day_dir, source=source, on_date_mismatch=on_date_mismatch)
        for day_dir in day_dirs
    ]
    combined = pd.concat(tables, ignore_index=True) if tables else empty_table()
    combined = combined.sort_values(["bird", "timestamp", "onset_s"], kind="stable")
    combined = combined.reset_index(drop=True)

    for key in (
        "n_audio_files",
        "n_annotated_files",
        "n_unannotated_files",
        "n_date_mismatch_files",
    ):
        combined.attrs[key] = sum(table.attrs[key] for table in tables)
    combined.attrs["n_days"] = len(day_dirs)
    return combined


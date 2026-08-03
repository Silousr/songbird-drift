"""Reproduce the Phase 0 inventory of the Duke juvenile zebra finch deposit.

Reads the per-bird ``{bird}_segs.zip`` archives from DOI 10.7924/r4j38x43h and derives,
without touching the 290 GB of raw audio: recorded days, day-post-hatch coverage, calendar
span, file counts, segmented-sound counts, and total segmented sound duration per day.

The archives are ~133 MB total. Download them first, e.g.::

    python scripts/phase0_duke_inventory.py --download --data-dir ./data/duke

Then::

    python scripts/phase0_duke_inventory.py --data-dir ./data/duke --out inventory.json

Three format quirks this handles, all of which silently corrupt a time axis if ignored:

* The top-level directory differs per bird (``segs1``, ``segs``, ``segs_undir``).
* Filename time fields are not zero-padded (``..._11_15_13_7_31.txt``).
* The numeric field is an Excel-style serial date (epoch 1899-12-30), and its *fractional*
  part is NOT time-of-day -- it disagrees with the explicit ``HH_MM_SS`` suffix by hours.
  Parse the suffix; use the integer part only as a cross-check.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

BIRDS = ("grn394", "grn395", "grn397", "grn475", "sil469")
BASE_URL = "https://duke.tind.io/record/135/files"
EXCEL_EPOCH = dt.date(1899, 12, 30)

# {anything}/{dph}/{bird}_{serial}.{frac}_{MM}_{DD}_{HH}_{MM}_{SS}.txt
SEG_NAME = re.compile(
    r"^[^/]+/(?P<dph>\d+)/[A-Za-z0-9]+_(?P<serial>\d+)\.(?P<frac>\d+)"
    r"_(?P<month>\d+)_(?P<day>\d+)_(?P<hour>\d+)_(?P<minute>\d+)_(?P<second>\d+)\.txt$"
)


@dataclass
class BirdInventory:
    """Per-day inventory for one bird, keyed by integer day-post-hatch."""

    bird: str
    files_per_day: dict[int, int] = field(default_factory=dict)
    sounds_per_day: dict[int, int] = field(default_factory=dict)
    seconds_per_day: dict[int, float] = field(default_factory=dict)
    dates_per_day: dict[int, str] = field(default_factory=dict)
    date_mismatches: int = 0
    unparsed_names: int = 0

    @property
    def days(self) -> list[int]:
        return sorted(self.files_per_day)

    @property
    def total_sounds(self) -> int:
        return sum(self.sounds_per_day.values())

    @property
    def total_minutes(self) -> float:
        return sum(self.seconds_per_day.values()) / 60.0

    def as_dict(self) -> dict:
        return {
            "bird": self.bird,
            "n_days": len(self.days),
            "dph_min": min(self.days) if self.days else None,
            "dph_max": max(self.days) if self.days else None,
            "calendar_span": [
                min(self.dates_per_day.values()),
                max(self.dates_per_day.values()),
            ]
            if self.dates_per_day
            else None,
            "n_wav_files": sum(self.files_per_day.values()),
            "n_segmented_sounds": self.total_sounds,
            "segmented_minutes": round(self.total_minutes, 1),
            "date_mismatches": self.date_mismatches,
            "unparsed_names": self.unparsed_names,
            "files_per_day": {str(d): self.files_per_day[d] for d in self.days},
            "sounds_per_day": {str(d): self.sounds_per_day.get(d, 0) for d in self.days},
            "seconds_per_day": {
                str(d): round(self.seconds_per_day.get(d, 0.0), 2) for d in self.days
            },
            "date_per_day": {str(d): self.dates_per_day.get(d) for d in self.days},
        }


def count_segments(payload: bytes) -> tuple[int, float]:
    """Return (n_segments, total_seconds) from one AVA onset/offset text file."""
    n, total = 0, 0.0
    for line in payload.decode(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            onset, offset = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        n += 1
        total += offset - onset
    return n, total


def inventory_bird(archive: Path, bird: str) -> BirdInventory:
    inv = BirdInventory(bird=bird)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            match = SEG_NAME.match(name)
            if match is None:
                inv.unparsed_names += 1
                continue
            dph = int(match["dph"])
            date = EXCEL_EPOCH + dt.timedelta(days=int(match["serial"]))
            if (date.month, date.day) != (int(match["month"]), int(match["day"])):
                inv.date_mismatches += 1

            n_seg, seconds = count_segments(zf.read(name))
            inv.files_per_day[dph] = inv.files_per_day.get(dph, 0) + 1
            inv.sounds_per_day[dph] = inv.sounds_per_day.get(dph, 0) + n_seg
            inv.seconds_per_day[dph] = inv.seconds_per_day.get(dph, 0.0) + seconds
            inv.dates_per_day.setdefault(dph, date.isoformat())
    return inv


def download(data_dir: Path, birds: tuple[str, ...]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for bird in birds:
        target = data_dir / f"{bird}_segs.zip"
        if target.exists():
            print(f"  {target.name} already present, skipping")
            continue
        url = f"{BASE_URL}/{bird}_segs.zip"
        print(f"  fetching {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=900) as response:
            target.write_bytes(response.read())
        print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory holding (or to receive) the {bird}_segs.zip archives",
    )
    parser.add_argument("--out", type=Path, help="write the inventory as JSON here")
    parser.add_argument(
        "--download", action="store_true", help="fetch missing archives from Duke first"
    )
    parser.add_argument("--birds", nargs="+", default=list(BIRDS))
    args = parser.parse_args()

    birds = tuple(args.birds)
    if args.download:
        download(args.data_dir, birds)

    results: dict[str, BirdInventory] = {}
    for bird in birds:
        archive = args.data_dir / f"{bird}_segs.zip"
        if not archive.exists():
            print(f"MISSING {archive} -- rerun with --download")
            continue
        results[bird] = inventory_bird(archive, bird)

    if not results:
        raise SystemExit("no archives found; nothing to inventory")

    header = f"{'bird':<8}{'days':>6}{'dph':>10}{'span':>26}{'files':>9}{'sounds':>10}{'minutes':>9}"
    print("\n" + header)
    print("-" * len(header))
    for bird, inv in results.items():
        span = f"{min(inv.dates_per_day.values())} -> {max(inv.dates_per_day.values())}"
        dph = f"{min(inv.days)}-{max(inv.days)}"
        print(
            f"{bird:<8}{len(inv.days):>6}{dph:>10}{span:>26}"
            f"{sum(inv.files_per_day.values()):>9}{inv.total_sounds:>10}"
            f"{inv.total_minutes:>9.1f}"
        )

    totals = {
        "bird_days": sum(len(i.days) for i in results.values()),
        "wav_files": sum(sum(i.files_per_day.values()) for i in results.values()),
        "segmented_sounds": sum(i.total_sounds for i in results.values()),
        "segmented_hours": round(sum(i.total_minutes for i in results.values()) / 60, 1),
        "date_mismatches": sum(i.date_mismatches for i in results.values()),
        "unparsed_names": sum(i.unparsed_names for i in results.values()),
    }
    print(
        f"\nTOTAL {totals['bird_days']} bird-days, {totals['wav_files']} wav files, "
        f"{totals['segmented_sounds']} sounds, {totals['segmented_hours']} h"
    )
    print(
        f"integrity: {totals['date_mismatches']} date mismatches, "
        f"{totals['unparsed_names']} unparsed filenames (both should be 0)"
    )

    if args.out:
        payload = {
            "source_doi": "10.7924/r4j38x43h",
            "totals": totals,
            "birds": {b: i.as_dict() for b, i in results.items()},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

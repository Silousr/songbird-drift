"""The canonical syllable table every loader produces.

One schema, so that a dataset-specific loader and a lab's own recordings are
interchangeable downstream. Anything that reads syllables — features, drift, dispersion,
power — depends on this and nothing else.

`day` is deliberately kept as ``datetime.date`` in an object column rather than letting
pandas coerce it to ``datetime64``: day identity is used as a grouping key throughout, and
it must compare equal to a plain date.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["SYLLABLE_COLUMNS", "empty_table", "finalise_table", "validate_table"]

SYLLABLE_COLUMNS = (
    "bird",        # individual identity
    "day",         # datetime.date of the recording
    "timestamp",   # full datetime of the bout
    "audio_file",  # bout identifier; the sampling unit for all inference
    "audio_path",  # resolvable path to the bout audio, for feature extraction
    "template",    # experimental condition tag from the recorder, if any
    "onset_s",     # syllable onset, seconds from the start of the bout
    "offset_s",
    "duration_s",
    "label",       # syllable type
    "source",      # dataset the row came from
)


def empty_table(columns=SYLLABLE_COLUMNS) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})


def finalise_table(rows: list[dict], columns=SYLLABLE_COLUMNS) -> pd.DataFrame:
    """Assemble rows into a sorted canonical table."""
    columns = list(columns)
    table = pd.DataFrame(rows, columns=columns)
    table["day"] = pd.Series([row["day"] for row in rows], dtype=object)
    table = table.sort_values(["bird", "timestamp", "onset_s"], kind="stable")
    return table.reset_index(drop=True)


def validate_table(table: pd.DataFrame) -> pd.DataFrame:
    """Raise if a table is missing required columns or contains impossible syllables."""
    missing = [c for c in SYLLABLE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"syllable table is missing column(s) {missing}")
    if len(table) and (table["offset_s"] <= table["onset_s"]).any():
        bad = int((table["offset_s"] <= table["onset_s"]).sum())
        raise ValueError(f"{bad} syllable(s) have offset at or before onset")
    return table

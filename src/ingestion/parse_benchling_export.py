"""Parse the spotting-run metadata table exported from the ELN (Benchling).

This is the one module to adjust when the real Benchling export format is known. It is
kept deliberately small and tolerant: column names are matched case- and
separator-insensitively and through a table of aliases, so a header of "Room Temp (C)"
or "room_temperature_c" both land on the same field.

Expected shape: one row per chip in the run. Run-level facts (run id, date, panel,
operator, conditions) repeat on every row, which is how a flat ELN table looks.

    spotting_run_id,run_date,panel,operator,chip_id,wafer_id,coating_batch,
    room_temp_c,room_humidity_pct,spotter_temp_c,spotter_humidity_pct,
    Anti-IL-6_lot,Anti-IL-6_lot_received_date, ... one pair per antibody

Antibody lots are constant within a run (one stock tube per antibody per spotting
session), so they are read from the run's rows and verified to be consistent.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd

LOT_SUFFIXES = ("_lot", "_lot_id", "_lot_number", "_lotno")
LOT_DATE_SUFFIXES = (
    "_lot_received_date",
    "_lot_received",
    "_lot_date",
    "_lot_opened_date",
    "_lot_opened",
)

# Canonical field -> accepted header spellings (normalized).
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "spotting_run_id": ("spotting_run_id", "run_id", "spotting_run", "run", "batch_id"),
    "run_date": ("run_date", "date", "spotting_date", "date_spotted"),
    "panel": ("panel", "panel_name", "assay_panel"),
    "operator": ("operator", "user", "spotter_user", "performed_by", "scientist"),
    "chip_id": ("chip_id", "chip", "chipid", "chip_no", "chip_number"),
    "wafer_id": ("wafer_id", "wafer", "wafer_lot", "wafer_no"),
    "coating_batch": (
        "coating_batch",
        "coating_lot",
        "surface_batch",
        "surface_coating_batch",
        "polymer_batch",  # accepted for backwards compatibility with older records
    ),
    "room_temp_c": ("room_temp_c", "room_temp", "room_temperature_c", "room_temperature"),
    "room_humidity_pct": (
        "room_humidity_pct",
        "room_humidity",
        "room_rh_pct",
        "room_rh",
    ),
    "spotter_temp_c": (
        "spotter_temp_c",
        "spotter_temp",
        "spotter_temperature_c",
        "spotter_temperature",
        "instrument_temp_c",
    ),
    "spotter_humidity_pct": (
        "spotter_humidity_pct",
        "spotter_humidity",
        "spotter_rh_pct",
        "spotter_rh",
        "instrument_humidity_pct",
    ),
}

ENV_FIELDS = ("room_temp_c", "room_humidity_pct", "spotter_temp_c", "spotter_humidity_pct")


class MetadataFileError(ValueError):
    """The run metadata export could not be parsed into the expected shape."""


@dataclass
class RunMetadata:
    """Run-level facts plus the per-chip and per-antibody-lot tables."""

    spotting_run_id: str
    run_date: dt.date
    panel: str | None
    operator: str | None
    room_temp_c: float | None
    room_humidity_pct: float | None
    spotter_temp_c: float | None
    spotter_humidity_pct: float | None
    chips: pd.DataFrame
    lots: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_file: str | None = None

    @property
    def chip_ids(self) -> list[str]:
        return self.chips["chip_id"].tolist()

    @property
    def coating_batches(self) -> list[str]:
        if "coating_batch" not in self.chips:
            return []
        return sorted(self.chips["coating_batch"].dropna().astype(str).unique().tolist())

    @property
    def wafer_ids(self) -> list[str]:
        if "wafer_id" not in self.chips:
            return []
        return sorted(self.chips["wafer_id"].dropna().astype(str).unique().tolist())


def _normalize(name: str) -> str:
    text = str(name).strip().lower()
    for ch in (" ", "-", ".", "/"):
        text = text.replace(ch, "_")
    for ch in ("(", ")", "%", "°", "µ"):
        text = text.replace(ch, "")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _resolve_columns(columns: list[str]) -> dict[str, str]:
    """Map canonical field names to the actual column labels present in the file."""
    lookup = {_normalize(c): c for c in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                break
    return resolved


def _single_value(frame: pd.DataFrame, column: str | None, label: str) -> object | None:
    """Read a run-level value that should be identical on every chip row."""
    if column is None or column not in frame:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    unique = values.astype(str).str.strip().unique()
    if len(unique) > 1:
        raise MetadataFileError(
            f"{label} differs between chip rows in the same run ({', '.join(unique[:4])}"
            f"{'...' if len(unique) > 4 else ''}). A single spotting run must carry one "
            f"{label.lower()}."
        )
    return values.iloc[0]


def _to_date(value: object, label: str) -> dt.date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise MetadataFileError(f"Could not read {label} from value {value!r}.")
    return parsed.date()


def _extract_lots(
    frame: pd.DataFrame, antibodies: list[str], run_date: dt.date
) -> pd.DataFrame:
    """Pull `<antibody>_lot` / `<antibody>_lot_received_date` column pairs.

    `lot_age_days` is derived here: the age of the stock on the day it was spotted,
    which is the x-axis of the shelf-degradation view in Part 2.
    """
    lookup = {_normalize(c): c for c in frame.columns}
    rows = []
    for antibody in antibodies:
        base = _normalize(antibody)
        lot_col = next((lookup[base + s] for s in LOT_SUFFIXES if base + s in lookup), None)
        date_col = next(
            (lookup[base + s] for s in LOT_DATE_SUFFIXES if base + s in lookup), None
        )
        if lot_col is None and date_col is None:
            continue

        lot = _single_value(frame, lot_col, f"Antibody lot for {antibody}")
        received = _to_date(
            _single_value(frame, date_col, f"Lot received date for {antibody}"),
            f"the lot received date for {antibody}",
        )
        age = (run_date - received).days if received else None
        if age is not None and age < 0:
            raise MetadataFileError(
                f"The lot received date for {antibody} ({received}) is after the run "
                f"date ({run_date})."
            )
        rows.append(
            {
                "antibody": antibody,
                "ab_lot": str(lot).strip() if lot is not None else None,
                "lot_received_date": received,
                "lot_age_days": age,
            }
        )
    return pd.DataFrame(rows, columns=["antibody", "ab_lot", "lot_received_date", "lot_age_days"])


def parse_benchling_export(
    source: str | Path | IO[bytes] | IO[str],
    antibodies: list[str] | None = None,
    source_name: str | None = None,
) -> RunMetadata:
    """Read a run metadata export into a `RunMetadata`.

    Args:
        source: path or file-like object holding the exported table (CSV).
        antibodies: panel antibody names, used to find their lot columns.
        source_name: original filename, recorded for provenance when `source` is a
            file-like object.

    Raises:
        MetadataFileError: when the run id, run date, or chip column is missing, when a
            run-level value is inconsistent across chip rows, or when chips repeat.
    """
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # pragma: no cover - pandas surfaces many parse errors
        raise MetadataFileError(f"Could not read the run metadata file: {exc}") from exc

    if raw.empty:
        raise MetadataFileError("The run metadata file has no data rows.")

    cols = _resolve_columns(list(raw.columns))

    for required, human in (("spotting_run_id", "spotting run ID"), ("chip_id", "chip ID")):
        if required not in cols:
            raise MetadataFileError(
                f"The run metadata file needs a {human} column (e.g. '{required}'). "
                f"Found columns: {', '.join(map(str, raw.columns))}"
            )

    run_id = _single_value(raw, cols["spotting_run_id"], "Spotting run ID")
    if run_id is None:
        raise MetadataFileError("The spotting run ID column is empty.")
    run_id = str(run_id).strip()

    if "run_date" not in cols:
        raise MetadataFileError(
            "The run metadata file needs a run date column (e.g. 'run_date'); the date "
            "orders every trend plot."
        )
    run_date = _to_date(_single_value(raw, cols["run_date"], "Run date"), "the run date")
    if run_date is None:
        raise MetadataFileError("The run date column is empty.")

    panel = _single_value(raw, cols.get("panel"), "Panel")
    operator = _single_value(raw, cols.get("operator"), "Operator")

    chips = pd.DataFrame({"chip_id": raw[cols["chip_id"]].astype(str).str.strip()})
    for field_name in ("wafer_id", "coating_batch"):
        if field_name in cols:
            series = raw[cols[field_name]]
            chips[field_name] = series.where(series.notna(), None)
            chips[field_name] = chips[field_name].map(
                lambda v: str(v).strip() if v is not None and not pd.isna(v) else None
            )
        else:
            chips[field_name] = None

    for env in ENV_FIELDS:
        chips[env] = (
            pd.to_numeric(raw[cols[env]], errors="coerce") if env in cols else pd.NA
        )

    if chips["chip_id"].duplicated().any():
        dupes = chips.loc[chips["chip_id"].duplicated(), "chip_id"].unique()[:5]
        raise MetadataFileError(
            f"The run metadata file lists the same chip more than once: {', '.join(dupes)}."
        )
    if (chips["chip_id"] == "").any() or chips["chip_id"].isin({"nan", "None"}).any():
        raise MetadataFileError("One or more chip ID values are blank.")

    def _run_level_env(column: str) -> float | None:
        values = pd.to_numeric(chips[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    lots = (
        _extract_lots(raw, antibodies, run_date)
        if antibodies
        else pd.DataFrame(columns=["antibody", "ab_lot", "lot_received_date", "lot_age_days"])
    )

    name = source_name
    if name is None and isinstance(source, (str, Path)):
        name = Path(source).name

    return RunMetadata(
        spotting_run_id=run_id,
        run_date=run_date,
        panel=str(panel).strip() if panel is not None else None,
        operator=str(operator).strip() if operator is not None else None,
        room_temp_c=_run_level_env("room_temp_c"),
        room_humidity_pct=_run_level_env("room_humidity_pct"),
        spotter_temp_c=_run_level_env("spotter_temp_c"),
        spotter_humidity_pct=_run_level_env("spotter_humidity_pct"),
        chips=chips.reset_index(drop=True),
        lots=lots,
        source_file=name,
    )

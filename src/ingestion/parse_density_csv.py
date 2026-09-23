"""Parse the spot-density CSV produced from a spotting run's chip readout.

Expected shape (one row per chip x replicate spot, one column per antibody):

    chip_id,replicate_num,Anti-IL-6,Anti-PCT,Anti-CRP,Anti-TNFa
    CHIP-01,1,4.21,3.88,5.02,3.31
    CHIP-01,2,4.05,3.92,4.87,3.44
    ...

Antibody names are read from the header, so a panel's real names (Anti-IL-6 and so on)
flow through the whole pipeline without any code change. The output is a long frame,
which is what the QC functions and the database both want.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

CHIP_COLUMN_ALIASES = {"chip_id", "chip", "chipid", "chip_no", "chip_number"}
REPLICATE_COLUMN_ALIASES = {
    "replicate_num",
    "replicate",
    "rep",
    "replicate_no",
    "replicate_number",
    "spot",
    "spot_num",
}


class DensityFileError(ValueError):
    """The density file could not be parsed into the expected shape."""


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def parse_density_csv(
    source: str | Path | IO[bytes] | IO[str],
    expected_antibodies: list[str] | None = None,
) -> pd.DataFrame:
    """Read a density CSV into a long frame.

    Args:
        source: path or file-like object holding the CSV.
        expected_antibodies: when given, the antibody columns are restricted and
            ordered to this list, and any missing or unexpected column is reported.

    Returns:
        Long frame with columns chip_id, antibody, replicate_num, density_ng_mm2.

    Raises:
        DensityFileError: on a missing identifier column, no antibody columns,
            non-numeric densities, or duplicated (chip, antibody, replicate) rows.
    """
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # pragma: no cover - pandas surfaces many parse errors
        raise DensityFileError(f"Could not read the density CSV: {exc}") from exc

    if raw.empty:
        raise DensityFileError("The density CSV has no data rows.")

    lookup = {_normalize(c): c for c in raw.columns}

    chip_col = next((lookup[a] for a in CHIP_COLUMN_ALIASES if a in lookup), None)
    if chip_col is None:
        raise DensityFileError(
            "The density CSV needs a chip identifier column (e.g. 'chip_id'). "
            f"Found columns: {', '.join(map(str, raw.columns))}"
        )

    replicate_col = next((lookup[a] for a in REPLICATE_COLUMN_ALIASES if a in lookup), None)

    antibody_cols = [c for c in raw.columns if c not in {chip_col, replicate_col}]
    if expected_antibodies:
        by_norm = {_normalize(c): c for c in antibody_cols}
        resolved, missing = [], []
        for antibody in expected_antibodies:
            match = by_norm.get(_normalize(antibody))
            (resolved.append(match) if match else missing.append(antibody))
        if missing:
            raise DensityFileError(
                "The density CSV is missing a column for these panel antibodies: "
                f"{', '.join(missing)}. Found: {', '.join(map(str, antibody_cols))}"
            )
        extra = [c for c in antibody_cols if c not in resolved]
        if extra:
            raise DensityFileError(
                "The density CSV has antibody column(s) that are not in the selected "
                f"panel: {', '.join(map(str, extra))}. Check the panel selection or "
                "add them to config/qc_config.yaml."
            )
        antibody_cols = resolved
        rename = dict(zip(resolved, expected_antibodies))
    else:
        rename = {}

    if not antibody_cols:
        raise DensityFileError("The density CSV has no antibody density columns.")

    frame = raw.copy()
    if replicate_col is None:
        # No replicate column: number the spots in file order within each chip, which
        # preserves the replicate count even when the ELN export omits the index.
        frame["__replicate_num"] = frame.groupby(chip_col).cumcount() + 1
        replicate_col = "__replicate_num"

    long = frame.melt(
        id_vars=[chip_col, replicate_col],
        value_vars=antibody_cols,
        var_name="antibody",
        value_name="density_ng_mm2",
    ).rename(columns={chip_col: "chip_id", replicate_col: "replicate_num"})

    if rename:
        long["antibody"] = long["antibody"].map(lambda c: rename.get(c, c))

    long["chip_id"] = long["chip_id"].astype(str).str.strip()
    long["replicate_num"] = pd.to_numeric(long["replicate_num"], errors="coerce").astype("Int64")
    long["density_ng_mm2"] = pd.to_numeric(long["density_ng_mm2"], errors="coerce")

    if long["replicate_num"].isna().any():
        raise DensityFileError("Some replicate numbers are missing or non-numeric.")

    bad = long[long["density_ng_mm2"].isna()]
    if not bad.empty:
        example = bad.iloc[0]
        raise DensityFileError(
            f"{len(bad)} density value(s) are blank or non-numeric, for example "
            f"chip {example.chip_id}, {example.antibody}, replicate "
            f"{example.replicate_num}. Blank spots must be removed or filled before "
            "ingestion so they cannot silently distort a chip's mean or CV."
        )
    if (long["density_ng_mm2"] < 0).any():
        raise DensityFileError("Density values cannot be negative.")

    duplicated = long.duplicated(subset=["chip_id", "antibody", "replicate_num"], keep=False)
    if duplicated.any():
        dupe = long[duplicated].iloc[0]
        raise DensityFileError(
            "The density CSV repeats the same chip/antibody/replicate, for example "
            f"chip {dupe.chip_id}, {dupe.antibody}, replicate {dupe.replicate_num}."
        )

    long["replicate_num"] = long["replicate_num"].astype(int)
    return long[["chip_id", "antibody", "replicate_num", "density_ng_mm2"]].reset_index(
        drop=True
    )

"""Read-side queries returning tidy dataframes.

Keeping every read in one module means the analysis and dashboard code works on plain
dataframes and never touches the ORM, which makes both of them straightforward to test
with hand-built frames.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select

from src.db.models import (
    AntibodyLot,
    AntibodyMeasurement,
    Chip,
    ReplicateMeasurement,
    Run,
    RunAntibodySummary,
)
from src.db.session import init_db, session_scope


def _read_sql(stmt, db_path: str | Path | None = None) -> pd.DataFrame:
    init_db(db_path)
    with session_scope(db_path) as session:
        rows = session.execute(stmt).mappings().all()
    return _normalize_nulls(pd.DataFrame([dict(row) for row in rows]))


def _normalize_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn missing text values back into None rather than NaN.

    A SQL NULL arrives as NaN, and NaN is truthy in Python, so a template writing
    `value or "-"` would print "nan" for a chip that simply has no failure reason.
    Numeric columns keep NaN, where it is the right missing marker.
    """
    for column in frame.columns:
        if frame[column].dtype == object or pd.api.types.is_string_dtype(frame[column]):
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def available_panels(db_path: str | Path | None = None) -> list[str]:
    """Panels that actually have ingested runs, newest activity first."""
    frame = _read_sql(select(Run.panel, Run.run_date), db_path)
    if frame.empty:
        return []
    order = frame.groupby("panel")["run_date"].max().sort_values(ascending=False)
    return order.index.tolist()


def load_runs(panel: str | None = None, db_path: str | Path | None = None) -> pd.DataFrame:
    """One row per run, ordered oldest first (the order every trend plot uses)."""
    stmt = select(
        Run.spotting_run_id,
        Run.panel,
        Run.run_date,
        Run.operator,
        Run.room_temp_c,
        Run.room_humidity_pct,
        Run.spotter_temp_c,
        Run.spotter_humidity_pct,
        Run.n_chips,
        Run.n_chips_failed,
        Run.failure_rate_pct,
        Run.density_min_ng_mm2,
        Run.cv_max_pct,
        Run.ingested_at,
        Run.source_density_file,
        Run.source_metadata_file,
        Run.notes,
    ).order_by(Run.run_date, Run.spotting_run_id)
    if panel:
        stmt = stmt.where(Run.panel == panel)
    frame = _read_sql(stmt, db_path)
    if not frame.empty:
        frame["run_date"] = pd.to_datetime(frame["run_date"])
        frame["run_index"] = range(len(frame))
    return frame


def load_run_antibody_summary(
    panel: str | None = None, db_path: str | Path | None = None
) -> pd.DataFrame:
    """Per run x antibody batch statistics, joined to run conditions and lot info.

    This is the frame the whole of Part 2 is built on: one row per antibody per run,
    carrying the metric (`pooled_mean_density`), the batch variability metric
    (`chip_to_chip_cv_pct`), the conditions, and which lot was in use.
    """
    stmt = (
        select(
            RunAntibodySummary.spotting_run_id,
            RunAntibodySummary.antibody,
            RunAntibodySummary.n_chips,
            RunAntibodySummary.n_chips_failed,
            RunAntibodySummary.n_failed_low_density,
            RunAntibodySummary.n_failed_high_cv,
            RunAntibodySummary.pooled_mean_density,
            RunAntibodySummary.median_density,
            RunAntibodySummary.chip_to_chip_sd,
            RunAntibodySummary.chip_to_chip_cv_pct,
            RunAntibodySummary.mean_within_chip_cv_pct,
            Run.panel,
            Run.run_date,
            Run.operator,
            Run.room_temp_c,
            Run.room_humidity_pct,
            Run.spotter_temp_c,
            Run.spotter_humidity_pct,
            Run.failure_rate_pct,
            AntibodyLot.ab_lot,
            AntibodyLot.lot_received_date,
            AntibodyLot.lot_age_days,
        )
        .join(Run, Run.spotting_run_id == RunAntibodySummary.spotting_run_id)
        .join(
            AntibodyLot,
            (AntibodyLot.spotting_run_id == RunAntibodySummary.spotting_run_id)
            & (AntibodyLot.antibody == RunAntibodySummary.antibody),
            isouter=True,
        )
        .order_by(Run.run_date, RunAntibodySummary.antibody)
    )
    if panel:
        stmt = stmt.where(Run.panel == panel)
    frame = _read_sql(stmt, db_path)
    if not frame.empty:
        frame["run_date"] = pd.to_datetime(frame["run_date"])
        run_order = {
            run_id: i
            for i, run_id in enumerate(
                frame.sort_values("run_date")["spotting_run_id"].unique()
            )
        }
        frame["run_index"] = frame["spotting_run_id"].map(run_order)
    return frame


def load_measurements(
    panel: str | None = None,
    spotting_run_id: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per chip x antibody verdicts joined to that chip's wafer, coating batch and
    conditions -- the frame behind the per-run report and the batch comparisons."""
    stmt = (
        select(
            AntibodyMeasurement.spotting_run_id,
            AntibodyMeasurement.chip_id,
            AntibodyMeasurement.antibody,
            AntibodyMeasurement.n_replicates,
            AntibodyMeasurement.mean_density,
            AntibodyMeasurement.sd_density,
            AntibodyMeasurement.cv_pct,
            AntibodyMeasurement.min_density,
            AntibodyMeasurement.max_density,
            AntibodyMeasurement.status,
            AntibodyMeasurement.fail_low_density,
            AntibodyMeasurement.fail_high_cv,
            AntibodyMeasurement.fail_reason,
            Chip.wafer_id,
            Chip.coating_batch,
            Chip.status.label("chip_status"),
            Chip.fail_reason_summary,
            Run.panel,
            Run.run_date,
            Run.operator,
            Run.room_temp_c,
            Run.room_humidity_pct,
            Run.spotter_temp_c,
            Run.spotter_humidity_pct,
        )
        .join(Chip, Chip.id == AntibodyMeasurement.chip_pk)
        .join(Run, Run.spotting_run_id == AntibodyMeasurement.spotting_run_id)
        .order_by(Run.run_date, AntibodyMeasurement.chip_id, AntibodyMeasurement.antibody)
    )
    if panel:
        stmt = stmt.where(Run.panel == panel)
    if spotting_run_id:
        stmt = stmt.where(AntibodyMeasurement.spotting_run_id == spotting_run_id)
    frame = _read_sql(stmt, db_path)
    if not frame.empty:
        frame["run_date"] = pd.to_datetime(frame["run_date"])
    return frame


def load_chips(
    panel: str | None = None,
    spotting_run_id: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """One row per chip with its QC verdict and provenance."""
    stmt = (
        select(
            Chip.spotting_run_id,
            Chip.chip_id,
            Chip.wafer_id,
            Chip.coating_batch,
            Chip.status,
            Chip.n_antibodies_failed,
            Chip.fail_reason_summary,
            Chip.room_temp_c,
            Chip.room_humidity_pct,
            Chip.spotter_temp_c,
            Chip.spotter_humidity_pct,
            Run.panel,
            Run.run_date,
        )
        .join(Run, Run.spotting_run_id == Chip.spotting_run_id)
        .order_by(Run.run_date, Chip.chip_id)
    )
    if panel:
        stmt = stmt.where(Run.panel == panel)
    if spotting_run_id:
        stmt = stmt.where(Chip.spotting_run_id == spotting_run_id)
    frame = _read_sql(stmt, db_path)
    if not frame.empty:
        frame["run_date"] = pd.to_datetime(frame["run_date"])
    return frame


def load_replicates(
    spotting_run_id: str,
    chip_id: str | None = None,
    antibody: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Raw per-spot densities, for tracing a flagged chip back to its spots."""
    stmt = select(
        ReplicateMeasurement.spotting_run_id,
        ReplicateMeasurement.chip_id,
        ReplicateMeasurement.antibody,
        ReplicateMeasurement.replicate_num,
        ReplicateMeasurement.density_ng_mm2,
    ).where(ReplicateMeasurement.spotting_run_id == spotting_run_id)
    if chip_id:
        stmt = stmt.where(ReplicateMeasurement.chip_id == chip_id)
    if antibody:
        stmt = stmt.where(ReplicateMeasurement.antibody == antibody)
    return _read_sql(
        stmt.order_by(
            ReplicateMeasurement.chip_id,
            ReplicateMeasurement.antibody,
            ReplicateMeasurement.replicate_num,
        ),
        db_path,
    )


def load_lots(panel: str | None = None, db_path: str | Path | None = None) -> pd.DataFrame:
    """Which antibody lot was used in which run, with its age on that day."""
    stmt = (
        select(
            AntibodyLot.spotting_run_id,
            AntibodyLot.antibody,
            AntibodyLot.ab_lot,
            AntibodyLot.lot_received_date,
            AntibodyLot.lot_age_days,
            Run.panel,
            Run.run_date,
        )
        .join(Run, Run.spotting_run_id == AntibodyLot.spotting_run_id)
        .order_by(Run.run_date, AntibodyLot.antibody)
    )
    if panel:
        stmt = stmt.where(Run.panel == panel)
    frame = _read_sql(stmt, db_path)
    if not frame.empty:
        frame["run_date"] = pd.to_datetime(frame["run_date"])
    return frame


def get_run(spotting_run_id: str, db_path: str | Path | None = None) -> dict | None:
    """A single run's header record, or None when it has not been ingested."""
    runs = load_runs(db_path=db_path)
    if runs.empty:
        return None
    match = runs[runs["spotting_run_id"] == spotting_run_id]
    if match.empty:
        return None
    record = match.iloc[0].to_dict()
    # A date reads better than a timestamp on a report header.
    if isinstance(record.get("run_date"), pd.Timestamp):
        record["run_date"] = record["run_date"].date()
    return record


def delete_run(spotting_run_id: str, db_path: str | Path | None = None) -> bool:
    """Remove a run and everything attached to it. Returns whether it existed."""
    init_db(db_path)
    with session_scope(db_path) as session:
        run = session.get(Run, spotting_run_id)
        if run is None:
            return False
        session.delete(run)
    return True

"""Ingest one spotting run: parse, validate, grade (Part 1), and persist.

This is the single write path into the QC database. Everything happens inside one
transaction, so a run that fails validation leaves no partial history behind -- which
matters because Part 2's trends are only trustworthy if every stored run is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd
from sqlalchemy import select

from src.config import QCConfig, get_config
from src.db.models import (
    FAIL,
    AntibodyLot,
    AntibodyMeasurement,
    Chip,
    ReplicateMeasurement,
    Run,
    RunAntibodySummary,
)
from src.db.session import init_db, session_scope
from src.ingestion.parse_benchling_export import RunMetadata, parse_benchling_export
from src.ingestion.parse_density_csv import parse_density_csv
from src.qc.part1_chip_qc import (
    compute_antibody_metrics,
    failure_rate_pct,
    summarize_chips,
    summarize_run_antibodies,
)

Source = str | Path | IO[bytes] | IO[str]


class IngestionError(ValueError):
    """Ingestion was refused. The message is written to be shown to the user."""


@dataclass
class IngestResult:
    """What a successful ingestion produced, for the dashboard to report back."""

    spotting_run_id: str
    panel: str
    run_date: object
    n_chips: int
    n_chips_failed: int
    failure_rate_pct: float
    antibody_summary: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    replaced_existing: bool = False


def run_exists(spotting_run_id: str, db_path: str | Path | None = None) -> bool:
    init_db(db_path)
    with session_scope(db_path) as session:
        return session.get(Run, spotting_run_id) is not None


def _resolve_panel(
    metadata: RunMetadata, panel: str | None, config: QCConfig, warnings: list[str]
) -> str:
    """Decide which panel the run belongs to, preferring the ELN export."""
    file_panel = metadata.panel or None
    chosen = file_panel or panel

    if chosen is None:
        raise IngestionError(
            "The run's panel could not be determined. Add a 'panel' column to the run "
            "metadata export, or select a panel when uploading."
        )
    if file_panel and panel and file_panel != panel:
        warnings.append(
            f"The metadata file names panel {file_panel!r} but {panel!r} was selected; "
            f"using {file_panel!r} from the file."
        )
    if not config.has_panel(chosen):
        raise IngestionError(
            f"Panel {chosen!r} is not defined in the QC config. Known panels: "
            f"{', '.join(config.panel_names) or 'none'}. Add it to "
            "config/qc_config.yaml with its antibody names and thresholds."
        )
    return chosen


def _validate_against_metadata(
    replicates: pd.DataFrame,
    metadata: RunMetadata,
    panel: str,
    config: QCConfig,
    warnings: list[str],
) -> pd.DataFrame:
    """Cross-check the two files and drop chips that cannot be graded.

    A measured chip with no ELN row is refused: without its coating batch and
    conditions, the chip's result could never be attributed to a cause, which is the
    point of the whole pipeline. The reverse (an ELN chip with no measurements) is only
    a warning, since a chip can legitimately be spotted but not read.
    """
    measured = set(replicates["chip_id"].unique())
    recorded = set(metadata.chip_ids)

    unrecorded = sorted(measured - recorded)
    if unrecorded:
        raise IngestionError(
            f"{len(unrecorded)} chip(s) in the density file have no row in the run "
            f"metadata export: {', '.join(unrecorded[:6])}"
            f"{'...' if len(unrecorded) > 6 else ''}. Every measured chip needs its "
            "wafer, coating batch, and conditions recorded."
        )

    unmeasured = sorted(recorded - measured)
    if unmeasured:
        warnings.append(
            f"{len(unmeasured)} chip(s) in the metadata export have no density data and "
            f"were skipped: {', '.join(unmeasured[:6])}{'...' if len(unmeasured) > 6 else ''}."
        )

    expected_reps = config.replicates_per_antibody(panel)
    counts = replicates.groupby(["chip_id", "antibody"]).size()
    off = counts[counts != expected_reps]
    if not off.empty:
        examples = ", ".join(
            f"{chip}/{antibody}: {n}" for (chip, antibody), n in off.head(4).items()
        )
        warnings.append(
            f"{len(off)} chip/antibody combination(s) do not have the expected "
            f"{expected_reps} replicate spots ({examples}). Their statistics were still "
            "computed from the spots present."
        )

    singletons = counts[counts < 2]
    if not singletons.empty:
        warnings.append(
            f"{len(singletons)} chip/antibody combination(s) have fewer than 2 spots, so "
            "no CV could be computed and they cannot fail the variability criterion."
        )

    expected_chips = config.chips_per_run(panel)
    if len(measured) != expected_chips:
        warnings.append(
            f"This run has {len(measured)} measured chips; the panel's configured run "
            f"size is {expected_chips}. Failure rate is a percentage, so it stays "
            "comparable across runs of different sizes."
        )

    return replicates[replicates["chip_id"].isin(recorded)].reset_index(drop=True)


def ingest_run(
    density_source: Source,
    metadata_source: Source,
    panel: str | None = None,
    config: QCConfig | None = None,
    db_path: str | Path | None = None,
    replace: bool = False,
    density_name: str | None = None,
    metadata_name: str | None = None,
    notes: str | None = None,
) -> IngestResult:
    """Parse, grade, and store one spotting run.

    Args:
        density_source: the per-spot density CSV (path or file-like).
        metadata_source: the ELN run metadata export (path or file-like).
        panel: fallback panel when the metadata export carries no panel column.
        config: loaded QC config; the shared one is used when omitted.
        db_path: target database; the configured one is used when omitted.
        replace: overwrite an already-ingested run of the same ID.
        density_name, metadata_name: original filenames, recorded for provenance.
        notes: free-text note stored with the run.

    Returns:
        IngestResult with the run's headline numbers and any non-fatal warnings.

    Raises:
        IngestionError: on any condition that makes the run unsafe to store.
    """
    config = config or get_config()
    warnings: list[str] = []

    metadata = parse_benchling_export(
        metadata_source, antibodies=None, source_name=metadata_name
    )
    resolved_panel = _resolve_panel(metadata, panel, config, warnings)
    antibodies = config.antibodies(resolved_panel)
    if not antibodies:
        raise IngestionError(
            f"Panel {resolved_panel!r} has no antibodies listed in config/qc_config.yaml."
        )

    # Re-parse now that the panel (and so the antibody names) is known, so the lot
    # columns for those antibodies can be picked up.
    if hasattr(metadata_source, "seek"):
        metadata_source.seek(0)  # type: ignore[union-attr]
    metadata = parse_benchling_export(
        metadata_source, antibodies=antibodies, source_name=metadata_name
    )

    replicates = parse_density_csv(density_source, expected_antibodies=antibodies)
    replicates = _validate_against_metadata(
        replicates, metadata, resolved_panel, config, warnings
    )
    if replicates.empty:
        raise IngestionError(
            "No chips could be graded: the density file and the metadata export have no "
            "chips in common."
        )

    if metadata.lots.empty:
        warnings.append(
            "No antibody lot columns were found in the metadata export. The run was "
            "stored, but lot-based diagnostics (step change at a lot switch, density vs. "
            "lot age) will be unavailable for it."
        )

    antibody_metrics = compute_antibody_metrics(replicates, resolved_panel, config)
    chip_summary = summarize_chips(antibody_metrics)
    antibody_summary = summarize_run_antibodies(antibody_metrics)
    run_failure_rate = failure_rate_pct(chip_summary)

    thresholds = config.resolve_thresholds(resolved_panel)

    init_db(db_path)
    with session_scope(db_path) as session:
        existing = session.get(Run, metadata.spotting_run_id)
        if existing is not None:
            if not replace:
                raise IngestionError(
                    f"Run {metadata.spotting_run_id!r} has already been ingested "
                    f"(on {existing.ingested_at:%Y-%m-%d}). Re-upload with 'replace' "
                    "enabled to overwrite it."
                )
            session.delete(existing)
            session.flush()
            warnings.append(
                f"Replaced the previously ingested copy of run {metadata.spotting_run_id}."
            )

        run = Run(
            spotting_run_id=metadata.spotting_run_id,
            panel=resolved_panel,
            run_date=metadata.run_date,
            operator=metadata.operator,
            room_temp_c=metadata.room_temp_c,
            room_humidity_pct=metadata.room_humidity_pct,
            spotter_temp_c=metadata.spotter_temp_c,
            spotter_humidity_pct=metadata.spotter_humidity_pct,
            n_chips=int(len(chip_summary)),
            n_chips_failed=int((chip_summary["status"] == FAIL).sum()),
            failure_rate_pct=run_failure_rate,
            density_min_ng_mm2=thresholds.density_min_ng_mm2,
            cv_max_pct=thresholds.cv_max_pct,
            source_density_file=density_name
            or (Path(density_source).name if isinstance(density_source, (str, Path)) else None),
            source_metadata_file=metadata.source_file,
            notes=notes,
        )
        session.add(run)

        chip_meta = metadata.chips.set_index("chip_id")
        chip_status = chip_summary.set_index("chip_id")
        chip_pks: dict[str, int] = {}

        for chip_id in chip_status.index:
            meta_row = chip_meta.loc[chip_id]
            status_row = chip_status.loc[chip_id]
            chip = Chip(
                spotting_run_id=run.spotting_run_id,
                chip_id=str(chip_id),
                wafer_id=meta_row.get("wafer_id"),
                coating_batch=meta_row.get("coating_batch"),
                room_temp_c=_as_float(meta_row.get("room_temp_c")),
                room_humidity_pct=_as_float(meta_row.get("room_humidity_pct")),
                spotter_temp_c=_as_float(meta_row.get("spotter_temp_c")),
                spotter_humidity_pct=_as_float(meta_row.get("spotter_humidity_pct")),
                status=str(status_row["status"]),
                n_antibodies_failed=int(status_row["n_antibodies_failed"]),
                fail_reason_summary=status_row["fail_reason_summary"],
            )
            session.add(chip)
            session.flush()
            chip_pks[str(chip_id)] = chip.id

        for row in metadata.lots.itertuples(index=False):
            session.add(
                AntibodyLot(
                    spotting_run_id=run.spotting_run_id,
                    antibody=row.antibody,
                    ab_lot=row.ab_lot,
                    lot_received_date=row.lot_received_date,
                    lot_age_days=(
                        int(row.lot_age_days) if pd.notna(row.lot_age_days) else None
                    ),
                )
            )

        for row in antibody_metrics.itertuples(index=False):
            session.add(
                AntibodyMeasurement(
                    spotting_run_id=run.spotting_run_id,
                    chip_pk=chip_pks[str(row.chip_id)],
                    chip_id=str(row.chip_id),
                    antibody=row.antibody,
                    n_replicates=int(row.n_replicates),
                    mean_density=float(row.mean_density),
                    sd_density=_as_float(row.sd_density),
                    cv_pct=_as_float(row.cv_pct),
                    min_density=_as_float(row.min_density),
                    max_density=_as_float(row.max_density),
                    status=str(row.status),
                    fail_low_density=bool(row.fail_low_density),
                    fail_high_cv=bool(row.fail_high_cv),
                    fail_reason=row.fail_reason,
                )
            )

        for row in antibody_summary.itertuples(index=False):
            session.add(
                RunAntibodySummary(
                    spotting_run_id=run.spotting_run_id,
                    antibody=row.antibody,
                    n_chips=int(row.n_chips),
                    n_chips_failed=int(row.n_chips_failed),
                    n_failed_low_density=int(row.n_failed_low_density),
                    n_failed_high_cv=int(row.n_failed_high_cv),
                    pooled_mean_density=_as_float(row.pooled_mean_density),
                    chip_to_chip_sd=_as_float(row.chip_to_chip_sd),
                    chip_to_chip_cv_pct=_as_float(row.chip_to_chip_cv_pct),
                    mean_within_chip_cv_pct=_as_float(row.mean_within_chip_cv_pct),
                    median_density=_as_float(row.median_density),
                )
            )

        for row in replicates.itertuples(index=False):
            session.add(
                ReplicateMeasurement(
                    spotting_run_id=run.spotting_run_id,
                    chip_id=str(row.chip_id),
                    antibody=row.antibody,
                    replicate_num=int(row.replicate_num),
                    density_ng_mm2=float(row.density_ng_mm2),
                )
            )

    return IngestResult(
        spotting_run_id=metadata.spotting_run_id,
        panel=resolved_panel,
        run_date=metadata.run_date,
        n_chips=int(len(chip_summary)),
        n_chips_failed=int((chip_summary["status"] == FAIL).sum()),
        failure_rate_pct=run_failure_rate,
        antibody_summary=antibody_summary,
        warnings=warnings,
        replaced_existing=any("Replaced the previously" in w for w in warnings),
    )


def _as_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def list_runs(
    panel: str | None = None, db_path: str | Path | None = None
) -> pd.DataFrame:
    """Every ingested run, newest last, optionally filtered to one panel."""
    init_db(db_path)
    with session_scope(db_path) as session:
        stmt = select(Run).order_by(Run.run_date, Run.spotting_run_id)
        if panel:
            stmt = stmt.where(Run.panel == panel)
        runs = session.scalars(stmt).all()
        return pd.DataFrame(
            [
                {
                    "spotting_run_id": r.spotting_run_id,
                    "panel": r.panel,
                    "run_date": r.run_date,
                    "operator": r.operator,
                    "n_chips": r.n_chips,
                    "n_chips_failed": r.n_chips_failed,
                    "failure_rate_pct": r.failure_rate_pct,
                    "room_temp_c": r.room_temp_c,
                    "room_humidity_pct": r.room_humidity_pct,
                    "spotter_temp_c": r.spotter_temp_c,
                    "spotter_humidity_pct": r.spotter_humidity_pct,
                    "ingested_at": r.ingested_at,
                }
                for r in runs
            ]
        )

"""SQLAlchemy models for the spotting QC database.

One SQLite file accumulates every ingested spotting run across every panel, which is
what makes the Part 2 trend analysis possible: each new run simply adds rows, and the
monitoring views always read the full history for the selected panel.

Grain of each table:
  runs                  -- one row per spotting run
  chips                 -- one row per chip within a run
  antibody_lots         -- one row per (run, antibody): which stock was used
  antibody_measurements -- one row per (run, chip, antibody): the QC verdict
  run_antibody_summary  -- one row per (run, antibody): batch-level stats for trending
  replicate_measurements-- one row per individual spot, kept for drill-down
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

PASS = "PASS"
FAIL = "FAIL"


class Base(DeclarativeBase):
    pass


class Run(Base):
    """A single spotting run. `spotting_run_id` comes from the Benchling export."""

    __tablename__ = "runs"

    spotting_run_id: Mapped[str] = mapped_column(String, primary_key=True)
    panel: Mapped[str] = mapped_column(String, index=True)
    run_date: Mapped[dt.date] = mapped_column(Date, index=True)
    operator: Mapped[str | None] = mapped_column(String)

    # Environmental conditions, averaged across the run's chips.
    room_temp_c: Mapped[float | None] = mapped_column(Float)
    room_humidity_pct: Mapped[float | None] = mapped_column(Float)
    spotter_temp_c: Mapped[float | None] = mapped_column(Float)
    spotter_humidity_pct: Mapped[float | None] = mapped_column(Float)

    n_chips: Mapped[int] = mapped_column(Integer, default=0)
    n_chips_failed: Mapped[int] = mapped_column(Integer, default=0)
    failure_rate_pct: Mapped[float] = mapped_column(Float, default=0.0)

    # Thresholds actually applied at ingestion time, recorded so a historical report
    # stays reproducible even if the config is later retuned.
    density_min_ng_mm2: Mapped[float | None] = mapped_column(Float)
    cv_max_pct: Mapped[float | None] = mapped_column(Float)

    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    source_density_file: Mapped[str | None] = mapped_column(String)
    source_metadata_file: Mapped[str | None] = mapped_column(String)
    notes: Mapped[str | None] = mapped_column(String)

    chips: Mapped[list["Chip"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    antibody_lots: Mapped[list["AntibodyLot"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    measurements: Mapped[list["AntibodyMeasurement"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    antibody_summaries: Mapped[list["RunAntibodySummary"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    replicates: Mapped[list["ReplicateMeasurement"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Run {self.spotting_run_id} panel={self.panel} date={self.run_date}>"


class Chip(Base):
    """One spotted chip. Chip numbering may repeat between runs, so the identity is
    the (run, chip) pair rather than chip_id alone."""

    __tablename__ = "chips"
    __table_args__ = (UniqueConstraint("spotting_run_id", "chip_id", name="uq_chip_per_run"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    spotting_run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.spotting_run_id", ondelete="CASCADE"), index=True
    )
    chip_id: Mapped[str] = mapped_column(String, index=True)

    wafer_id: Mapped[str | None] = mapped_column(String, index=True)
    coating_batch: Mapped[str | None] = mapped_column(String, index=True)

    # Per-chip environmental readings, when the ELN records them at chip level.
    room_temp_c: Mapped[float | None] = mapped_column(Float)
    room_humidity_pct: Mapped[float | None] = mapped_column(Float)
    spotter_temp_c: Mapped[float | None] = mapped_column(Float)
    spotter_humidity_pct: Mapped[float | None] = mapped_column(Float)

    status: Mapped[str] = mapped_column(String, index=True, default=PASS)
    n_antibodies_failed: Mapped[int] = mapped_column(Integer, default=0)
    fail_reason_summary: Mapped[str | None] = mapped_column(String)

    run: Mapped[Run] = relationship(back_populates="chips")
    measurements: Mapped[list["AntibodyMeasurement"]] = relationship(
        back_populates="chip", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Chip {self.spotting_run_id}/{self.chip_id} {self.status}>"


class AntibodyLot(Base):
    """Which antibody stock was used for an antibody in a run.

    This is what lets the dashboard separate 'the antibody is degrading' from 'this is
    simply a different lot': a density step exactly at a lot boundary implicates the
    lot, while a gradual slide inside one lot implicates the stock going bad.
    """

    __tablename__ = "antibody_lots"
    __table_args__ = (
        UniqueConstraint("spotting_run_id", "antibody", name="uq_lot_per_run_antibody"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    spotting_run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.spotting_run_id", ondelete="CASCADE"), index=True
    )
    antibody: Mapped[str] = mapped_column(String, index=True)
    ab_lot: Mapped[str | None] = mapped_column(String, index=True)
    lot_received_date: Mapped[dt.date | None] = mapped_column(Date)
    # Days between the lot being received/opened and this run: the x-axis of the
    # shelf-degradation view.
    lot_age_days: Mapped[int | None] = mapped_column(Integer)

    run: Mapped[Run] = relationship(back_populates="antibody_lots")


class AntibodyMeasurement(Base):
    """The Part 1 verdict for one antibody on one chip, over its replicate spots."""

    __tablename__ = "antibody_measurements"
    __table_args__ = (
        UniqueConstraint(
            "spotting_run_id", "chip_id", "antibody", name="uq_measurement_grain"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    spotting_run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.spotting_run_id", ondelete="CASCADE"), index=True
    )
    chip_pk: Mapped[int] = mapped_column(ForeignKey("chips.id", ondelete="CASCADE"), index=True)
    chip_id: Mapped[str] = mapped_column(String, index=True)
    antibody: Mapped[str] = mapped_column(String, index=True)

    n_replicates: Mapped[int] = mapped_column(Integer)
    mean_density: Mapped[float] = mapped_column(Float)
    sd_density: Mapped[float | None] = mapped_column(Float)
    cv_pct: Mapped[float | None] = mapped_column(Float)
    min_density: Mapped[float | None] = mapped_column(Float)
    max_density: Mapped[float | None] = mapped_column(Float)

    status: Mapped[str] = mapped_column(String, index=True)
    fail_low_density: Mapped[bool] = mapped_column(Boolean, default=False)
    fail_high_cv: Mapped[bool] = mapped_column(Boolean, default=False)
    fail_reason: Mapped[str | None] = mapped_column(String)

    run: Mapped[Run] = relationship(back_populates="measurements")
    chip: Mapped[Chip] = relationship(back_populates="measurements")


class RunAntibodySummary(Base):
    """Batch-level statistics for one antibody in one run.

    `chip_to_chip_cv_pct` is the Part 2 variability metric: the CV across the per-chip
    mean densities, which is a different quantity from the within-chip replicate CV
    used for pass/fail in Part 1.
    """

    __tablename__ = "run_antibody_summary"
    __table_args__ = (
        UniqueConstraint("spotting_run_id", "antibody", name="uq_summary_per_run_antibody"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    spotting_run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.spotting_run_id", ondelete="CASCADE"), index=True
    )
    antibody: Mapped[str] = mapped_column(String, index=True)

    n_chips: Mapped[int] = mapped_column(Integer)
    n_chips_failed: Mapped[int] = mapped_column(Integer, default=0)
    n_failed_low_density: Mapped[int] = mapped_column(Integer, default=0)
    n_failed_high_cv: Mapped[int] = mapped_column(Integer, default=0)

    pooled_mean_density: Mapped[float | None] = mapped_column(Float)
    chip_to_chip_sd: Mapped[float | None] = mapped_column(Float)
    chip_to_chip_cv_pct: Mapped[float | None] = mapped_column(Float)
    mean_within_chip_cv_pct: Mapped[float | None] = mapped_column(Float)
    median_density: Mapped[float | None] = mapped_column(Float)

    run: Mapped[Run] = relationship(back_populates="antibody_summaries")


class ReplicateMeasurement(Base):
    """A single spot's density, retained so any flagged chip can be traced back to
    the raw replicate values."""

    __tablename__ = "replicate_measurements"
    __table_args__ = (
        UniqueConstraint(
            "spotting_run_id",
            "chip_id",
            "antibody",
            "replicate_num",
            name="uq_replicate_grain",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    spotting_run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.spotting_run_id", ondelete="CASCADE"), index=True
    )
    chip_id: Mapped[str] = mapped_column(String, index=True)
    antibody: Mapped[str] = mapped_column(String, index=True)
    replicate_num: Mapped[int] = mapped_column(Integer)
    density_ng_mm2: Mapped[float] = mapped_column(Float)

    run: Mapped[Run] = relationship(back_populates="replicates")

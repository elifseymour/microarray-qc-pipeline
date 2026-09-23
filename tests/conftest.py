"""Shared fixtures.

The heavier fixtures build a real database from the synthetic generator once per test
session, so the Part 2 tests assert against the same ground truth the dashboard shows.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.config import QCConfig
from src.data_generation.generate_synthetic_data import (
    build_sepsis_panel,
    write_dataset,
)
from src.db.session import reset_engine_cache
from src.ingestion.bulk_load import discover_run_files
from src.ingestion.ingest_run import ingest_run

TEST_PANEL = "TestPanel"


@pytest.fixture
def config() -> QCConfig:
    """A minimal two-antibody panel with a per-antibody threshold override."""
    return QCConfig(
        {
            "defaults": {
                "density_min_ng_mm2": 3.0,
                "cv_max_pct": 20.0,
                "replicates_per_antibody": 6,
                "chips_per_run": 4,
            },
            "panels": {
                TEST_PANEL: {
                    "antibodies": ["Ab-A", "Ab-B"],
                    "antibody_overrides": {"Ab-B": {"density_min_ng_mm2": 2.0}},
                }
            },
            "monitoring": {"min_runs_for_trend": 4, "trend_window_runs": 6},
        }
    )


def replicate_frame(values: dict[tuple[str, str], list[float]]) -> pd.DataFrame:
    """Build a long replicate frame from {(chip, antibody): [densities]}."""
    rows = []
    for (chip_id, antibody), densities in values.items():
        for i, density in enumerate(densities, start=1):
            rows.append(
                {
                    "chip_id": chip_id,
                    "antibody": antibody,
                    "replicate_num": i,
                    "density_ng_mm2": density,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def make_replicates():
    return replicate_frame


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory) -> dict:
    """Generate the sepsis dataset, ingest it, and hand back the database path."""
    raw_dir = tmp_path_factory.mktemp("raw")
    db_path = tmp_path_factory.mktemp("db") / "qc.db"
    write_dataset(raw_dir, panels=[build_sepsis_panel()])

    reset_engine_cache()
    run_ids = []
    for run_id, density, metadata in discover_run_files(raw_dir):
        ingest_run(density, metadata, db_path=db_path)
        run_ids.append(run_id)

    return {"db_path": db_path, "raw_dir": raw_dir, "run_ids": run_ids}


@pytest.fixture
def ground_truth_dates() -> list[dt.date]:
    return [dt.date(2026, 7, 6) + dt.timedelta(weeks=i) for i in range(10)]

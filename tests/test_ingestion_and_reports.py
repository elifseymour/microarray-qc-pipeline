"""Ingestion guards and report rendering.

The ingestion tests matter more than they look: Part 2's trends are only meaningful if
every stored run is complete and attributable, so the checks that refuse a run are what
keep the monitoring view honest.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.config import load_config
from src.db import queries as q
from src.db.session import reset_engine_cache
from src.ingestion.ingest_run import IngestionError, ingest_run
from src.ingestion.parse_benchling_export import MetadataFileError, parse_benchling_export
from src.ingestion.parse_density_csv import DensityFileError, parse_density_csv
from src.reports.report_generator import build_run_report_html

PANEL = "Sepsis-4plex"


@pytest.fixture
def run_files(synthetic_db):
    raw = synthetic_db["raw_dir"]
    return (
        raw / "SR-2026-W28_densities.csv",
        raw / "SR-2026-W28_run_metadata.csv",
    )


@pytest.fixture
def fresh_db(tmp_path):
    reset_engine_cache()
    return tmp_path / "fresh.db"


# -- density file ------------------------------------------------------------------


def test_density_csv_parses_to_long_form(run_files):
    density, _ = run_files
    frame = parse_density_csv(density)
    assert set(frame.columns) == {"chip_id", "antibody", "replicate_num", "density_ng_mm2"}
    assert len(frame) == 20 * 4 * 6
    assert frame["replicate_num"].between(1, 6).all()


def test_density_csv_rejects_blank_values():
    csv = io.StringIO("chip_id,replicate_num,Ab-A\nC1,1,4.0\nC1,2,\n")
    with pytest.raises(DensityFileError, match="blank or non-numeric"):
        parse_density_csv(csv)


def test_density_csv_rejects_negative_values():
    csv = io.StringIO("chip_id,replicate_num,Ab-A\nC1,1,4.0\nC1,2,-1.0\n")
    with pytest.raises(DensityFileError, match="negative"):
        parse_density_csv(csv)


def test_density_csv_rejects_duplicate_replicates():
    csv = io.StringIO("chip_id,replicate_num,Ab-A\nC1,1,4.0\nC1,1,4.2\n")
    with pytest.raises(DensityFileError, match="repeats the same chip"):
        parse_density_csv(csv)


def test_density_csv_reports_antibodies_missing_for_the_panel(run_files):
    density, _ = run_files
    with pytest.raises(DensityFileError, match="missing a column"):
        parse_density_csv(density, expected_antibodies=["Anti-IL-6", "Anti-Nonexistent"])


def test_density_csv_header_matching_ignores_case_and_separators():
    csv = io.StringIO("Chip ID,Replicate,anti_il_6\nC1,1,4.0\nC1,2,4.2\n")
    frame = parse_density_csv(csv, expected_antibodies=["Anti-IL-6"])
    assert frame["antibody"].unique().tolist() == ["Anti-IL-6"]
    assert frame["chip_id"].tolist() == ["C1", "C1"]


# -- metadata file -----------------------------------------------------------------


def test_metadata_export_reads_run_facts_and_lots(run_files):
    _, metadata = run_files
    parsed = parse_benchling_export(
        metadata, antibodies=["Anti-IL-6", "Anti-PCT", "Anti-TNFa", "Isotype-IgG1"]
    )
    assert parsed.spotting_run_id == "SR-2026-W28"
    assert parsed.panel == PANEL
    assert len(parsed.chips) == 20
    assert parsed.coating_batches == ["CB-2601"]
    lots = parsed.lots.set_index("antibody")
    assert lots.loc["Anti-PCT", "ab_lot"] == "L-PCT-2604"
    # Stock age is derived from the run date, and is what the shelf-life view plots.
    assert lots.loc["Anti-PCT", "lot_age_days"] == 91


def test_metadata_export_rejects_inconsistent_run_level_values(run_files):
    _, metadata = run_files
    frame = pd.read_csv(metadata)
    frame.loc[0, "spotting_run_id"] = "SR-OTHER"
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    with pytest.raises(MetadataFileError, match="differs between chip rows"):
        parse_benchling_export(buffer)


def test_metadata_export_requires_a_run_date(run_files):
    _, metadata = run_files
    frame = pd.read_csv(metadata).drop(columns=["run_date"])
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    with pytest.raises(MetadataFileError, match="run date column"):
        parse_benchling_export(buffer)


# -- ingestion ---------------------------------------------------------------------


def test_ingest_stores_every_level(run_files, fresh_db):
    density, metadata = run_files
    result = ingest_run(density, metadata, db_path=fresh_db)

    assert result.spotting_run_id == "SR-2026-W28"
    assert result.n_chips == 20
    assert q.load_chips(db_path=fresh_db).shape[0] == 20
    assert q.load_measurements(db_path=fresh_db).shape[0] == 20 * 4
    assert q.load_replicates("SR-2026-W28", db_path=fresh_db).shape[0] == 20 * 4 * 6
    assert q.load_lots(db_path=fresh_db).shape[0] == 4


def test_ingest_refuses_a_duplicate_run(run_files, fresh_db):
    density, metadata = run_files
    ingest_run(density, metadata, db_path=fresh_db)
    with pytest.raises(IngestionError, match="already been ingested"):
        ingest_run(density, metadata, db_path=fresh_db)


def test_ingest_replaces_a_run_on_request(run_files, fresh_db):
    density, metadata = run_files
    ingest_run(density, metadata, db_path=fresh_db)
    result = ingest_run(density, metadata, db_path=fresh_db, replace=True)
    assert result.replaced_existing
    # Replacing must not duplicate the run's rows.
    assert q.load_chips(db_path=fresh_db).shape[0] == 20


def test_ingest_refuses_a_chip_with_no_eln_row(run_files, fresh_db):
    """A measured chip without its coating batch and conditions could never be
    attributed to a cause, so storing it would weaken the monitoring view."""
    density, metadata = run_files
    frame = pd.read_csv(metadata)
    reduced = io.StringIO()
    frame[frame["chip_id"] != "CHIP-05"].to_csv(reduced, index=False)
    reduced.seek(0)

    with pytest.raises(IngestionError, match="no row in the run metadata"):
        ingest_run(density, reduced, db_path=fresh_db)
    assert q.load_runs(db_path=fresh_db).empty


def test_failed_ingestion_leaves_no_partial_run(run_files, fresh_db):
    density, metadata = run_files
    bad_density = io.StringIO("chip_id,replicate_num,Anti-IL-6\nCHIP-01,1,4.0\n")
    with pytest.raises((IngestionError, DensityFileError)):
        ingest_run(bad_density, metadata, db_path=fresh_db)
    assert q.load_runs(db_path=fresh_db).empty
    assert q.load_chips(db_path=fresh_db).empty


def test_ingest_warns_when_a_chip_has_no_density_data(run_files, fresh_db):
    """The reverse case is legitimate: a chip can be spotted but not read."""
    density, metadata = run_files
    frame = pd.read_csv(density)
    reduced = io.StringIO()
    frame[frame["chip_id"] != "CHIP-05"].to_csv(reduced, index=False)
    reduced.seek(0)

    result = ingest_run(reduced, metadata, db_path=fresh_db)
    assert result.n_chips == 19
    assert any("no density data" in w for w in result.warnings)


def test_ingest_refuses_an_unconfigured_panel(run_files, fresh_db, tmp_path):
    density, metadata = run_files
    config_path = tmp_path / "other.yaml"
    config_path.write_text(
        "defaults: {density_min_ng_mm2: 3.0, cv_max_pct: 20.0}\n"
        "panels:\n  Other-2plex:\n    antibodies: [Ab-A, Ab-B]\n"
    )
    with pytest.raises(IngestionError, match="not defined in the QC config"):
        ingest_run(density, metadata, config=load_config(config_path), db_path=fresh_db)


def test_thresholds_recorded_on_the_run_stay_reproducible(run_files, fresh_db):
    """A report rebuilt later must show the limits that were actually applied."""
    density, metadata = run_files
    ingest_run(density, metadata, db_path=fresh_db)
    run = q.get_run("SR-2026-W28", db_path=fresh_db)
    assert run["density_min_ng_mm2"] == 3.0
    assert run["cv_max_pct"] == 20.0


# -- report ------------------------------------------------------------------------


def test_report_renders_with_the_run_s_facts(synthetic_db):
    db = synthetic_db["db_path"]
    html = build_run_report_html("SR-2026-W37", db_path=db, embed_plotly=False)

    assert "SR-2026-W37" in html
    assert PANEL in html
    assert "Anti-PCT" in html
    assert "L-PCT-2604" in html  # the lot in use, for traceability
    assert "CHIP-01" in html
    # A passing chip has no failure reason, and that must not surface as "nan".
    assert ">nan<" not in html
    assert "nan " not in html.replace("\n", " ")


def test_report_refuses_an_unknown_run(synthetic_db):
    with pytest.raises(ValueError, match="has not been ingested"):
        build_run_report_html("SR-DOES-NOT-EXIST", db_path=synthetic_db["db_path"])


def test_report_states_the_limits_it_applied(synthetic_db):
    html = build_run_report_html(
        "SR-2026-W28", db_path=synthetic_db["db_path"], embed_plotly=False
    )
    assert "3 ng/mm²" in html
    assert "20%" in html

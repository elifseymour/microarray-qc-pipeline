"""Part 2 monitoring, checked against the synthetic dataset's injected ground truth.

The dataset builds in one antibody that degrades inside a single lot and another that
steps down at a lot change. Telling those two apart is the point of the analysis, so
these tests assert the attribution, not just that something was flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MonitoringRules, get_config
from src.db import queries as q
from src.qc.part2_trend_analysis import (
    DEGRADING,
    LOT_CHANGE,
    STABLE,
    analyze_panel,
    fit_trend,
    group_effects,
)

PANEL = "Sepsis-4plex"


@pytest.fixture(scope="module")
def analysis(synthetic_db):
    db = synthetic_db["db_path"]
    config = get_config()
    return analyze_panel(
        q.load_runs(PANEL, db_path=db),
        q.load_run_antibody_summary(PANEL, db_path=db),
        q.load_measurements(PANEL, db_path=db),
        config.monitoring,
        PANEL,
        config.reference_antibody(PANEL),
    )


# -- the fitting primitive ----------------------------------------------------------


def test_fit_trend_recovers_a_known_slope():
    fit = fit_trend(range(10), [10 - 0.5 * i for i in range(10)])
    assert fit.slope == pytest.approx(-0.5)
    assert fit.pct_change == pytest.approx(-45.0)  # 10 -> 5.5 over the fitted range
    assert fit.p < 1e-9


def test_fit_trend_needs_three_points():
    assert fit_trend([0, 1], [1.0, 2.0]).slope is None


def test_fit_trend_ignores_missing_values():
    fit = fit_trend([0, 1, 2, 3], [1.0, np.nan, 3.0, 4.0])
    assert fit.n == 3


# -- degradation vs. lot change -----------------------------------------------------


def test_degrading_antibody_is_attributed_to_the_stock(analysis):
    """Anti-PCT declines inside one lot, so it must read as degradation."""
    row = analysis.density_trends.set_index("antibody").loc["Anti-PCT"]
    assert row["attribution"] == DEGRADING
    assert row["n_lots"] == 1
    assert row["within_lot_pct_change"] < -25.0
    assert row["within_lot_p"] < 0.01


def test_lot_step_is_not_mistaken_for_degradation(analysis):
    """Anti-TNFa steps down at a fresh lot, so it must read as a lot difference."""
    row = analysis.density_trends.set_index("antibody").loc["Anti-TNFa"]
    assert row["attribution"] == LOT_CHANGE
    assert row["n_lots"] == 2
    assert row["lot_step_pct"] < -15.0
    # No meaningful slide inside the current lot: that is what rules out degradation.
    assert row["within_lot_p"] > 0.2


def test_stable_antibodies_are_not_flagged(analysis):
    trends = analysis.density_trends.set_index("antibody")
    assert trends.loc["Anti-IL-6", "attribution"] == STABLE
    assert trends.loc["Isotype-IgG1", "attribution"] == STABLE

    flagged = {flag.subject for flag in analysis.flags}
    assert "Anti-IL-6" not in flagged
    assert "Isotype-IgG1" not in flagged


def test_degradation_flag_cites_the_stable_negative_control(analysis):
    flag = next(f for f in analysis.flags if f.category == "antibody-degradation")
    assert flag.subject == "Anti-PCT"
    assert flag.level == "alert"
    assert flag.evidence["reference_stable"] is True
    assert "Isotype-IgG1" in flag.detail


def test_lot_age_tracks_the_degrading_antibody_only(analysis):
    lot_age = analysis.lot_age.set_index("antibody")
    assert lot_age.loc["Anti-PCT", "r"] < -0.7
    assert abs(lot_age.loc["Anti-IL-6", "r"]) < 0.5


# -- drivers ------------------------------------------------------------------------


def test_defective_coating_batch_is_identified_via_the_control(analysis):
    """CB-2602 is the defective batch, used for runs 4 and 5.

    Every chip in a run shares one coating batch, so the batch cannot be compared
    against another batch inside the same run. What identifies it is that the whole
    panel spotted low on those runs, negative control included, while temperature and
    humidity were normal.
    """
    evidence = analysis.surface_evidence.set_index("coating_batch")
    bad = evidence.loc["CB-2602"]
    # The control is the clean signal: no antibody trend can move it. The panel-wide
    # figure is diluted because Anti-PCT was still comparatively high on these early runs.
    assert bad["control_deficit_pct"] < -8.0
    assert bad["density_deficit_pct"] < 0.0
    assert bad["max_room_temp_c"] < 24.0  # conditions cannot explain it

    flag = next(f for f in analysis.flags if f.category == "coating-batch")
    assert flag.subject == "CB-2602"
    assert flag.level == "alert"
    assert "negative control" in flag.detail


def test_dry_runs_are_not_blamed_on_their_coating_batch(analysis):
    """CB-2603 covers the dry winter runs but is not itself defective.

    Those runs show inflated CV, not lower density, and the control sits at normal, so
    the coating batch must not be implicated.
    """
    evidence = analysis.surface_evidence.set_index("coating_batch")
    hot_batch = evidence.loc["CB-2603"]
    assert hot_batch["density_deficit_pct"] > -8.0
    assert hot_batch["control_deficit_pct"] > -5.0
    assert "CB-2603" not in {f.subject for f in analysis.flags}


def test_coating_batch_is_not_separable_within_a_run(analysis):
    """One coating batch per run means no within-run comparison is available."""
    batches = analysis.coating_batches
    assert not batches["separable_from_run"].any()


def test_control_density_is_watched_in_absolute_terms_too(analysis):
    """The control must be trended on its own, not only used as a denominator.

    Otherwise a process-wide decline that moved every antibody together would leave
    every ratio flat and go unnoticed.
    """
    trends = analysis.density_trends.set_index("antibody")
    assert "Isotype-IgG1" in trends.index
    assert pd.notna(trends.loc["Isotype-IgG1", "within_lot_pct_change"])
    # The control has no ratio-to-itself column value.
    assert pd.isna(trends.loc["Isotype-IgG1", "ratio_to_control_pct_change"])


def test_operator_has_no_effect_and_is_separable(analysis):
    """Operators alternate between runs, so the comparison is meaningful and null."""
    operators = analysis.operators
    assert operators["relative_density_pct_diff"].abs().max() < 10.0


def test_wafer_lots_are_marked_as_not_separable_from_the_run(analysis):
    """One wafer per run means a wafer column cannot be read as a wafer effect.

    Anti-PCT declines over the series, so early wafers necessarily look better than late
    ones. The analysis must say the comparison is confounded rather than imply a wafer
    difference, and no wafer may be flagged.
    """
    wafers = analysis.wafers
    assert not wafers["separable_from_run"].any()
    assert (wafers["n_within_run_comparisons"] == 0).all()
    assert not any(f.category == "wafer" for f in analysis.flags)


def test_degradation_is_confirmed_against_the_control_ratio(analysis):
    """The ratio is supporting evidence: the absolute decline is what triggers the flag."""
    row = analysis.density_trends.set_index("antibody").loc["Anti-PCT"]
    assert row["within_lot_pct_change"] < -25.0  # absolute, drives the attribution
    assert row["ratio_to_control_pct_change"] < -25.0  # survives cancelling run effects


def test_low_room_humidity_is_identified_as_the_driver(analysis):
    """The dry winter runs are the injected cause of poor spot uniformity."""
    notable = analysis.correlations[analysis.correlations["notable"]]
    humidity_cv = notable[
        (notable["driver"] == "room_humidity_pct")
        & (notable["metric"] == "mean_within_chip_cv_pct")
    ]
    assert not humidity_cv.empty
    # Drier room, higher CV: the correlation must be negative.
    assert (humidity_cv["r"] < -0.6).all()


def test_room_temperature_is_not_flagged(analysis):
    """Temperature is controlled all year and carries no injected effect.

    It is only separable from humidity because the two were deliberately not moved
    together. If they had been, both would correlate and neither could be named.
    """
    temperature = analysis.correlations[
        analysis.correlations["driver"].isin(["room_temp_c", "spotter_temp_c"])
    ]
    assert not temperature["notable"].any()
    assert "Room temperature" not in {f.subject for f in analysis.flags}


def test_environment_drivers_are_not_confounded_with_time(analysis):
    """The dry spell sits mid-series, so humidity must not track the calendar."""
    humidity = analysis.correlations[analysis.correlations["driver"] == "room_humidity_pct"]
    assert humidity["driver_vs_time_r"].abs().max() < 0.6


def test_outlier_runs_include_the_excursion_runs(analysis):
    flagged_runs = set(analysis.outliers["spotting_run_id"])
    # Runs 7 and 8 of the series are the injected temperature/humidity excursion.
    assert {"SR-2026-W34", "SR-2026-W35"} & flagged_runs


def test_failure_rate_rise_is_reported_with_a_significance_test(analysis):
    trend = analysis.failure_trend
    assert trend["recent_mean"] > trend["baseline_mean"]
    assert trend["proportion_p"] is not None
    flag = next(f for f in analysis.flags if f.category == "failure-rate")
    assert "Fisher" in flag.detail


# -- normalization behaviour --------------------------------------------------------


def test_group_effects_normalizes_away_run_level_differences():
    """A group used only in a low-signal run must not look defective for that reason."""
    frame = pd.DataFrame(
        [
            # Run 1 spotted low across the board; only group X was used.
            *[
                {
                    "spotting_run_id": "R1",
                    "chip_id": f"C{i}",
                    "antibody": "Ab",
                    "mean_density": 3.0,
                    "cv_pct": 5.0,
                    "chip_status": "PASS",
                    "coating_batch": "X",
                }
                for i in range(5)
            ],
            # Runs 2 and 3 spotted normally with group Y.
            *[
                {
                    "spotting_run_id": run,
                    "chip_id": f"C{i}",
                    "antibody": "Ab",
                    "mean_density": 5.0,
                    "cv_pct": 5.0,
                    "chip_status": "PASS",
                    "coating_batch": "Y",
                }
                for run in ("R2", "R3")
                for i in range(5)
            ],
        ]
    )
    effects = group_effects(frame, "coating_batch").set_index("coating_batch")
    # X does look low against the series norm, but it has no within-run comparison at
    # all, which is exactly the caveat the flag text has to carry.
    assert effects.loc["X", "relative_density_pct_diff"] < -30.0
    assert effects.loc["X", "n_within_run_comparisons"] == 0


def test_trend_flags_are_suppressed_without_enough_runs(synthetic_db):
    db = synthetic_db["db_path"]
    summary = q.load_run_antibody_summary(PANEL, db_path=db)
    runs = q.load_runs(PANEL, db_path=db)
    first_two = runs["spotting_run_id"].head(2)

    analysis = analyze_panel(
        runs[runs["spotting_run_id"].isin(first_two)],
        summary[summary["spotting_run_id"].isin(first_two)],
        q.load_measurements(PANEL, db_path=db),
        MonitoringRules(min_runs_for_trend=4),
        PANEL,
    )
    assert analysis.density_trends["attribution"].eq("insufficient data").all()
    assert not any(f.category == "antibody-degradation" for f in analysis.flags)

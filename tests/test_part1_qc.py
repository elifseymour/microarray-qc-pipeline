"""Part 1 pass/fail rules, including the behaviour exactly at the thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.qc.part1_chip_qc import (
    HIGH_CV_LABEL,
    LOW_DENSITY_LABEL,
    coefficient_of_variation,
    compute_antibody_metrics,
    failure_rate_pct,
    summarize_chips,
    summarize_run_antibodies,
)
from tests.conftest import TEST_PANEL


def test_cv_uses_sample_standard_deviation():
    # Sample SD of [9, 11] is 1.4142; population SD would be 1.0 and give 10%.
    assert coefficient_of_variation([9.0, 11.0]) == pytest.approx(14.142, abs=1e-3)


def test_cv_undefined_for_single_or_nonpositive():
    assert coefficient_of_variation([4.0]) is None
    assert coefficient_of_variation([]) is None
    assert coefficient_of_variation([0.0, 0.0]) is None


def test_identical_replicates_give_zero_cv(config, make_replicates):
    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-A"): [5.0] * 6}), TEST_PANEL, config
    )
    assert metrics.loc[0, "cv_pct"] == pytest.approx(0.0)
    assert metrics.loc[0, "status"] == "PASS"


def test_density_exactly_at_threshold_passes(config, make_replicates):
    """The rule is 'below 3.0 fails', so a mean of exactly 3.0 must pass."""
    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-A"): [3.0] * 6}), TEST_PANEL, config
    )
    assert metrics.loc[0, "mean_density"] == pytest.approx(3.0)
    assert not metrics.loc[0, "fail_low_density"]
    assert metrics.loc[0, "status"] == "PASS"


def test_density_just_below_threshold_fails(config, make_replicates):
    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-A"): [2.99] * 6}), TEST_PANEL, config
    )
    assert metrics.loc[0, "fail_low_density"]
    assert metrics.loc[0, "fail_reason"] == LOW_DENSITY_LABEL


def test_cv_exactly_at_threshold_passes(config, make_replicates):
    """Construct replicates whose sample CV is exactly 20%."""
    mean = 5.0
    target_sd = mean * 0.20
    # Two-point sample: SD = |a - b| / sqrt(2).
    half = target_sd * np.sqrt(2) / 2
    values = [mean - half, mean + half]
    assert coefficient_of_variation(values) == pytest.approx(20.0)

    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-A"): values}), TEST_PANEL, config
    )
    assert metrics.loc[0, "cv_pct"] == pytest.approx(20.0)
    assert not metrics.loc[0, "fail_high_cv"]


def test_cv_above_threshold_fails(config, make_replicates):
    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-A"): [5.0, 5.0, 5.0, 5.0, 5.0, 1.0]}),
        TEST_PANEL,
        config,
    )
    assert metrics.loc[0, "fail_high_cv"]
    assert metrics.loc[0, "cv_pct"] > 20.0


def test_both_criteria_failing_records_both_reasons(config, make_replicates):
    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-A"): [2.6, 2.6, 2.6, 2.6, 2.6, 0.4]}),
        TEST_PANEL,
        config,
    )
    row = metrics.loc[0]
    assert row["fail_low_density"] and row["fail_high_cv"]
    assert LOW_DENSITY_LABEL in row["fail_reason"] and HIGH_CV_LABEL in row["fail_reason"]


def test_per_antibody_threshold_override_is_applied(config, make_replicates):
    """Ab-B is configured with a 2.0 limit, so 2.5 passes for it but fails for Ab-A."""
    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-A"): [2.5] * 6, ("C1", "Ab-B"): [2.5] * 6}),
        TEST_PANEL,
        config,
    ).set_index("antibody")
    assert metrics.loc["Ab-A", "status"] == "FAIL"
    assert metrics.loc["Ab-B", "status"] == "PASS"


def test_single_replicate_cannot_fail_on_variability(config, make_replicates):
    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-A"): [5.0]}), TEST_PANEL, config
    )
    assert metrics.loc[0, "cv_pct"] is None or pd.isna(metrics.loc[0, "cv_pct"])
    assert not metrics.loc[0, "fail_high_cv"]


def test_chip_fails_when_any_antibody_fails(config, make_replicates):
    metrics = compute_antibody_metrics(
        make_replicates(
            {
                ("C1", "Ab-A"): [5.0] * 6,
                ("C1", "Ab-B"): [1.0] * 6,  # below even the 2.0 override
                ("C2", "Ab-A"): [5.0] * 6,
                ("C2", "Ab-B"): [5.0] * 6,
            }
        ),
        TEST_PANEL,
        config,
    )
    chips = summarize_chips(metrics).set_index("chip_id")
    assert chips.loc["C1", "status"] == "FAIL"
    assert chips.loc["C1", "n_antibodies_failed"] == 1
    assert "Ab-B" in chips.loc["C1", "fail_reason_summary"]
    assert chips.loc["C2", "status"] == "PASS"
    assert chips.loc["C2", "fail_reason_summary"] is None
    assert failure_rate_pct(chips.reset_index()) == pytest.approx(50.0)


def test_chip_to_chip_cv_is_computed_across_chip_means(config, make_replicates):
    """The Part 2 metric must use per-chip means, not pooled replicates."""
    metrics = compute_antibody_metrics(
        make_replicates(
            {
                ("C1", "Ab-A"): [4.0] * 6,
                ("C2", "Ab-A"): [5.0] * 6,
                ("C3", "Ab-A"): [6.0] * 6,
            }
        ),
        TEST_PANEL,
        config,
    )
    summary = summarize_run_antibodies(metrics).set_index("antibody")
    # Chip means are 4, 5, 6: mean 5, sample SD 1, so CV = 20%.
    assert summary.loc["Ab-A", "pooled_mean_density"] == pytest.approx(5.0)
    assert summary.loc["Ab-A", "chip_to_chip_cv_pct"] == pytest.approx(20.0)
    # Within-chip CV is zero for every chip, and must not be confused with the above.
    assert summary.loc["Ab-A", "mean_within_chip_cv_pct"] == pytest.approx(0.0)


def test_antibody_order_follows_panel_config(config, make_replicates):
    metrics = compute_antibody_metrics(
        make_replicates({("C1", "Ab-B"): [5.0] * 6, ("C1", "Ab-A"): [5.0] * 6}),
        TEST_PANEL,
        config,
    )
    assert metrics["antibody"].tolist() == ["Ab-A", "Ab-B"]


def test_empty_input_is_rejected(config):
    with pytest.raises(ValueError):
        compute_antibody_metrics(
            pd.DataFrame(columns=["chip_id", "antibody", "replicate_num", "density_ng_mm2"]),
            TEST_PANEL,
            config,
        )

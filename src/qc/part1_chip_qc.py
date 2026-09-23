"""Part 1: deterministic per-chip QC.

Pure functions over dataframes -- no database, no I/O -- so the pass/fail rules can be
unit tested directly and reused by the ingestion pipeline, the report generator, and
any ad-hoc notebook analysis.

The rules, in one place:
  * For each antibody on each chip, summarize its replicate spot densities
    (mean, sample SD, CV).
  * That antibody FAILS if its mean density is below the density threshold, or its
    replicate CV is above the CV threshold. Both reasons are recorded when both apply.
  * A chip FAILS if any of its antibodies fails.
  * A run's failure rate is the percentage of its chips that failed.

Note on terminology: the instrument measures spot height, which is proportional to
antibody surface density; densities in ng/mm^2 are the single metric used throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import QCConfig, Thresholds
from src.db.models import FAIL, PASS

LOW_DENSITY_LABEL = "low density"
HIGH_CV_LABEL = "high CV"

# Columns of the long replicate frame that every entry point expects.
REPLICATE_COLUMNS = ["chip_id", "antibody", "replicate_num", "density_ng_mm2"]


def coefficient_of_variation(values: pd.Series | np.ndarray) -> float | None:
    """Sample CV as a percentage, or None when it is undefined.

    Uses the sample standard deviation (ddof=1): the replicate spots are a sample of
    the spotting process, not the whole population. Undefined for fewer than two
    replicates or a non-positive mean.
    """
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if arr.size < 2:
        return None
    mean = float(arr.mean())
    if mean <= 0:
        return None
    return float(arr.std(ddof=1) / mean * 100.0)


def _fail_reason(fails_density: bool, fails_cv: bool) -> str | None:
    reasons = []
    if fails_density:
        reasons.append(LOW_DENSITY_LABEL)
    if fails_cv:
        reasons.append(HIGH_CV_LABEL)
    return " + ".join(reasons) if reasons else None


def _keep_none(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Preserve None in text columns instead of letting it become NaN.

    Building a frame from dicts turns None into NaN, which would be written to the
    database as a float and shown in reports as the string "nan". A passing chip has no
    failure reason, and that absence should stay null all the way through.
    """
    for column in columns:
        if column in frame:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def compute_antibody_metrics(
    replicates: pd.DataFrame,
    panel: str,
    config: QCConfig,
) -> pd.DataFrame:
    """Summarize and grade every (chip, antibody) from long replicate data.

    Args:
        replicates: long frame with columns chip_id, antibody, replicate_num,
            density_ng_mm2 (one row per spot).
        panel: panel name, used to resolve thresholds and antibody display order.
        config: loaded QC config.

    Returns:
        One row per (chip, antibody) with summary statistics and the QC verdict.
    """
    missing = [c for c in REPLICATE_COLUMNS if c not in replicates.columns]
    if missing:
        raise ValueError(f"Replicate data is missing column(s): {', '.join(missing)}")
    if replicates.empty:
        raise ValueError("Replicate data is empty; nothing to grade.")

    grouped = replicates.groupby(["chip_id", "antibody"], dropna=False)["density_ng_mm2"]
    metrics = grouped.agg(
        n_replicates="count",
        mean_density="mean",
        sd_density=lambda s: s.std(ddof=1),
        min_density="min",
        max_density="max",
    ).reset_index()

    metrics["cv_pct"] = [
        coefficient_of_variation(group)
        for _, group in replicates.groupby(["chip_id", "antibody"], dropna=False)[
            "density_ng_mm2"
        ]
    ]

    # Thresholds are resolved per antibody so a panel can carry a justified exception
    # for one antibody without loosening the criteria for the rest.
    thresholds: dict[str, Thresholds] = {
        antibody: config.resolve_thresholds(panel, antibody)
        for antibody in metrics["antibody"].unique()
    }

    verdicts = []
    for row in metrics.itertuples(index=False):
        rule = thresholds[row.antibody]
        fails_density, fails_cv = rule.evaluate(float(row.mean_density), row.cv_pct)
        verdicts.append(
            {
                "density_min_ng_mm2": rule.density_min_ng_mm2,
                "cv_max_pct": rule.cv_max_pct,
                "fail_low_density": fails_density,
                "fail_high_cv": fails_cv,
                "status": FAIL if (fails_density or fails_cv) else PASS,
                "fail_reason": _fail_reason(fails_density, fails_cv),
            }
        )
    metrics = pd.concat([metrics, pd.DataFrame(verdicts, index=metrics.index)], axis=1)
    metrics = _keep_none(metrics, "fail_reason")

    return _sort_by_panel_order(metrics, panel, config)


def _sort_by_panel_order(
    frame: pd.DataFrame, panel: str, config: QCConfig
) -> pd.DataFrame:
    """Order rows by chip, then by the antibody order declared in the config, so
    reports and plots read consistently rather than alphabetically."""
    declared = config.antibodies(panel)
    order = {name: i for i, name in enumerate(declared)}
    frame = frame.copy()
    frame["_ab_order"] = frame["antibody"].map(lambda a: order.get(a, len(order)))
    frame = frame.sort_values(["chip_id", "_ab_order", "antibody"]).drop(columns="_ab_order")
    return frame.reset_index(drop=True)


def summarize_chips(antibody_metrics: pd.DataFrame) -> pd.DataFrame:
    """Roll antibody verdicts up to a chip-level pass/fail with readable reasons.

    A chip fails if any antibody fails. The summary names each failing antibody and
    why, e.g. "Anti-IL-6: low density; Anti-CRP: high CV", so the report says what to
    investigate rather than only that the chip was rejected.
    """
    rows = []
    for chip_id, group in antibody_metrics.groupby("chip_id", sort=False):
        failures = group[group["status"] == FAIL]
        reason = "; ".join(
            f"{row.antibody}: {row.fail_reason}" for row in failures.itertuples(index=False)
        )
        rows.append(
            {
                "chip_id": chip_id,
                "status": FAIL if not failures.empty else PASS,
                "n_antibodies_failed": int(len(failures)),
                "fail_reason_summary": reason or None,
            }
        )
    return _keep_none(pd.DataFrame(rows), "fail_reason_summary")


def summarize_run_antibodies(antibody_metrics: pd.DataFrame) -> pd.DataFrame:
    """Batch-level statistics per antibody for one run.

    `chip_to_chip_cv_pct` -- the CV across per-chip mean densities -- is the
    variability metric tracked over time in Part 2. It answers "how uniform were the
    chips in this batch?", which is a different question from the within-chip
    replicate CV that decides pass/fail.
    """
    rows = []
    for antibody, group in antibody_metrics.groupby("antibody", sort=False):
        chip_means = group["mean_density"].astype(float)
        rows.append(
            {
                "antibody": antibody,
                "n_chips": int(len(group)),
                "n_chips_failed": int((group["status"] == FAIL).sum()),
                "n_failed_low_density": int(group["fail_low_density"].sum()),
                "n_failed_high_cv": int(group["fail_high_cv"].sum()),
                "pooled_mean_density": float(chip_means.mean()),
                "median_density": float(chip_means.median()),
                "chip_to_chip_sd": (
                    float(chip_means.std(ddof=1)) if len(chip_means) > 1 else None
                ),
                "chip_to_chip_cv_pct": coefficient_of_variation(chip_means),
                "mean_within_chip_cv_pct": (
                    float(group["cv_pct"].mean()) if group["cv_pct"].notna().any() else None
                ),
            }
        )
    return pd.DataFrame(rows)


def failure_rate_pct(chip_summary: pd.DataFrame) -> float:
    """Percentage of chips in the run that failed QC."""
    if chip_summary.empty:
        return 0.0
    return float((chip_summary["status"] == FAIL).mean() * 100.0)


def run_qc(
    replicates: pd.DataFrame, panel: str, config: QCConfig
) -> dict[str, pd.DataFrame | float]:
    """Convenience wrapper returning every Part 1 artifact for one run."""
    antibody_metrics = compute_antibody_metrics(replicates, panel, config)
    chip_summary = summarize_chips(antibody_metrics)
    return {
        "antibody_metrics": antibody_metrics,
        "chip_summary": chip_summary,
        "antibody_summary": summarize_run_antibodies(antibody_metrics),
        "failure_rate_pct": failure_rate_pct(chip_summary),
    }

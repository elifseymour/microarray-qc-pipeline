"""Part 2: monitoring the spotting process across runs.

Everything here is a pure function over the tidy frames produced by `src.db.queries`,
so the analysis can be tested against hand-built data.

The analytical questions this answers, in the order the original QC process document
poses them:

  Is an antibody going bad?      `antibody_density_trends` fits the density trend
                                 *within the lot that is currently in use*, and
                                 separately measures the step across a lot boundary.
                                 A slide inside one lot means the stock is degrading;
                                 a step at the boundary means the new lot is simply
                                 different. Conflating the two is what leads to
                                 discarding good antibody stock.
  Is chip-to-chip variability
  getting worse?                 `variability_trends`, plus `outlier_runs` for the
                                 sporadic bad run rather than a slow drift.
  Is the failure rate rising?    `failure_rate_trend`.
  What is driving it?            `environment_correlations` for temperature and
                                 humidity, `group_effects` for coating batch, wafer
                                 lot and operator.

A note on statistics that the dashboard repeats to the user: with a handful of runs
these tests have little power, and a correlation across runs can be confounded by
anything else that drifts with time. Every result carries its n, and environmental
correlations also report how strongly the driver itself tracks time, so a spurious
association is visible rather than hidden. Findings are leads to investigate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.config import MonitoringRules

DENSITY_METRIC = "pooled_mean_density"
CHIP_TO_CHIP_CV = "chip_to_chip_cv_pct"
WITHIN_CHIP_CV = "mean_within_chip_cv_pct"

ENVIRONMENT_DRIVERS = {
    "room_temp_c": "Room temperature",
    "room_humidity_pct": "Room humidity",
    "spotter_temp_c": "Spotter temperature",
    "spotter_humidity_pct": "Spotter humidity",
}

METRIC_LABELS = {
    DENSITY_METRIC: "Mean spot height (density)",
    CHIP_TO_CHIP_CV: "Chip-to-chip CV",
    WITHIN_CHIP_CV: "Within-chip CV",
}

# Mid-sentence forms: lowercasing the labels would mangle "CV".
METRIC_LABELS_INLINE = {
    DENSITY_METRIC: "mean spot height",
    CHIP_TO_CHIP_CV: "chip-to-chip CV",
    WITHIN_CHIP_CV: "within-chip CV",
}

# Attribution outcomes for an antibody's density trend.
STABLE = "stable"
DEGRADING = "degrading"
LOT_CHANGE = "lot change"
DECLINING_UNATTRIBUTED = "declining"
INSUFFICIENT = "insufficient data"


@dataclass
class TrendFit:
    """A least-squares fit of a metric against run order."""

    n: int
    slope: float | None = None
    intercept: float | None = None
    r: float | None = None
    p: float | None = None
    start_fitted: float | None = None
    end_fitted: float | None = None
    pct_change: float | None = None

    @property
    def is_significant_decline(self) -> bool:
        return (
            self.slope is not None
            and self.slope < 0
            and self.p is not None
            and self.pct_change is not None
        )


@dataclass
class Flag:
    """One finding for the Alerts panel, written to be read by a scientist."""

    level: str  # "alert" | "watch" | "info"
    category: str
    subject: str
    headline: str
    detail: str
    evidence: dict = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple[int, str]:
        return ({"alert": 0, "watch": 1, "info": 2}.get(self.level, 3), self.subject)


def fit_trend(x: Iterable[float], y: Iterable[float]) -> TrendFit:
    """Least-squares fit of y on x, with the change expressed as a percentage.

    `pct_change` is taken between the fitted endpoints rather than the raw first and
    last points, so a single noisy run cannot masquerade as a trend.
    """
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr, y_arr = x_arr[mask], y_arr[mask]

    if x_arr.size < 3 or np.ptp(x_arr) == 0:
        return TrendFit(n=int(x_arr.size))

    result = stats.linregress(x_arr, y_arr)
    start = float(result.intercept + result.slope * x_arr.min())
    end = float(result.intercept + result.slope * x_arr.max())
    pct = float((end - start) / start * 100.0) if start not in (0.0,) else None

    return TrendFit(
        n=int(x_arr.size),
        slope=float(result.slope),
        intercept=float(result.intercept),
        r=float(result.rvalue),
        p=float(result.pvalue),
        start_fitted=start,
        end_fitted=end,
        pct_change=pct,
    )


# --------------------------------------------------------------------------------
# Antibody density: degradation vs. lot change
# --------------------------------------------------------------------------------


def antibody_density_trends(
    summary: pd.DataFrame,
    rules: MonitoringRules,
    reference_antibody: str | None = None,
) -> pd.DataFrame:
    """Per antibody: is the spot height drifting, and is the lot or the stock to blame?

    The key comparison is deliberately *not* a single regression over all runs. When a
    lot changes mid-series, one regression through the step reports a decline that is
    neither degradation nor reproducible. Instead:

      * the trend is fitted within the lot currently in use, over every run that used
        it -- this is the degradation test;
      * the step between the current lot's mean and the previous lot's mean is measured
        separately -- this is the lot potency test;
      * a recent-window fit is reported alongside for context.
    """
    if summary.empty:
        return pd.DataFrame()

    # Density of the negative control in each run. Dividing an antibody by it cancels
    # anything that affected the whole run's deposition -- the coated surface, the
    # spotter, the day's conditions -- leaving only what is specific to that antibody.
    control_by_run: dict[str, float] = {}
    if reference_antibody:
        control = summary[summary["antibody"] == reference_antibody]
        control_by_run = dict(
            zip(control["spotting_run_id"], control[DENSITY_METRIC].astype(float))
        )

    rows = []
    for antibody, group in summary.groupby("antibody", sort=False):
        group = group.sort_values("run_index")
        n_runs = len(group)

        window = group.tail(rules.trend_window_runs)
        window_fit = fit_trend(window["run_index"], window[DENSITY_METRIC])
        overall_fit = fit_trend(group["run_index"], group[DENSITY_METRIC])

        lots = [lot for lot in group["ab_lot"].dropna().unique()]
        current_lot = group["ab_lot"].iloc[-1] if lots else None
        lot_runs = (
            group[group["ab_lot"] == current_lot] if current_lot else group.iloc[0:0]
        )

        # Degradation test: within the lot in use now, across every run that used it.
        # With no lot information at all, the whole series is the best available proxy.
        if current_lot is None:
            within_lot_fit = overall_fit
            lot_runs = group
        else:
            within_lot_fit = fit_trend(lot_runs["run_index"], lot_runs[DENSITY_METRIC])

        lot_means = (
            group.dropna(subset=["ab_lot"])
            .groupby("ab_lot", sort=False)[DENSITY_METRIC]
            .agg(["mean", "size"])
        )
        previous_lot = None
        lot_step_pct = None
        if len(lot_means) >= 2:
            ordered = [lot for lot in group["ab_lot"].dropna().unique()]
            previous_lot = ordered[-2]
            prev_mean = float(lot_means.loc[previous_lot, "mean"])
            curr_mean = float(lot_means.loc[current_lot, "mean"])
            if prev_mean:
                lot_step_pct = (curr_mean - prev_mean) / prev_mean * 100.0

        # The same within-lot trend, but as a ratio to the negative control.
        ratio_fit = TrendFit(n=0)
        if control_by_run and antibody != reference_antibody and not lot_runs.empty:
            denominator = lot_runs["spotting_run_id"].map(control_by_run)
            ratio = lot_runs[DENSITY_METRIC].astype(float) / denominator
            ratio_fit = fit_trend(lot_runs["run_index"], ratio)

        attribution = _attribute_density_trend(
            n_runs=n_runs,
            within_lot_fit=within_lot_fit,
            overall_fit=overall_fit,
            lot_step_pct=lot_step_pct,
            rules=rules,
        )

        rows.append(
            {
                "antibody": antibody,
                "n_runs": n_runs,
                "latest_density": float(group[DENSITY_METRIC].iloc[-1]),
                "first_density": float(group[DENSITY_METRIC].iloc[0]),
                "current_lot": current_lot,
                "previous_lot": previous_lot,
                "n_lots": len(lot_means),
                "n_runs_current_lot": int(len(lot_runs)),
                "lot_step_pct": lot_step_pct,
                "current_lot_age_days": (
                    float(lot_runs["lot_age_days"].iloc[-1])
                    if "lot_age_days" in lot_runs and lot_runs["lot_age_days"].notna().any()
                    else None
                ),
                "within_lot_slope": within_lot_fit.slope,
                "within_lot_pct_change": within_lot_fit.pct_change,
                "within_lot_p": within_lot_fit.p,
                "within_lot_n": within_lot_fit.n,
                "ratio_to_control_pct_change": ratio_fit.pct_change,
                "ratio_to_control_p": ratio_fit.p,
                "window_pct_change": window_fit.pct_change,
                "window_p": window_fit.p,
                "window_n": window_fit.n,
                "overall_pct_change": overall_fit.pct_change,
                "attribution": attribution,
            }
        )

    return pd.DataFrame(rows)


def _attribute_density_trend(
    n_runs: int,
    within_lot_fit: TrendFit,
    overall_fit: TrendFit,
    lot_step_pct: float | None,
    rules: MonitoringRules,
) -> str:
    """Decide what a density change should be blamed on.

    Order matters: a significant decline *inside* the current lot is degradation even
    if a lot change also happened earlier, because a fresh lot should not be falling.
    """
    if n_runs < rules.min_runs_for_trend:
        return INSUFFICIENT

    declining_within_lot = (
        within_lot_fit.n >= rules.min_runs_for_trend
        and within_lot_fit.pct_change is not None
        and within_lot_fit.pct_change <= -rules.density_decline_pct_flag
        and within_lot_fit.p is not None
        and within_lot_fit.p <= rules.slope_p_max
    )
    if declining_within_lot:
        return DEGRADING

    if lot_step_pct is not None and abs(lot_step_pct) >= rules.lot_step_change_pct:
        return LOT_CHANGE

    if (
        overall_fit.pct_change is not None
        and overall_fit.pct_change <= -rules.density_decline_pct_flag
        and overall_fit.p is not None
        and overall_fit.p <= rules.slope_p_max
    ):
        return DECLINING_UNATTRIBUTED

    return STABLE


def lot_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Per antibody x lot: mean density, runs used, lot age span, failure counts."""
    if summary.empty or "ab_lot" not in summary:
        return pd.DataFrame()
    frame = summary.dropna(subset=["ab_lot"])
    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby(["antibody", "ab_lot"], sort=False).agg(
        n_runs=("spotting_run_id", "nunique"),
        first_run=("run_date", "min"),
        last_run=("run_date", "max"),
        mean_density=(DENSITY_METRIC, "mean"),
        sd_density=(DENSITY_METRIC, "std"),
        mean_chip_to_chip_cv=(CHIP_TO_CHIP_CV, "mean"),
        chips_failed=("n_chips_failed", "sum"),
        chips=("n_chips", "sum"),
        min_lot_age_days=("lot_age_days", "min"),
        max_lot_age_days=("lot_age_days", "max"),
    ).reset_index()
    grouped["failure_rate_pct"] = grouped["chips_failed"] / grouped["chips"] * 100.0

    # Percentage step against the antibody's previous lot, in run order.
    grouped["step_vs_previous_lot_pct"] = (
        grouped.groupby("antibody")["mean_density"].pct_change() * 100.0
    )
    return grouped


def lot_age_relationship(summary: pd.DataFrame) -> pd.DataFrame:
    """Correlate density against how old the stock was on the day it was spotted.

    This is the direct shelf-degradation view. It only carries weight when a lot was
    used across a decent span of days, so the span is reported with the correlation.
    """
    if summary.empty or "lot_age_days" not in summary:
        return pd.DataFrame()

    rows = []
    frame = summary.dropna(subset=["lot_age_days", DENSITY_METRIC])
    for antibody, group in frame.groupby("antibody", sort=False):
        if group["lot_age_days"].nunique() < 3:
            continue
        r, p = stats.pearsonr(group["lot_age_days"], group[DENSITY_METRIC])
        rows.append(
            {
                "antibody": antibody,
                "n": int(len(group)),
                "n_lots": int(group["ab_lot"].nunique()),
                "age_span_days": float(
                    group["lot_age_days"].max() - group["lot_age_days"].min()
                ),
                "r": float(r),
                "p": float(p),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------
# Variability and failure rate
# --------------------------------------------------------------------------------


def variability_trends(summary: pd.DataFrame, rules: MonitoringRules) -> pd.DataFrame:
    """Per antibody: is chip-to-chip (and within-chip) variability drifting upward?

    Reported as a change in percentage points, because a CV is already a percentage
    and a "20% rise in a CV" would be ambiguous.
    """
    if summary.empty:
        return pd.DataFrame()

    rows = []
    for antibody, group in summary.groupby("antibody", sort=False):
        group = group.sort_values("run_index")
        window = group.tail(rules.trend_window_runs)
        baseline = group.head(max(len(group) - len(window), 1))

        record = {"antibody": antibody, "n_runs": len(group)}
        for metric, prefix in ((CHIP_TO_CHIP_CV, "chip_to_chip"), (WITHIN_CHIP_CV, "within_chip")):
            fit = fit_trend(group["run_index"], group[metric])
            recent_mean = float(window[metric].mean()) if window[metric].notna().any() else None
            baseline_mean = (
                float(baseline[metric].mean()) if baseline[metric].notna().any() else None
            )
            record.update(
                {
                    f"{prefix}_latest": (
                        float(group[metric].iloc[-1])
                        if pd.notna(group[metric].iloc[-1])
                        else None
                    ),
                    f"{prefix}_recent_mean": recent_mean,
                    f"{prefix}_baseline_mean": baseline_mean,
                    f"{prefix}_rise_points": (
                        recent_mean - baseline_mean
                        if recent_mean is not None and baseline_mean is not None
                        else None
                    ),
                    f"{prefix}_slope_per_run": fit.slope,
                    f"{prefix}_p": fit.p,
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def failure_rate_trend(runs: pd.DataFrame, rules: MonitoringRules) -> dict:
    """Run-level failure rate: the fitted trend, recent vs. earlier, and a proper test.

    A percentage-point rise on its own is easy to over-read: with 20 chips a run, a few
    extra rejects move the rate several points by chance. So the recent window and the
    earlier baseline are also compared as chip counts with Fisher's exact test, which is
    what decides whether the rise is flagged. This keeps the alert honest instead of
    firing whenever a noisy run lands at the end of the series.
    """
    if runs.empty:
        return {"n_runs": 0}

    ordered = runs.sort_values("run_date")
    fit = fit_trend(range(len(ordered)), ordered["failure_rate_pct"])
    window = ordered.tail(rules.trend_window_runs)
    baseline = ordered.head(max(len(ordered) - len(window), 1))

    recent_mean = float(window["failure_rate_pct"].mean())
    baseline_mean = float(baseline["failure_rate_pct"].mean())

    recent_failed = int(window["n_chips_failed"].sum())
    recent_total = int(window["n_chips"].sum())
    baseline_failed = int(baseline["n_chips_failed"].sum())
    baseline_total = int(baseline["n_chips"].sum())

    proportion_p = None
    if recent_total and baseline_total and (recent_failed + baseline_failed) > 0:
        _, proportion_p = stats.fisher_exact(
            [
                [recent_failed, recent_total - recent_failed],
                [baseline_failed, baseline_total - baseline_failed],
            ],
            alternative="greater",
        )
        proportion_p = float(proportion_p)

    return {
        "n_runs": int(len(ordered)),
        "latest": float(ordered["failure_rate_pct"].iloc[-1]),
        "recent_mean": recent_mean,
        "baseline_mean": baseline_mean,
        "rise_points": recent_mean - baseline_mean,
        "slope_per_run": fit.slope,
        "p": fit.p,
        "mean": float(ordered["failure_rate_pct"].mean()),
        "recent_failed": recent_failed,
        "recent_total": recent_total,
        "baseline_failed": baseline_failed,
        "baseline_total": baseline_total,
        "proportion_p": proportion_p,
        "n_recent_runs": int(len(window)),
        "n_baseline_runs": int(len(baseline)),
    }


def outlier_runs(summary: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Runs that stand out from the series rather than continuing a drift.

    Uses a median/MAD rule instead of mean/SD: with a handful of runs, one bad run
    inflates an SD enough to hide itself. A robust z of 3 is the usual convention for
    "this does not belong to the same distribution as the rest".
    """
    findings = []

    def robust_z(series: pd.Series) -> pd.Series:
        median = series.median()
        mad = (series - median).abs().median()
        scale = mad * 1.4826
        if not np.isfinite(scale) or scale == 0:
            scale = series.std(ddof=1)
        if not np.isfinite(scale) or scale == 0:
            return pd.Series(0.0, index=series.index)
        return (series - median) / scale

    if not summary.empty:
        for antibody, group in summary.groupby("antibody", sort=False):
            if len(group) < 4:
                continue
            for metric, label in (
                (CHIP_TO_CHIP_CV, "chip-to-chip CV"),
                (WITHIN_CHIP_CV, "within-chip CV"),
            ):
                values = group[metric].astype(float)
                if values.notna().sum() < 4:
                    continue
                z = robust_z(values)
                for idx in z[z >= 3.0].index:
                    findings.append(
                        {
                            "spotting_run_id": group.loc[idx, "spotting_run_id"],
                            "run_date": group.loc[idx, "run_date"],
                            "subject": antibody,
                            "metric": label,
                            "value": float(values.loc[idx]),
                            "median": float(values.median()),
                            "robust_z": float(z.loc[idx]),
                        }
                    )

    if not runs.empty and len(runs) >= 4:
        values = runs["failure_rate_pct"].astype(float)
        z = robust_z(values)
        for idx in z[z >= 3.0].index:
            findings.append(
                {
                    "spotting_run_id": runs.loc[idx, "spotting_run_id"],
                    "run_date": runs.loc[idx, "run_date"],
                    "subject": "All antibodies",
                    "metric": "failure rate",
                    "value": float(values.loc[idx]),
                    "median": float(values.median()),
                    "robust_z": float(z.loc[idx]),
                }
            )

    return pd.DataFrame(findings)


# --------------------------------------------------------------------------------
# Drivers: environment, coating batch, wafer lot, operator
# --------------------------------------------------------------------------------


def environment_correlations(summary: pd.DataFrame, rules: MonitoringRules) -> pd.DataFrame:
    """Correlate each run-level metric against each environmental reading.

    `driver_vs_time_r` is included for a reason: over a run series, anything that
    drifts with time correlates with anything else that drifts with time. If the
    driver itself tracks the calendar, a strong correlation with a declining metric
    may be coincidence, and the dashboard says so rather than implying causation.
    """
    if summary.empty:
        return pd.DataFrame()

    rows = []
    for antibody, group in summary.groupby("antibody", sort=False):
        for driver, driver_label in ENVIRONMENT_DRIVERS.items():
            if driver not in group or group[driver].notna().sum() < 3:
                continue
            driver_time_r = (
                float(stats.pearsonr(group["run_index"], group[driver])[0])
                if group[driver].nunique() > 1
                else np.nan
            )
            for metric, metric_label in METRIC_LABELS.items():
                pair = group[[driver, metric, "run_index"]].dropna()
                if len(pair) < 3 or pair[driver].nunique() < 2 or pair[metric].nunique() < 2:
                    continue
                r, p = stats.pearsonr(pair[driver], pair[metric])
                rows.append(
                    {
                        "antibody": antibody,
                        "driver": driver,
                        "driver_label": driver_label,
                        "metric": metric,
                        "metric_label": metric_label,
                        "metric_label_inline": METRIC_LABELS_INLINE[metric],
                        "n": int(len(pair)),
                        "r": float(r),
                        "p": float(p),
                        "driver_vs_time_r": driver_time_r,
                        "notable": bool(abs(r) >= rules.correlation_r_min and p <= 0.05),
                    }
                )
    return pd.DataFrame(rows)


def run_panel_effects(
    summary: pd.DataFrame, reference_antibody: str | None = None
) -> pd.DataFrame:
    """Per run: how the whole panel sat relative to its own historical norm.

    This is the discriminator between a problem in the spotting process and a problem
    with one antibody. Each antibody's run mean is divided by that antibody's median
    across the series, so 1.00 is "normal for this antibody". Then:

      * every antibody low together, negative control included, means the deposition
        itself under-performed -- the coated surface, the spotter, or that day's
        conditions;
      * one antibody low while the rest sit at 1.00 means that antibody's stock.

    The negative control carries the most weight of all, because it has no biomarker
    specificity: nothing about an antibody going bad can move it.
    """
    if summary.empty:
        return pd.DataFrame()

    frame = summary.copy()
    norm = frame.groupby("antibody")[DENSITY_METRIC].transform("median")
    frame["relative_density"] = frame[DENSITY_METRIC] / norm

    rows = []
    for run_id, group in frame.groupby("spotting_run_id", sort=False):
        control = (
            group.loc[group["antibody"] == reference_antibody, "relative_density"]
            if reference_antibody
            else pd.Series(dtype=float)
        )
        specific = (
            group[group["antibody"] != reference_antibody] if reference_antibody else group
        )
        rows.append(
            {
                "spotting_run_id": run_id,
                "run_date": group["run_date"].iloc[0],
                "run_index": group["run_index"].iloc[0],
                "n_antibodies": int(len(group)),
                "n_antibodies_low": int((group["relative_density"] < 0.90).sum()),
                "median_relative_density": float(specific["relative_density"].median()),
                "min_relative_density": float(group["relative_density"].min()),
                "control_relative_density": (
                    float(control.iloc[0]) if len(control) else None
                ),
                "median_chip_to_chip_cv": float(group[CHIP_TO_CHIP_CV].median()),
                "median_within_chip_cv": float(group[WITHIN_CHIP_CV].median()),
                "room_temp_c": group["room_temp_c"].iloc[0],
                "room_humidity_pct": group["room_humidity_pct"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def run_group_map(measurements: pd.DataFrame, by: str) -> pd.DataFrame:
    """Which group (coating batch, wafer lot, operator) each run used."""
    if measurements.empty or by not in measurements:
        return pd.DataFrame(columns=["spotting_run_id", by])
    return (
        measurements.dropna(subset=[by])
        .groupby("spotting_run_id")[by]
        .agg(lambda s: ", ".join(sorted(set(s.astype(str)))))
        .reset_index()
    )


def coating_batch_surface_evidence(
    run_panel: pd.DataFrame, run_batches: pd.DataFrame, by: str = "coating_batch"
) -> pd.DataFrame:
    """Aggregate the panel-wide evidence per coating batch.

    Because every chip in a run shares one coating batch, a batch cannot be compared
    against another batch inside the same run. What can still be established is whether
    the runs that used a batch came out low across the *whole* panel, control included.
    That pattern rules out any single antibody as the cause and leaves the surface, the
    spotter, or the day's conditions -- and the conditions are recorded, so they can be
    checked separately.
    """
    if run_panel.empty or run_batches.empty:
        return pd.DataFrame()

    merged = run_panel.merge(run_batches, on="spotting_run_id", how="left").dropna(
        subset=[by]
    )
    if merged.empty:
        return pd.DataFrame()

    grouped = (
        merged.groupby(by)
        .agg(
            n_runs=("spotting_run_id", "nunique"),
            runs=("spotting_run_id", lambda s: ", ".join(sorted(set(s)))),
            median_relative_density=("median_relative_density", "mean"),
            control_relative_density=("control_relative_density", "mean"),
            median_within_chip_cv=("median_within_chip_cv", "mean"),
            median_chip_to_chip_cv=("median_chip_to_chip_cv", "mean"),
            max_room_temp_c=("room_temp_c", "max"),
            min_room_humidity_pct=("room_humidity_pct", "min"),
        )
        .reset_index()
    )
    grouped["density_deficit_pct"] = (grouped["median_relative_density"] - 1.0) * 100.0
    grouped["control_deficit_pct"] = (grouped["control_relative_density"] - 1.0) * 100.0
    return grouped.sort_values("median_relative_density")


def group_effects(measurements: pd.DataFrame, by: str) -> pd.DataFrame:
    """Compare chip-level results across coating batches, wafer lots or operators.

    Two baselines are reported, because each answers a different question and each has
    a weakness the other covers.

    `relative_density` divides a chip's density by the median for that antibody across
    the whole series, so 1.00 means "normal for this antibody". This is what identifies
    *which* group is abnormal, but it can be misled by a group that happens to have been
    used only during, say, a hot week.

    `within_run_pct_vs_others` compares the group only against the other groups present
    in the very same run and antibody. Those chips shared the day's temperature,
    humidity, operator and antibody lots, so a difference there is attributable to the
    group itself. It is the stronger evidence, but it is only defined when groups
    co-occur in a run, and it is symmetric: it shows two groups differ, not which one is
    at fault. Reading the two together resolves both weaknesses.
    """
    if measurements.empty or by not in measurements:
        return pd.DataFrame()

    frame = measurements.dropna(subset=[by]).copy()
    if frame.empty:
        return pd.DataFrame()

    series_reference = frame.groupby("antibody")["mean_density"].transform("median")
    frame["relative_density"] = frame["mean_density"] / series_reference

    # Leave-this-group-out reference, within each run and antibody.
    frame["within_run_reference"] = np.nan
    for _, block in frame.groupby(["spotting_run_id", "antibody"], sort=False):
        present = block[by].unique()
        if len(present) < 2:
            continue
        for name in present:
            mask = block.index[block[by] == name]
            others = block.loc[block[by] != name, "mean_density"]
            frame.loc[mask, "within_run_reference"] = others.median()
    frame["within_run_relative"] = frame["mean_density"] / frame["within_run_reference"]

    chips = frame.drop_duplicates(subset=["spotting_run_id", "chip_id"])
    chip_fail = (
        chips.assign(failed=lambda d: d["chip_status"].eq("FAIL"))
        .groupby(by)["failed"]
        .agg(["sum", "size"])
        .rename(columns={"sum": "chips_failed", "size": "n_chips"})
    )

    grouped = (
        frame.groupby(by)
        .agg(
            n_runs=("spotting_run_id", "nunique"),
            n_measurements=("mean_density", "size"),
            relative_density=("relative_density", "mean"),
            within_run_relative=("within_run_relative", "mean"),
            n_within_run_comparisons=("within_run_relative", "count"),
            mean_within_chip_cv=("cv_pct", "mean"),
            mean_density=("mean_density", "mean"),
        )
        .join(chip_fail)
        .reset_index()
    )
    grouped["failure_rate_pct"] = grouped["chips_failed"] / grouped["n_chips"] * 100.0
    grouped["relative_density_pct_diff"] = (grouped["relative_density"] - 1.0) * 100.0
    grouped["within_run_pct_vs_others"] = (grouped["within_run_relative"] - 1.0) * 100.0

    # A group that never shares a run with another group is perfectly aligned with the
    # runs it was used in, so its effect cannot be separated from whatever else changed
    # over those runs -- including an antibody drifting. Wafer lots typically fall in
    # this category (one wafer per run), and reading their column as a wafer effect is a
    # mistake the dashboard has to warn about rather than invite.
    grouped["separable_from_run"] = grouped["n_within_run_comparisons"] > 0
    return grouped.sort_values("relative_density")


def within_run_group_contrast(measurements: pd.DataFrame, by: str) -> pd.DataFrame:
    """Compare groups that appear together in the same run.

    When two coating batches are spotted in one run, they share the day's temperature,
    humidity, operator and antibody lots. Any difference between them is therefore
    attributable to the batch itself -- a much stronger comparison than pooling across
    runs, and the reason the synthetic dataset splits batches within a run.
    """
    if measurements.empty or by not in measurements:
        return pd.DataFrame()

    rows = []
    for (run_id, antibody), group in measurements.dropna(subset=[by]).groupby(
        ["spotting_run_id", "antibody"], sort=False
    ):
        if group[by].nunique() < 2:
            continue
        stats_by_group = group.groupby(by).agg(
            mean_density=("mean_density", "mean"),
            mean_cv=("cv_pct", "mean"),
            n_chips=("chip_id", "nunique"),
        )
        overall = float(group["mean_density"].mean())
        for name, row in stats_by_group.iterrows():
            rows.append(
                {
                    "spotting_run_id": run_id,
                    "antibody": antibody,
                    by: name,
                    "mean_density": float(row["mean_density"]),
                    "mean_cv": float(row["mean_cv"]),
                    "n_chips": int(row["n_chips"]),
                    "pct_vs_run_mean": (row["mean_density"] - overall) / overall * 100.0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------------


def build_flags(
    density_trends: pd.DataFrame,
    variability: pd.DataFrame,
    failure_trend: dict,
    correlations: pd.DataFrame,
    batch_effects: pd.DataFrame,
    outliers: pd.DataFrame,
    lot_age: pd.DataFrame,
    rules: MonitoringRules,
    reference_antibody: str | None = None,
    surface_evidence: pd.DataFrame | None = None,
) -> list[Flag]:
    """Turn the analyses into a ranked list of plain-language findings."""
    flags: list[Flag] = []

    flags += _density_flags(density_trends, lot_age, rules, reference_antibody)
    flags += _variability_flags(variability, rules)
    flags += _failure_rate_flags(failure_trend, rules)
    flags += _outlier_flags(outliers)
    flags += _batch_flags(
        surface_evidence if surface_evidence is not None else pd.DataFrame(),
        batch_effects,
        reference_antibody,
        rules,
    )
    flags += _correlation_flags(correlations)

    return sorted(flags, key=lambda f: f.sort_key)


def _fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _reference_is_stable(trends: pd.DataFrame, reference_antibody: str | None) -> bool:
    """Whether the panel's negative control held steady over the same runs."""
    if not reference_antibody or trends.empty:
        return False
    row = trends[trends["antibody"] == reference_antibody]
    return not row.empty and row.iloc[0]["attribution"] == STABLE


def _reference_is_declining(trends: pd.DataFrame, reference_antibody: str | None) -> bool:
    """Whether the negative control is itself losing deposited density.

    This matters for how a capture antibody's decline should be read. If the control is
    falling too, a ratio between them is flat and says nothing, and the shared cause
    (surface, spotter, conditions) has to be excluded before blaming the antibody.
    """
    if not reference_antibody or trends.empty:
        return False
    row = trends[trends["antibody"] == reference_antibody]
    return not row.empty and row.iloc[0]["attribution"] in (
        DEGRADING,
        DECLINING_UNATTRIBUTED,
    )


def _density_flags(
    trends: pd.DataFrame,
    lot_age: pd.DataFrame,
    rules: MonitoringRules,
    reference_antibody: str | None = None,
) -> list[Flag]:
    if trends.empty:
        return []
    age_lookup = (
        lot_age.set_index("antibody").to_dict("index") if not lot_age.empty else {}
    )
    reference_stable = _reference_is_stable(trends, reference_antibody)
    reference_declining = _reference_is_declining(trends, reference_antibody)
    flags = []
    for row in trends.itertuples(index=False):
        is_reference = reference_antibody is not None and row.antibody == reference_antibody

        if row.attribution == DEGRADING and is_reference:
            # The negative control has no specificity, so it cannot "go bad" in the way
            # a capture antibody does. If its deposited density is falling, what changed
            # is upstream of the antibodies.
            flags.append(
                Flag(
                    level="alert",
                    category="process-reference",
                    subject=row.antibody,
                    headline=f"The {row.antibody} negative control is losing signal too",
                    detail=(
                        f"Deposited density on the negative control spot fell "
                        f"{_fmt(abs(row.within_lot_pct_change), 1, '%')} across "
                        f"{int(row.within_lot_n)} runs (p = {_fmt(row.within_lot_p, 3)}). "
                        "The control has no biomarker specificity, so this points at the "
                        "spotting process itself rather than at any antibody: check the "
                        "spotter's dispensed volume and nozzle, and the coated surface. "
                        "Any decline seen on the capture antibodies over the same runs may "
                        "share this cause."
                    ),
                    evidence={
                        "within_lot_pct_change": row.within_lot_pct_change,
                        "within_lot_p": row.within_lot_p,
                        "n_runs": row.within_lot_n,
                    },
                )
            )
        elif row.attribution == DEGRADING:
            age_note = ""
            age = age_lookup.get(row.antibody)
            if age and abs(age["r"]) >= 0.6:
                age_note = (
                    f" Density also tracks stock age across {int(age['age_span_days'])} days "
                    f"(r = {age['r']:.2f}, n = {age['n']})."
                )
            lot_note = (
                f"lot {row.current_lot} has been in use for all "
                f"{int(row.n_runs_current_lot)} of those runs"
                if row.current_lot
                else "no lot information is recorded"
            )
            control_note = ""
            ratio_change = getattr(row, "ratio_to_control_pct_change", None)
            ratio_p = getattr(row, "ratio_to_control_p", None)
            ratio_confirms = (
                reference_antibody
                and ratio_change is not None
                and pd.notna(ratio_change)
                and ratio_change < 0
                and ratio_p is not None
                and pd.notna(ratio_p)
                and ratio_p <= rules.slope_p_max
            )
            if reference_declining:
                # The control is falling too, so the ratio between them is flat and
                # carries no information. Saying so is important: a flat ratio here must
                # not be read as "normal".
                control_note = (
                    f" Note that the {reference_antibody} negative control is losing "
                    "density over the same runs, so this decline is at least partly shared "
                    "with the spotting process rather than specific to this antibody. "
                    "Resolve the control's own alert first: if deposition is under-"
                    "performing generally, that may account for some or all of this drop."
                )
            elif ratio_confirms:
                # Dividing by the control removes every run-level effect, so a decline
                # that survives it belongs to this antibody alone. This is reported
                # alongside the absolute decline above, never in place of it.
                control_note = (
                    f" Measured as a ratio to the {reference_antibody} control spot, which "
                    "cancels any run-to-run difference in how much protein was deposited, "
                    f"the decline is still {_fmt(abs(ratio_change), 1, '%')} "
                    f"(p = {_fmt(ratio_p, 3)}). That isolates it to this antibody rather "
                    "than to the surface or the spotter."
                )
            elif reference_stable:
                control_note = (
                    f" The {reference_antibody} negative control was stable over the same "
                    "runs, so the coated surface and the spotter are unlikely to explain "
                    "this on their own."
                )
            flags.append(
                Flag(
                    level="alert",
                    category="antibody-degradation",
                    subject=row.antibody,
                    headline=f"{row.antibody} is losing signal within one lot",
                    detail=(
                        f"Mean spot height fell {_fmt(abs(row.within_lot_pct_change), 1, '%')} "
                        f"across {int(row.within_lot_n)} runs "
                        f"(p = {_fmt(row.within_lot_p, 3)}), and {lot_note}. A decline "
                        "inside a single lot points to the stock degrading rather than "
                        f"lot-to-lot variation.{age_note}{control_note} Latest mean is "
                        f"{_fmt(row.latest_density, 2)} ng/mm². Consider testing a fresh "
                        "vial of this antibody before discarding the stock."
                    ),
                    evidence={
                        "within_lot_pct_change": row.within_lot_pct_change,
                        "within_lot_p": row.within_lot_p,
                        "n_runs": row.within_lot_n,
                        "current_lot": row.current_lot,
                        "lot_age_days": row.current_lot_age_days,
                        "reference_stable": reference_stable,
                        "ratio_to_control_pct_change": ratio_change,
                    },
                )
            )
        elif row.attribution == LOT_CHANGE:
            direction = "lower" if (row.lot_step_pct or 0) < 0 else "higher"
            flags.append(
                Flag(
                    level="watch",
                    category="antibody-lot",
                    subject=row.antibody,
                    headline=f"{row.antibody} shifted at a lot change, not gradually",
                    detail=(
                        f"Mean spot height is {_fmt(abs(row.lot_step_pct), 1, '%')} {direction} "
                        f"on lot {row.current_lot} than on {row.previous_lot}, with no "
                        "significant trend inside the current lot "
                        f"(p = {_fmt(row.within_lot_p, 3)}). This looks like a difference in "
                        "lot potency rather than antibody degradation, so the stock itself "
                        "is probably fine. Check the new lot's certificate of analysis and, "
                        "if the panel is calibrated, whether the shift needs compensating."
                    ),
                    evidence={
                        "lot_step_pct": row.lot_step_pct,
                        "current_lot": row.current_lot,
                        "previous_lot": row.previous_lot,
                        "within_lot_p": row.within_lot_p,
                    },
                )
            )
        elif row.attribution == DECLINING_UNATTRIBUTED:
            flags.append(
                Flag(
                    level="watch",
                    category="antibody-density",
                    subject=row.antibody,
                    headline=f"{row.antibody} spot height is drifting down",
                    detail=(
                        f"Mean spot height fell {_fmt(abs(row.overall_pct_change), 1, '%')} "
                        f"across {int(row.n_runs)} runs, but the decline does not sit cleanly "
                        "inside one lot or at a lot boundary. Check the spotter and the "
                        "coating batches used over this period as well as the antibody."
                    ),
                    evidence={"overall_pct_change": row.overall_pct_change},
                )
            )
    return flags


def _variability_flags(variability: pd.DataFrame, rules: MonitoringRules) -> list[Flag]:
    if variability.empty:
        return []
    flags = []
    for row in variability.itertuples(index=False):
        rise = getattr(row, "chip_to_chip_rise_points", None)
        if (
            rise is not None
            and pd.notna(rise)
            and rise >= rules.cv_rise_pct_points_flag
            and row.n_runs >= rules.min_runs_for_trend
        ):
            flags.append(
                Flag(
                    level="alert",
                    category="variability",
                    subject=row.antibody,
                    headline=f"Chip-to-chip variability is rising for {row.antibody}",
                    detail=(
                        f"Chip-to-chip CV averaged {_fmt(row.chip_to_chip_recent_mean, 1, '%')} "
                        f"over the recent runs against {_fmt(row.chip_to_chip_baseline_mean, 1, '%')} "
                        f"earlier, a rise of {_fmt(rise, 1)} percentage points. Chips within a "
                        "batch are becoming less alike: check the spotter nozzle and droplet "
                        "consistency, and whether recent coating batches differ."
                    ),
                    evidence={
                        "recent_mean": row.chip_to_chip_recent_mean,
                        "baseline_mean": row.chip_to_chip_baseline_mean,
                        "rise_points": rise,
                    },
                )
            )
        within_rise = getattr(row, "within_chip_rise_points", None)
        if (
            within_rise is not None
            and pd.notna(within_rise)
            and within_rise >= rules.cv_rise_pct_points_flag
            and row.n_runs >= rules.min_runs_for_trend
        ):
            flags.append(
                Flag(
                    level="watch",
                    category="variability",
                    subject=row.antibody,
                    headline=f"Within-chip spot uniformity is worsening for {row.antibody}",
                    detail=(
                        f"Replicate CV within a chip averaged "
                        f"{_fmt(row.within_chip_recent_mean, 1, '%')} recently against "
                        f"{_fmt(row.within_chip_baseline_mean, 1, '%')} earlier "
                        f"({_fmt(within_rise, 1)} points higher). Replicate spots on the same "
                        "chip are drifting apart, which usually points at the spotter or at "
                        "evaporation during spotting rather than at the antibody."
                    ),
                    evidence={
                        "recent_mean": row.within_chip_recent_mean,
                        "baseline_mean": row.within_chip_baseline_mean,
                        "rise_points": within_rise,
                    },
                )
            )
    return flags


def _failure_rate_flags(failure_trend: dict, rules: MonitoringRules) -> list[Flag]:
    if failure_trend.get("n_runs", 0) < rules.min_runs_for_trend:
        return []
    rise = failure_trend.get("rise_points")
    if rise is None or rise < rules.failure_rate_rise_points_flag:
        return []

    p = failure_trend.get("proportion_p")
    significant = p is not None and p <= 0.05
    common = (
        f"The last {failure_trend['n_recent_runs']} runs rejected "
        f"{failure_trend['recent_failed']} of {failure_trend['recent_total']} chips "
        f"({_fmt(failure_trend['recent_mean'], 1, '%')}) against "
        f"{failure_trend['baseline_failed']} of {failure_trend['baseline_total']} "
        f"({_fmt(failure_trend['baseline_mean'], 1, '%')}) before that, a rise of "
        f"{_fmt(rise, 1)} percentage points"
    )

    if significant:
        return [
            Flag(
                level="alert",
                category="failure-rate",
                subject="Process",
                headline="Chip failure rate is rising",
                detail=(
                    f"{common} (Fisher's exact p = {_fmt(p, 3)}). More chips are being "
                    "discarded than earlier in the series. Check the per-antibody failure "
                    "breakdown below to see whether one antibody accounts for it before "
                    "looking at the spotter."
                ),
                evidence=failure_trend,
            )
        ]

    return [
        Flag(
            level="watch",
            category="failure-rate",
            subject="Process",
            headline="Chip failure rate may be drifting up",
            detail=(
                f"{common}, but the difference is not statistically established "
                f"(Fisher's exact p = {_fmt(p, 3)}). With this many chips a swing of this "
                "size can still be chance. Worth watching over the next runs rather than "
                "acting on now."
            ),
            evidence=failure_trend,
        )
    ]


def _outlier_flags(outliers: pd.DataFrame) -> list[Flag]:
    if outliers.empty:
        return []
    flags = []
    for run_id, group in outliers.groupby("spotting_run_id", sort=False):
        items = "; ".join(
            f"{row.subject} {row.metric} {_fmt(row.value, 1)} against a series median of "
            f"{_fmt(row.median, 1)}"
            for row in group.itertuples(index=False)
        )
        flags.append(
            Flag(
                level="watch",
                category="outlier-run",
                subject=str(run_id),
                headline=f"Run {run_id} stands out from the series",
                detail=(
                    f"{items}. A one-off excursion like this usually traces to that day's "
                    "conditions, coating batch or spotter state rather than to a gradual "
                    "process drift. Check the run's report and its recorded conditions."
                ),
                evidence={"findings": group.to_dict("records")},
            )
        )
    return flags


def _batch_flags(
    surface_evidence: pd.DataFrame,
    batch_effects: pd.DataFrame,
    reference_antibody: str | None,
    rules: MonitoringRules,
) -> list[Flag]:
    """Flag a coating batch whose runs came out low across the entire panel.

    The argument is one of elimination. Every chip in a run shares its coating batch, so
    a batch cannot be compared against another batch within a run. But a drop that
    includes the negative control cannot be caused by any antibody, and the runs'
    temperature and humidity are recorded, so those can be ruled in or out explicitly.
    What remains is the coated surface or the spotter.
    """
    if surface_evidence.empty:
        return []

    chip_stats = (
        batch_effects.set_index("coating_batch")
        if not batch_effects.empty and "coating_batch" in batch_effects
        else pd.DataFrame()
    )
    flags = []

    for row in surface_evidence.itertuples(index=False):
        batch = str(row.coating_batch)
        deficit = row.density_deficit_pct
        control_deficit = row.control_deficit_pct
        has_control = bool(reference_antibody) and pd.notna(control_deficit)

        # The negative control is the primary measure of how well the surface took
        # protein, because it is the one spot no antibody trend can move. The panel
        # median is diluted whenever a capture antibody is drifting over the same
        # period, so leading on it would miss real surface problems.
        primary_deficit = control_deficit if has_control else deficit
        if not pd.notna(primary_deficit) or primary_deficit > -8.0:
            continue

        control_agrees = has_control and control_deficit <= -5.0

        if has_control and control_agrees:
            control_note = (
                f"The {reference_antibody} negative control spotted "
                f"{_fmt(abs(control_deficit), 1, '%')} low on these runs. The control has "
                "no biomarker specificity, so no antibody problem can explain it: the "
                "deposited protein itself came out low. (The panel-wide figure of "
                f"{_fmt(abs(deficit), 1, '%')} understates this where a capture antibody "
                "was drifting over the same period.)"
            )
        elif reference_antibody and pd.notna(control_deficit):
            control_note = (
                f"The {reference_antibody} negative control was close to normal on these "
                f"runs ({_fmt(control_deficit, 1, '%')}), so the deposition itself may be "
                "fine and the drop may sit with individual antibodies instead."
            )
        else:
            control_note = (
                "No negative control spot is configured for this panel, so a surface "
                "problem cannot be separated from an antibody problem. Adding an "
                "isotype control to the panel would make this distinguishable."
            )

        # The recorded conditions are the other candidate explanation for a whole-panel
        # drop, so they are checked rather than left implicit.
        hot = pd.notna(row.max_room_temp_c) and row.max_room_temp_c > 24.0
        dry = pd.notna(row.min_room_humidity_pct) and row.min_room_humidity_pct < 35.0
        if hot or dry:
            condition_note = (
                f"Conditions on these runs were not normal (up to "
                f"{_fmt(row.max_room_temp_c, 1)} °C, down to "
                f"{_fmt(row.min_room_humidity_pct, 1, '%')} RH), so the environment is an "
                "equally plausible cause and should be excluded first."
            )
            level = "watch"
        else:
            condition_note = (
                f"Temperature and humidity on these runs were within the normal range "
                f"(up to {_fmt(row.max_room_temp_c, 1)} °C, down to "
                f"{_fmt(row.min_room_humidity_pct, 1, '%')} RH), so the conditions do not "
                "explain it."
            )
            level = "alert" if control_agrees else "watch"

        failure_note = ""
        if batch in chip_stats.index:
            failure_note = (
                f" Chips on this batch failed at "
                f"{_fmt(chip_stats.loc[batch, 'failure_rate_pct'], 1, '%')}."
            )

        flags.append(
            Flag(
                level=level,
                category="coating-batch",
                subject=batch,
                headline=f"Coating batch {batch} spotted low across the whole panel",
                detail=(
                    f"On run(s) {row.runs}, coated in batch {batch}, spot height sat "
                    f"{_fmt(abs(primary_deficit), 1, '%')} below normal, with a "
                    f"median within-chip CV of {_fmt(row.median_within_chip_cv, 1, '%')}."
                    f"{failure_note} {control_note} {condition_note} Quarantine remaining "
                    "chips from this coating batch and review its coating record before "
                    "spotting more."
                ),
                evidence={
                    "density_deficit_pct": deficit,
                    "control_deficit_pct": control_deficit,
                    "n_runs": row.n_runs,
                    "runs": row.runs,
                    "max_room_temp_c": row.max_room_temp_c,
                    "min_room_humidity_pct": row.min_room_humidity_pct,
                },
            )
        )
    return flags


def _correlation_flags(correlations: pd.DataFrame) -> list[Flag]:
    if correlations.empty:
        return []
    notable = correlations[correlations["notable"]]
    if notable.empty:
        return []

    flags = []
    for (driver, metric), group in notable.groupby(["driver", "metric"], sort=False):
        antibodies = ", ".join(sorted(group["antibody"].unique()))
        strongest = group.reindex(group["r"].abs().sort_values(ascending=False).index).iloc[0]
        direction = "rises" if strongest["r"] > 0 else "falls"
        confound = ""
        if abs(strongest.get("driver_vs_time_r", 0) or 0) >= 0.6:
            confound = (
                " Note that this driver also tracks time across the series "
                f"(r = {strongest['driver_vs_time_r']:.2f}), so the association may reflect "
                "anything else that changed over the same period."
            )
        flags.append(
            Flag(
                level="info",
                category="environment",
                subject=str(strongest["driver_label"]),
                headline=(
                    f"{strongest['metric_label']} {direction} with "
                    f"{strongest['driver_label'].lower()}"
                ),
                detail=(
                    f"Across {int(strongest['n'])} runs, {strongest['metric_label_inline']} "
                    f"correlates with {strongest['driver_label'].lower()} for {antibodies} "
                    f"(strongest r = {strongest['r']:.2f}, p = {strongest['p']:.3f}). "
                    "With this few runs a correlation is a lead, not proof, but it points at "
                    "where to look: control the condition for the next run and see whether "
                    f"the metric follows.{confound}"
                ),
                evidence=group.to_dict("records"),
            )
        )
    return flags


# --------------------------------------------------------------------------------
# One-call analysis
# --------------------------------------------------------------------------------


@dataclass
class PanelAnalysis:
    """Everything the Trends page and the report need for one panel."""

    panel: str
    n_runs: int
    runs: pd.DataFrame
    summary: pd.DataFrame
    density_trends: pd.DataFrame
    variability: pd.DataFrame
    failure_trend: dict
    correlations: pd.DataFrame
    lots: pd.DataFrame
    lot_age: pd.DataFrame
    coating_batches: pd.DataFrame
    wafers: pd.DataFrame
    operators: pd.DataFrame
    batch_contrast: pd.DataFrame
    outliers: pd.DataFrame
    flags: list[Flag]
    reference_antibody: str | None = None
    run_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    surface_evidence: pd.DataFrame = field(default_factory=pd.DataFrame)


def analyze_panel(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
    rules: MonitoringRules,
    panel: str = "",
    reference_antibody: str | None = None,
) -> PanelAnalysis:
    """Run every Part 2 analysis over one panel's accumulated history."""
    density_trends = antibody_density_trends(summary, rules, reference_antibody)
    variability = variability_trends(summary, rules)
    failure_trend = failure_rate_trend(runs, rules)
    correlations = environment_correlations(summary, rules)
    lots = lot_comparison(summary)
    lot_age = lot_age_relationship(summary)
    coating_batches = group_effects(measurements, "coating_batch")
    wafers = group_effects(measurements, "wafer_id")
    operators = group_effects(measurements, "operator")
    batch_contrast = within_run_group_contrast(measurements, "coating_batch")
    outliers = outlier_runs(summary, runs.reset_index(drop=True))

    run_panel = run_panel_effects(summary, reference_antibody)
    surface_evidence = coating_batch_surface_evidence(
        run_panel, run_group_map(measurements, "coating_batch")
    )

    flags = build_flags(
        density_trends=density_trends,
        variability=variability,
        failure_trend=failure_trend,
        correlations=correlations,
        batch_effects=coating_batches,
        outliers=outliers,
        lot_age=lot_age,
        rules=rules,
        reference_antibody=reference_antibody,
        surface_evidence=surface_evidence,
    )

    return PanelAnalysis(
        panel=panel,
        n_runs=int(len(runs)),
        runs=runs,
        summary=summary,
        density_trends=density_trends,
        variability=variability,
        failure_trend=failure_trend,
        correlations=correlations,
        lots=lots,
        lot_age=lot_age,
        coating_batches=coating_batches,
        wafers=wafers,
        operators=operators,
        batch_contrast=batch_contrast,
        outliers=outliers,
        flags=flags,
        reference_antibody=reference_antibody,
        run_panel=run_panel,
        surface_evidence=surface_evidence,
    )

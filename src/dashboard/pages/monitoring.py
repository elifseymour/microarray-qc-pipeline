"""Monitoring page (Part 2): what has been happening across runs.

Ordered the way a scientist reads it: what needs attention, then the trend that shows
it, then the drivers that might explain it. The alerts come first because the charts
are there to check a finding, not to be scanned for one.
"""

from __future__ import annotations

import dash
import pandas as pd
from dash import Input, Output, callback, html

from src.config import get_config
from src.dashboard import components as ui
from src.dashboard import figures as fig_builders
from src.dashboard.theme import get_theme
from src.db import queries as queries_module
from src.qc.part2_trend_analysis import (
    CHIP_TO_CHIP_CV,
    DENSITY_METRIC,
    WITHIN_CHIP_CV,
    analyze_panel,
)

dash.register_page(__name__, path="/", name="Monitoring", title="Microarray Spotting QC · Monitoring")


def layout(**_kwargs):
    return html.Div(id="monitoring-body", className="page")


@callback(
    Output("monitoring-body", "children"),
    Input("panel-store", "data"),
    Input("theme-store", "data"),
    Input("data-version", "data"),
)
def render(panel, theme_name, _version):
    theme = get_theme(theme_name)
    config = get_config()

    if not panel:
        return _empty("No panel selected.", "Add a panel to config/qc_config.yaml.")

    runs = queries_module.load_runs(panel)
    if runs.empty:
        return _empty(
            f"No runs ingested for {panel} yet.",
            "Upload a run's density CSV and its ELN metadata export on the "
            "“Add a run” page to start building the history.",
        )

    summary = queries_module.load_run_antibody_summary(panel)
    measurements = queries_module.load_measurements(panel)
    antibodies = config.antibodies(panel)
    reference = config.reference_antibody(panel)
    thresholds = config.resolve_thresholds(panel)

    analysis = analyze_panel(
        runs, summary, measurements, config.monitoring, panel, reference
    )

    return html.Div(
        [
            html.H1(f"{panel} process monitoring", className="page-title"),
            html.P(
                f"{len(runs)} run(s) between {runs['run_date'].min():%d %b %Y} and "
                f"{runs['run_date'].max():%d %b %Y}. "
                "Findings below are leads to investigate: each one states the size of the "
                "effect and how many runs it rests on, and with a handful of runs a "
                "correlation is never proof on its own.",
                className="page-subtitle",
            ),
            _headline_tiles(runs, analysis, reference),
            ui.section(
                "What needs attention",
                None,
                [
                    ui.flag_list(
                        analysis.flags,
                        "Nothing stands out across the runs ingested so far.",
                    )
                ],
            ),
            _trend_section(summary, antibodies, reference, theme, thresholds, analysis),
            _variability_section(summary, antibodies, reference, theme, thresholds, runs),
            _attribution_section(analysis, reference, theme),
            _driver_section(summary, antibodies, reference, theme, analysis),
        ]
    )


def _empty(title: str, message: str):
    return html.Div(
        [html.H1(title, className="page-title"), html.P(message, className="page-subtitle")]
    )


def _headline_tiles(runs, analysis, reference):
    latest = runs.iloc[-1]
    failure = analysis.failure_trend
    alerts = [f for f in analysis.flags if f.level == "alert"]

    rise = failure.get("rise_points")
    if rise is None:
        failure_sub = "not enough runs to compare"
    elif rise > 0:
        failure_sub = f"{rise:+.1f} points vs. earlier runs"
    else:
        failure_sub = f"{rise:.1f} points vs. earlier runs"

    return ui.tile_row(
        [
            ui.stat_tile("Runs ingested", f"{len(runs)}", f"latest {latest['spotting_run_id']}"),
            ui.stat_tile(
                "Failure rate, recent runs",
                f"{failure.get('recent_mean', 0):.0f}%",
                failure_sub,
                tone="critical" if (rise or 0) >= 10 else None,
            ),
            ui.stat_tile(
                "Latest run",
                f"{latest['failure_rate_pct']:.0f}%",
                f"{int(latest['n_chips_failed'])} of {int(latest['n_chips'])} chips failed",
                tone="critical" if latest["failure_rate_pct"] >= 25 else "good",
            ),
            ui.stat_tile(
                "Findings needing action",
                f"{len(alerts)}",
                f"{len(analysis.flags)} findings in total",
                tone="critical" if alerts else "good",
            ),
        ]
    )


def _trend_section(summary, antibodies, reference, theme, thresholds, analysis):
    table = analysis.density_trends[
        [
            "antibody",
            "n_runs",
            "first_density",
            "latest_density",
            "current_lot",
            "lot_step_pct",
            "within_lot_pct_change",
            "within_lot_p",
            "attribution",
        ]
    ]
    return ui.section(
        "Spot height over time",
        "Each antibody's mean across the chips of a run. Lines break at a lot change and "
        "the marker shape changes with it, because a step at a lot boundary and a slide "
        "within one lot mean different things: the first is a lot of different potency, "
        "the second is stock losing activity.",
        [
            ui.chart_panel(
                fig_builders.density_trend(
                    summary, antibodies, reference, theme, thresholds.density_min_ng_mm2
                ),
                table,
                theme,
                note=(
                    "The attribution column is decided from absolute spot height. "
                    "Where a lot changed mid-series, the trend is fitted within the lot "
                    "in use now rather than straight through the step."
                ),
                numeric_format={
                    "first_density": 2,
                    "latest_density": 2,
                    "lot_step_pct": 1,
                    "within_lot_pct_change": 1,
                    "within_lot_p": 3,
                },
                key="density-trend",
            )
        ],
    )


def _variability_section(summary, antibodies, reference, theme, thresholds, runs):
    cv_table = summary[
        [
            "spotting_run_id",
            "antibody",
            CHIP_TO_CHIP_CV,
            WITHIN_CHIP_CV,
            "n_chips_failed",
        ]
    ]
    return ui.section(
        "Variability and failure rate",
        "Chip-to-chip CV asks whether the chips in a batch resemble each other. "
        "Within-chip CV asks whether the replicate spots on one chip do, and it is the "
        "one with a QC limit. They fail for different reasons: the first points at the "
        "coated surface or a drifting spotter, the second at the spotter or evaporation "
        "during spotting.",
        [
            html.Div(
                [
                    ui.chart_panel(
                        fig_builders.variability_trend(
                            summary, antibodies, reference, theme, CHIP_TO_CHIP_CV
                        ),
                        None,
                        theme,
                        key="cc-cv",
                    ),
                    ui.chart_panel(
                        fig_builders.variability_trend(
                            summary,
                            antibodies,
                            reference,
                            theme,
                            WITHIN_CHIP_CV,
                            thresholds.cv_max_pct,
                        ),
                        cv_table,
                        theme,
                        numeric_format={CHIP_TO_CHIP_CV: 1, WITHIN_CHIP_CV: 1},
                        key="wc-cv",
                    ),
                ],
                className="grid-2",
            ),
            ui.chart_panel(
                fig_builders.failure_rate_trend(runs, theme),
                runs[
                    [
                        "spotting_run_id",
                        "run_date",
                        "n_chips",
                        "n_chips_failed",
                        "failure_rate_pct",
                        "operator",
                    ]
                ],
                theme,
                numeric_format={"failure_rate_pct": 1},
                key="failure-rate",
            ),
        ],
    )


def _attribution_section(analysis, reference, theme):
    """The surface-versus-antibody question, which is what the control spot answers."""
    children = [
        ui.chart_panel(
            fig_builders.control_vs_panel(analysis.run_panel, reference, theme),
            analysis.run_panel[
                [
                    "spotting_run_id",
                    "median_relative_density",
                    "control_relative_density",
                    "n_antibodies_low",
                    "median_within_chip_cv",
                ]
            ]
            if not analysis.run_panel.empty
            else None,
            theme,
            note=(
                "Both series are relative to their own norm across the series, so 1.00 is "
                "normal. The two moving together means the deposition under-performed; "
                "only the capture antibodies moving means the cause sits with an antibody."
            ),
            numeric_format={
                "median_relative_density": 2,
                "control_relative_density": 2,
                "median_within_chip_cv": 1,
            },
            key="control-vs-panel",
        )
    ]

    if not analysis.surface_evidence.empty:
        children.append(
            ui.chart_panel(
                fig_builders.group_effect_bars(
                    analysis.coating_batches, "coating_batch", "Coating batch", theme
                ),
                analysis.surface_evidence[
                    [
                        "coating_batch",
                        "n_runs",
                        "runs",
                        "density_deficit_pct",
                        "control_deficit_pct",
                        "median_within_chip_cv",
                        "max_room_temp_c",
                        "min_room_humidity_pct",
                    ]
                ],
                theme,
                note=(
                    "Every chip in a run shares one coating batch, so a batch cannot be "
                    "compared against another batch inside the same run and its bar is "
                    "drawn in neutral grey to say so. The control deficit column in the "
                    "table is the reliable signal: no antibody trend can move the control "
                    "spot, so a batch that pulls it down implicates the surface."
                ),
                numeric_format={
                    "density_deficit_pct": 1,
                    "control_deficit_pct": 1,
                    "median_within_chip_cv": 1,
                    "max_room_temp_c": 1,
                    "min_room_humidity_pct": 1,
                },
                key="coating-batches",
            )
        )

    return ui.section(
        "Is it the surface, the spotter, or the antibody?",
        "The negative control spot is deposited like any other antibody but binds none of "
        "the biomarkers, so it measures the spotting process on its own. It is what makes "
        "these two causes separable at all."
        if reference
        else "No negative control antibody is configured for this panel, so a surface "
        "problem cannot be told apart from an antibody problem. Adding an isotype control "
        "to the panel would make it separable.",
        children,
    )


def _driver_section(summary, antibodies, reference, theme, analysis):
    notable = (
        analysis.correlations[analysis.correlations["notable"]]
        if not analysis.correlations.empty
        else pd.DataFrame()
    )
    driver, driver_label = "room_temp_c", "Room temperature (°C)"
    metric, metric_label = WITHIN_CHIP_CV, "Within-chip CV (%)"
    if not notable.empty:
        strongest = notable.reindex(
            notable["r"].abs().sort_values(ascending=False).index
        ).iloc[0]
        driver, driver_label = strongest["driver"], strongest["driver_label"]
        metric, metric_label = strongest["metric"], strongest["metric_label"]

    correlation_table = (
        analysis.correlations[
            ["antibody", "driver_label", "metric_label", "n", "r", "p", "driver_vs_time_r"]
        ]
        if not analysis.correlations.empty
        else None
    )

    return ui.section(
        "Conditions, stock age and other variables",
        "Correlations across a run series are easy to over-read: anything that drifts with "
        "time correlates with anything else that does. The table reports how strongly each "
        "condition itself tracks the calendar, so a coincidence is visible rather than "
        "hidden.",
        [
            ui.chart_panel(
                fig_builders.environment_facets(
                    summary,
                    antibodies,
                    reference,
                    driver,
                    driver_label,
                    metric,
                    metric_label,
                    theme,
                    analysis.correlations,
                ),
                correlation_table,
                theme,
                note=(
                    "One panel per antibody rather than four colored series in one frame: "
                    "on a scatter, any two series can end up side by side, and beyond three "
                    "hues they stop being reliably distinguishable."
                ),
                numeric_format={"r": 2, "p": 3, "driver_vs_time_r": 2},
                key="environment",
            ),
            ui.chart_panel(
                fig_builders.lot_age_facets(
                    summary, antibodies, reference, theme, analysis.lot_age
                ),
                analysis.lots[
                    [
                        "antibody",
                        "ab_lot",
                        "n_runs",
                        "mean_density",
                        "mean_chip_to_chip_cv",
                        "failure_rate_pct",
                        "step_vs_previous_lot_pct",
                        "min_lot_age_days",
                        "max_lot_age_days",
                    ]
                ]
                if not analysis.lots.empty
                else None,
                theme,
                note=(
                    "Points are shaped by lot. A slope built from two lots of different "
                    "potency would say nothing about degradation, so only the within-lot "
                    "spread is evidence of a stock ageing."
                ),
                numeric_format={
                    "mean_density": 2,
                    "mean_chip_to_chip_cv": 1,
                    "failure_rate_pct": 1,
                    "step_vs_previous_lot_pct": 1,
                },
                key="lot-age",
            ),
            html.Div(
                [
                    ui.chart_panel(
                        fig_builders.group_effect_bars(
                            analysis.wafers, "wafer_id", "Wafer lot", theme
                        ),
                        None,
                        theme,
                        note=(
                            "Grey bars cannot be separated from the runs they were used "
                            "in. With one wafer lot per run, a difference here may simply "
                            "be whatever else changed over those weeks."
                        ),
                        key="wafers",
                    ),
                    ui.chart_panel(
                        fig_builders.group_effect_bars(
                            analysis.operators, "operator", "Operator", theme
                        ),
                        None,
                        theme,
                        key="operators",
                    ),
                ],
                className="grid-2",
            ),
        ],
    )

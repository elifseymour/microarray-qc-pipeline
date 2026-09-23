"""Run report page (Part 1): the pass/fail record for one spotting run.

This is the page a scientist opens right after a run: which chips can go downstream,
which cannot, and why. The report can be downloaded as a self-contained file for the
run's record.
"""

from __future__ import annotations

import datetime as dt

import dash
from dash import Input, Output, State, callback, dcc, html, no_update

from src.config import get_config
from src.dashboard import components as ui
from src.dashboard import figures as fig_builders
from src.dashboard.theme import get_theme
from src.db import queries as queries_module
from src.reports.report_generator import build_run_report_html

dash.register_page(
    __name__, path="/runs", name="Run reports", title="Microarray Spotting QC · Run reports"
)


def layout(**_kwargs):
    return html.Div(
        [
            html.H1("Run QC report", className="page-title"),
            html.P(
                "Chips are released or rejected here. A chip fails if any of its "
                "antibodies spots below the density limit or above the CV limit.",
                className="page-subtitle",
            ),
            html.Div(
                [
                    html.Label("Spotting run", htmlFor="run-select"),
                    dcc.Dropdown(id="run-select", clearable=False, style={"width": "280px"}),
                    html.Button(
                        "Download report",
                        id="download-report",
                        className="btn btn-secondary",
                        n_clicks=0,
                    ),
                    dcc.Download(id="report-download"),
                ],
                className="header-controls",
                style={"marginBottom": "8px"},
            ),
            html.Div(id="run-report-body"),
        ],
        className="page",
    )


@callback(
    Output("run-select", "options"),
    Output("run-select", "value"),
    Input("panel-store", "data"),
    Input("data-version", "data"),
    State("run-select", "value"),
)
def _populate_runs(panel, _version, current):
    if not panel:
        return [], None
    runs = queries_module.load_runs(panel)
    if runs.empty:
        return [], None
    # Newest first: the run just spotted is the one being looked at.
    ordered = runs.sort_values("run_date", ascending=False)
    options = [
        {
            "label": f"{row.spotting_run_id}  ·  {row.run_date:%d %b %Y}"
            f"  ·  {row.failure_rate_pct:.0f}% failed",
            "value": row.spotting_run_id,
        }
        for row in ordered.itertuples(index=False)
    ]
    values = {o["value"] for o in options}
    return options, (current if current in values else options[0]["value"])


@callback(
    Output("run-report-body", "children"),
    Input("run-select", "value"),
    Input("theme-store", "data"),
    Input("data-version", "data"),
    State("panel-store", "data"),
)
def _render(run_id, theme_name, _version, panel):
    if not run_id:
        return ui.note("No runs have been ingested for this panel yet.")

    theme = get_theme(theme_name)
    config = get_config()
    run = queries_module.get_run(run_id)
    if run is None:
        return ui.note("That run is no longer in the database.")

    panel = run["panel"]
    antibodies = config.antibodies(panel)
    reference = config.reference_antibody(panel)
    thresholds = config.resolve_thresholds(panel)

    measurements = queries_module.load_measurements(panel=panel, spotting_run_id=run_id)
    chips = queries_module.load_chips(panel=panel, spotting_run_id=run_id)
    lots = queries_module.load_lots(panel=panel)
    lots = lots[lots["spotting_run_id"] == run_id] if not lots.empty else lots
    summary = queries_module.load_run_antibody_summary(panel=panel)
    run_summary = summary[summary["spotting_run_id"] == run_id]

    failed = int(run["n_chips_failed"])
    tone = "good" if failed == 0 else ("critical" if run["failure_rate_pct"] >= 25 else None)

    return html.Div(
        [
            ui.tile_row(
                [
                    ui.stat_tile(
                        "Chips released",
                        f"{int(run['n_chips']) - failed}",
                        f"of {int(run['n_chips'])} spotted",
                        tone="good" if failed == 0 else None,
                    ),
                    ui.stat_tile(
                        "Chips failed",
                        f"{failed}",
                        f"{run['failure_rate_pct']:.1f}% of the run",
                        tone=tone,
                    ),
                    ui.stat_tile(
                        "QC limits",
                        f"{run['density_min_ng_mm2']:g} ng/mm²",
                        f"minimum density · {run['cv_max_pct']:g}% CV maximum",
                    ),
                    ui.stat_tile(
                        "Conditions",
                        f"{run['room_temp_c']:.1f} °C"
                        if run["room_temp_c"] is not None
                        else "not recorded",
                        f"{run['room_humidity_pct']:.0f}% RH in the room"
                        if run["room_humidity_pct"] is not None
                        else "",
                    ),
                ]
            ),
            ui.section(
                "Run record",
                None,
                [
                    html.Div(
                        ui.data_table(
                            _run_facts(run, chips, lots),
                            theme,
                            page_size=12,
                        ),
                        className="card",
                        style={"padding": "12px"},
                    )
                ],
            ),
            ui.section(
                "Chip results",
                "Every chip, with the antibody and criterion behind each rejection.",
                [
                    html.Div(
                        ui.data_table(
                            chips[
                                [
                                    "chip_id",
                                    "status",
                                    "n_antibodies_failed",
                                    "wafer_id",
                                    "coating_batch",
                                    "fail_reason_summary",
                                ]
                            ],
                            theme,
                            page_size=20,
                            highlight_fail_column="status",
                        ),
                        className="card",
                        style={"padding": "12px"},
                    )
                ],
            ),
            ui.section(
                "Distributions across the run's chips",
                "One point per chip. Points in red are the ones that failed that "
                "criterion, so a single outlier is immediately distinguishable from a "
                "whole batch sitting low.",
                [
                    html.Div(
                        [
                            ui.chart_panel(
                                fig_builders.run_distribution(
                                    measurements,
                                    antibodies,
                                    reference,
                                    theme,
                                    "mean_density",
                                    limit=thresholds.density_min_ng_mm2,
                                ),
                                run_summary[
                                    [
                                        "antibody",
                                        "pooled_mean_density",
                                        "chip_to_chip_cv_pct",
                                        "mean_within_chip_cv_pct",
                                        "n_chips_failed",
                                        "n_failed_low_density",
                                        "n_failed_high_cv",
                                    ]
                                ],
                                theme,
                                numeric_format={
                                    "pooled_mean_density": 2,
                                    "chip_to_chip_cv_pct": 1,
                                    "mean_within_chip_cv_pct": 1,
                                },
                                key="run-density",
                            ),
                            ui.chart_panel(
                                fig_builders.run_distribution(
                                    measurements,
                                    antibodies,
                                    reference,
                                    theme,
                                    "cv_pct",
                                    limit=thresholds.cv_max_pct,
                                    limit_is_maximum=True,
                                ),
                                None,
                                theme,
                                key="run-cv",
                            ),
                        ],
                        className="grid-2",
                    ),
                    ui.chart_panel(
                        fig_builders.chip_density_heatmap(measurements, antibodies, theme),
                        measurements[
                            ["chip_id", "antibody", "mean_density", "cv_pct", "status"]
                        ],
                        theme,
                        note=(
                            "A low row is one bad chip; a low column is one antibody across "
                            "the batch. The table carries the numbers, since a color scale "
                            "should never be the only way to read a value."
                        ),
                        numeric_format={"mean_density": 2, "cv_pct": 1},
                        key="run-heatmap",
                    ),
                ],
            ),
        ]
    )


def _run_facts(run, chips, lots):
    """The run's provenance as a two-column table: what was used, and under what."""
    import pandas as pd

    rows = [
        ("Spotting run", run["spotting_run_id"]),
        ("Panel", run["panel"]),
        ("Run date", f"{run['run_date']:%d %b %Y}"),
        ("Operator", run["operator"] or "not recorded"),
        (
            "Coating batch",
            ", ".join(sorted(chips["coating_batch"].dropna().unique())) or "not recorded",
        ),
        (
            "Wafer lot",
            ", ".join(sorted(chips["wafer_id"].dropna().unique())) or "not recorded",
        ),
        ("Room temperature", _fmt(run["room_temp_c"], " °C")),
        ("Room humidity", _fmt(run["room_humidity_pct"], "%")),
        ("Spotter temperature", _fmt(run["spotter_temp_c"], " °C")),
        ("Spotter humidity", _fmt(run["spotter_humidity_pct"], "%")),
    ]
    for lot in lots.itertuples(index=False):
        age = f" · {int(lot.lot_age_days)} days old" if lot.lot_age_days is not None else ""
        rows.append((f"{lot.antibody} lot", f"{lot.ab_lot or 'not recorded'}{age}"))
    rows.append(("Source files", run.get("source_density_file") or "uploaded"))
    return pd.DataFrame(rows, columns=["field", "value"])


def _fmt(value, suffix=""):
    return "not recorded" if value is None else f"{value:.1f}{suffix}"


@callback(
    Output("report-download", "data"),
    Input("download-report", "n_clicks"),
    State("run-select", "value"),
    prevent_initial_call=True,
)
def _download(n_clicks, run_id):
    if not n_clicks or not run_id:
        return no_update
    # Light theme and an embedded plotting library: the file is a record, so it has to
    # print cleanly and open later without a network connection.
    html_text = build_run_report_html(run_id, theme_name="light", embed_plotly=True)
    stamp = dt.date.today().isoformat()
    return dict(content=html_text, filename=f"QC_report_{run_id}_{stamp}.html")

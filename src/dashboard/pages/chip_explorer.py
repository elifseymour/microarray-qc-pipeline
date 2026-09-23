"""Chip explorer: trace a flagged chip down to its individual spots.

The monitoring page answers "is something wrong across runs" and the run report answers
"which chips do I release". This page answers the third question, which is the one asked
while troubleshooting: what actually happened on this chip. A high CV caused by one
starved spot points at a nozzle; six uniformly low spots point at the antibody or the
surface, and only the raw spots distinguish them.
"""

from __future__ import annotations

import dash
from dash import Input, Output, State, callback, dcc, html

from src.config import get_config
from src.dashboard import components as ui
from src.dashboard import figures as fig_builders
from src.dashboard.theme import get_theme
from src.db import queries as queries_module

dash.register_page(
    __name__, path="/chips", name="Chip explorer", title="Microarray Spotting QC · Chip explorer"
)


def layout(**_kwargs):
    return html.Div(
        [
            html.H1("Chip explorer", className="page-title"),
            html.P(
                "Look at one chip's replicate spots to see why it passed or failed.",
                className="page-subtitle",
            ),
            html.Div(
                [
                    html.Label("Run", htmlFor="explorer-run"),
                    dcc.Dropdown(
                        id="explorer-run", clearable=False, style={"width": "250px"}
                    ),
                    html.Label("Chip", htmlFor="explorer-chip"),
                    dcc.Dropdown(
                        id="explorer-chip", clearable=False, style={"width": "220px"}
                    ),
                    dcc.Checklist(
                        id="failed-only",
                        options=[{"label": "  Failed chips only", "value": "failed"}],
                        value=[],
                        style={"fontSize": "14px", "color": "var(--text-secondary)"},
                    ),
                ],
                className="header-controls",
            ),
            html.Div(id="explorer-body"),
        ],
        className="page",
    )


@callback(
    Output("explorer-run", "options"),
    Output("explorer-run", "value"),
    Input("panel-store", "data"),
    Input("data-version", "data"),
    State("explorer-run", "value"),
)
def _runs(panel, _version, current):
    if not panel:
        return [], None
    runs = queries_module.load_runs(panel)
    if runs.empty:
        return [], None
    ordered = runs.sort_values("run_date", ascending=False)
    options = [
        {"label": f"{row.spotting_run_id}  ·  {row.run_date:%d %b %Y}", "value": row.spotting_run_id}
        for row in ordered.itertuples(index=False)
    ]
    values = {o["value"] for o in options}
    return options, (current if current in values else options[0]["value"])


@callback(
    Output("explorer-chip", "options"),
    Output("explorer-chip", "value"),
    Input("explorer-run", "value"),
    Input("failed-only", "value"),
    State("explorer-chip", "value"),
)
def _chips(run_id, failed_only, current):
    if not run_id:
        return [], None
    chips = queries_module.load_chips(spotting_run_id=run_id)
    if chips.empty:
        return [], None
    if "failed" in (failed_only or []):
        chips = chips[chips["status"] == "FAIL"]
        if chips.empty:
            return [], None
    options = [
        {
            "label": f"{row.chip_id}  ·  {row.status}"
            + (f"  ·  {row.fail_reason_summary}" if row.fail_reason_summary else ""),
            "value": row.chip_id,
        }
        for row in chips.itertuples(index=False)
    ]
    values = {o["value"] for o in options}
    return options, (current if current in values else options[0]["value"])


@callback(
    Output("explorer-body", "children"),
    Input("explorer-run", "value"),
    Input("explorer-chip", "value"),
    Input("theme-store", "data"),
)
def _render(run_id, chip_id, theme_name):
    if not run_id or not chip_id:
        return ui.note("Select a run and a chip.")

    theme = get_theme(theme_name)
    config = get_config()
    run = queries_module.get_run(run_id)
    if run is None:
        return ui.note("That run is no longer in the database.")

    panel = run["panel"]
    antibodies = config.antibodies(panel)
    reference = config.reference_antibody(panel)
    thresholds = config.resolve_thresholds(panel)

    replicates = queries_module.load_replicates(run_id, chip_id=chip_id)
    measurements = queries_module.load_measurements(panel=panel, spotting_run_id=run_id)
    chip_measurements = measurements[measurements["chip_id"] == chip_id]
    chips = queries_module.load_chips(spotting_run_id=run_id)
    chip = chips[chips["chip_id"] == chip_id].iloc[0]

    return html.Div(
        [
            ui.tile_row(
                [
                    ui.stat_tile(
                        "Chip verdict",
                        chip["status"],
                        chip["fail_reason_summary"] or "all antibodies within limits",
                        tone="critical" if chip["status"] == "FAIL" else "good",
                    ),
                    ui.stat_tile(
                        "Coating batch", chip["coating_batch"] or "not recorded", ""
                    ),
                    ui.stat_tile("Wafer lot", chip["wafer_id"] or "not recorded", ""),
                    ui.stat_tile(
                        "Run", run_id, f"{run['run_date']:%d %b %Y} · {panel}"
                    ),
                ]
            ),
            ui.section(
                "This chip's antibodies",
                "The mean and CV each verdict was made on, next to the limits that "
                "applied.",
                [
                    html.Div(
                        ui.data_table(
                            chip_measurements[
                                [
                                    "antibody",
                                    "n_replicates",
                                    "mean_density",
                                    "sd_density",
                                    "cv_pct",
                                    "min_density",
                                    "max_density",
                                    "status",
                                    "fail_reason",
                                ]
                            ],
                            theme,
                            page_size=8,
                            numeric_format={
                                "mean_density": 2,
                                "sd_density": 3,
                                "cv_pct": 1,
                                "min_density": 2,
                                "max_density": 2,
                            },
                            highlight_fail_column="status",
                        ),
                        className="card",
                        style={"padding": "12px"},
                    )
                ],
            ),
            ui.section(
                "Individual spots",
                "One high CV caused by a single starved spot usually points at the "
                "spotter nozzle. A whole group sitting low points at the antibody or the "
                "coated surface instead.",
                [
                    ui.chart_panel(
                        fig_builders.replicate_strip(
                            replicates,
                            antibodies,
                            reference,
                            theme,
                            thresholds.density_min_ng_mm2,
                        ),
                        replicates[
                            ["antibody", "replicate_num", "density_ng_mm2"]
                        ],
                        theme,
                        numeric_format={"density_ng_mm2": 3},
                        key="replicates",
                    )
                ],
            ),
            ui.section(
                "How this chip compares within its run",
                "The same chip marked against every other chip spotted that day.",
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
                        None,
                        theme,
                        key="explorer-run-dist",
                    )
                ],
            ),
        ]
    )

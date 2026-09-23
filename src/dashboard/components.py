"""Reusable UI pieces.

Two conventions are enforced here rather than left to each page:

  * Every chart ships with a table view in a tab beside it. A value should never be
    reachable only by hovering, and no distinction should rest on color alone.
  * Every alert states its evidence and what to do, not just that something happened.
"""

from __future__ import annotations

import pandas as pd
from dash import dash_table, dcc, html

from src.dashboard.theme import STATUS, Theme

LEVEL_META = {
    "alert": {"color": STATUS["critical"], "label": "Action", "icon": "●"},
    "watch": {"color": STATUS["warning"], "label": "Watch", "icon": "◐"},
    "info": {"color": STATUS["serious"], "label": "Context", "icon": "○"},
}


def section(title: str, note: str | None = None, children: list | None = None) -> html.Div:
    parts: list = [html.H2(title, className="section-title")]
    if note:
        parts.append(html.P(note, className="section-note"))
    parts.extend(children or [])
    return html.Div(parts, className="section")


def stat_tile(label: str, value: str, sub: str | None = None, tone: str | None = None):
    """A single headline number. Tone is a status color, used with its own words."""
    style = {"color": STATUS[tone]} if tone in STATUS else {}
    return html.Div(
        [
            html.Div(label, className="tile-label"),
            html.Div(value, className="tile-value", style=style),
            html.Div(sub or "", className="tile-sub"),
        ],
        className="tile",
    )


def tile_row(tiles: list) -> html.Div:
    return html.Div(tiles, className="tile-row")


def data_table(
    frame: pd.DataFrame,
    theme: Theme,
    table_id: str | None = None,
    page_size: int = 12,
    numeric_format: dict[str, int] | None = None,
    highlight_fail_column: str | None = None,
):
    """A compact, sortable table view of whatever a chart is showing."""
    if frame is None or frame.empty:
        return html.P("No data yet.", className="section-note")

    display = frame.copy()
    for column, digits in (numeric_format or {}).items():
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(digits)
    for column in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[column]):
            display[column] = display[column].dt.strftime("%Y-%m-%d")

    conditional = [
        {
            "if": {"row_index": "odd"},
            "backgroundColor": theme.surface,
        }
    ]
    if highlight_fail_column and highlight_fail_column in display:
        conditional.append(
            {
                "if": {
                    "filter_query": f"{{{highlight_fail_column}}} = 'FAIL'",
                    "column_id": highlight_fail_column,
                },
                "color": STATUS["critical"],
                "fontWeight": "600",
            }
        )

    return dash_table.DataTable(
        id=table_id or f"table-{id(frame)}",
        data=display.to_dict("records"),
        columns=[
            {"name": c.replace("_", " "), "id": c, "type": "numeric"}
            if pd.api.types.is_numeric_dtype(display[c])
            else {"name": c.replace("_", " "), "id": c}
            for c in display.columns
        ],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": theme.surface_raised,
            "color": theme.text_muted,
            "fontWeight": "600",
            "fontSize": "12px",
            "textTransform": "uppercase",
            "letterSpacing": "0.04em",
            "border": "none",
            "borderBottom": f"1px solid {theme.border}",
        },
        style_cell={
            "backgroundColor": theme.surface_raised,
            "color": theme.text_primary,
            "fontSize": "13px",
            "padding": "8px 10px",
            "border": "none",
            "borderBottom": f"1px solid {theme.border}",
            "fontFamily": "inherit",
            "textAlign": "left",
            "maxWidth": "320px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_data_conditional=conditional,
        style_filter={
            "backgroundColor": theme.surface,
            "color": theme.text_primary,
            "border": "none",
        },
    )


def chart_panel(
    figure,
    table: pd.DataFrame | None,
    theme: Theme,
    note: str | None = None,
    numeric_format: dict[str, int] | None = None,
    key: str = "",
):
    """A chart with its numbers one click away.

    The table is not an afterthought: several palette steps sit below the contrast a
    small label needs, and some readers cannot separate two hues at all, so the numbers
    have to be available without interpreting the picture.
    """
    tabs = [
        dcc.Tab(
            label="Chart",
            value="chart",
            className="tab",
            selected_className="tab-selected",
            children=[
                dcc.Graph(
                    figure=figure,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                        "doubleClick": "reset",
                    },
                    className="graph",
                )
            ],
        )
    ]
    if table is not None and not table.empty:
        tabs.append(
            dcc.Tab(
                label="Table",
                value="table",
                className="tab",
                selected_className="tab-selected",
                children=[
                    html.Div(
                        data_table(table, theme, numeric_format=numeric_format),
                        className="table-wrap",
                    )
                ],
            )
        )

    children = [
        dcc.Tabs(
            id=f"tabs-{key}" if key else None,
            value="chart",
            children=tabs,
            className="chart-tabs",
        )
    ]
    if note:
        children.append(html.P(note, className="chart-note"))
    return html.Div(children, className="card")


def flag_item(flag) -> html.Div:
    """One finding, with its level, its evidence and what to do about it."""
    meta = LEVEL_META.get(flag.level, LEVEL_META["info"])
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        f"{meta['icon']} {meta['label']}",
                        className="flag-badge",
                        style={"color": meta["color"], "borderColor": meta["color"]},
                    ),
                    html.Span(flag.subject, className="flag-subject"),
                ],
                className="flag-head",
            ),
            html.H3(flag.headline, className="flag-headline"),
            html.P(flag.detail, className="flag-detail"),
        ],
        className="flag",
        style={"borderLeftColor": meta["color"]},
    )


def flag_list(flags: list, empty_message: str) -> html.Div:
    if not flags:
        return html.Div(
            [
                html.Span("● ", style={"color": STATUS["good"]}),
                html.Span(empty_message),
            ],
            className="flag flag-clear",
            style={"borderLeftColor": STATUS["good"]},
        )
    return html.Div([flag_item(f) for f in flags], className="flag-list")


def status_pill(status: str) -> html.Span:
    """Pass/fail with a word attached, never color alone."""
    is_fail = str(status).upper() == "FAIL"
    color = STATUS["critical"] if is_fail else STATUS["good"]
    return html.Span(
        [html.Span("● ", style={"color": color}), status],
        className="status-pill",
    )


def note(text: str) -> html.P:
    return html.P(text, className="section-note")

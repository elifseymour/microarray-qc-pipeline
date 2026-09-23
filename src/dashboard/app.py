"""Dash application entry point.

    python -m src.dashboard.app

The panel selector and the theme choice live here, above the pages, because a filter
belongs in one row at the top rather than inside individual chart cards. Both are kept
in browser storage so a reload does not lose the scientist's context.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dash
from dash import Input, Output, State, callback, dcc, html

from src.config import get_config
from src.db import queries as queries_module
from src.db.session import init_db

PAGES_FOLDER = Path(__file__).parent / "pages"
ASSETS_FOLDER = Path(__file__).parent / "assets"

NAV_ITEMS = [
    ("/", "Monitoring"),
    ("/runs", "Run reports"),
    ("/chips", "Chip explorer"),
    ("/upload", "Add a run"),
]


def available_panel_options() -> tuple[list[dict], str | None]:
    """Panels that have data, falling back to those merely configured."""
    config = get_config()
    ingested = queries_module.available_panels()
    configured = config.panel_names
    names = ingested + [p for p in configured if p not in ingested]
    options = [
        {
            "label": name + ("" if name in ingested else "  (no runs yet)"),
            "value": name,
        }
        for name in names
    ]
    return options, (names[0] if names else None)


def create_app() -> dash.Dash:
    init_db()
    options, default_panel = available_panel_options()

    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder=str(PAGES_FOLDER),
        assets_folder=str(ASSETS_FOLDER),
        title="Microarray Spotting QC",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        [
            dcc.Store(id="panel-store", storage_type="session", data=default_panel),
            dcc.Store(id="theme-store", storage_type="local", data="light"),
            dcc.Store(id="theme-applied"),
            # Bumped whenever a run is ingested or deleted, so every page reloads its
            # data instead of showing a stale view of the database.
            dcc.Store(id="data-version", storage_type="memory", data=0),
            dcc.Location(id="url"),
            html.Header(
                html.Div(
                    [
                        html.Div(
                            [
                                "Microarray Spotting QC",
                                html.Span("Biosensor chip spotting process"),
                            ],
                            className="brand",
                        ),
                        html.Nav(
                            [
                                dcc.Link(label, href=href, id={"type": "nav", "href": href})
                                for href, label in NAV_ITEMS
                            ],
                            className="nav",
                        ),
                        html.Div(
                            [
                                html.Label("Panel", htmlFor="panel-select"),
                                dcc.Dropdown(
                                    id="panel-select",
                                    options=options,
                                    value=default_panel,
                                    clearable=False,
                                    style={"width": "210px"},
                                ),
                                html.Button(
                                    "Dark",
                                    id="theme-toggle",
                                    className="theme-toggle",
                                    n_clicks=0,
                                ),
                            ],
                            className="header-controls",
                        ),
                    ],
                    className="header-inner",
                ),
                className="app-header",
            ),
            dash.page_container,
        ]
    )

    _register_shell_callbacks(app)
    return app


def _register_shell_callbacks(app: dash.Dash) -> None:
    @callback(
        Output("panel-store", "data"),
        Input("panel-select", "value"),
        prevent_initial_call=True,
    )
    def _store_panel(value):
        return value

    @callback(
        Output("theme-store", "data"),
        Output("theme-toggle", "children"),
        Input("theme-toggle", "n_clicks"),
        State("theme-store", "data"),
    )
    def _toggle_theme(n_clicks, current):
        theme = current or "light"
        if n_clicks:
            theme = "dark" if theme == "light" else "light"
        # The button names the mode it switches to, not the one in use.
        return theme, ("Light" if theme == "dark" else "Dark")

    # Applying the theme to the document is the one thing that has to happen in the
    # browser; the figures are re-rendered server-side from the same stored value.
    app.clientside_callback(
        """
        function(theme) {
            document.documentElement.dataset.theme = theme || 'light';
            return theme;
        }
        """,
        Output("theme-applied", "data"),
        Input("theme-store", "data"),
    )

    @callback(
        Output({"type": "nav", "href": dash.ALL}, "className"),
        Input("url", "pathname"),
    )
    def _highlight_nav(pathname):
        current = pathname or "/"
        return ["active" if href == current else "" for href, _ in NAV_ITEMS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Microarray Spotting QC dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    print(f"Microarray Spotting QC dashboard: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


app = None

if __name__ == "__main__":
    main()

"""Add a run: upload the two files, validate, grade, store.

The uploaded files are archived to `data/raw/<run id>/` before anything is written to
the database, so a run's result can always be traced back to the exact files it came
from. Validation refuses anything that would make the stored history misleading, and
says what to fix rather than only that something failed.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import dash
from dash import Input, Output, State, callback, dcc, html, no_update

from src.config import PROJECT_ROOT, get_config
from src.dashboard import components as ui
from src.ingestion.ingest_run import IngestionError, ingest_run
from src.ingestion.parse_benchling_export import MetadataFileError
from src.ingestion.parse_density_csv import DensityFileError

dash.register_page(__name__, path="/upload", name="Add a run", title="Microarray Spotting QC · Add a run")

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def layout(**_kwargs):
    return html.Div(
        [
            html.H1("Add a spotting run", className="page-title"),
            html.P(
                "Two files per run. The density CSV holds one row per chip and replicate "
                "spot with one column per antibody. The ELN export holds one row per chip "
                "with the run ID, date, wafer, coating batch, conditions and the antibody "
                "lot used. The run ID and date are read from the ELN export, so the "
                "notebook stays the single source of truth.",
                className="page-subtitle",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Spot density CSV", className="tile-label"),
                            dcc.Upload(
                                id="upload-density",
                                children=html.Div(
                                    "Drop the density CSV here, or click to choose"
                                ),
                                className="upload-drop",
                                multiple=False,
                                accept=".csv,text/csv",
                            ),
                            html.Div(id="density-filename", className="chart-note"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("ELN run metadata export", className="tile-label"),
                            dcc.Upload(
                                id="upload-metadata",
                                children=html.Div(
                                    "Drop the Benchling export here, or click to choose"
                                ),
                                className="upload-drop",
                                multiple=False,
                                accept=".csv,text/csv",
                            ),
                            html.Div(id="metadata-filename", className="chart-note"),
                        ]
                    ),
                ],
                className="grid-2",
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="replace-existing",
                        options=[
                            {
                                "label": "  Replace this run if it has already been ingested",
                                "value": "replace",
                            }
                        ],
                        value=[],
                        style={"fontSize": "14px", "color": "var(--text-secondary)"},
                    ),
                    html.Button(
                        "Validate and add run",
                        id="ingest-button",
                        className="btn",
                        n_clicks=0,
                        disabled=True,
                    ),
                ],
                className="header-controls",
                style={"marginTop": "22px"},
            ),
            dcc.Loading(html.Div(id="ingest-result"), type="default"),
            ui.section(
                "What is checked before a run is stored",
                None,
                [
                    html.Ul(
                        [
                            html.Li(
                                "Every antibody column in the density file matches the "
                                "selected panel's antibodies, so a file from a different "
                                "panel cannot be filed under this one."
                            ),
                            html.Li(
                                "Every measured chip has a row in the ELN export. Without "
                                "its coating batch and conditions a chip's result could "
                                "never be attributed to a cause, which is the point of the "
                                "monitoring view."
                            ),
                            html.Li(
                                "No blank or non-numeric densities, no negative values and "
                                "no repeated chip, antibody and replicate combinations, "
                                "since any of those would quietly distort a mean or a CV."
                            ),
                            html.Li(
                                "The run has not already been ingested, unless replacing "
                                "is explicitly ticked above."
                            ),
                            html.Li(
                                "Replicate counts match the panel's expectation. A "
                                "mismatch is reported as a warning rather than a refusal, "
                                "and the statistics are computed from the spots present."
                            ),
                        ],
                        className="section-note",
                    )
                ],
            ),
        ],
        className="page",
    )


def _decode(contents: str) -> bytes:
    _header, _, payload = contents.partition(",")
    return base64.b64decode(payload)


@callback(
    Output("density-filename", "children"),
    Output("upload-density", "className"),
    Input("upload-density", "filename"),
)
def _show_density_name(filename):
    if not filename:
        return "", "upload-drop"
    return f"Selected: {filename}", "upload-drop upload-ready"


@callback(
    Output("metadata-filename", "children"),
    Output("upload-metadata", "className"),
    Input("upload-metadata", "filename"),
)
def _show_metadata_name(filename):
    if not filename:
        return "", "upload-drop"
    return f"Selected: {filename}", "upload-drop upload-ready"


@callback(
    Output("ingest-button", "disabled"),
    Input("upload-density", "contents"),
    Input("upload-metadata", "contents"),
)
def _enable_button(density, metadata):
    return not (density and metadata)


@callback(
    Output("ingest-result", "children"),
    Output("data-version", "data"),
    Input("ingest-button", "n_clicks"),
    State("upload-density", "contents"),
    State("upload-density", "filename"),
    State("upload-metadata", "contents"),
    State("upload-metadata", "filename"),
    State("replace-existing", "value"),
    State("panel-store", "data"),
    State("data-version", "data"),
    prevent_initial_call=True,
)
def _ingest(
    n_clicks,
    density_contents,
    density_name,
    metadata_contents,
    metadata_name,
    replace,
    panel,
    version,
):
    if not n_clicks or not density_contents or not metadata_contents:
        return no_update, no_update

    density_bytes = _decode(density_contents)
    metadata_bytes = _decode(metadata_contents)

    try:
        result = ingest_run(
            io.BytesIO(density_bytes),
            io.BytesIO(metadata_bytes),
            panel=panel,
            replace="replace" in (replace or []),
            density_name=density_name,
            metadata_name=metadata_name,
        )
    except (IngestionError, DensityFileError, MetadataFileError) as exc:
        return _error(str(exc)), no_update
    except Exception as exc:  # pragma: no cover - unexpected parse failures
        return _error(f"The run could not be read: {exc}"), no_update

    archived = _archive(
        result.spotting_run_id,
        (density_name or "densities.csv", density_bytes),
        (metadata_name or "run_metadata.csv", metadata_bytes),
    )

    tone = "result-warn" if result.warnings else "result-ok"
    body = [
        html.Strong(
            f"Run {result.spotting_run_id} added: {result.n_chips_failed} of "
            f"{result.n_chips} chips failed ({result.failure_rate_pct:.1f}%)."
        ),
        html.Div(
            f"{result.panel} · spotted {result.run_date}. "
            f"Files archived to {archived}.",
            className="chart-note",
            style={"margin": "6px 0 0"},
        ),
        html.Div(
            [
                dcc.Link("Open the QC report", href="/runs", className="btn btn-secondary"),
                dcc.Link(
                    "See it in monitoring", href="/", className="btn btn-secondary"
                ),
            ],
            className="header-controls",
            style={"marginTop": "12px"},
        ),
    ]
    if result.warnings:
        body.append(html.Ul([html.Li(w) for w in result.warnings]))

    return html.Div(body, className=tone), (version or 0) + 1


def _archive(run_id: str, *files: tuple[str, bytes]) -> str:
    """Keep the exact uploaded bytes beside the run's results."""
    folder = RAW_DIR / run_id
    folder.mkdir(parents=True, exist_ok=True)
    for name, payload in files:
        (folder / Path(name).name).write_bytes(payload)
    return str(folder.relative_to(PROJECT_ROOT))


def _error(message: str):
    return html.Div(
        [
            html.Strong("The run was not added."),
            html.Div(message, style={"marginTop": "6px"}),
            html.Div(
                "Nothing was written to the database, so the monitoring history is "
                "unchanged.",
                className="chart-note",
                style={"margin": "8px 0 0"},
            ),
        ],
        className="result-error",
    )

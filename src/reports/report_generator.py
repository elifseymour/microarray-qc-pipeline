"""Build the per-run QC report.

The report is a single self-contained HTML file: the Plotly library is embedded rather
than linked, so an archived report still opens years later on a machine with no network
access, which is what makes it usable as a QC record. Printing it from the browser
("Print → Save as PDF") produces a clean paginated PDF; the stylesheet carries the page
rules for that.

Rendering goes through the same figure builders the dashboard uses, so a report can
never show something different from the screen it was generated from.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import PROJECT_ROOT, QCConfig, get_config
from src.dashboard import figures as figures_module
from src.dashboard.theme import STATUS, get_theme
from src.db import queries as queries_module

TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


def _format_optional(value, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "not recorded"
    if isinstance(value, (int, float)):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["fmt"] = _format_optional
    return env


def _figure_html(fig, include_js: bool) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={"displayModeBar": False, "responsive": True},
    )


def _context_notes(
    run: dict, runs: pd.DataFrame, measurements: pd.DataFrame, config: QCConfig
) -> list[str]:
    """A few sentences putting this run beside the rest of the series.

    Deliberately short and factual. The full cross-run analysis lives in the monitoring
    view; what belongs on a run's own record is whether this run was ordinary.
    """
    notes: list[str] = []
    if runs.empty or len(runs) < 2:
        return notes

    others = runs[runs["spotting_run_id"] != run["spotting_run_id"]]
    if others.empty:
        return notes

    average = float(others["failure_rate_pct"].mean())
    delta = float(run["failure_rate_pct"]) - average
    if abs(delta) < 5:
        notes.append(
            f"The failure rate of {run['failure_rate_pct']:.1f}% is in line with the "
            f"{average:.1f}% average of the other {len(others)} run(s) for this panel."
        )
    else:
        direction = "above" if delta > 0 else "below"
        notes.append(
            f"The failure rate of {run['failure_rate_pct']:.1f}% is "
            f"{abs(delta):.1f} percentage points {direction} the {average:.1f}% average "
            f"of the other {len(others)} run(s) for this panel."
        )

    reference = config.reference_antibody(run["panel"])
    if reference and not measurements.empty:
        control = measurements[measurements["antibody"] == reference]
        if not control.empty:
            notes.append(
                f"The {reference} negative control averaged "
                f"{control['mean_density'].mean():.2f} ng/mm² on this run. It has no "
                "biomarker specificity, so it reflects how well the surface and the "
                "spotter deposited protein, independently of any antibody."
            )

    if run["room_temp_c"] is not None and not pd.isna(run["room_temp_c"]):
        temps = others["room_temp_c"].dropna()
        if not temps.empty and abs(run["room_temp_c"] - temps.mean()) > 2.0:
            notes.append(
                f"Room temperature was {run['room_temp_c']:.1f} °C against a usual "
                f"{temps.mean():.1f} °C, which is worth noting when reading this run's "
                "variability."
            )
    return notes


def build_run_report_html(
    spotting_run_id: str,
    config: QCConfig | None = None,
    db_path: str | Path | None = None,
    theme_name: str = "light",
    embed_plotly: bool = True,
) -> str:
    """Render one run's QC report to an HTML string.

    Args:
        spotting_run_id: the run to report on.
        config: loaded QC config; the shared one is used when omitted.
        db_path: database to read; the configured one is used when omitted.
        theme_name: chart theme, "light" for print.
        embed_plotly: embed the plotting library for a self-contained file. Set False
            when the report is being shown inside a page that already loaded it.

    Raises:
        ValueError: when the run has not been ingested.
    """
    config = config or get_config()
    run = queries_module.get_run(spotting_run_id, db_path=db_path)
    if run is None:
        raise ValueError(
            f"Run {spotting_run_id!r} has not been ingested, so no report can be built."
        )

    panel = run["panel"]
    antibodies = config.antibodies(panel)
    reference = config.reference_antibody(panel)
    theme = get_theme(theme_name)

    measurements = queries_module.load_measurements(
        panel=panel, spotting_run_id=spotting_run_id, db_path=db_path
    )
    chips = queries_module.load_chips(
        panel=panel, spotting_run_id=spotting_run_id, db_path=db_path
    )
    lots = queries_module.load_lots(panel=panel, db_path=db_path)
    lots = lots[lots["spotting_run_id"] == spotting_run_id] if not lots.empty else lots
    runs = queries_module.load_runs(panel=panel, db_path=db_path)

    summary = queries_module.load_run_antibody_summary(panel=panel, db_path=db_path)
    antibody_summary = (
        summary[summary["spotting_run_id"] == spotting_run_id] if not summary.empty else summary
    )
    # Keep the panel's declared order rather than whatever the query returned.
    if not antibody_summary.empty:
        order = {name: i for i, name in enumerate(antibodies)}
        antibody_summary = antibody_summary.assign(
            _order=antibody_summary["antibody"].map(lambda a: order.get(a, len(order)))
        ).sort_values("_order")

    thresholds = config.resolve_thresholds(panel)
    density_fig = figures_module.run_distribution(
        measurements, antibodies, reference, theme, "mean_density",
        limit=thresholds.density_min_ng_mm2,
    )
    cv_fig = figures_module.run_distribution(
        measurements, antibodies, reference, theme, "cv_pct",
        limit=thresholds.cv_max_pct, limit_is_maximum=True,
    )
    heatmap_fig = figures_module.chip_density_heatmap(measurements, antibodies, theme)

    failed = int(run["n_chips_failed"])
    if failed == 0:
        verdict_text, verdict_color = "All chips passed QC", STATUS["good"]
    elif run["failure_rate_pct"] >= 25:
        verdict_text = f"{failed} of {run['n_chips']} chips failed QC"
        verdict_color = STATUS["critical"]
    else:
        verdict_text = f"{failed} of {run['n_chips']} chips failed QC"
        verdict_color = STATUS["warning"]

    others = runs[runs["spotting_run_id"] != spotting_run_id] if not runs.empty else runs
    series_average = float(others["failure_rate_pct"].mean()) if not others.empty else None

    template = _environment().get_template("run_report.html.j2")
    return template.render(
        run=run,
        chips=chips.to_dict("records"),
        lots=lots.to_dict("records") if not lots.empty else [],
        antibody_summary=antibody_summary.to_dict("records"),
        coating_batches=", ".join(sorted(chips["coating_batch"].dropna().unique()))
        if not chips.empty
        else "",
        wafer_ids=", ".join(sorted(chips["wafer_id"].dropna().unique()))
        if not chips.empty
        else "",
        density_figure=_figure_html(density_fig, embed_plotly),
        cv_figure=_figure_html(cv_fig, False),
        heatmap_figure=_figure_html(heatmap_fig, False),
        verdict_text=verdict_text,
        verdict_color=verdict_color,
        good_color=STATUS["good"],
        critical_color=STATUS["critical"],
        series_average=series_average,
        context_notes=_context_notes(run, runs, measurements, config),
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        source_files=", ".join(
            filter(
                None,
                [run.get("source_density_file"), run.get("source_metadata_file")],
            )
        )
        or "uploaded files",
    )


def write_run_report(
    spotting_run_id: str,
    output_dir: Path | None = None,
    config: QCConfig | None = None,
    db_path: str | Path | None = None,
) -> Path:
    """Render a run's report and save it to the reports archive."""
    output_dir = output_dir or DEFAULT_REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    html = build_run_report_html(spotting_run_id, config=config, db_path=db_path)
    path = output_dir / f"QC_report_{spotting_run_id}.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write a run's QC report to reports/.")
    parser.add_argument("run_id", help="spotting run ID, e.g. SR-2026-W37")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    path = write_run_report(args.run_id, output_dir=args.output_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

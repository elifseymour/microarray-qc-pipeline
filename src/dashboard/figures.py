"""Figure builders shared by the dashboard pages and the printable QC report.

Each function takes tidy data plus a theme and returns a Plotly figure. Keeping them
here, rather than inline in callbacks, means the report and the dashboard cannot drift
apart, and a figure can be rendered in a notebook while working on it.

Conventions applied throughout, so the charts read as one system:
  * an antibody keeps its color everywhere; the negative control is a dashed neutral;
  * QC limits are drawn as a quiet labelled rule, never left implicit;
  * lines are 2px, markers are at least 8px with a surface ring where marks overlap;
  * trend charts use a shared crosshair tooltip, so values are readable without
    hunting for a single point;
  * every chart in the dashboard is paired with a table view, so no value is
    reachable only by hovering and no meaning rests on color alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.theme import (
    Theme,
    antibody_colors,
    antibody_dash,
    plotly_template,
    reference_line,
    status_color,
)
from src.qc.part2_trend_analysis import (
    CHIP_TO_CHIP_CV,
    DENSITY_METRIC,
    WITHIN_CHIP_CV,
)

DENSITY_AXIS = "Spot height (density, ng/mm²)"
MARKER_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x"]


def _base(
    fig: go.Figure,
    theme: Theme,
    height: int = 380,
    legend: bool = False,
    bottom: int | None = None,
    **layout,
) -> go.Figure:
    """Apply the shared template, reserving room for a legend row when there is one."""
    fig.update_layout(template=plotly_template(theme), height=height, **layout)
    overrides: dict[str, int] = {}
    if legend:
        # Room for the title line and the legend row beneath it.
        overrides["t"] = 94
    if bottom is not None:
        overrides["b"] = bottom
    if overrides:
        fig.update_layout(margin=overrides)
    return fig


def empty_figure(message: str, theme: Theme, height: int = 300) -> go.Figure:
    """A placeholder that says why there is nothing to show."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=13, color=theme.text_muted),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, theme, height=height)


def _marker(theme: Theme, color: str, symbol: str = "circle", size: int = 9) -> dict:
    # The surface-colored ring separates overlapping points without drawing a border
    # that reads as part of the data.
    return dict(
        size=size,
        symbol=symbol,
        color=color,
        line=dict(width=1.5, color=theme.surface_raised),
    )


# --------------------------------------------------------------------------------
# Part 2: trends across runs
# --------------------------------------------------------------------------------


def density_trend(
    summary: pd.DataFrame,
    antibodies: list[str],
    reference_antibody: str | None,
    theme: Theme,
    density_min: float | None = None,
) -> go.Figure:
    """Mean spot height per antibody across runs, split at each lot change.

    Each lot is drawn as its own segment in the antibody's color, with the marker shape
    changing at the switch. That makes the difference between a step at a lot boundary
    and a slide within one lot visible directly, which is the distinction the alerts
    are built on.
    """
    if summary.empty:
        return empty_figure("No runs ingested yet.", theme)

    colors = antibody_colors(antibodies, reference_antibody, theme)
    fig = go.Figure()

    for antibody in antibodies:
        group = summary[summary["antibody"] == antibody].sort_values("run_date")
        if group.empty:
            continue
        color = colors.get(antibody, theme.text_secondary)
        dash = antibody_dash(antibody, reference_antibody)

        lots = group["ab_lot"].fillna("unknown")
        # Segment ids increment whenever the lot changes between consecutive runs.
        segment_ids = (lots != lots.shift()).cumsum()
        first = True
        for seg_index, (_, segment) in enumerate(group.groupby(segment_ids, sort=True)):
            lot = segment["ab_lot"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=segment["run_date"],
                    y=segment[DENSITY_METRIC],
                    name=antibody,
                    legendgroup=antibody,
                    showlegend=first,
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash=dash),
                    marker=_marker(
                        theme, color, MARKER_SYMBOLS[seg_index % len(MARKER_SYMBOLS)]
                    ),
                    customdata=np.stack(
                        [
                            segment["spotting_run_id"],
                            segment["ab_lot"].fillna("not recorded"),
                            segment["chip_to_chip_cv_pct"].round(1),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        f"<b>{antibody}</b><br>%{{customdata[0]}}<br>"
                        "Spot height %{y:.2f} ng/mm²<br>"
                        "Lot %{customdata[1]}<br>"
                        "Chip-to-chip CV %{customdata[2]}%<extra></extra>"
                    ),
                )
            )
            first = False

            # Mark the switch itself, so a step is attributable at a glance.
            if seg_index > 0:
                fig.add_annotation(
                    x=segment["run_date"].iloc[0],
                    y=segment[DENSITY_METRIC].iloc[0],
                    text=f"new lot {lot}",
                    showarrow=True,
                    arrowhead=0,
                    arrowwidth=1,
                    arrowcolor=theme.text_muted,
                    ax=0,
                    ay=-28,
                    font=dict(size=10, color=theme.text_muted),
                )

    if density_min is not None:
        reference_line(fig, density_min, f"QC limit {density_min:g} ng/mm²", theme)

    return _base(
        fig,
        theme,
        height=420,
        legend=True,
        title="Mean spot height per antibody, by spotting run",
        yaxis_title=DENSITY_AXIS,
        xaxis_title="Spotting run date",
        hovermode="x unified",
    )


def variability_trend(
    summary: pd.DataFrame,
    antibodies: list[str],
    reference_antibody: str | None,
    theme: Theme,
    metric: str = CHIP_TO_CHIP_CV,
    cv_max: float | None = None,
) -> go.Figure:
    """CV per antibody across runs.

    Two different metrics share this builder. Chip-to-chip CV asks whether chips within
    a batch resemble each other; within-chip CV asks whether the replicate spots on one
    chip do. Only the second has a QC limit, so the limit line is drawn only for it.
    """
    if summary.empty:
        return empty_figure("No runs ingested yet.", theme)

    titles = {
        CHIP_TO_CHIP_CV: "Chip-to-chip variability (CV of per-chip means)",
        WITHIN_CHIP_CV: "Within-chip variability (mean CV across replicate spots)",
    }
    colors = antibody_colors(antibodies, reference_antibody, theme)
    fig = go.Figure()

    for antibody in antibodies:
        group = summary[summary["antibody"] == antibody].sort_values("run_date")
        if group.empty:
            continue
        color = colors.get(antibody, theme.text_secondary)
        fig.add_trace(
            go.Scatter(
                x=group["run_date"],
                y=group[metric],
                name=antibody,
                mode="lines+markers",
                line=dict(
                    color=color, width=2, dash=antibody_dash(antibody, reference_antibody)
                ),
                marker=_marker(theme, color),
                customdata=group[["spotting_run_id"]],
                hovertemplate=(
                    f"<b>{antibody}</b><br>%{{customdata[0]}}<br>"
                    "CV %{y:.1f}%<extra></extra>"
                ),
            )
        )

    if cv_max is not None:
        reference_line(fig, cv_max, f"QC limit {cv_max:g}%", theme)

    return _base(
        fig,
        theme,
        legend=True,
        title=titles.get(metric, "Variability by run"),
        yaxis_title="CV (%)",
        xaxis_title="Spotting run date",
        hovermode="x unified",
    )


def failure_rate_trend(runs: pd.DataFrame, theme: Theme) -> go.Figure:
    """Chip failure rate per run, with the series average for context.

    Bars rather than a line: each run is a discrete event, and the rate is a proportion
    of that run's chips rather than a continuously varying quantity.
    """
    if runs.empty:
        return empty_figure("No runs ingested yet.", theme)

    ordered = runs.sort_values("run_date")
    fig = go.Figure(
        go.Bar(
            x=ordered["run_date"],
            y=ordered["failure_rate_pct"],
            marker=dict(
                color=[
                    status_color("FAIL") if rate >= 25 else theme.series_color(0)
                    for rate in ordered["failure_rate_pct"]
                ],
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    ordered["spotting_run_id"],
                    ordered["n_chips_failed"],
                    ordered["n_chips"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{y:.1f}% failed"
                "<br>%{customdata[1]} of %{customdata[2]} chips<extra></extra>"
            ),
            name="Failure rate",
            showlegend=False,
        )
    )
    reference_line(
        fig,
        float(ordered["failure_rate_pct"].mean()),
        f"series average {ordered['failure_rate_pct'].mean():.1f}%",
        theme,
    )
    return _base(
        fig,
        theme,
        title="Chip failure rate per spotting run",
        yaxis_title="Chips failed (%)",
        xaxis_title="Spotting run date",
        bargap=0.35,
    )


def control_vs_panel(
    run_panel: pd.DataFrame, reference_antibody: str | None, theme: Theme
) -> go.Figure:
    """The surface diagnostic: how the control and the capture antibodies moved together.

    Both series are relative to their own norm, so 1.00 is "normal". When the control
    drops with the rest, the deposition itself under-performed and the coated surface,
    the spotter or the day's conditions are implicated. When only the capture antibodies
    drop, the cause sits with an antibody.
    """
    if run_panel.empty:
        return empty_figure("No runs ingested yet.", theme)
    if not reference_antibody or run_panel["control_relative_density"].isna().all():
        return empty_figure(
            "No negative control antibody is configured for this panel, so a surface "
            "problem cannot be separated from an antibody problem.",
            theme,
        )

    ordered = run_panel.sort_values("run_date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ordered["run_date"],
            y=ordered["median_relative_density"],
            name="Capture antibodies (median)",
            mode="lines+markers",
            line=dict(color=theme.series_color(0), width=2),
            marker=_marker(theme, theme.series_color(0)),
            hovertemplate="Capture antibodies %{y:.2f}×<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ordered["run_date"],
            y=ordered["control_relative_density"],
            name=f"{reference_antibody} (negative control)",
            mode="lines+markers",
            line=dict(color=theme.reference, width=2, dash="dash"),
            marker=_marker(theme, theme.reference, "square"),
            customdata=ordered[["spotting_run_id"]],
            hovertemplate=(
                "%{customdata[0]}<br>Control %{y:.2f}×<extra></extra>"
            ),
        )
    )
    reference_line(fig, 1.0, "normal for this panel", theme)
    return _base(
        fig,
        theme,
        legend=True,
        title="Deposited density relative to normal: control vs. capture antibodies",
        yaxis_title="Relative to own series norm (×)",
        xaxis_title="Spotting run date",
        hovermode="x unified",
    )


def environment_facets(
    summary: pd.DataFrame,
    antibodies: list[str],
    reference_antibody: str | None,
    driver: str,
    driver_label: str,
    metric: str,
    metric_label: str,
    theme: Theme,
    correlations: pd.DataFrame | None = None,
) -> go.Figure:
    """One small scatter per antibody: a metric against an environmental reading.

    Faceting rather than four colored series in one frame: on a scatter every pair of
    series can end up adjacent, and beyond three hues they stop being reliably
    distinguishable. One antibody per panel also lets each carry its own fitted line
    and correlation without a legend to decode.
    """
    present = [a for a in antibodies if a in set(summary["antibody"])]
    if summary.empty or not present or driver not in summary:
        return empty_figure("Not enough data for this comparison yet.", theme)

    colors = antibody_colors(antibodies, reference_antibody, theme)
    cols = min(len(present), 2)
    rows = int(np.ceil(len(present) / cols))

    stats_lookup = {}
    if correlations is not None and not correlations.empty:
        subset = correlations[
            (correlations["driver"] == driver) & (correlations["metric"] == metric)
        ]
        stats_lookup = {row.antibody: row for row in subset.itertuples(index=False)}

    subplot_titles = []
    for antibody in present:
        stat = stats_lookup.get(antibody)
        subplot_titles.append(
            f"{antibody}   r = {stat.r:.2f}, n = {int(stat.n)}" if stat else antibody
        )

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.24,
    )

    for i, antibody in enumerate(present):
        row, col = divmod(i, cols)
        row, col = row + 1, col + 1
        group = summary[summary["antibody"] == antibody].dropna(subset=[driver, metric])
        color = colors.get(antibody, theme.text_secondary)

        fig.add_trace(
            go.Scatter(
                x=group[driver],
                y=group[metric],
                mode="markers",
                marker=_marker(theme, color, size=10),
                name=antibody,
                showlegend=False,
                customdata=group[["spotting_run_id"]],
                hovertemplate=(
                    f"<b>{antibody}</b><br>%{{customdata[0]}}<br>"
                    f"{driver_label} %{{x:.1f}}<br>{metric_label} %{{y:.2f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # A fitted line makes the direction legible; the correlation in the facet title
        # carries the strength, so the line needs no label of its own.
        if len(group) >= 3 and group[driver].nunique() > 1:
            slope, intercept = np.polyfit(group[driver], group[metric], 1)
            xs = np.linspace(group[driver].min(), group[driver].max(), 50)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=slope * xs + intercept,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    opacity=0.65,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text=driver_label, row=row, col=col)
        fig.update_yaxes(title_text=metric_label if col == 1 else None, row=row, col=col)

    fig = _base(
        fig,
        theme,
        height=300 * rows + 40,
        bottom=64,
        title=f"{metric_label} vs. {driver_label}",
    )
    # One shared y scale: the panels are meant to be compared with each other, and
    # independent scales would make a low-variability antibody look like a high one.
    values = summary[summary["antibody"].isin(present)][metric].dropna()
    if not values.empty:
        span = float(values.max() - values.min()) or 1.0
        fig.update_yaxes(range=[values.min() - 0.1 * span, values.max() + 0.1 * span])
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color=theme.text_secondary)
    return fig


def lot_age_facets(
    summary: pd.DataFrame,
    antibodies: list[str],
    reference_antibody: str | None,
    theme: Theme,
    lot_age: pd.DataFrame | None = None,
) -> go.Figure:
    """Spot height against how old the antibody stock was on the day it was spotted.

    This is the direct shelf-life view: a stock that loses activity in storage shows a
    downward slope here. Points from different lots are shaped differently, because a
    slope built from two lots of different potency says nothing about degradation.
    """
    present = [a for a in antibodies if a in set(summary["antibody"])]
    if summary.empty or "lot_age_days" not in summary or summary["lot_age_days"].isna().all():
        return empty_figure(
            "No antibody lot dates are recorded, so stock age cannot be assessed.", theme
        )

    colors = antibody_colors(antibodies, reference_antibody, theme)
    cols = min(len(present), 2)
    rows = int(np.ceil(len(present) / cols))
    stats_lookup = (
        {row.antibody: row for row in lot_age.itertuples(index=False)}
        if lot_age is not None and not lot_age.empty
        else {}
    )

    titles = []
    for antibody in present:
        stat = stats_lookup.get(antibody)
        titles.append(f"{antibody}   r = {stat.r:.2f}, n = {int(stat.n)}" if stat else antibody)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.24,
    )

    for i, antibody in enumerate(present):
        row, col = divmod(i, cols)
        row, col = row + 1, col + 1
        group = summary[summary["antibody"] == antibody].dropna(
            subset=["lot_age_days", DENSITY_METRIC]
        )
        color = colors.get(antibody, theme.text_secondary)

        for lot_index, (lot, lot_group) in enumerate(
            group.groupby(group["ab_lot"].fillna("not recorded"), sort=False)
        ):
            fig.add_trace(
                go.Scatter(
                    x=lot_group["lot_age_days"],
                    y=lot_group[DENSITY_METRIC],
                    mode="markers",
                    marker=_marker(
                        theme, color, MARKER_SYMBOLS[lot_index % len(MARKER_SYMBOLS)], size=10
                    ),
                    name=str(lot),
                    showlegend=False,
                    customdata=lot_group[["spotting_run_id"]],
                    hovertemplate=(
                        f"<b>{antibody}</b> lot {lot}<br>%{{customdata[0]}}<br>"
                        "Stock age %{x:.0f} days<br>Spot height %{y:.2f} ng/mm²"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Stock age at spotting (days)", row=row, col=col)
        fig.update_yaxes(title_text=DENSITY_AXIS if col == 1 else None, row=row, col=col)

    fig = _base(
        fig,
        theme,
        height=300 * rows + 40,
        bottom=64,
        title="Spot height vs. antibody stock age",
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color=theme.text_secondary)
    return fig


def group_effect_bars(
    effects: pd.DataFrame, by: str, label: str, theme: Theme
) -> go.Figure:
    """How each coating batch, wafer lot or operator compares with the series norm.

    Bars that cannot be separated from the runs they were used in are drawn in a
    neutral ink and labelled as such. With one coating batch per run, "this batch" and
    "these runs" are the same thing, so a difference here is a lead to check against
    the control spot rather than a conclusion about the surface.
    """
    if effects.empty:
        return empty_figure(f"No {label.lower()} information recorded.", theme)

    frame = effects.sort_values("relative_density_pct_diff")
    separable = frame.get("separable_from_run", pd.Series(True, index=frame.index))

    colors = [
        theme.series_color(0) if sep else theme.text_muted for sep in separable.astype(bool)
    ]
    fig = go.Figure(
        go.Bar(
            x=frame["relative_density_pct_diff"],
            y=frame[by].astype(str),
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            customdata=np.stack(
                [
                    frame["n_chips"].fillna(0),
                    frame["n_runs"],
                    frame["failure_rate_pct"].fillna(0).round(1),
                    frame["mean_within_chip_cv"].fillna(0).round(1),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>%{x:+.1f}% vs. series norm<br>"
                "%{customdata[0]:.0f} chips across %{customdata[1]:.0f} run(s)<br>"
                "Failure rate %{customdata[2]:.1f}%<br>"
                "Mean within-chip CV %{customdata[3]:.1f}%<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line=dict(color=theme.axis, width=1))
    return _base(
        fig,
        theme,
        height=max(240, 46 * len(frame) + 120),
        title=f"Spot height by {label.lower()}, relative to each antibody's norm",
        xaxis_title="Difference from series norm (%)",
    )


# --------------------------------------------------------------------------------
# Part 1: a single run
# --------------------------------------------------------------------------------


def run_distribution(
    measurements: pd.DataFrame,
    antibodies: list[str],
    reference_antibody: str | None,
    theme: Theme,
    value: str = "mean_density",
    limit: float | None = None,
    limit_is_maximum: bool = False,
) -> go.Figure:
    """Per-antibody distribution across the run's chips, with every chip shown.

    Every chip is plotted rather than only a box: with 20 chips the individual points
    are what let a scientist see whether a failure was one outlier or a whole batch
    shifting, and failing chips are colored with the reserved status color so they are
    identifiable without reading the axis.
    """
    if measurements.empty:
        return empty_figure("No measurements for this run.", theme)

    titles = {
        "mean_density": "Spot height by antibody, one point per chip",
        "cv_pct": "Within-chip replicate CV by antibody, one point per chip",
    }
    axis_titles = {"mean_density": DENSITY_AXIS, "cv_pct": "Within-chip CV (%)"}
    colors = antibody_colors(antibodies, reference_antibody, theme)
    present = [a for a in antibodies if a in set(measurements["antibody"])]
    fail_column = "fail_high_cv" if value == "cv_pct" else "fail_low_density"

    # The jitter is seeded so points do not jump between renders of the same run.
    rng = np.random.default_rng(7)
    fig = go.Figure()

    for position, antibody in enumerate(present):
        group = measurements[measurements["antibody"] == antibody]
        color = colors.get(antibody, theme.text_secondary)

        # The box carries the distribution; the points are a separate layer because only
        # a scatter can color each chip individually, which is what marks the failures.
        fig.add_trace(
            go.Box(
                y=group[value],
                x0=position,
                name=antibody,
                boxpoints=False,
                width=0.5,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=position + rng.uniform(-0.16, 0.16, len(group)),
                y=group[value],
                mode="markers",
                marker=dict(
                    size=9,
                    color=[
                        status_color("FAIL") if failed else color
                        for failed in group[fail_column]
                    ],
                    line=dict(width=1.5, color=theme.surface_raised),
                ),
                name=antibody,
                showlegend=False,
                customdata=np.stack(
                    [group["chip_id"], group["status"], group[fail_column]], axis=-1
                ),
                hovertemplate=(
                    f"<b>{antibody}</b><br>Chip %{{customdata[0]}}<br>"
                    "%{y:.2f}<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )

    if limit is not None:
        word = "maximum" if limit_is_maximum else "minimum"
        reference_line(fig, limit, f"QC {word} {limit:g}", theme)

    fig = _base(
        fig,
        theme,
        title=titles.get(value, "Distribution by antibody"),
        yaxis_title=axis_titles.get(value, value),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(present))),
        ticktext=present,
        range=[-0.6, len(present) - 0.4],
        showline=False,
        ticks="",
    )
    return fig


def chip_density_heatmap(
    measurements: pd.DataFrame, antibodies: list[str], theme: Theme
) -> go.Figure:
    """Chip by antibody grid of mean spot height for one run.

    A single glance answers whether a problem is one chip (a row), one antibody (a
    column) or the whole run. A sequential single-hue ramp encodes magnitude; the paired
    table view carries the numbers, since a color scale alone should never be the only
    way to read a value.
    """
    if measurements.empty:
        return empty_figure("No measurements for this run.", theme)

    present = [a for a in antibodies if a in set(measurements["antibody"])]
    grid = measurements.pivot_table(
        index="chip_id", columns="antibody", values="mean_density"
    ).reindex(columns=present)
    status = measurements.pivot_table(
        index="chip_id", columns="antibody", values="status", aggfunc="first"
    ).reindex(columns=present)

    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=list(grid.columns),
            y=list(grid.index),
            colorscale="Blues",
            colorbar=dict(
                title=dict(text="ng/mm²", font=dict(size=11, color=theme.text_secondary)),
                tickfont=dict(size=11, color=theme.text_muted),
                outlinewidth=0,
                thickness=12,
            ),
            xgap=2,
            ygap=2,
            customdata=status.to_numpy(),
            hovertemplate=(
                "Chip %{y}<br>%{x}<br>%{z:.2f} ng/mm²<br>%{customdata}<extra></extra>"
            ),
        )
    )
    return _base(
        fig,
        theme,
        height=max(320, 22 * len(grid.index) + 140),
        title="Mean spot height per chip and antibody",
        xaxis_title=None,
        yaxis_title="Chip",
    )


def replicate_strip(
    replicates: pd.DataFrame,
    antibodies: list[str],
    reference_antibody: str | None,
    theme: Theme,
    density_min: float | None = None,
) -> go.Figure:
    """Every replicate spot for one chip, which is where a CV failure is explained.

    A high CV is usually one bad spot rather than a uniformly poor group, and that
    distinction changes the diagnosis: a single low spot points at a nozzle, a whole
    group points at the antibody or the surface.
    """
    if replicates.empty:
        return empty_figure("Select a chip to see its individual spots.", theme)

    colors = antibody_colors(antibodies, reference_antibody, theme)
    present = [a for a in antibodies if a in set(replicates["antibody"])]
    rng = np.random.default_rng(11)
    fig = go.Figure()

    for position, antibody in enumerate(present):
        group = replicates[replicates["antibody"] == antibody].sort_values("replicate_num")
        color = colors.get(antibody, theme.text_secondary)
        fig.add_trace(
            go.Scatter(
                x=position + rng.uniform(-0.12, 0.12, len(group)),
                y=group["density_ng_mm2"],
                mode="markers",
                marker=_marker(theme, color, size=11),
                name=antibody,
                showlegend=False,
                customdata=group[["replicate_num"]],
                hovertemplate=(
                    f"<b>{antibody}</b><br>Spot %{{customdata[0]}}<br>"
                    "%{y:.2f} ng/mm²<extra></extra>"
                ),
            )
        )
        # A short rule at the chip's mean for this antibody: it is the value the QC
        # decision was made on, so it belongs beside the spots it came from.
        mean = float(group["density_ng_mm2"].mean())
        fig.add_shape(
            type="line",
            x0=position - 0.24,
            x1=position + 0.24,
            y0=mean,
            y1=mean,
            xref="x",
            yref="y",
            line=dict(color=color, width=2),
        )

    if density_min is not None:
        reference_line(fig, density_min, f"QC limit {density_min:g} ng/mm²", theme)

    fig = _base(
        fig,
        theme,
        height=340,
        title="Individual replicate spots on this chip (rule marks the mean)",
        yaxis_title=DENSITY_AXIS,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(present))),
        ticktext=present,
        range=[-0.6, len(present) - 0.4],
        showline=False,
        ticks="",
    )
    return fig

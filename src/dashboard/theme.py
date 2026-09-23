"""Visual tokens and the Plotly template for every chart in the dashboard.

Colors are assigned by the job they do, not picked per chart:

  * capture antibodies take the categorical slots in panel order, so an antibody
    keeps its color everywhere and filtering a panel never repaints the survivors;
  * the negative control takes a neutral ink with a dashed line, because it is a
    reference rather than one more biomarker. That also keeps the number of hues on a
    scatter at three, which is the limit that stays distinguishable for colorblind
    readers when every pair of series can appear side by side;
  * pass/fail uses the reserved status colors, which are never used for a series and
    always ship with a word beside them, never color alone.

Dark mode is a selected set of steps for the dark surface, not an inversion of the
light one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import plotly.graph_objects as go

# Validated categorical order: adjacent pairs clear the colorblind separation gate in
# both modes, and the first three clear it for every pair (the scatter case).
CATEGORICAL_LIGHT = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

# Reserved status colors, identical in both modes.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
]

FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    "Arial, sans-serif"
)


@dataclass(frozen=True)
class Theme:
    """One resolved set of visual tokens."""

    name: str
    surface: str
    surface_raised: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    border: str
    reference: str  # the negative control's ink
    categorical: list[str] = field(default_factory=list)

    def series_color(self, index: int) -> str:
        """Categorical slot by position, never cycled past the available hues."""
        if index < len(self.categorical):
            return self.categorical[index]
        # Past the palette, fall back to a neutral rather than inventing a hue; the
        # caller should be folding extra series into "Other" or faceting instead.
        return self.text_secondary


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    surface_raised="#ffffff",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#77756f",
    grid="#e8e7e3",
    axis="#c9c8c3",
    border="#e2e1dc",
    reference="#52514e",
    categorical=CATEGORICAL_LIGHT,
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    surface_raised="#232322",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#95948b",
    grid="#33332f",
    axis="#4a4a45",
    border="#33332f",
    reference="#c3c2b7",
    categorical=CATEGORICAL_DARK,
)

THEMES = {"light": LIGHT, "dark": DARK}


def get_theme(name: str | None) -> Theme:
    return THEMES.get((name or "light").lower(), LIGHT)


def antibody_colors(
    antibodies: list[str], reference_antibody: str | None, theme: Theme
) -> dict[str, str]:
    """Map antibody name to color, stable regardless of what is currently shown.

    The reference antibody is deliberately excluded from the categorical slots: it is a
    process control, and giving it a hue would both imply it is a biomarker and use up a
    slot that colorblind separation needs.
    """
    mapping: dict[str, str] = {}
    slot = 0
    for antibody in antibodies:
        if reference_antibody and antibody == reference_antibody:
            mapping[antibody] = theme.reference
            continue
        mapping[antibody] = theme.series_color(slot)
        slot += 1
    return mapping


def antibody_dash(antibody: str, reference_antibody: str | None) -> str:
    """The control is dashed so identity never rests on color alone."""
    return "dash" if reference_antibody and antibody == reference_antibody else "solid"


def status_color(status: str) -> str:
    return STATUS["critical"] if str(status).upper() == "FAIL" else STATUS["good"]


def plotly_template(theme: Theme) -> go.layout.Template:
    """A recessive template: thin marks, quiet grid, no chart junk."""
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_FAMILY, size=13, color=theme.text_secondary),
            # The title sits at the top of the margin band and the legend just above the
            # plot area, so the two never occupy the same line.
            title=dict(
                font=dict(size=15, color=theme.text_primary),
                x=0,
                xanchor="left",
                y=0.97,
                yanchor="top",
            ),
            paper_bgcolor=theme.surface_raised,
            plot_bgcolor=theme.surface_raised,
            colorway=theme.categorical,
            margin=dict(l=62, r=28, t=56, b=56),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                linecolor=theme.axis,
                linewidth=1,
                ticks="outside",
                tickcolor=theme.axis,
                ticklen=4,
                tickfont=dict(size=12, color=theme.text_muted),
                title=dict(font=dict(size=12, color=theme.text_secondary)),
                automargin=True,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=theme.grid,
                gridwidth=1,
                zeroline=False,
                showline=False,
                ticks="",
                tickfont=dict(size=12, color=theme.text_muted),
                title=dict(font=dict(size=12, color=theme.text_secondary)),
                automargin=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=12, color=theme.text_secondary),
                bgcolor="rgba(0,0,0,0)",
                itemsizing="constant",
            ),
            hoverlabel=dict(
                bgcolor=theme.surface_raised,
                bordercolor=theme.border,
                font=dict(family=FONT_FAMILY, size=12, color=theme.text_primary),
            ),
            hovermode="closest",
        )
    )


def reference_line(
    fig: go.Figure,
    y: float,
    label: str,
    theme: Theme,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """A QC limit drawn as a quiet solid rule with a word attached.

    Kept solid and thin: a dashed rule competes with the data, and an unlabeled rule
    leaves the reader guessing which limit it is.
    """
    kwargs = {"row": row, "col": col} if row is not None else {}
    fig.add_hline(
        y=y,
        line=dict(color=theme.text_muted, width=1),
        annotation_text=label,
        annotation_position="top left",
        annotation_font=dict(size=11, color=theme.text_muted),
        **kwargs,
    )

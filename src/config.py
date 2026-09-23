"""Configuration loading and threshold resolution.

The QC thresholds are deliberately data-driven rather than hard-coded: a panel can
override the defaults, and a single antibody within a panel can override its panel.
`resolve_thresholds` is the one place that decides which numbers actually apply.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "qc_config.yaml"


# A value sitting exactly on a limit must land on the same side of it every time. A CV
# of "20%" computed from real numbers can come out as 20.000000000000004, so comparisons
# are made with a relative tolerance: the criteria are "below the density limit fails"
# and "above the CV limit fails", and a value at the limit passes.
BOUNDARY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Thresholds:
    """The pass/fail criteria applied to one antibody on one chip."""

    density_min_ng_mm2: float
    cv_max_pct: float

    def evaluate(self, mean_density: float, cv_pct: float | None) -> tuple[bool, bool]:
        """Return (fails_low_density, fails_high_cv).

        A CV of None (single replicate, or a zero mean) cannot fail the variability
        criterion -- it is reported as missing rather than silently passing.
        """
        density_limit = self.density_min_ng_mm2 * (1.0 - BOUNDARY_TOLERANCE)
        cv_limit = self.cv_max_pct * (1.0 + BOUNDARY_TOLERANCE)
        fails_density = bool(mean_density < density_limit)
        fails_cv = bool(cv_pct is not None and cv_pct > cv_limit)
        return fails_density, fails_cv


@dataclass(frozen=True)
class MonitoringRules:
    """Thresholds that govern when Part 2 raises a flag."""

    min_runs_for_trend: int = 4
    trend_window_runs: int = 6
    density_decline_pct_flag: float = 12.0
    slope_p_max: float = 0.10
    cv_rise_pct_points_flag: float = 5.0
    failure_rate_rise_points_flag: float = 10.0
    correlation_r_min: float = 0.6
    lot_step_change_pct: float = 10.0


class QCConfig:
    """Parsed `qc_config.yaml`."""

    def __init__(self, raw: dict[str, Any], path: Path | None = None):
        self._raw = raw
        self.path = path
        self._defaults = raw.get("defaults", {}) or {}
        self._panels = raw.get("panels", {}) or {}
        self.monitoring = MonitoringRules(**(raw.get("monitoring", {}) or {}))

    # -- database ---------------------------------------------------------------

    @property
    def database_path(self) -> Path:
        configured = (self._raw.get("database", {}) or {}).get("path", "data/qc.db")
        p = Path(configured)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # -- panels -----------------------------------------------------------------

    @property
    def panel_names(self) -> list[str]:
        return sorted(self._panels)

    def has_panel(self, panel: str) -> bool:
        return panel in self._panels

    def panel(self, panel: str) -> dict[str, Any]:
        try:
            return self._panels[panel] or {}
        except KeyError:
            raise KeyError(
                f"Panel {panel!r} is not defined in {self.path or 'the QC config'}. "
                f"Known panels: {', '.join(self.panel_names) or 'none'}."
            ) from None

    def antibodies(self, panel: str) -> list[str]:
        """Expected antibody names for a panel, in panel (report/plot) order."""
        return list(self.panel(panel).get("antibodies", []))

    def panel_description(self, panel: str) -> str:
        return self.panel(panel).get("description", "")

    def reference_antibody(self, panel: str) -> str | None:
        """The panel's negative/isotype control spot, when one is declared.

        It is spotted like any other antibody, so it acts as a process reference: if it
        moves with the others, the surface or the spotter is implicated rather than an
        individual antibody stock.
        """
        return self.panel(panel).get("reference_antibody")

    def replicates_per_antibody(self, panel: str) -> int:
        return int(
            self.panel(panel).get(
                "replicates_per_antibody", self._defaults.get("replicates_per_antibody", 6)
            )
        )

    def chips_per_run(self, panel: str) -> int:
        return int(
            self.panel(panel).get("chips_per_run", self._defaults.get("chips_per_run", 20))
        )

    # -- thresholds -------------------------------------------------------------

    def resolve_thresholds(self, panel: str, antibody: str | None = None) -> Thresholds:
        """Resolve thresholds with antibody overriding panel overriding defaults."""
        panel_cfg = self.panel(panel)
        density = panel_cfg.get(
            "density_min_ng_mm2", self._defaults.get("density_min_ng_mm2", 3.0)
        )
        cv = panel_cfg.get("cv_max_pct", self._defaults.get("cv_max_pct", 20.0))

        if antibody:
            override = (panel_cfg.get("antibody_overrides", {}) or {}).get(antibody, {}) or {}
            density = override.get("density_min_ng_mm2", density)
            cv = override.get("cv_max_pct", cv)

        return Thresholds(density_min_ng_mm2=float(density), cv_max_pct=float(cv))


def load_config(path: str | Path | None = None) -> QCConfig:
    """Load the QC config from disk (uncached; use `get_config` for the shared one)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"QC config not found at {cfg_path}")
    with cfg_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return QCConfig(raw, path=cfg_path)


@functools.lru_cache(maxsize=1)
def get_config() -> QCConfig:
    """Process-wide config, loaded once. Call `get_config.cache_clear()` to reload."""
    return load_config()

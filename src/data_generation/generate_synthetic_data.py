"""Generate a synthetic spotting-run dataset for development and testing.

The dataset is not random noise: each injected pattern targets one piece of downstream
logic, so the dashboard can be validated against known ground truth before any real
data exists. What is deliberately built in for the Sepsis-4plex panel over 10 weekly
runs:

  Anti-IL-6     stable throughout. Nothing should ever be flagged for it.
  Anti-PCT      gradual density decline inside ONE lot (4.6 -> ~2.9 ng/mm^2).
                Reads as the stock degrading on the shelf, and drives a late rise in
                failure rate as chips start dropping below 3 ng/mm^2.
  Anti-TNFa     flat, then a step DOWN at a lot switch before run 6 (5.0 -> 3.9).
                The new lot is fresh, so this must read as a lot potency difference,
                not degradation. Together with Anti-PCT it tests the discrimination
                that motivated tracking lots at all. Also intrinsically more variable.
  Isotype-IgG1  the negative control spot, stable throughout. It is affected by the
                coating batch and the environment like every other spot but by no
                antibody-specific effect, so it separates a surface or spotter problem
                from an antibody problem.

  Environment a dry winter spell in runs 7 and 8 (room RH ~28-30% against a usual
              ~45%) inflates within-chip and chip-to-chip CV for every antibody.
              Room temperature stays normal throughout, so humidity and temperature
              are not collinear and the analysis can name which one actually
              matters. The spotter enclosure is humidified near 58% and barely
              follows the room, which is why the room reading is the informative one.
  Coating     batch CB-2602 is defective and was used for runs 4 and 5, lowering
              density and inflating CV. As in the lab, every chip in a run shares one
              coating batch, so the batch cannot be compared against another batch
              inside the same run. What identifies it instead is that the drop hits
              every antibody INCLUDING the negative control, which no antibody-specific
              problem could do.
  Room temp   controlled all year, with no real effect (a control variable).
  Wafer lot   changes once mid-series, with no real effect (a control variable).
  Operator    alternates with no real effect (a control variable).
  Sporadic    occasional nozzle dropouts give isolated low-density or high-CV chips.

Output, per run, in `data/raw/`:
  <run_id>_densities.csv      one row per chip x replicate, one column per antibody
  <run_id>_run_metadata.csv   one row per chip, ELN-style, incl. per-antibody lots

Ground truth is also written to `data/raw/synthetic_ground_truth.json` so tests and
the README can assert against it rather than restating it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SEED = 20260919

# Baseline environment: comfortable, controlled lab conditions.
BASE_ROOM_TEMP_C = 21.5
BASE_ROOM_RH_PCT = 45.0
BASE_SPOTTER_TEMP_C = 22.5
BASE_SPOTTER_RH_PCT = 58.0  # the enclosure is actively humidified for protein spotting


@dataclass
class AntibodySpec:
    """How one antibody behaves across the run series."""

    name: str
    base_density: float
    within_chip_cv_pct: float
    chip_to_chip_cv_pct: float
    # Per-run multiplicative trend applied to the baseline (index 0 = first run).
    decline_per_run_pct: float = 0.0
    # Lot schedule: (lot_id, received_date, first_run_index, potency_multiplier)
    lots: list[tuple[str, dt.date, int, float]] = field(default_factory=list)

    def lot_for_run(self, run_index: int) -> tuple[str, dt.date, float]:
        applicable = [lot for lot in self.lots if lot[2] <= run_index]
        lot_id, received, _, potency = max(applicable, key=lambda lot: lot[2])
        return lot_id, received, potency

    def density_for_run(self, run_index: int) -> float:
        _, _, potency = self.lot_for_run(run_index)
        trend = (1.0 - self.decline_per_run_pct / 100.0) ** run_index
        return self.base_density * trend * potency


@dataclass
class RunSpec:
    """Conditions for one spotting run."""

    run_id: str
    run_date: dt.date
    operator: str
    room_temp_c: float
    room_humidity_pct: float
    spotter_temp_c: float
    spotter_humidity_pct: float
    wafer_id: str
    coating_batch: str
    n_chips: int


@dataclass
class PanelSpec:
    name: str
    antibodies: list[AntibodySpec]
    runs: list[RunSpec]
    replicates: int = 6
    # Coating batches that were badly coated: density multiplier, CV inflation points.
    bad_coating_batches: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Per chip x antibody. Across 4 antibodies these compound to a baseline chip
    # failure rate around 8%, which is the sporadic background a healthy process has.
    dropout_low_density_prob: float = 0.010
    dropout_high_cv_prob: float = 0.010


def _weekly_dates(start: dt.date, n: int) -> list[dt.date]:
    return [start + dt.timedelta(weeks=i) for i in range(n)]


def build_sepsis_panel() -> PanelSpec:
    """The main 10-run development dataset."""
    start = dt.date(2026, 7, 6)  # a Monday
    dates = _weekly_dates(start, 10)

    antibodies = [
        AntibodySpec(
            name="Anti-IL-6",
            base_density=4.8,
            within_chip_cv_pct=6.0,
            chip_to_chip_cv_pct=5.0,
            lots=[("L-IL6-2605", dt.date(2026, 5, 4), 0, 1.0)],
        ),
        AntibodySpec(
            name="Anti-PCT",
            base_density=4.6,
            within_chip_cv_pct=6.5,
            chip_to_chip_cv_pct=5.5,
            # ~4.8% lost per run compounds to roughly -36% by run 10, ending just above
            # the 3 ng/mm^2 limit. A slide inside one lot is the signature of a stock
            # going bad, and it pushes a growing share of chips under the limit, so the
            # failure rate climbs in the last runs too.
            decline_per_run_pct=4.8,
            lots=[("L-PCT-2604", dt.date(2026, 4, 6), 0, 1.0)],
        ),
        AntibodySpec(
            name="Anti-TNFa",
            base_density=5.0,
            within_chip_cv_pct=9.0,
            chip_to_chip_cv_pct=6.5,
            lots=[
                ("L-TNF-2606", dt.date(2026, 6, 1), 0, 1.00),
                # Fresh lot from run 6 onward, ~23% less potent.
                ("L-TNF-2608", dt.date(2026, 8, 5), 5, 0.77),
            ],
        ),
        AntibodySpec(
            name="Isotype-IgG1",
            base_density=4.4,
            within_chip_cv_pct=6.0,
            chip_to_chip_cv_pct=5.0,
            lots=[("L-IGG1-2605", dt.date(2026, 5, 18), 0, 1.0)],
        ),
    ]

    # Placeholder names. The operator alternates between runs and carries no injected
    # effect, so it acts as a control variable in the monitoring view.
    operators = ["Operator A", "Operator B"]
    # One wafer lot per run, changing halfway through the series.
    wafers = ["WF-2607"] * 5 + ["WF-2608"] * 5
    # One coating batch per run, as in the lab: every chip in a run shares its surface
    # chemistry. CB-2602 (runs 4 and 5) is the defective batch. Nothing else is injected
    # into those two runs, so the coating is the only candidate explanation for them.
    coating_schedule = [
        "CB-2601", "CB-2601", "CB-2601",
        "CB-2602", "CB-2602",
        "CB-2603", "CB-2603", "CB-2603",
        "CB-2604", "CB-2604",
    ]
    # Room temperature is well controlled all year and carries no injected effect:
    # it is a second control variable, and nothing should ever be flagged for it.
    room_temps = [21.4, 21.6, 21.3, 21.8, 21.5, 21.6, 21.9, 21.4, 22.0, 21.7]
    # Runs 7 and 8 (indices 6, 7) are a dry winter spell in the room. This is the
    # real-world pattern: variability rises when room humidity falls, while the
    # temperature stays put. Keeping the two uncoupled is what lets the dashboard
    # name humidity as the driver instead of reporting both as correlated.
    room_rh = [45.5, 44.8, 46.0, 45.2, 44.6, 45.0, 29.5, 27.8, 43.8, 45.4]

    runs = []
    for i, date in enumerate(dates):
        room_t = room_temps[i]
        room_h = room_rh[i]
        runs.append(
            RunSpec(
                run_id=f"SR-2026-W{date.isocalendar().week:02d}",
                run_date=date,
                operator=operators[i % len(operators)],
                room_temp_c=room_t,
                room_humidity_pct=room_h,
                # The enclosure damps but does not eliminate what the room does.
                spotter_temp_c=round(BASE_SPOTTER_TEMP_C + 0.55 * (room_t - BASE_ROOM_TEMP_C), 1),
                # Humidity is actively held near the setpoint, so a dry room moves the
                # enclosure only slightly. That weak coupling is exactly why the room
                # reading, not the spotter reading, is the one that tracks spot
                # uniformity: the humidifier absorbs most of the swing but not its effect
                # on the droplet between leaving the nozzle and landing.
                spotter_humidity_pct=round(
                    BASE_SPOTTER_RH_PCT + 0.15 * (room_h - BASE_ROOM_RH_PCT), 1
                ),
                wafer_id=wafers[i],
                coating_batch=coating_schedule[i],
                n_chips=20,
            )
        )

    return PanelSpec(
        name="Sepsis-4plex",
        antibodies=antibodies,
        runs=runs,
        replicates=6,
        bad_coating_batches={"CB-2602": (0.85, 4.0)},
    )


def build_cardiac_panel() -> PanelSpec:
    """A smaller, well-behaved second panel, to confirm multi-panel support."""
    dates = _weekly_dates(dt.date(2026, 8, 11), 4)
    antibodies = [
        AntibodySpec(
            name="Anti-cTnI",
            base_density=5.0,
            within_chip_cv_pct=5.5,
            chip_to_chip_cv_pct=4.5,
            lots=[("L-CTNI-2607", dt.date(2026, 7, 2), 0, 1.0)],
        ),
        AntibodySpec(
            name="Anti-BNP",
            base_density=4.4,
            within_chip_cv_pct=6.0,
            chip_to_chip_cv_pct=5.0,
            lots=[("L-BNP-2607", dt.date(2026, 7, 2), 0, 1.0)],
        ),
        AntibodySpec(
            name="Anti-Myoglobin",
            base_density=4.6,
            within_chip_cv_pct=6.5,
            chip_to_chip_cv_pct=5.0,
            lots=[("L-MYO-2606", dt.date(2026, 6, 20), 0, 1.0)],
        ),
    ]
    runs = [
        RunSpec(
            run_id=f"CARD-2026-W{date.isocalendar().week:02d}",
            run_date=date,
            operator="Operator B",
            room_temp_c=21.5 + 0.2 * i,
            room_humidity_pct=45.0 - 0.3 * i,
            spotter_temp_c=22.5,
            spotter_humidity_pct=BASE_SPOTTER_RH_PCT,
            wafer_id="WF-2612",
            coating_batch="CB-2605",
            n_chips=12,
        )
        for i, date in enumerate(dates)
    ]
    return PanelSpec(name="Cardiac-3plex", antibodies=antibodies, runs=runs, replicates=6)


def _env_cv_inflation(run: RunSpec) -> float:
    """Extra CV (percentage points) attributable to the run's conditions.

    Dry room air speeds droplet evaporation during spotting, which leaves the deposited
    spots less uniform. This is the dominant term and matches what the lab sees in
    winter, when room humidity falls. Heat would do the same, but the room is
    temperature-controlled, so in practice that term never fires here.

    Only conditions past the comfortable range contribute, so an ordinary run is
    unaffected rather than carrying a small penalty for no reason.
    """
    heat = max(0.0, run.room_temp_c - (BASE_ROOM_TEMP_C + 3.0)) * 1.05
    dryness = max(0.0, (BASE_ROOM_RH_PCT - 5.0) - run.room_humidity_pct) * 0.55
    return heat + dryness


def generate_panel(
    spec: PanelSpec, rng: np.random.Generator
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict]]:
    """Produce density and metadata frames per run, plus a ground-truth record."""
    densities: dict[str, pd.DataFrame] = {}
    metadata: dict[str, pd.DataFrame] = {}
    truth: list[dict] = []

    for run_index, run in enumerate(spec.runs):
        chip_ids = [f"CHIP-{i + 1:02d}" for i in range(run.n_chips)]
        # Coating batches alternate across chips so a bad batch is separable from the
        # run-level environment it shares a run with.
        chip_batches = [run.coating_batch] * run.n_chips

        env_inflation = _env_cv_inflation(run)

        density_rows = []
        for chip_id, batch in zip(chip_ids, chip_batches):
            batch_density_mult, batch_cv_inflation = spec.bad_coating_batches.get(
                batch, (1.0, 0.0)
            )
            for antibody in spec.antibodies:
                target = antibody.density_for_run(run_index) * batch_density_mult

                chip_cv = antibody.chip_to_chip_cv_pct + 0.35 * (
                    env_inflation + batch_cv_inflation
                )
                chip_mean = target * (1.0 + rng.normal(0.0, chip_cv / 100.0))

                within_cv = (
                    antibody.within_chip_cv_pct + env_inflation + batch_cv_inflation
                )

                # Sporadic spotter faults: a partially clogged nozzle either starves
                # the whole spot group (low density) or makes one spot an outlier.
                if rng.random() < spec.dropout_low_density_prob:
                    chip_mean *= rng.uniform(0.45, 0.70)
                outlier_replicate = (
                    int(rng.integers(1, spec.replicates + 1))
                    if rng.random() < spec.dropout_high_cv_prob
                    else None
                )

                for replicate in range(1, spec.replicates + 1):
                    value = chip_mean * (1.0 + rng.normal(0.0, within_cv / 100.0))
                    if replicate == outlier_replicate:
                        value *= rng.uniform(0.30, 0.55)
                    density_rows.append(
                        {
                            "chip_id": chip_id,
                            "replicate_num": replicate,
                            "antibody": antibody.name,
                            "density_ng_mm2": round(max(value, 0.01), 3),
                        }
                    )

        wide = (
            pd.DataFrame(density_rows)
            .pivot(index=["chip_id", "replicate_num"], columns="antibody", values="density_ng_mm2")
            .reset_index()
        )
        wide.columns.name = None
        wide = wide[["chip_id", "replicate_num"] + [ab.name for ab in spec.antibodies]]
        densities[run.run_id] = wide

        meta = pd.DataFrame(
            {
                "spotting_run_id": run.run_id,
                "run_date": run.run_date.isoformat(),
                "panel": spec.name,
                "operator": run.operator,
                "chip_id": chip_ids,
                "wafer_id": run.wafer_id,
                "coating_batch": chip_batches,
                "room_temp_c": np.round(
                    run.room_temp_c + rng.normal(0, 0.12, run.n_chips), 2
                ),
                "room_humidity_pct": np.round(
                    run.room_humidity_pct + rng.normal(0, 0.4, run.n_chips), 2
                ),
                "spotter_temp_c": np.round(
                    run.spotter_temp_c + rng.normal(0, 0.08, run.n_chips), 2
                ),
                "spotter_humidity_pct": np.round(
                    run.spotter_humidity_pct + rng.normal(0, 0.3, run.n_chips), 2
                ),
            }
        )
        for antibody in spec.antibodies:
            lot_id, received, _ = antibody.lot_for_run(run_index)
            meta[f"{antibody.name}_lot"] = lot_id
            meta[f"{antibody.name}_lot_received_date"] = received.isoformat()
        metadata[run.run_id] = meta

        truth.append(
            {
                "panel": spec.name,
                "spotting_run_id": run.run_id,
                "run_date": run.run_date.isoformat(),
                "room_temp_c": run.room_temp_c,
                "room_humidity_pct": run.room_humidity_pct,
                "coating_batch": run.coating_batch,
                "coating_batch_defective": run.coating_batch in spec.bad_coating_batches,
                "env_cv_inflation_points": round(env_inflation, 2),
                "target_density_ng_mm2": {
                    ab.name: round(ab.density_for_run(run_index), 3) for ab in spec.antibodies
                },
                "antibody_lots": {
                    ab.name: ab.lot_for_run(run_index)[0] for ab in spec.antibodies
                },
            }
        )

    return densities, metadata, truth


def write_dataset(
    output_dir: Path, seed: int = DEFAULT_SEED, panels: list[PanelSpec] | None = None
) -> dict[str, list[str]]:
    """Generate every panel's runs and write the CSVs plus the ground-truth JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    panels = panels if panels is not None else [build_sepsis_panel(), build_cardiac_panel()]

    written: dict[str, list[str]] = {}
    all_truth: list[dict] = []

    for spec in panels:
        densities, metadata, truth = generate_panel(spec, rng)
        files: list[str] = []
        for run_id in densities:
            density_path = output_dir / f"{run_id}_densities.csv"
            metadata_path = output_dir / f"{run_id}_run_metadata.csv"
            densities[run_id].to_csv(density_path, index=False)
            metadata[run_id].to_csv(metadata_path, index=False)
            files += [density_path.name, metadata_path.name]
        written[spec.name] = files
        all_truth += truth

    truth_payload = {
        "seed": seed,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "injected_effects": {
            "Anti-PCT": "gradual density decline within a single lot (degrading stock)",
            "Anti-TNFa": "step decrease at the lot switch before run 6 (less potent new lot)",
            "environment": (
                "low room humidity in runs 7-8 (a dry winter spell) inflates CV; "
                "room temperature is controlled and carries no effect"
            ),
            "coating_batch": (
                "CB-2602 is defective and was used for runs 4-5; every antibody "
                "including the negative control spots low on those runs"
            ),
            "controls": (
                "Anti-IL-6, the Isotype-IgG1 negative control, wafer lot and operator "
                "carry no antibody-specific effect"
            ),
        },
        "runs": all_truth,
    }
    (output_dir / "synthetic_ground_truth.json").write_text(
        json.dumps(truth_payload, indent=2) + "\n"
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where to write the CSVs"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument(
        "--panel",
        choices=["sepsis", "cardiac", "both"],
        default="both",
        help="which panel(s) to generate",
    )
    args = parser.parse_args()

    selected = {
        "sepsis": [build_sepsis_panel()],
        "cardiac": [build_cardiac_panel()],
        "both": [build_sepsis_panel(), build_cardiac_panel()],
    }[args.panel]

    written = write_dataset(args.output_dir, seed=args.seed, panels=selected)
    for panel, files in written.items():
        print(f"{panel}: {len(files) // 2} runs -> {args.output_dir}")
    print(f"Ground truth: {args.output_dir / 'synthetic_ground_truth.json'}")


if __name__ == "__main__":
    main()

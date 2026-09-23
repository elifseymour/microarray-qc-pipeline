# Microarray Spotting QC

A dashboard and analysis pipeline for robotic microarray spotting of biosensor chips.
It does two jobs:

1. **Release or reject each chip after a run.** Every antibody on every chip is graded
   against a surface-density minimum and a replicate-CV maximum, and the run gets a QC
   report listing each chip's verdict and the reason behind it.
2. **Monitor the process across runs.** Spot height, chip-to-chip variability and
   failure rate are tracked over time, and the dashboard attributes a change to the
   most likely cause: a degrading antibody stock, a new antibody lot, a bad coating
   batch, the room's temperature and humidity, or the spotter itself.

Spot height and surface density are the same measurement throughout. The instrument
reads spot height; it is stored and reported as surface density in ng/mm².

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Build a 10-run synthetic dataset and load it, so there is something to look at
.venv/bin/python -m src.data_generation.generate_synthetic_data
.venv/bin/python -m src.ingestion.bulk_load --fresh

.venv/bin/python -m src.dashboard.app      # http://127.0.0.1:8050
```

Run the tests with `.venv/bin/python -m pytest tests/`.

## The four pages

| Page | Question it answers |
|---|---|
| **Monitoring** | Is anything drifting, and what is causing it? Alerts first, then the trends and drivers behind them. |
| **Run reports** | Which chips from this run can go downstream, and why did the rest fail? Downloadable as a self-contained record. |
| **Chip explorer** | What actually happened on this chip, spot by spot? |
| **Add a run** | Upload a run's two files, validate them, and store the result. |

## Adding a run

Two files per run, uploaded on the **Add a run** page.

**1. Spot density CSV** — one row per chip and replicate spot, one column per antibody:

```csv
chip_id,replicate_num,Anti-IL-6,Anti-PCT,Anti-TNFa,Isotype-IgG1
CHIP-01,1,4.92,4.44,5.13,4.38
CHIP-01,2,4.22,4.85,4.87,4.51
```

**2. ELN run metadata export** — one row per chip. Run-level facts repeat on every row,
which is how a flat Benchling table looks:

```csv
spotting_run_id,run_date,panel,operator,chip_id,wafer_id,coating_batch,
room_temp_c,room_humidity_pct,spotter_temp_c,spotter_humidity_pct,
Anti-IL-6_lot,Anti-IL-6_lot_received_date, ... one pair per antibody
```

The run ID and date come from this file, so the notebook stays the single source of
truth. Column names are matched case- and separator-insensitively, and common
alternatives are accepted ("Room Temp (C)", "room_temperature_c", and so on).

A run is refused, with nothing written to the database, if its antibody columns do not
match the panel, if a measured chip has no ELN row, if any density is blank, negative
or non-numeric, if a chip/antibody/replicate combination repeats, or if the run has
already been ingested. Mismatched replicate counts are reported as warnings and the
statistics are computed from the spots present.

**To use a real Benchling export in a different shape, edit
`src/ingestion/parse_benchling_export.py`.** It is isolated for exactly this, and the
rest of the pipeline is unaffected.

## QC rules

An antibody on a chip fails if its mean spot height is below the density limit or its
replicate CV is above the CV limit. A chip fails if any of its antibodies fails. Both
reasons are recorded when both apply, and the limits in force at ingestion are stored
on the run so an old report stays reproducible if the limits are later retuned.

Defaults are 3 ng/mm² and 20% CV, set in `config/qc_config.yaml`. A panel can override
them, and a single antibody within a panel can override its panel.

## Configuring a panel

Everything panel-specific lives in `config/qc_config.yaml`; adding a panel needs no
code. Antibody names are read from the density CSV header, so real names work as they
are.

```yaml
panels:
  Sepsis-4plex:
    antibodies: [Anti-IL-6, Anti-PCT, Anti-TNFa, Isotype-IgG1]
    reference_antibody: Isotype-IgG1
```

Multiple panels live in one database and are switched with the panel selector, so there
is no risk of pointing the dashboard at the wrong file.

### Why the negative control earns its slot

`reference_antibody` names an isotype-matched control IgG that binds none of the
biomarkers. (Deliberately not anti-BSA, which would bind the BSA in most blocking
buffers.) It is spotted and measured like any other antibody, so its deposited density
reflects the spotting process alone, and that makes two otherwise inseparable causes
separable:

- spot height falls on the capture antibodies **and** the control → the coated surface,
  the spotter, or that day's conditions;
- spot height falls on one antibody while the control holds → that antibody's stock.

The control is always trended in absolute terms as well as used as a denominator. If
both fell together, a ratio between them would be flat, and the dashboard says so
explicitly rather than reporting "normal".

## What the monitoring page can and cannot establish

Two limits are stated on the page itself rather than left for the reader to infer.

**Correlations across a run series are weak evidence.** With ten runs, anything that
drifts with time correlates with anything else that does. Every correlation is reported
with its n, and alongside how strongly the driver itself tracks the calendar, so a
coincidence is visible.

**A variable that never changes within a run cannot be separated from the run.** Every
chip in a run shares one coating batch and one wafer lot, so "this batch" and "these
runs" are the same thing. Those comparisons are drawn in neutral grey and labelled as
not separable, and a coating batch is only flagged when the negative control confirms
the drop and the recorded conditions do not explain it. To test a variable properly,
vary it within a run.

## Antibody lot tracking

Lot ID and received date are stored per antibody per run, which is what separates a
degrading stock from a less potent new lot:

- a gradual slide **inside one lot** → the stock is going bad;
- a step **at a lot boundary**, flat within each lot → the new lot is simply different.

The distinction decides whether stock gets discarded, so the trend is fitted within the
lot currently in use rather than straight through a lot change, and spot height is also
plotted against how old the stock was on the day it was spotted.

## The synthetic dataset

`src/data_generation/generate_synthetic_data.py` builds a 10-run dataset with known
ground truth, so the analysis can be validated before real data exists. It is what the
tests assert against.

| Injected | Should be reported as |
|---|---|
| Anti-PCT declines ~35% inside one lot | a degrading stock, confirmed against the control |
| Anti-TNFa steps down at a fresh lot before run 6 | a lot potency difference, not degradation |
| Coating batch CB-2602 used for runs 4 and 5 | an under-performing coating batch, evidenced by the control |
| Low room humidity in runs 7 and 8, a dry winter spell | inflated CV correlating with room humidity |
| Anti-IL-6, the control, room temperature, wafer lot, operator | nothing |

Room humidity and room temperature are deliberately **not** moved together. In the lab
the room is temperature-controlled year round and the variability appears in winter
when the room dries out, so only humidity carries an effect here. Had both been moved
at once they would correlate with each other and neither could be named as the cause.
The spotter enclosure is humidified near 58%, so it follows the room only weakly, which
is why the room reading is the informative one.

Regenerate and reload with:

```bash
.venv/bin/python -m src.data_generation.generate_synthetic_data
.venv/bin/python -m src.ingestion.bulk_load --fresh
```

## Reports

A run's report is a single self-contained HTML file: the plotting library is embedded,
so an archived report opens later without a network connection. Print it from the
browser ("Save as PDF") for a paginated copy; the stylesheet carries the page rules.

```bash
.venv/bin/python -m src.reports.report_generator SR-2026-W37   # writes to reports/
```

## Layout

```
config/qc_config.yaml     panels, QC limits, monitoring rules
data/raw/                 uploaded run files, archived per run
data/qc.db                SQLite: every run, chip, measurement and spot
src/config.py             config loading and threshold resolution
src/db/                   models, session handling, read queries
src/ingestion/            file parsing, validation, the single write path
src/qc/part1_chip_qc.py   pass/fail rules
src/qc/part2_trend_analysis.py   cross-run trends, attribution, alerts
src/dashboard/            Dash app, pages, figures, theme
src/reports/              HTML report generator
tests/                    pass/fail edge cases and the injected ground truth
```

The QC functions are pure functions over dataframes, so they can be used directly in a
notebook without the dashboard.

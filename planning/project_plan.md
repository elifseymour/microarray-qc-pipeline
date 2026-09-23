# Microarray Spotting QC — project plan of record

This is the built design, with the parameters as agreed. It supersedes the open
questions in `Robotic Microarray Spotting QC Process.md`, which remains the statement
of intent.

## Goal

Establish a QC process for biosensor chip spotting that (1) keeps low-quality chips out
of downstream work and (2) monitors the spotting process over time well enough to
pinpoint *why* something changed. The second part is what avoids the expensive failure
modes: discarding antibody stock that was never the problem, or continuing to spot onto
a bad coating batch.

## Scope decisions

| Decision | Value | Why |
|---|---|---|
| Framework | Dash (Plotly) | A standalone web app with multi-page layout control, servable to the lab later without a rewrite. |
| Storage | SQLite, one file | A run adds ~500 rows. No server, and the whole QC history is one file that can be copied or backed up. |
| Ingestion | Upload two files in the dashboard | Matches how a run actually ends: the scientist has a density CSV and an ELN export. |
| Run identity | `spotting_run_id` read from the ELN export | Keeps the notebook as the single source of truth and prevents typos creating duplicate runs. |
| Development data | 10 synthetic runs with known ground truth | Lets the analysis be validated before real data exists, and gives the tests something to assert against. |
| Density limit | 3 ng/mm² mean per antibody per chip | From the process document. |
| CV limit | 20% within-chip replicate CV | Agreed default; per-panel and per-antibody overrides exist in config. |
| Chip verdict | Fails if **any** antibody fails | A chip is only usable if all of its spots are. |
| Surface-treatment field | `coating_batch` | Chemistry-agnostic: polymer, epoxy, silane. |
| Coating batch per run | One batch for every chip in a run | Matches the lab. Its analytical consequence is handled below. |
| Panels | One database, `panel` field, panel selector | Multiple panels coexist; no risk of pointing at the wrong database. Antibody names come from the CSV header. |
| Negative control | `Isotype-IgG1`, declared per panel | See below. Not anti-BSA, which would bind the BSA in blocking buffers. |
| Antibody lot | Lot ID + received date, per antibody per run | Separates a degrading stock from a less potent new lot. |
| Report | Self-contained HTML, prints to PDF | No system dependencies (no pango, cairo or headless Chrome), and it opens years later offline. |

## Data model

One SQLite file accumulates everything, which is what makes cross-run trends possible.

- `runs` — one per spotting run: panel, date, operator, conditions, failure rate, and
  the QC limits actually applied (so an old report stays reproducible if limits change).
- `chips` — one per chip per run: wafer, coating batch, verdict, failure reasons.
- `antibody_lots` — one per (run, antibody): lot, received date, age at spotting.
- `antibody_measurements` — one per (run, chip, antibody): mean, SD, CV, verdict.
- `run_antibody_summary` — one per (run, antibody): the batch-level statistics that
  Part 2 trends, including chip-to-chip CV.
- `replicate_measurements` — every individual spot, kept so a flagged chip can be
  traced to its raw values.

## Part 1 — per-chip QC

For each chip and antibody, summarize the replicate spots and apply the two criteria.
Roll up to a chip verdict and a run failure rate. Validation refuses anything that
would make the stored history misleading, and a failed ingestion writes nothing at all.

Two details worth recording:

- **Boundary behaviour.** A value exactly at a limit passes. Comparisons carry a small
  relative tolerance, because a CV of "20%" computed from real numbers can come out as
  20.000000000000004 and a verdict must not turn on binary representation.
- **CV is the sample CV** (n−1), since the replicate spots are a sample of the process.

## Part 2 — monitoring, and the attribution problem

The hard part is not plotting the trends; it is saying what caused one. Three
mechanisms do the work.

### 1. Within-lot trend versus lot step

A single regression through a lot change reports a decline that is neither degradation
nor reproducible. Instead the trend is fitted **within the lot currently in use**, and
the step between lot means is measured separately. A slide inside one lot is a
degrading stock; a step at the boundary with flat segments either side is a lot of
different potency. Spot height is also plotted against stock age in days.

### 2. The negative control as a process reference

An isotype control IgG is spotted like any other antibody but binds no biomarker, so
its deposited density measures the spotting process alone.

- Every antibody low **including** the control → the surface, the spotter, or the day's
  conditions.
- One antibody low while the control holds → that antibody's stock.

The control is trended in absolute terms *and* used as a denominator. The ratio is
supporting evidence only and never an input to a verdict: if the antibody and the
control fell together the ratio would be flat, and the dashboard states that explicitly
instead of reporting "normal".

### 3. Naming what cannot be established

Because every chip in a run shares one coating batch and one wafer lot, those variables
are perfectly aligned with the run. Anti-PCT declines over the series, so early wafers
would otherwise appear ~11% "better" for no reason of their own. The dashboard:

- draws such comparisons in neutral grey and labels them not separable from the run;
- flags a coating batch only when the negative control confirms the drop **and** the
  recorded temperature and humidity do not explain it;
- reports each correlation with its n and with how strongly the driver itself tracks
  the calendar, so a time-confounded coincidence is visible;
- suppresses trend flags below a configured minimum number of runs;
- tests a rising failure rate with Fisher's exact test on chip counts rather than
  firing on a percentage-point threshold that a noisy run could trip.

To test a coating batch or wafer lot properly, vary it within a single run. That is a
lab-process recommendation, not a software change.

## Alerts

Each finding carries a level (action / watch / context), the effect size, the number of
runs behind it, and what to do next. Covered: antibody degradation, a lot change, a
declining negative control, rising chip-to-chip or within-chip variability, a rising
failure rate, outlier runs (median/MAD, so one bad run cannot hide inside an inflated
SD), under-performing coating batches, and environmental associations.

## Dashboard

Four pages: Monitoring (alerts, then trends, then drivers), Run reports (chip verdicts,
distributions, download), Chip explorer (individual spots), Add a run (upload and
validate). The panel selector and theme sit in one row at the top rather than inside
chart cards.

Charts follow a fixed system: an antibody keeps its color everywhere; the negative
control is a dashed neutral rather than a hue, which also keeps scatter plots within
the three-hue limit that stays distinguishable for colorblind readers; QC limits are
drawn as labelled rules; and every chart has a table view beside it, so no value is
reachable only by hovering and no distinction rests on color alone.

## Synthetic dataset

10 weekly runs, 20 chips, 4 antibodies, 6 replicate spots, with ground truth written to
`data/raw/synthetic_ground_truth.json`. Injected: Anti-PCT degrading within one lot;
Anti-TNFa stepping down at a fresh lot from run 6; coating batch CB-2602 defective on
runs 4 and 5; a dry winter spell on runs 7 and 8 (room humidity ~28% against a usual
~45%); sporadic nozzle dropouts. Controls with nothing injected: Anti-IL-6, the isotype
control, room temperature, wafer lot and operator.

Room temperature is held flat while humidity moves, because moving both together would
make them collinear and leave the analysis unable to name either. The spotter enclosure
is humidified near 58% and follows the room only weakly, matching a protein-spotting
setup, so the room reading is the one that tracks spot uniformity. Densities sit in the 2–6 ng/mm² range seen in practice.

## Verification

`pytest tests/` covers the pass/fail boundaries, the ingestion guards, report
rendering, and Part 2's attribution against the injected ground truth — including that
the degrading antibody and the lot step are told apart, that the defective coating
batch is identified through the control, that the dry runs are *not* blamed on their
coating batch, and that the wafer lot is marked inseparable and never flagged.

End to end: generate the dataset, load it, run the dashboard, upload a run through the
UI, and confirm the alerts match the table above.

## Not built, and why

- **One-click PDF.** WeasyPrint needs pango and cairo, and kaleido 1.x needs a headless
  Chrome. The HTML report prints to PDF from the browser with page rules already in the
  stylesheet. If a true PDF button is wanted, WeasyPrint plus a documented `brew`
  dependency is the smallest addition.
- **Authentication and multi-user deployment.** Runs locally for now. Dash is a WSGI
  app, so serving it to the lab is a deployment step rather than a rewrite.
- **Benchling API integration.** Files are exported and uploaded. The parser is
  isolated in one module so either the export format or an API feed can replace it.

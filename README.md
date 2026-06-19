# Week-Ahead (and Multi-Week) California Traffic Forecasting

A study of one-week-ahead and multi-week-ahead forecasting of California highway
traffic **flow**, using the [LargeST](https://github.com/liuxu77/LargeST) dataset
(Caltrans PeMS, hourly, 2017–2021). Greater Bay Area (District 4, 2,352 sensors) is
the main working set; Greater LA and San Diego are used for a cross-regional test.

## Headline findings

- A simple **calendar-adjusted seasonal baseline** — a recency-weighted (EWMA)
  average of the same hour-of-week, times a learned **holiday/calendar factor** —
  reaches **R² ≈ 0.949** and a weekly-aggregate MAPE of **3.55%** one week ahead.
- It is **at the practical ceiling**: autoregression, gradient boosting, pooled
  panel models, and cross-regional pooling do **not** beat it on endogenous
  information — the residual is dominated by irreducible incident noise.
- The interesting frontier is **multi-week** forecasting, where a slowly-drifting
  **level** governs the error. We reframe it as forecasting the per-sensor weekly
  **level** series and show where (little) and how (covariates, not fancier models
  of the same data) further gains might come.

## Repository layout

```
src/
  core/         shared library: data loading, baselines, holiday model, metrics
  data_prep/    download & process the raw LargeST files to hourly arrays
  experiments/  the analysis & figure-generating scripts (one per result)
docs/     research notes & reports (LaTeX + PDF), plus shareable advice notes
  research_notes/   notes 01–03, the unified paper, the best-rule report & slides,
                    the granularity-invariance math note, figures/
results/  small JSON/CSV backtest summaries
data/     NOT tracked — large & reproducible (see below)
```

Experiment scripts import the shared code from `src/core/` and can be run directly,
e.g. `python src/experiments/weekly_ladder.py`. See `src/README.md` for a one-line
description of every script.

### Key documents (in `docs/research_notes/`)
- `paper_multiweek_forecasting.pdf` — the unified paper (the full story).
- `note01…03_*.pdf` — the working notes (seasonal baseline → calendar-aware → horizons/trend).
- `best_rule_report.pdf` / `best_rule_slides.pdf` — focused report + slide deck on the rule.
- `granularity_invariance.pdf` — why a linear seasonal baseline is accurate at any granularity.

## Reproducing the data

The `data/` directory is git-ignored (≈0.6 GB, fully reproducible). LargeST is on
Kaggle; with a Kaggle API token configured (`~/.kaggle/kaggle.json` or
`KAGGLE_API_TOKEN`), the yearly raw files are downloaded and streamed to compact
hourly per-region arrays:

```bash
# metadata + one yearly raw, then process a region (gba | gla | sd)
kaggle datasets download liuxu77/largest -f ca_meta.csv      -p data/ca
kaggle datasets download liuxu77/largest -f ca_his_raw_2019.h5 -p data/ca
python src/data_prep/process_largest.py 2019 gba --delete-raw
```

The PEMS-BAY speed dataset (used for one comparison) is on
[Zenodo](https://zenodo.org/records/4263971).

## Requirements

Python 3.12 with `numpy`, `pandas`, `h5py`, `scikit-learn`, `statsmodels`,
`matplotlib`. LaTeX (TeX Live) to compile the documents.

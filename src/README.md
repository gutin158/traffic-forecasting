# `src/` — code guide

Three tiers: shared **library** (`core/`), **data prep**, and **experiments**.
Experiment and prep scripts put `core/` on the path and import from it; each can be
run directly (`python src/experiments/<name>.py`). Paths to `data/` and the figure
folder are resolved relative to the repo root, so scripts run from anywhere.

## `core/` — shared library (imported by the experiments)
| file | provides |
|---|---|
| `run_gba.py` | `load_years`, `impute_hour_of_week` — load cached hourly region arrays |
| `harness.py` | forecaster classes (hour-of-week / last-week), metrics, constants |
| `eda.py` | `seasonal_baseline` (the 4-week hour-of-week baseline) + EDA figures |
| `ewma_baseline.py` | EWMA (decaying-weight) weekly baseline |
| `holiday_model.py` | `special_day_keys` + the pooled holiday-factor model |
| `run_holiday_backtest.py` | `WeeklyBaseline`, `HolidayAdjusted`, `learn_factors` + rolling backtest |
| `holiday_region.py` | per-sensor (shrunk) holiday factors |

## `data_prep/` — get & shape the data
| file | does |
|---|---|
| `process_largest.py` | stream a raw LargeST year → hourly per-region array (`gba`/`gla`/`sd`) |
| `inspect_h5.py` | probe the structure of a raw LargeST `.h5` file |

## `experiments/` — one script per result
| file | result |
|---|---|
| `eda_holiday.py` | holiday daily-profile & corridor-heterogeneity figures |
| `residual_acf.py` | residual autocorrelation at short & weekly lags |
| `seasonal_ar.py` | seasonal-residual AR (one-week-ahead) |
| `yoy_baseline.py` | year-over-year growth-adjusted baseline |
| `holiday_diagnostic.py` | how much the YoY/holiday gain concentrates on holidays |
| `feature_merit.py` | ridge/XGBoost feature merit on the residual (mostly ~0) |
| `floor_and_mape.py` | irreducible-noise floor + weekly-aggregate MAPE |
| `horizon_ceiling.py` | per-horizon recoverable-residual predictability |
| `weekly_aggregate_headroom.py` | weekly-aggregate-residual ARIMA headroom |
| `multiweek_horizon.py` | trailing vs annual anchor at 1/2/4/8 weeks |
| `horizon_sweep.py` | fine H=1..12-week staleness sweep vs an oracle floor |
| `multiyear_trend.py` | 2017–2021 trend/seasonal decomposition (+ COVID) |
| `weekly_level.py` | STL decomposition of the weekly-level series |
| `weekly_ladder.py` | rules vs Holt vs global GBM (the ladder) |
| `pooled_h4.py` | pooled panel AR at H=4 weeks |
| `pooled_h4_fourier.py` | pooled AR with a smooth Fourier seasonal |
| `focus_h6.py` | 6-week forecasting with honest metrics |
| `co_movement.py` | cross-regional (GBA/GLA/SD) level co-movement test |

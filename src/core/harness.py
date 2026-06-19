"""
One-week-ahead (H=168) rolling out-of-sample forecasting harness for
California highway traffic (PEMS-BAY hourly speed).

Baseline under test: average of the same hour-of-week over the prior N weeks.

Forecaster interface is fit(y_train) / predict(H) so trainable models
(stat / ML) can be dropped in later without touching the backtest loop.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "pems-bay.h5"
RESULTS = ROOT / "results"
PERIOD = 168  # hours in a week
H = 168       # forecast horizon (one week ahead, hourly)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_hourly() -> pd.DataFrame:
    """5-min speed -> regular hourly mean speed. Returns (T x 325) DataFrame."""
    df = pd.read_hdf(DATA)
    df = df.sort_index()
    # collapse to hourly mean; resample yields a regular hourly grid and
    # absorbs the single 5-min gap in the raw series.
    hourly = df.resample("1h").mean()
    # PEMS speed sometimes encodes missing as 0; treat exact 0 as NaN then
    # fill by the same-hour-of-week mean per sensor so the array is complete.
    hourly = hourly.mask(hourly == 0.0)
    if hourly.isna().any().any():
        how = hourly.index.dayofweek * 24 + hourly.index.hour
        for col in hourly.columns:
            s = hourly[col]
            if s.isna().any():
                filler = s.groupby(how).transform("mean")
                hourly[col] = s.fillna(filler)
    hourly = hourly.ffill().bfill()
    return hourly


# --------------------------------------------------------------------------
# Forecasters  (operate on 2D arrays: y is (T, N_sensors))
# --------------------------------------------------------------------------
class Forecaster:
    name = "base"

    def fit(self, y_train: np.ndarray) -> "Forecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError


class HourOfWeekAverage(Forecaster):
    """Predict each future hour as the mean of the same hour-of-week over the
    most recent `n_weeks` weeks available at the training cutoff."""

    def __init__(self, n_weeks: int = 4, period: int = PERIOD):
        self.n_weeks = n_weeks
        self.period = period
        self.name = f"hour_of_week_avg(n_weeks={n_weeks})"

    def fit(self, y_train: np.ndarray) -> "HourOfWeekAverage":
        self.y = y_train
        self.T = y_train.shape[0]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        T, p, nw = self.T, self.period, self.n_weeks
        out = np.empty((horizon, self.y.shape[1]), dtype=float)
        for h in range(1, horizon + 1):
            t_future = (T - 1) + h           # absolute index of target
            lags = [t_future - p * k for k in range(1, nw + 1)]
            out[h - 1] = self.y[lags].mean(axis=0)
        return out


class LastWeek(HourOfWeekAverage):
    """Same-hour-last-week persistence (n_weeks=1)."""
    def __init__(self, period: int = PERIOD):
        super().__init__(n_weeks=1, period=period)
        self.name = "last_week"


# --------------------------------------------------------------------------
# Rolling-origin backtest
# --------------------------------------------------------------------------
def rolling_backtest(y: np.ndarray, forecaster: Forecaster,
                     horizon: int = H, warmup: int = 4 * PERIOD,
                     stride: int = 24):
    """Walk the origin t from `warmup-1` forward by `stride`; at each origin
    fit on y[:t+1] and forecast t+1..t+horizon. Returns stacked preds/actuals
    of shape (n_origins, horizon, N)."""
    T = y.shape[0]
    preds, actuals, origins = [], [], []
    o = warmup - 1
    while o + horizon < T:
        f = forecaster.fit(y[: o + 1])
        preds.append(f.predict(horizon))
        actuals.append(y[o + 1: o + 1 + horizon])
        origins.append(o)
        o += stride
    return np.stack(preds), np.stack(actuals), np.array(origins)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _rmse(p, a):
    return float(np.sqrt(np.mean((p - a) ** 2)))

def _r2(p, a):
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1 - ss_res / ss_tot)

def _mae(p, a):
    return float(np.mean(np.abs(p - a)))


def summarize(preds, actuals, label):
    p, a = preds.ravel(), actuals.ravel()
    headline = {
        "forecaster": label,
        "n_origins": int(preds.shape[0]),
        "horizon": int(preds.shape[1]),
        "n_sensors": int(preds.shape[2]),
        "n_point_predictions": int(p.size),
        "RMSE": _rmse(p, a),
        "MAE": _mae(p, a),
        "R2": _r2(p, a),
    }
    # per-horizon-hour
    per_h = []
    for h in range(preds.shape[1]):
        ph, ah = preds[:, h, :].ravel(), actuals[:, h, :].ravel()
        per_h.append({"h": h + 1, "RMSE": _rmse(ph, ah), "R2": _r2(ph, ah)})
    # per-sensor R2 distribution
    sensor_r2 = np.array([
        _r2(preds[:, :, s].ravel(), actuals[:, :, s].ravel())
        for s in range(preds.shape[2])
    ])
    sensor_rmse = np.array([
        _rmse(preds[:, :, s].ravel(), actuals[:, :, s].ravel())
        for s in range(preds.shape[2])
    ])
    headline["sensor_R2_median"] = float(np.median(sensor_r2))
    headline["sensor_R2_p10"] = float(np.percentile(sensor_r2, 10))
    headline["sensor_R2_p90"] = float(np.percentile(sensor_r2, 90))
    headline["sensor_RMSE_median"] = float(np.median(sensor_rmse))
    return headline, per_h, sensor_r2


# --------------------------------------------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    print("Loading hourly data ...")
    df = load_hourly()
    y = df.values.astype(float)
    print(f"  hourly grid: {y.shape[0]} hours x {y.shape[1]} sensors "
          f"({df.index.min()} -> {df.index.max()})")
    weeks = y.shape[0] / PERIOD
    print(f"  ~{weeks:.1f} weeks total; warmup=4 weeks -> ~{weeks-4:.1f} weeks testable\n")

    models = [HourOfWeekAverage(n_weeks=4), LastWeek(), HourOfWeekAverage(n_weeks=8)]
    all_summaries = []
    per_horizon_store = {}
    for m in models:
        preds, actuals, origins = rolling_backtest(y, m, horizon=H, stride=24)
        s, per_h, _ = summarize(preds, actuals, m.name)
        all_summaries.append(s)
        per_horizon_store[m.name] = per_h
        print(f"=== {m.name} ===")
        print(f"  origins={s['n_origins']}  points={s['n_point_predictions']:,}")
        print(f"  RMSE={s['RMSE']:.3f} mph   MAE={s['MAE']:.3f} mph   R2={s['R2']:.4f}")
        print(f"  per-sensor R2: median={s['sensor_R2_median']:.4f} "
              f"[p10={s['sensor_R2_p10']:.4f}, p90={s['sensor_R2_p90']:.4f}]\n")

    (RESULTS / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    pd.DataFrame(per_horizon_store["hour_of_week_avg(n_weeks=4)"]).to_csv(
        RESULTS / "per_horizon_hourofweek4.csv", index=False)
    print(f"Saved results/ summary.json + per-horizon csv  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

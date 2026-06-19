"""Multi-year GBA (Greater Bay Area) hourly-flow rolling backtest.

Loads cached gba_hourly_<year>.npz files, imputes missing hours via per-sensor
hour-of-week means, then runs the one-week-ahead (H=168) rolling baselines with
a memory-bounded streaming evaluator (no giant prediction tensor) and a per-year
RMSE/R2 breakdown.

Usage:  python src/run_gba.py 2017 2018 2019
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import HourOfWeekAverage, LastWeek, PERIOD, H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CA = ROOT / "data" / "ca"
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------
def load_years(years):
    vals, idx, ids = [], [], None
    for y in years:
        d = np.load(CA / f"gba_hourly_{y}.npz", allow_pickle=True)
        vals.append(d["vals"])
        idx.append(d["index"])
        if ids is None:
            ids = d["ids"]
        else:
            assert np.array_equal(ids, d["ids"]), f"sensor set mismatch in {y}"
    v = np.concatenate(vals, axis=0).astype(np.float32)
    index = pd.to_datetime(np.concatenate(idx))
    return v, index, ids


def impute_hour_of_week(v, index):
    """Fill NaN entries with the per-sensor mean for that hour-of-week."""
    how = index.dayofweek.values * 24 + index.hour.values
    for h in range(PERIOD):
        rows = np.where(how == h)[0]
        sub = v[rows]
        if np.isfinite(sub).all():
            continue
        cm = np.nanmean(sub, axis=0)              # per-sensor mean at this hour-of-week
        cm = np.where(np.isfinite(cm), cm, 0.0)
        mask = ~np.isfinite(sub)
        sub[mask] = np.broadcast_to(cm, sub.shape)[mask]
        v[rows] = sub
    return v


# --------------------------------------------------------------------------
def _new_acc(shape):
    return {k: np.zeros(shape) for k in ("n", "sa", "sa2", "sse")}

def _metrics(acc):
    n, sse = acc["n"], acc["sse"]
    mean = acc["sa"] / n
    ss_tot = acc["sa2"] - n * mean ** 2
    rmse = np.sqrt(sse / n)
    r2 = 1 - sse / ss_tot
    return rmse, r2


def evaluate(y, index, forecaster, horizon=H, warmup=4 * PERIOD, stride=24):
    T, N = y.shape
    years = index.year.values
    pooled = dict(n=0.0, sa=0.0, sa2=0.0, sse=0.0)
    ph = _new_acc(horizon)        # per-horizon-step (summed over sensors)
    ps = _new_acc(N)              # per-sensor (summed over horizon)
    py = {}                       # per calendar year of the target
    n_origins = 0
    o = warmup - 1
    while o + horizon < T:
        f = forecaster.fit(y[: o + 1])
        p = f.predict(horizon)
        a = y[o + 1: o + 1 + horizon]
        d2 = (p - a) ** 2
        a2 = a * a
        # pooled
        pooled["n"] += a.size; pooled["sa"] += a.sum()
        pooled["sa2"] += a2.sum(); pooled["sse"] += d2.sum()
        # per-horizon (over sensors)
        ph["n"] += N; ph["sa"] += a.sum(1); ph["sa2"] += a2.sum(1); ph["sse"] += d2.sum(1)
        # per-sensor (over horizon)
        ps["n"] += horizon; ps["sa"] += a.sum(0); ps["sa2"] += a2.sum(0); ps["sse"] += d2.sum(0)
        # per-year
        yrs = years[o + 1: o + 1 + horizon]
        for yr in np.unique(yrs):
            m = yrs == yr
            d = py.setdefault(int(yr), dict(n=0.0, sa=0.0, sa2=0.0, sse=0.0))
            am, dm, am2 = a[m], d2[m], a2[m]
            d["n"] += am.size; d["sa"] += am.sum(); d["sa2"] += am2.sum(); d["sse"] += dm.sum()
        n_origins += 1
        o += stride
    return pooled, ph, ps, py, n_origins


# --------------------------------------------------------------------------
def main():
    years = [int(x) for x in sys.argv[1:]] or [2017, 2018, 2019]
    RESULTS.mkdir(exist_ok=True)
    print(f"Loading GBA years: {years}")
    v, index, ids = load_years(years)
    nan_frac = 1 - np.isfinite(v).mean()
    print(f"  matrix: {v.shape[0]} hours x {v.shape[1]} sensors "
          f"({index[0]} -> {index[-1]})")
    print(f"  NaN-hour fraction before impute: {nan_frac:.4f}")
    v = impute_hour_of_week(v, index)
    print(f"  NaN after impute: {1-np.isfinite(v).mean():.6f}\n")

    models = [HourOfWeekAverage(n_weeks=4), LastWeek(), HourOfWeekAverage(n_weeks=8)]
    out = {}
    for m in models:
        pooled, ph, ps, py, no = evaluate(v, index, m, stride=24)
        rmse, r2 = _metrics(pooled)
        s_rmse, s_r2 = _metrics(ps)        # per-sensor vectors
        print(f"=== {m.name} ===  ({no} origins)")
        print(f"  POOLED  RMSE={rmse:.2f} veh/hr   R2={r2:.4f}")
        print(f"  per-sensor R2: median={np.median(s_r2):.4f} "
              f"[p10={np.percentile(s_r2,10):.4f}, p90={np.percentile(s_r2,90):.4f}]")
        print("  per-year:")
        for yr in sorted(py):
            yr_rmse, yr_r2 = _metrics(py[yr])
            print(f"     {yr}:  RMSE={yr_rmse:6.2f}   R2={yr_r2:.4f}")
        print()
        out[m.name] = {
            "n_origins": no, "pooled_RMSE": float(rmse), "pooled_R2": float(r2),
            "sensor_R2_median": float(np.median(s_r2)),
            "per_year": {int(yr): dict(zip(("RMSE", "R2"), map(float, _metrics(py[yr]))))
                         for yr in sorted(py)},
        }
    (RESULTS / "gba_summary.json").write_text(json.dumps(out, indent=2))
    print("Saved results/gba_summary.json")


if __name__ == "__main__":
    main()

"""Does an XGBoost/ridge model on the seasonal residual with calendar + sensor-
type fixed effects have headroom?

Decisive cheap test: for a categorical feature, the best predictor of the
residual is the group mean. We learn group means on 2017-18 and measure the
out-of-sample residual variance they explain on 2019. This bounds what ANY model
(ridge or XGBoost) using that feature can achieve. Aggregate-R^2 lift = (residual
R^2) x (residual share of total variance, ~0.066).

Features tested (each on the SEASONAL RESIDUAL r = y - b):
  - hour-of-week (168)         : removed by construction -> expect ~0
  - day-of-week, hour-of-day   : subsumed by hour-of-week -> expect ~0
  - week-of-year (52), pooled  : the real candidate (annual signal baseline misses)
  - week-of-year x lanes       : sensor-type interaction (XGBoost's edge)
  - week-of-year x sensor      : full per-sensor seasonal (ceiling, may overfit)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week
from eda import seasonal_baseline

ROOT = Path(__file__).resolve().parents[2]
P = 168


def oos_group_r2(resid, groups, train, test, total_var=None):
    """Residual variance explained OOS by predicting each cell with its training
    group mean. groups: int array aligning with resid rows (per-time) -- here we
    pass a flattened (time,sensor) scheme via a closure instead. Returns r2."""
    raise NotImplementedError


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    b, start = seasonal_baseline(v)
    r = (v - b)
    T, N = v.shape
    sl = slice(start, T)
    r = r[sl]; vv = v[sl]
    idx = index[sl]
    year = idx.year.values
    tr = year <= 2018
    te = year == 2019

    meta = pd.read_csv(ROOT / "data" / "ca" / "ca_meta.csv").set_index("ID")
    lanes = np.array([meta.loc[int(s), "Lanes"] if int(s) in meta.index else 0 for s in ids])

    total_var = float(vv[te].var())
    resid_share = float(r[te].var() / total_var)

    # time-level categorical codes
    how = idx.dayofweek.values * 24 + idx.hour.values      # hour-of-week 0..167
    dow = idx.dayofweek.values
    hod = idx.hour.values
    woy = idx.isocalendar().week.values.astype(int) - 1    # 0..52

    def r2_timefeat(code, per_sensor=False, sensor_group=None):
        """Predict r[t,i] by mean over training rows sharing the same feature
        level. per_sensor: separate mean per sensor. sensor_group: array over
        sensors giving a grouping (e.g. lanes) for interaction."""
        levels = code.max() + 1
        if per_sensor:                                     # level x sensor means
            num = np.zeros((levels, N)); den = np.zeros((levels, N))
            np.add.at(num, code[tr], r[tr]);
            cnt = np.bincount(code[tr], minlength=levels)
            mean = np.where(cnt[:, None] > 0, num / np.maximum(cnt[:, None], 1), 0.0)
            pred = mean[code[te]]
        elif sensor_group is not None:                     # level x group means
            G = sensor_group.max() + 1
            num = np.zeros((levels, G)); cnt = np.zeros((levels, G))
            for g in range(G):
                cols = np.where(sensor_group == g)[0]
                if len(cols) == 0:
                    continue
                sub = r[tr][:, cols]
                np.add.at(num[:, g], code[tr], sub.sum(1))
                c = np.bincount(code[tr], minlength=levels) * len(cols)
                cnt[:, g] = c
            mean = np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)
            pred = mean[code[te]][:, sensor_group]
        else:                                              # pooled level mean
            num = np.zeros(levels);
            np.add.at(num, code[tr], r[tr].sum(1))
            cnt = np.bincount(code[tr], minlength=levels) * N
            mean = np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)
            pred = mean[code[te]][:, None] * np.ones((1, N))
        a = r[te]
        sse = np.sum((a - pred) ** 2)
        sstot = np.sum((a - a.mean()) ** 2)
        return 1 - sse / sstot

    # encode lanes into compact group ids
    uniq = {l: i for i, l in enumerate(sorted(set(lanes)))}
    lane_g = np.array([uniq[l] for l in lanes])

    print(f"residual share of total variance: {resid_share:.3f}")
    print(f"(aggregate-R^2 lift = residual_R2 x {resid_share:.3f})\n")
    print(f"{'feature on residual':<34}{'OOS resid R2':>14}{'agg R2 lift':>13}")
    print("-" * 61)
    tests = [
        ("hour-of-week (168), pooled",      lambda: r2_timefeat(how)),
        ("day-of-week (7), pooled",         lambda: r2_timefeat(dow)),
        ("hour-of-day (24), pooled",        lambda: r2_timefeat(hod)),
        ("week-of-year (52), pooled",       lambda: r2_timefeat(woy)),
        ("week-of-year x lanes",            lambda: r2_timefeat(woy, sensor_group=lane_g)),
        ("week-of-year x sensor",           lambda: r2_timefeat(woy, per_sensor=True)),
        ("hour-of-week x sensor",           lambda: r2_timefeat(how, per_sensor=True)),
    ]
    res = {}
    for name, fn in tests:
        rr = fn()
        res[name] = rr
        print(f"{name:<34}{rr:>14.4f}{rr*resid_share:>13.4f}")

    print("\nInterpretation: features with ~0 here cannot help any model (ridge or")
    print("XGBoost). The aggregate-R^2 lift column is the best-case headroom, to be")
    print("compared against the EWMA+holiday model already at R^2=0.9493.")


if __name__ == "__main__":
    main()

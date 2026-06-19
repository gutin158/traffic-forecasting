"""Explicit holiday-adjusted baseline.

For each 'special day' = (federal holiday identity, offset in {-1,0,+1}) and each
hour-of-day, learn a pooled multiplicative factor

    factor[key, hod] = sum_{sensors, past instances} y  /  sum b

applied to the seasonal baseline on matching test days:

    y_hat = b * factor[key, hod]   on special days,    b   otherwise.

Pooling over all 2,352 sensors denoises the factor; keying by holiday identity
(not a 52-week lag) aligns holidays correctly; the per-hour-of-day resolution
captures the flattening of commute peaks. Train 2017-18, test 2019.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week
from eda import seasonal_baseline
from ewma_baseline import baseline as ewma_baseline

P = 168
Y = 52 * P


def rmse(p, a):
    return float(np.sqrt(np.mean((p - a) ** 2)))

def r2(p, a):
    return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)


def special_day_keys(index):
    """Map each timestamp -> 'HolidayName+off' (or '' if ordinary)."""
    cal = USFederalHolidayCalendar()
    named = cal.holidays(start="2016-12-15", end="2020-01-20", return_name=True)
    by_date = {}
    for date, name in named.items():
        for off in (-1, 0, 1):
            by_date[(date + pd.Timedelta(days=off)).date()] = f"{name}{off:+d}"
    dates = index.normalize()
    keys = np.array([by_date.get(d.date(), "") for d in dates], dtype=object)
    return keys


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    year = index.year.values
    hod = index.hour.values
    keys = special_day_keys(index)

    b, _ = seasonal_baseline(v)                       # flat 4-week baseline
    w = 0.8 ** np.arange(8); w /= w.sum()
    b_ewma, _ = ewma_baseline(v, w)                   # best base so far

    finite = np.isfinite(b[:, 0])                     # b defined (t >= 672)

    # ---- learn factors on 2017-2018 ----
    sy, sb = defaultdict(float), defaultdict(float)
    train_t = np.where((year <= 2018) & (keys != "") & finite)[0]
    for t in train_t:
        cell = (keys[t], hod[t])
        sy[cell] += v[t].sum()
        sb[cell] += b[t].sum()
    factor = {c: sy[c] / sb[c] for c in sy if sb[c] > 0}

    # ---- apply on 2019 ----
    def apply_holiday(base):
        out = base.copy()
        test_t = np.where((year == 2019) & (keys != ""))[0]
        for t in test_t:
            f = factor.get((keys[t], hod[t]))
            if f is not None:
                out[t] = base[t] * f
        return out

    pred_hol = apply_holiday(b)
    pred_ewma_hol = apply_holiday(b_ewma)

    # ---- evaluate on 2019, bucketed ----
    ym = (year == 2019)
    is_hol = ym & (keys != "")
    ordin = ym & (keys == "")
    a = v

    models = {
        "flat baseline":            b,
        "EWMA a=0.8":               b_ewma,
        "flat + holiday model":     pred_hol,
        "EWMA + holiday model":     pred_ewma_hol,
    }
    print(f"learned factors for {len(factor)} (holiday,hour) cells from 2017-18\n")
    print(f"{'model':<24}{'ALL':>9}{'ordinary':>11}{'holiday':>10}{'  R2(all)':>10}")
    print("-" * 64)
    base_all = rmse(b[ym].ravel(), a[ym].ravel())
    for name, p in models.items():
        ra = rmse(p[ym].ravel(), a[ym].ravel())
        ro = rmse(p[ordin].ravel(), a[ordin].ravel())
        rh = rmse(p[is_hol].ravel(), a[is_hol].ravel())
        print(f"{name:<24}{ra:>9.2f}{ro:>11.2f}{rh:>10.2f}{r2(p[ym].ravel(), a[ym].ravel()):>10.4f}")

    print(f"\nholiday windows = {100*is_hol.sum()/ym.sum():.1f}% of 2019 hours")
    # a few interpretable factors (evening commute hour 17)
    print("\nlearned multiplicative factors (hour 17:00):")
    for name in ["Thanksgiving+0", "Independence Day+0", "Thanksgiving-1", "Christmas Day+0"]:
        f = factor.get((name, 17))
        if f is not None:
            print(f"  {name:<22} x{f:.2f}")


if __name__ == "__main__":
    main()

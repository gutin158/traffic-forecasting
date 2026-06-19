"""Does the YoY term's improvement concentrate on holidays?

Compare flat 4-week baseline vs the optimally-blended YoY model on 2019,
bucketed into holiday-window hours vs ordinary hours. If the gain is
concentrated on holidays, the YoY term is effectively a (crude) holiday model.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week
from eda import seasonal_baseline

P = 168
Y = 52 * P


def rmse(p, a):
    return float(np.sqrt(np.mean((p - a) ** 2)))


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    year = index.year.values

    b, _ = seasonal_baseline(v)
    cs = np.cumsum(v, axis=0)
    m = np.full_like(v, np.nan)
    m[672:] = (cs[672:] - cs[:-672]) / 672.0

    vs = Y + 672
    sl, slY = slice(vs, T), slice(vs - Y, T - Y)
    bt, yY, yt = b[sl], v[slY], v[sl]
    g = np.clip(np.where(m[slY] > 1.0, m[sl] / m[slY], 1.0), 0.5, 2.0)
    d = g * yY - bt                                    # YoY term - baseline

    # optimal blend weight, fit on 2018
    tr = (year[sl] == 2018)
    lam = np.sum((yt - bt)[tr] * d[tr]) / np.sum((d * d)[tr])
    pred = bt + lam * d
    print(f"optimal YoY blend weight (fit 2018): lambda* = {lam:.3f}")

    # holiday windows (+/- 1 day) for the 2019 test year
    ts = index[vs:T]
    cal = USFederalHolidayCalendar()
    hols = cal.holidays(start="2018-12-15", end="2020-01-15")
    hol_days = set()
    for h in hols:
        for off in (-1, 0, 1):
            hol_days.add((h + pd.Timedelta(days=off)).date())
    is_hol = np.array([d_.date() in hol_days for d_ in ts])

    ymask = (year[vs:T] == 2019)
    hol = ymask & is_hol
    ord_ = ymask & ~is_hol
    a = yt

    def bucket(mask, label):
        af = a[mask].ravel()
        rf, ry = rmse(bt[mask].ravel(), af), rmse(pred[mask].ravel(), af)
        frac = 100 * mask.sum() / ymask.sum()
        print(f"  {label:<22} ({frac:4.1f}% of hours):  "
              f"flat RMSE {rf:6.2f}  ->  YoY RMSE {ry:6.2f}   "
              f"({100*(1-ry/rf):+5.1f}%)")

    print("\n2019 error, flat baseline vs optimal-YoY, by bucket:")
    bucket(ymask, "ALL 2019")
    bucket(ord_, "ordinary hours")
    bucket(hol,  "holiday +/-1d hours")


if __name__ == "__main__":
    main()

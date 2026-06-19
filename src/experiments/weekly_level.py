"""Operationalize 'the week is the right unit': collapse hourly flow to a per-week
LEVEL series and decompose it. The hourly within-week shape is deterministic
scaffolding; the forecastable multi-week signal lives in this weekly-level series.

  L_w = mean flow over all hours of week w (network mean for the illustration)

We STL-decompose L_w (annual period = 52 weeks) into trend + annual-seasonal +
remainder, report the variance split and the remainder's week-to-week dynamics,
and contrast the clean pre-COVID stats with the full 2017-2021 picture.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "research_notes" / "figures"
P = 168


def weekly_level(years):
    v, index, ids = load_years(years)
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    W = T // P
    Lw_sensor = v[:W * P].reshape(W, P, N).mean(axis=1)     # (weeks, sensors)
    week_start = index[:W * P:P]
    return Lw_sensor, week_start, ids


def var_shares(stl_res):
    t, s, r = stl_res.trend, stl_res.seasonal, stl_res.resid
    comps = {"trend": np.var(t), "annual-seasonal": np.var(s), "remainder": np.var(r)}
    tot = sum(comps.values())
    return {k: v / tot for k, v in comps.items()}


def main():
    # full picture for the figure
    Lw, wk, ids = weekly_level([2017, 2018, 2019, 2020, 2021])
    net = Lw.mean(axis=1)                                   # network-mean weekly level
    s_full = pd.Series(net, index=pd.DatetimeIndex(wk))
    stl = STL(s_full, period=52, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(s_full.index, s_full.values, color="#222"); axes[0].set_ylabel("level\n(veh/h)")
    axes[0].set_title("Weekly traffic LEVEL series $L_w$ and its STL decomposition "
                      "(network mean, 2017–2021)")
    axes[1].plot(s_full.index, stl.trend, color="#06a"); axes[1].set_ylabel("trend")
    axes[2].plot(s_full.index, stl.seasonal, color="#3a7"); axes[2].set_ylabel("annual\nseasonal")
    axes[3].plot(s_full.index, stl.resid, color="#a36"); axes[3].set_ylabel("remainder")
    axes[3].axhline(0, color="0.6", lw=0.8); axes[3].set_xlabel("week")
    fig.tight_layout(); fig.savefig(FIG / "fig14_weekly_level.png", dpi=130); plt.close(fig)

    print("FULL 2017-2021 weekly-level variance shares:")
    for k, val in var_shares(stl).items():
        print(f"   {k:<16}: {100*val:5.1f}%")

    # clean pre-COVID dynamics: is there week-to-week structure beyond trend+seasonal?
    Lw3, wk3, _ = weekly_level([2017, 2018, 2019])
    net3 = pd.Series(Lw3.mean(1), index=pd.DatetimeIndex(wk3))
    stl3 = STL(net3, period=52, robust=True).fit()
    rem = stl3.resid.values
    ac = acf(rem, nlags=4, fft=False)
    print("\nPRE-COVID 2017-2019 weekly-level:")
    for k, val in var_shares(stl3).items():
        print(f"   {k:<16}: {100*val:5.1f}%")
    print(f"   remainder ACF lag1..4: {np.round(ac[1:5],3)}")
    print(f"   weekly-level series length: {len(net3)} weeks (vs {len(net3)*P} hours) "
          f"-- a short, model-able macro series")
    # dimensionality note: per-sensor it's (weeks x 2352)
    print(f"   per-sensor object: {Lw3.shape[0]} weeks x {Lw3.shape[1]} sensors")
    print("wrote fig14_weekly_level.png")


if __name__ == "__main__":
    main()

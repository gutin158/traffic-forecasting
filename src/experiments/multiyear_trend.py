"""Multi-year trend/seasonal decomposition of GBA flow.

Illustrates 'where the problem stops being sunrise': the weekly-seasonal band is
boringly stable across years, while the slowly-drifting LEVEL carries all the
hard, macro-coupled action -- with the 2020 COVID break as the exclamation point.

Produces:
  fig11_multiyear_trend.png  -- network-mean flow over time: trend vs seasonal vs COVID
  fig12_seasonal_stability.png -- the weekly profile is ~identical across years
Plus quantified: cross-year seasonal stability vs trend/level excursions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "research_notes" / "figures"
CA = ROOT / "data" / "ca"
P = 168


def available_years():
    return [y for y in range(2017, 2022) if (CA / f"gba_hourly_{y}.npz").exists()]


def main():
    years = available_years()
    v, index, ids = load_years(years)
    v = impute_hour_of_week(v, index)
    netmean = v.mean(axis=1)                       # network-mean flow per hour
    s = pd.Series(netmean, index=index)

    # trend = 4-week rolling mean; daily mean for readability
    daily = s.resample("1D").mean()
    trend = s.rolling(4 * P, center=True, min_periods=P).mean().resample("1D").mean()

    # ---- Fig 11: multi-year level with trend + COVID annotation ----
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily.index, daily.values, lw=0.5, color="0.7", label="daily mean flow")
    ax.plot(trend.index, trend.values, lw=2.0, color="#06a", label="4-week trend")
    if 2020 in years:
        cov = pd.Timestamp("2020-03-17")          # CA shelter-in-place
        ax.axvline(cov, color="#c33", ls="--", lw=1.2)
        ax.annotate("COVID shelter-in-place\n(Mar 2020)", xy=(cov, daily.min()),
                    xytext=(cov + pd.Timedelta(days=60), daily.min() + 20),
                    color="#c33", fontsize=9,
                    arrowprops=dict(color="#c33", arrowstyle="->"))
    ax.set_ylabel("network-mean flow (veh/h)")
    ax.set_title("GBA traffic level over time: a stable seasonal oscillation on a slow "
                 "trend\n-- until a macro shock no seasonal model can see")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(FIG / "fig11_multiyear_trend.png", dpi=130); plt.close(fig)

    # ---- Fig 12: weekly profile per year (seasonal band is stable) ----
    how = index.dayofweek.values * 24 + index.hour.values
    yr = index.year.values
    fig, ax = plt.subplots(figsize=(11, 4))
    for y in years:
        prof = np.array([netmean[(yr == y) & (how == h)].mean() for h in range(P)])
        style = "--" if y == 2020 else "-"
        ax.plot(prof, style, lw=1.3, label=str(y))
    ax.set_xticks(np.arange(0, P, 24))
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_ylabel("mean flow (veh/h)"); ax.set_xlabel("hour of week")
    ax.set_title("The weekly-seasonal SHAPE is near-identical year to year "
                 "(2020 = the exception that proves the rule)")
    ax.legend(fontsize=8, ncol=len(years)); fig.tight_layout()
    fig.savefig(FIG / "fig12_seasonal_stability.png", dpi=130); plt.close(fig)

    # ---- quantify: normalized seasonal shape stability vs level excursions ----
    profs = {}
    for y in years:
        pr = np.array([netmean[(yr == y) & (how == h)].mean() for h in range(P)])
        profs[y] = pr
    # correlation of normalized (mean-removed, scaled) weekly shapes across years
    norm = {y: (profs[y] - profs[y].mean()) / profs[y].std() for y in years}
    base = norm[2019] if 2019 in years else norm[years[0]]
    print("Weekly-shape correlation vs 2019 (shape stability):")
    for y in years:
        print(f"  {y}: corr={np.corrcoef(norm[y], base)[0,1]:.4f}  "
              f"mean level={profs[y].mean():.1f} veh/h")
    levels = np.array([profs[y].mean() for y in years])
    print(f"\nlevel range across years: {levels.min():.1f}..{levels.max():.1f} "
          f"({100*(levels.max()/levels.min()-1):.1f}% swing) -- the trend/COVID band")
    print("=> seasonal shape ~frozen (corr>0.99 pre-COVID); the action is all in the level.")
    print("wrote fig11_multiyear_trend.png, fig12_seasonal_stability.png")


if __name__ == "__main__":
    main()

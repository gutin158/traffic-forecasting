"""Figures for research note 02: holiday daily-profile reshaping (pooled factor)
and per-sensor holiday heterogeneity."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week
from holiday_model import special_day_keys
from run_holiday_backtest import WeeklyBaseline, learn_factors
from holiday_region import learn_scale

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "research_notes" / "figures"


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    year = index.year.values; hod = index.hour.values
    keys = special_day_keys(index)
    train = year <= 2018
    w = 0.8 ** np.arange(8); w /= w.sum()
    ewma = WeeklyBaseline(v, w)
    shape = learn_factors(ewma, v, keys, hod, train)

    # Fig A: pooled multiplicative factor vs hour-of-day, several holidays
    fig, ax = plt.subplots(figsize=(9, 4))
    for hol in ["Independence Day", "Thanksgiving Day", "Christmas Day", "Labor Day"]:
        prof = [shape.get((hol + "+0", h), np.nan) for h in range(24)]
        ax.plot(range(24), prof, marker="o", ms=3, label=hol)
    ax.axhline(1.0, color="0.6", ls="--", lw=1, label="normal (=1)")
    ax.set_xlabel("hour of day"); ax.set_ylabel("holiday factor  (actual / baseline)")
    ax.set_title("Learned holiday factors reshape the daily profile\n"
                 "(values <1 = commute peaks suppressed)")
    ax.legend(fontsize=8); ax.set_xticks(range(0, 24, 3))
    fig.tight_layout(); fig.savefig(FIG / "fig6_holiday_shape.png", dpi=130); plt.close(fig)

    # Fig B: per-sensor scale heterogeneity
    id_idx, scale = learn_scale(ewma, v, keys, hod, shape, train, tau=50)
    fig, ax = plt.subplots(figsize=(9, 4))
    for hol, c in [("Independence Day", "#06a"), ("Thanksgiving Day", "#a36")]:
        sc = scale[:, id_idx[hol]]
        ax.hist(sc, bins=50, alpha=0.55, color=c, label=f"{hol} (median {np.median(sc):.2f})")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("per-sensor holiday scale (>1 = corridor gains traffic vs pooled)")
    ax.set_ylabel("# sensors")
    ax.set_title("Holiday response is heterogeneous across corridors")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "fig7_holiday_heterogeneity.png", dpi=130); plt.close(fig)
    print("wrote fig6_holiday_shape.png, fig7_holiday_heterogeneity.png")


if __name__ == "__main__":
    main()

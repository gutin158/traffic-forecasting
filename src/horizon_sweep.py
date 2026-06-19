"""At what horizon H (weeks) does week-ahead flow forecasting become 'interesting'
in STEADY STATE (pre-COVID 2017-2019, no regime breaks)?

Decompose the trailing baseline's error vs H into:
  - a horizon-INDEPENDENT floor (centered oracle: seasonal shape + concurrent level
    using weeks +/-1,+/-2 around the target -- uses future, so no staleness; this is
    ~the irreducible+seasonal floor)
  - a horizon-GROWING staleness cost (trailing uses the 4 most recent AVAILABLE
    weeks {H..H+3}, which age with H)
'Interesting' = when the recoverable gap (what a smarter level/trend model could
buy) exceeds the scale of gains we already bothered to chase (EWMA/holiday ~2% RMSE).
We bracket the recoverable gap by (a) a simple annual-anchor blend and (b) the
centered oracle (upper bound).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "research_notes" / "figures"
P = 168


def r2(p, a): return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
def rmse(p, a): return float(np.sqrt(np.mean((p - a) ** 2)))


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    year = index.year.values
    Hmax = 12
    S0 = 56 * P                                  # room for trailing {H..H+3} and annual {52..55}
    S1 = T - 2 * P                               # room for centered oracle (+2 weeks)
    yrs = year[S0:S1]
    tr, te = yrs <= 2018, yrs == 2019

    def pred(Ks):                                # mean of weekly lags Ks (k>0 back, k<0 future)
        acc = np.zeros((S1 - S0, N))
        for k in Ks:
            acc += v[S0 - P * k: S1 - P * k]
        return acc / len(Ks)

    a_te = v[S0:S1][te]
    annual = pred([52, 53, 54, 55])              # horizon-independent calendar anchor
    oracle = pred([-2, -1, 1, 2])                # centered: no staleness (uses future) -> floor

    Hs = list(range(1, Hmax + 1))
    trail_rmse, blend_rmse, blend_w = [], [], []
    for Hw in Hs:
        trailing = pred([Hw, Hw + 1, Hw + 2, Hw + 3])
        t_te = trailing[te]
        # optimal blend weight on annual, fit 2018
        t_tr, an_tr, y_tr = trailing[tr], annual[tr], v[S0:S1][tr]
        d = (an_tr - t_tr).ravel()
        w = float(np.clip(np.sum((y_tr - t_tr).ravel() * d) / np.sum(d * d), 0, 1))
        blend = (1 - w) * t_te + w * annual[te]
        trail_rmse.append(rmse(t_te, a_te)); blend_rmse.append(rmse(blend, a_te))
        blend_w.append(w)

    floor = rmse(oracle[te], a_te)               # horizon-independent
    print(f"centered-oracle floor (no staleness): RMSE={floor:.2f}  R2={r2(oracle[te],a_te):.4f}\n")
    print(f"{'H(wk)':<6}{'trail RMSE':>11}{'stale cost':>11}{'blend gain':>11}"
          f"{'oracle gap':>11}{'annual wt':>10}")
    print("-" * 60)
    for i, Hw in enumerate(Hs):
        stale = trail_rmse[i] - trail_rmse[0]                  # vs fresh (H=1)
        bgain = 100 * (1 - blend_rmse[i] / trail_rmse[i])      # % RMSE recovered by blend
        ogap = 100 * (1 - floor / trail_rmse[i])               # % gap to oracle floor
        print(f"{Hw:<6}{trail_rmse[i]:>11.2f}{stale:>11.2f}{bgain:>10.1f}%"
              f"{ogap:>10.1f}%{blend_w[i]:>10.2f}")

    # ---- figure: error vs horizon with floor + recoverable band ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(Hs, trail_rmse, "o-", color="#c33", label="trailing baseline (naive)")
    ax.plot(Hs, blend_rmse, "s-", color="#06a", label="+ annual anchor (simple)")
    ax.axhline(floor, ls="--", color="0.4", label="centered oracle (no staleness, ~floor)")
    ax.fill_between(Hs, blend_rmse, trail_rmse, color="#06a", alpha=0.12)
    # threshold: where staleness cost exceeds a holiday-scale gain (~2% of RMSE ~ 0.8 veh/h)
    thr = trail_rmse[0] * 1.02
    ax.axhline(thr, ls=":", color="#3a7", lw=1)
    ax.text(8.4, thr + 0.2, "+2% RMSE (holiday-scale)", color="#3a7", fontsize=8)
    ax.set_xlabel("forecast horizon H (weeks)"); ax.set_ylabel("RMSE (veh/h)")
    ax.set_title("Steady-state (pre-COVID): when does staleness make richer modeling worth it?")
    ax.legend(fontsize=9); ax.set_xticks(Hs)
    fig.tight_layout(); fig.savefig(FIG / "fig13_horizon_sweep.png", dpi=130)
    # crossover horizon
    cross = next((Hs[i] for i in range(len(Hs)) if trail_rmse[i] >= thr), None)
    print(f"\ntrailing error crosses +2% (holiday-scale) staleness at H = {cross} weeks")
    print("wrote fig13_horizon_sweep.png")


if __name__ == "__main__":
    main()

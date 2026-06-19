"""Are 4-8 week horizons still 'seasonal-anchor dominates', or does annual/trend
info become exploitable?

At horizon H weeks, a weekly lag k (weeks before target) is observable iff k>=H.
So the trailing-weeks baseline uses the 4 most recent AVAILABLE weeks {H..H+3},
which grow stale as H grows. We compare:
  - trailing(H):  mean of weeks {H, H+1, H+2, H+3} back   (fresh at H=1, stale at H=8)
  - annual:       mean of weeks {52..55} back (prior-year, calendar-aligned, H-independent)
  - blend:        optimal convex combo (weight fit on 2018)
Pooled R^2/RMSE on 2019, for H in {1,2,4,8} weeks. Hypothesis: trailing degrades
with H while annual stays flat, so the optimal blend leans on annual at long H ->
annual/sub-annual-seasonal info becomes exploitable beyond ~1 week.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week

P = 168


def r2(p, a): return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
def rmse(p, a): return float(np.sqrt(np.mean((p - a) ** 2)))


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    year = index.year.values
    S0 = 56 * P                                   # need up to 55 weeks of history
    yrs = year[S0:T]
    tr, te = yrs <= 2018, yrs == 2019

    def predictor(Ks):
        acc = np.zeros((T - S0, N))
        for k in Ks:
            acc += v[S0 - P * k: T - P * k]
        return acc / len(Ks)

    annual = predictor([52, 53, 54, 55])          # H-independent calendar anchor
    a_te = v[S0:T][te]

    print(f"{'H (weeks)':<11}{'trailing R2':>13}{'trail RMSE':>12}"
          f"{'blend R2':>11}{'blend RMSE':>12}{'annual wt':>11}")
    print("-" * 70)
    rows = []
    for Hw in [1, 2, 4, 8]:
        trailing = predictor([Hw, Hw + 1, Hw + 2, Hw + 3])
        t_te = trailing[te]; an_te = annual[te]
        # optimal convex blend weight w on annual, fit on 2018
        t_tr = trailing[tr]; an_tr = annual[tr]; y_tr = v[S0:T][tr]
        d = (an_tr - t_tr).ravel()
        num = float(np.sum((y_tr - t_tr).ravel() * d)); den = float(np.sum(d * d))
        w = np.clip(num / den, 0.0, 1.0)
        blend = (1 - w) * t_te + w * an_te
        rows.append((Hw, r2(t_te, a_te), rmse(t_te, a_te),
                     r2(an_te, a_te), rmse(an_te, a_te),
                     r2(blend, a_te), rmse(blend, a_te), w))
        print(f"{Hw:<11}{r2(t_te,a_te):>13.4f}{rmse(t_te,a_te):>12.2f}"
              f"{r2(blend,a_te):>11.4f}{rmse(blend,a_te):>12.2f}{w:>11.2f}")

    print("\nannual-only baseline (H-independent):"
          f"  R2={r2(annual[te],a_te):.4f}  RMSE={rmse(annual[te],a_te):.2f}")
    print("\nreading: trailing R2 drop from H=1 to H=8 = cost of level staleness;")
    print("rising annual weight = annual/sub-annual-seasonal info becoming exploitable.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = Path(__file__).resolve().parents[2] / "docs" / "research_notes" / "figures"
    Hs = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(Hs, [r[2] for r in rows], "o-", label="trailing baseline", color="#c33")
    ax1.plot(Hs, [r[6] for r in rows], "s-", label="+ annual blend", color="#06a")
    ax1.axhline(rmse(annual[te], a_te), ls="--", color="0.5", label="annual-only")
    ax1.set_xlabel("forecast horizon (weeks)"); ax1.set_ylabel("RMSE (veh/h)")
    ax1.set_title("Trailing baseline goes stale;\nannual anchor stays flat"); ax1.legend(fontsize=8)
    ax2.plot(Hs, [r[7] for r in rows], "D-", color="#3a7")
    ax2.set_xlabel("forecast horizon (weeks)")
    ax2.set_ylabel("optimal weight on annual anchor")
    ax2.set_title("Annual / sub-annual-seasonal info\nbecomes exploitable past ~1 week")
    ax2.set_ylim(0, 0.45)
    fig.tight_layout(); fig.savefig(FIG / "fig10_multiweek.png", dpi=130)
    print("wrote", FIG / "fig10_multiweek.png")


if __name__ == "__main__":
    main()

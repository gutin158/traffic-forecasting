"""How much can recent-lag (ARIMA-style) modeling of the seasonal residual help,
as a function of horizon, and what does it sum to under a week-aggregate metric?

At horizon h, only residual lags ell >= h are observable. For each h we fit the
best linear predictor of r_{t+h} from the available lags (train 2017-18, test
2019) and report the residual variance it explains. We then average over
h=1..168 and scale by the residual's share of total variance to get the best-
case lift in aggregate R^2.
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
from eda import seasonal_baseline

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "research_notes" / "figures"
P = 168
CAND = [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 96, 120, 144, 168, 336]


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    b, start = seasonal_baseline(v)
    r = v - b
    T, N = v.shape
    year = index.year.values

    rng = np.random.default_rng(0)
    sub = rng.choice(N, 150, replace=False)
    rr = r[:, sub]

    s0 = start + max(CAND)                      # first target with all candidate lags
    s = np.arange(s0, T)
    yr_s = year[s0:T]
    # design: intercept + candidate lags
    cols = [np.ones((len(s), len(sub)))]
    for L in CAND:
        cols.append(rr[s0 - L: T - L])
    X = np.stack(cols, -1).reshape(-1, len(CAND) + 1)     # (n, 1+ncand)
    Y = rr[s0:T].reshape(-1)
    yrf = np.repeat(yr_s, len(sub))
    tr, te = yrf <= 2018, yrf == 2019

    # precompute Gram matrices
    Gtr = X[tr].T @ X[tr]; ctr = X[tr].T @ Y[tr]
    Gte = X[te].T @ X[te]; cte = X[te].T @ Y[te]
    yyte = float(Y[te] @ Y[te]); nte = int(te.sum()); ybar = Y[te].mean()
    sstot = yyte - nte * ybar ** 2

    def resid_r2(cols_sel):
        beta = np.linalg.solve(Gtr[np.ix_(cols_sel, cols_sel)], ctr[cols_sel])
        sse = yyte - 2 * beta @ cte[cols_sel] + beta @ Gte[np.ix_(cols_sel, cols_sel)] @ beta
        return 1 - sse / sstot

    # R^2 of residual at each horizon h (lags >= h, plus intercept col 0)
    r2_of_h = np.zeros(P + 1)
    for h in range(1, P + 1):
        sel = [0] + [j + 1 for j, L in enumerate(CAND) if L >= h]
        r2_of_h[h] = resid_r2(sel) if len(sel) > 1 else 0.0

    resid_frac = float(rr[s0:T].var() / v[s0:T, sub].var())
    avg_resid_r2 = float(np.mean(r2_of_h[1:P + 1]))
    agg_lift = avg_resid_r2 * resid_frac

    print("residual variance explained by best available-lag predictor, by horizon:")
    for h in [1, 2, 3, 6, 12, 24, 48, 72, 120, 168]:
        print(f"   h={h:3d}h :  resid R^2 = {r2_of_h[h]:.3f}")
    print(f"\nresidual share of total variance: {resid_frac:.3f}")
    print(f"horizon-averaged residual R^2 (h=1..168): {avg_resid_r2:.3f}")
    print(f"=> best-case lift in AGGREGATE R^2 from all recent-lag info: "
          f"+{agg_lift:.4f}")
    print(f"   (i.e. ~{100*agg_lift:.2f}% of total variance; aggregate R^2 "
          f"0.9493 -> ~{0.9493+agg_lift:.4f})")
    # how concentrated: share of the gain in the first day
    front = np.mean(r2_of_h[1:25]); back = np.mean(r2_of_h[25:P + 1])
    print(f"\nconcentration: avg resid R^2 over h=1..24 = {front:.3f} vs "
          f"h=25..168 = {back:.3f}")

    hs = np.arange(1, P + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hs, r2_of_h[1:P + 1], color="#06a")
    ax.fill_between(hs, r2_of_h[1:P + 1], alpha=0.2, color="#06a")
    ax.axvline(24, color="0.6", ls="--", lw=1)
    ax.text(27, 0.45, "1 day", color="0.4")
    ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168])
    ax.set_xlabel("forecast horizon $h$ (hours ahead)")
    ax.set_ylabel("residual variance explained\nby best available-lag predictor")
    ax.set_title("Recent-lag predictability of the seasonal residual collapses with horizon\n"
                 f"(horizon-averaged {avg_resid_r2:.3f} -> aggregate $R^2$ lift only +{agg_lift:.4f})")
    fig.tight_layout(); fig.savefig(FIG / "fig8_horizon_predictability.png", dpi=130)
    print("wrote", FIG / "fig8_horizon_predictability.png")


if __name__ == "__main__":
    main()

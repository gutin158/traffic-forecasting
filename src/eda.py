"""EDA for research note 01: seasonal structure of Bay Area traffic flow and
the seasonal-baseline residual decomposition.

Produces figures in docs/research_notes/figures/ and prints key statistics.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week  # noqa: E402
from harness import PERIOD  # 168

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "research_notes" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
META = ROOT / "data" / "ca" / "ca_meta.csv"
NW = 4  # weeks in the seasonal baseline


def seasonal_baseline(y, n_weeks=NW, period=PERIOD):
    """b[t] = mean_{k=1..nw} y[t-period*k], defined for t >= nw*period."""
    T = y.shape[0]
    start = n_weeks * period
    b = np.full_like(y, np.nan)
    acc = np.zeros((T - start, y.shape[1]), dtype=np.float64)
    for k in range(1, n_weeks + 1):
        acc += y[start - period * k: T - period * k]
    b[start:] = (acc / n_weeks).astype(y.dtype)
    return b, start


def main():
    years = [2017, 2018, 2019]
    v, index, ids = load_years(years)
    v = impute_hour_of_week(v, index)
    meta = pd.read_csv(META).set_index("ID")
    ids_int = ids.astype(int)
    fwy = [f"{meta.loc[i,'Fwy']} {meta.loc[i,'Direction']}" if i in meta.index else str(i)
           for i in ids_int]
    T, N = v.shape
    print(f"matrix {v.shape}  {index[0]} -> {index[-1]}")

    how = index.dayofweek.values * 24 + index.hour.values

    # --- baseline / residual decomposition ---
    b, start = seasonal_baseline(v)
    r = v - b                                  # residual; valid for t>=start
    rv = r[start:]
    yv = v[start:]
    var_y = yv.var(0)
    var_r = rv.var(0)
    ve = 1 - var_r / var_y                     # per-sensor variance explained
    pooled_ve = 1 - rv.var() / yv.var()
    print(f"pooled variance explained by 4wk seasonal baseline: {pooled_ve:.4f}")
    print(f"per-sensor VE: median={np.median(ve):.4f} "
          f"p10={np.percentile(ve,10):.4f} p90={np.percentile(ve,90):.4f}")
    print(f"std(y)={yv.std():.1f}  std(resid)={rv.std():.1f}  "
          f"ratio={rv.std()/yv.std():.3f}")

    mean_flow = v.mean(0)
    order = np.argsort(mean_flow)
    busy = order[int(0.90 * N)]
    med = order[int(0.50 * N)]
    quiet = order[int(0.10 * N)]

    # ---- Fig 1: weekly profile (hour-of-week mean) ----
    fig, ax = plt.subplots(figsize=(10, 4))
    for s, lab in [(busy, "busy"), (med, "median"), (quiet, "quiet")]:
        prof = np.array([v[how == h, s].mean() for h in range(PERIOD)])
        ax.plot(prof, label=f"{lab}: sensor {ids[s]} (Fwy {fwy[s]})")
    for d in range(1, 7):
        ax.axvline(d * 24, color="0.85", lw=0.8, zorder=0)
    ax.set_xticks(np.arange(0, PERIOD, 24))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("hour of week"); ax.set_ylabel("mean flow (veh/hr)")
    ax.set_title("Weekly flow profile (averaged over 2017–2019)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig1_weekly_profile.png", dpi=130)
    plt.close(fig)

    # ---- Fig 2: weekday x hour heatmap for the busy sensor ----
    grid = np.array([[v[(index.dayofweek.values == d) & (index.hour.values == h), busy].mean()
                      for h in range(24)] for d in range(7)])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(7)); ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hour of day")
    ax.set_title(f"Mean flow heatmap — sensor {ids[busy]} (Fwy {fwy[busy]})")
    fig.colorbar(im, label="veh/hr"); fig.tight_layout()
    fig.savefig(FIG / "fig2_heatmap.png", dpi=130); plt.close(fig)

    # ---- Fig 3: per-sensor variance-explained histogram ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ve, bins=40, color="#3a7", edgecolor="k", lw=0.3)
    ax.axvline(np.median(ve), color="k", ls="--", label=f"median {np.median(ve):.3f}")
    ax.set_xlabel("variance explained by 4-week seasonal baseline (per sensor)")
    ax.set_ylabel("# sensors"); ax.set_title("How much variance the seasonal baseline removes")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig3_ve_hist.png", dpi=130); plt.close(fig)

    # ---- Fig 4: decomposition for busy sensor over 3 weeks ----
    seg = slice(start + 1000, start + 1000 + 3 * PERIOD)
    t = np.arange(3 * PERIOD)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(t, v[seg, busy], label="observed $y$", color="#222", lw=1)
    axes[0].plot(t, b[seg, busy], label="seasonal baseline $b$", color="#e07", lw=1.2)
    axes[0].set_ylabel("veh/hr"); axes[0].legend(fontsize=9)
    axes[0].set_title(f"Decomposition — sensor {ids[busy]} (Fwy {fwy[busy]}), 3 weeks")
    axes[1].plot(t, r[seg, busy], color="#06a", lw=0.9)
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set_ylabel("residual $r=y-b$"); axes[1].set_xlabel("hour")
    for d in range(1, 21):
        for ax in axes: ax.axvline(d * 24, color="0.92", lw=0.6, zorder=0)
    fig.tight_layout(); fig.savefig(FIG / "fig4_decomposition.png", dpi=130); plt.close(fig)

    # ---- Fig 5: residual ACF (mean over sample of sensors) ----
    rng = np.random.default_rng(0)
    sample = rng.choice(N, size=200, replace=False)
    L = 72
    acfs = np.array([acf(r[start:, s], nlags=L, fft=True) for s in sample])
    macf = acfs.mean(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(L + 1), macf, color="#a36", width=0.8)
    ax.set_xlabel("lag (hours)"); ax.set_ylabel("autocorrelation")
    ax.set_title("Residual ACF (mean over 200 sensors) — motivates an AR model on residuals")
    fig.tight_layout(); fig.savefig(FIG / "fig5_residual_acf.png", dpi=130); plt.close(fig)

    print("\nresidual ACF: lag1={:.3f} lag2={:.3f} lag24={:.3f} lag168(approx via {})={:.3f}"
          .format(macf[1], macf[2], macf[24], L, macf[-1]))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()

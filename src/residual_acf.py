"""Characterise residual autocorrelation at SHORT and WEEKLY (seasonal) lags,
to drive the SARIMA-style residual-model specification.

residual r_{i,t} = y_{i,t} - (1/4) sum_{k=1..4} y_{i,t-168k}
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week
from eda import seasonal_baseline
from harness import PERIOD  # 168

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "research_notes" / "figures"


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    b, start = seasonal_baseline(v)
    r = (v - b)[start:]                      # (T', N) residuals
    T2, N = r.shape

    rng = np.random.default_rng(0)
    sample = rng.choice(N, size=250, replace=False)
    L = 24 * 7 * 4 + 5                       # ~4 weeks of lags
    A = np.array([acf(r[:, s], nlags=L, fft=True) for s in sample])
    macf = A.mean(0)

    keys = [1, 2, 3, 6, 12, 24, 25, 48, 72, 167, 168, 169, 336, 504, 672]
    print("mean residual ACF at key lags:")
    for k in keys:
        tag = ""
        if k % PERIOD == 0:
            tag = f"  <-- weekly lag {k // PERIOD}w"
        elif k % 24 == 0:
            tag = f"  <-- daily lag {k // 24}d"
        print(f"  lag {k:4d}: {macf[k]:+.3f}{tag}")

    # PACF to see whether weekly lag carries PARTIAL autocorrelation beyond short lags
    Lp = 175
    P = np.array([pacf(r[:, s], nlags=Lp, method="ywm") for s in sample])
    mpacf = P.mean(0)
    print("\nmean residual PACF at key lags:")
    for k in [1, 2, 3, 24, 168]:
        print(f"  lag {k:4d}: {mpacf[k]:+.3f}")

    # ---- plot: full ACF with weekly markers + zoom around weekly lags ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].bar(np.arange(L + 1), macf, width=1.0, color="#a36")
    for w in range(1, 5):
        axes[0].axvline(w * PERIOD, color="#06a", lw=1.0, ls="--",
                        label="weekly lag" if w == 1 else None)
    axes[0].axhline(0, color="0.5", lw=0.6)
    axes[0].set_xlabel("lag (hours)"); axes[0].set_ylabel("ACF")
    axes[0].set_title("Residual ACF out to 4 weeks (mean over 250 sensors)")
    axes[0].legend(fontsize=8)

    # zoom: bars at weekly lags vs neighbourhood
    zl = np.arange(150, 200)
    axes[1].bar(zl, macf[150:200], width=0.8, color="#a36")
    axes[1].axvline(PERIOD, color="#06a", lw=1.0, ls="--")
    axes[1].axhline(0, color="0.5", lw=0.6)
    axes[1].set_xlabel("lag (hours)"); axes[1].set_ylabel("ACF")
    axes[1].set_title("Zoom near the 1-week lag (168h): is there a seasonal spike?")
    fig.tight_layout(); fig.savefig(FIG / "fig5_residual_acf.png", dpi=130)
    plt.close(fig)
    print("\nupdated", FIG / "fig5_residual_acf.png")


if __name__ == "__main__":
    main()

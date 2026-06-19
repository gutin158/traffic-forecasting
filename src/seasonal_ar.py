"""Test the seasonal-residual AR for week-ahead forecasting.

seasonal residual:  r_s = y_s - (1/4) sum_{k=1..4} y_{s-168k}
model:              r_hat_s = psi0 + sum_{m=1..M} psi_m r_{s-168m}
combined forecast:  y_hat_s = b_s + r_hat_s

All features r_{s-168m} are known at a forecast origin <= s-168, so this is a
valid one-week-ahead model at every horizon h in 1..168 (no recursion).

Train on 2017-2018, test out-of-sample on 2019.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week
from eda import seasonal_baseline

PERIOD = 168


def r2(p, a):
    return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)

def rmse(p, a):
    return float(np.sqrt(np.mean((p - a) ** 2)))


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4   # # of weekly lags
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    b, start = seasonal_baseline(v)          # start = 672
    r = v - b
    T, N = v.shape
    S0 = start + M * PERIOD                   # first target with all M weekly lags

    rng = np.random.default_rng(0)
    sub = rng.choice(N, size=600, replace=False)
    rr, vv, bb = r[:, sub], v[:, sub], b[:, sub]

    Y = rr[S0:T]                              # (n, ns) target seasonal residual
    F = np.stack([rr[S0 - PERIOD * m: T - PERIOD * m] for m in range(1, M + 1)], -1)
    yr = index[S0:T].year.values
    train, test = yr <= 2018, yr == 2019

    Xtr = F[train].reshape(-1, M); ytr = Y[train].reshape(-1)
    Xte = F[test].reshape(-1, M)
    rte = Y[test].reshape(-1)
    bte = bb[S0:T][test].reshape(-1)
    vte = vv[S0:T][test].reshape(-1)

    # OLS with intercept (pooled across sensors)
    Atr = np.column_stack([np.ones(len(ytr)), Xtr])
    coef, *_ = np.linalg.lstsq(Atr, ytr, rcond=None)
    rhat = np.column_stack([np.ones(len(rte)), Xte]) @ coef

    print(f"=== seasonal-residual AR with M={M} weekly lags (train 2017-18, test 2019) ===")
    print(f"  coefficients: intercept={coef[0]:+.3f}  " +
          "  ".join(f"psi_{m}(lag {m}w)={coef[m]:+.3f}" for m in range(1, M + 1)))
    print(f"  residual R^2 on test (predicting r):      {r2(rhat, rte):+.4f}")
    print()
    # forecast accuracy on y
    y_base = bte                              # baseline only (r_hat = 0)
    y_model = bte + rhat
    print(f"  one-week-ahead y  R^2 :  baseline={r2(y_base, vte):.4f}   "
          f"+seasonal-AR={r2(y_model, vte):.4f}")
    print(f"  one-week-ahead y RMSE :  baseline={rmse(y_base, vte):.2f}   "
          f"+seasonal-AR={rmse(y_model, vte):.2f}   "
          f"({100*(1-rmse(y_model,vte)/rmse(y_base,vte)):.1f}% lower)")

    # single-lag reference: how good is just the 1-week lag alone?
    if M >= 1:
        a1 = np.column_stack([np.ones(len(ytr)), Xtr[:, :1]])
        c1, *_ = np.linalg.lstsq(a1, ytr, rcond=None)
        rh1 = np.column_stack([np.ones(len(rte)), Xte[:, :1]]) @ c1
        print(f"\n  (reference) 1 weekly lag only: resid R^2={r2(rh1, rte):+.4f}, "
              f"y RMSE={rmse(bte+rh1, vte):.2f}")


if __name__ == "__main__":
    main()

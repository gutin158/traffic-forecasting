"""Pooled (panel) models for forecasting the weekly level L_{i,w} at H=4 weeks.

Short per-sensor series (~104 train weeks) make per-sensor ARIMA hopeless; the fix
is POOLING -- one shared set of coefficients estimated across all 2352 sensors.
We work in deseasonalized space (remove the trained week-of-year mean), forecast
the level/remainder with a pooled direct AR, then re-add the CORRECT target-week
seasonal -- which is exactly the seasonal-phase staleness the trailing rule suffers.

Models (all forecast L_{i,target}, origin = target-4, test 2019):
  trailing (rule)        : mean of raw L over weeks {4,5,6,7} back
  seasonal-naive (rule)  : L[target-52]
  blend (rule, 1 param)  : optimal convex trailing + seasonal-naive (weight fit 2018)
  pooled AR (direct OLS) : deseason-AR on lags {4..7}, shared coefs, + correct season
  pooled AR + ridge      : same, ridge-regularized
  global GBM (trained ML): HistGBM, raw-level features, all sensors
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
P = 168; SY = 52; H = 4


def r2(p, a): return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
def rmse(p, a): return float(np.sqrt(np.mean((p - a) ** 2)))


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape; W = T // P
    L = v[:W * P].reshape(W, P, N).mean(1)                  # (W, N) weekly level
    year = index[:W * P:P].year.values
    woy = np.arange(W) % SY
    tr = year <= 2018

    # trained week-of-year seasonal means (2017-18), deseasonalize
    seas = np.zeros((SY, N))
    for s in range(SY):
        m = tr & (woy == s); seas[s] = L[m].mean(0) if m.any() else L[tr].mean(0)
    ds = L - seas[woy]

    tt = np.where((year == 2019) & (np.arange(W) - H - 3 - SY >= 0))[0]   # test targets
    a = L[tt]                                                # actual level

    def stacklags(weeks, lags, arr):
        return np.stack([arr[weeks - l].ravel() for l in lags], 1)

    lags = [H, H + 1, H + 2, H + 3]                          # available at origin

    # ---- rules ----
    trailing = np.mean([L[tt - l] for l in lags], 0)
    snaive = L[tt - SY]
    # optimal blend weight (fit 2018 targets)
    bt = np.where((year == 2018) & (np.arange(W) - H - 3 - SY >= 0))[0]
    tb = np.mean([L[bt - l] for l in lags], 0); sb = L[bt - SY]; yb = L[bt]
    d = (sb - tb).ravel(); wgt = float(np.clip(np.sum((yb - tb).ravel() * d) / np.sum(d * d), 0, 1))
    blend = (1 - wgt) * trailing + wgt * snaive

    # ---- pooled AR (direct, deseason), shared coefs fit on 2017-18 ----
    trw = np.where(tr & (np.arange(W) - H - 3 - SY >= 0))[0]
    Xtr = stacklags(trw, lags, ds); ytr = ds[trw].ravel()
    Xtr = np.column_stack([np.ones(len(ytr)), Xtr])
    Xte = np.column_stack([np.ones(tt.size * N), stacklags(tt, lags, ds)])
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    pooled = (Xte @ beta).reshape(len(tt), N) + seas[woy[tt]]
    # ridge
    lam = 10.0; G = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    betar = np.linalg.solve(G, Xtr.T @ ytr)
    pooledr = (Xte @ betar).reshape(len(tt), N) + seas[woy[tt]]

    # ---- global GBM (raw level features) ----
    def feats(weeks):
        woy_t = woy[weeks]
        base = [L[weeks - l] for l in lags] + [L[weeks - SY],
                L[weeks - H] - L[weeks - H - 3]]
        extra = [np.sin(2*np.pi*woy_t/SY)[:,None]*np.ones((1,N)),
                 np.cos(2*np.pi*woy_t/SY)[:,None]*np.ones((1,N)),
                 np.ones((len(weeks),1))*L[tr].mean(0)[None,:]]
        return np.stack([c.ravel() for c in base+extra], 1)
    gbm = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05,
                                        max_iter=400, l2_regularization=1.0)
    gbm.fit(feats(trw), L[trw].ravel())
    gbmp = gbm.predict(feats(tt)).reshape(len(tt), N)

    print(f"Weekly-level forecasting at H={H} weeks, test 2019, all {N} sensors")
    print(f"(pooled-AR coefs on deseason lags {lags}: "
          f"{np.round(beta[1:],3)}, intercept {beta[0]:.2f}; blend weight {wgt:.2f})\n")
    print(f"{'model':<34}{'R2':>8}{'RMSE':>9}")
    print("-" * 51)
    for name, p in [("trailing (rule)", trailing),
                    ("seasonal-naive (rule)", snaive),
                    ("blend trailing+annual (1-param)", blend),
                    ("pooled AR direct (OLS)", pooled),
                    ("pooled AR direct (ridge)", pooledr),
                    ("global GBM (trained ML)", gbmp)]:
        print(f"{name:<34}{r2(p.ravel(),a.ravel()):>8.4f}{rmse(p.ravel(),a.ravel()):>9.2f}")


if __name__ == "__main__":
    main()

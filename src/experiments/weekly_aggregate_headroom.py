"""Headroom for a week-level model: ARIMA on the weekly-aggregate seasonal
residual, then distribute across hours by historical shape.

m_{i,w} = mean over the 168 hours of week w of r_{i,t}  (weekly-aggregate residual)

We measure: (1) how much of the hourly residual variance is week-level/coherent
(between-week share), (2) the ACF of m at weekly lags (does aggregation reveal
clean week-over-week structure?), (3) out-of-sample predictability of m from its
own weekly lags, and (4) the implied best-case lift in AGGREGATE R^2 -- which,
unlike the hourly-lag idea, applies uniformly to all 168 horizons.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week
from eda import seasonal_baseline

P = 168


def r2(p, a):
    return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    b, start = seasonal_baseline(v)
    r = v - b
    T, N = v.shape

    W = (T - start) // P
    rr = r[start:start + W * P].reshape(W, P, N)
    m = rr.mean(axis=1)                                  # (W weeks, N) weekly-mean residual
    week_year = index[start:start + W * P:P].year.values  # year of each week's start

    rng = np.random.default_rng(0)
    sub = rng.choice(N, 400, replace=False)
    m_s = m[:, sub]

    # (1) between-week share of residual variance
    var_hourly = r[start:start + W * P, sub].var(0)       # per sensor
    var_weekly = m_s.var(0)
    between_share = float(np.mean(var_weekly / var_hourly))
    resid_share = float(r[start:, sub].var() / v[start:, sub].var())
    print(f"between-week share of residual variance: {between_share:.3f} "
          f"(rest is within-week/hourly)")
    print(f"residual share of total variance:        {resid_share:.3f}\n")

    # (2) ACF of weekly-aggregate residual (mean over sensors)
    A = np.array([acf(m_s[:, j], nlags=8, fft=False) for j in range(m_s.shape[1])])
    macf = A.mean(0)
    print("weekly-aggregate residual ACF (mean over sensors):")
    for L in range(1, 9):
        print(f"   lag {L}w: {macf[L]:+.3f}")
    print(f"  (compare: hourly residual ACF at lag 168h was +0.094)\n")

    # (3) OOS predictability: m_w ~ m_{w-1..w-4}, train weeks <=2018, test 2019
    M = 4
    tgt = m_s[M:]                                          # (W-M, ns)
    yrs = week_year[M:]
    feats = np.stack([m_s[M - k: W - k] for k in range(1, M + 1)], -1)  # (W-M, ns, M)
    tr, te = yrs <= 2018, yrs == 2019
    Xtr = feats[tr].reshape(-1, M); ytr = tgt[tr].reshape(-1)
    Xte = feats[te].reshape(-1, M); yte = tgt[te].reshape(-1)
    A1 = np.column_stack([np.ones(len(ytr)), Xtr])
    coef, *_ = np.linalg.lstsq(A1, ytr, rcond=None)
    pred = np.column_stack([np.ones(len(yte)), Xte]) @ coef
    wk_r2 = r2(pred, yte)
    # single lag-1 reference
    a1 = np.column_stack([np.ones(len(ytr)), Xtr[:, :1]])
    c1, *_ = np.linalg.lstsq(a1, ytr, rcond=None)
    p1 = np.column_stack([np.ones(len(yte)), Xte[:, :1]]) @ c1
    print(f"OOS predictability of weekly-mean residual (test 2019):")
    print(f"   1 weekly lag : resid R^2 = {r2(p1, yte):+.3f}")
    print(f"   4 weekly lags: resid R^2 = {wk_r2:+.3f}")
    print(f"   coefs (lag1..4): {np.round(coef[1:],3)}")

    headroom = max(wk_r2, 0.0) * between_share * resid_share
    print(f"\n=> best-case AGGREGATE R^2 lift from a week-level model: +{headroom:.4f}")
    print(f"   (applies to ALL 168 horizons; aggregate R^2 0.9493 -> ~{0.9493+headroom:.4f})")


if __name__ == "__main__":
    main()

"""Test EWMA (decaying-weight) weekly baselines against the flat 4-week mean.

baseline:  b_{i,t} = sum_{k=1..L} w_k y_{i,t-168k},  sum_k w_k = 1
  - flat:  w_k = 1/4 (k<=4)        <- current baseline
  - EWMA:  w_k ∝ alpha^{k-1}        <- decaying toward recent weeks
  - OLS:   free weights, fit by least squares (matches the seasonal-AR finding)

Only weekly lags are used, so every variant is a valid one-week-ahead forecast.
Select hyper-params on 2017-2018, report out-of-sample on 2019.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week

P = 168


def r2(p, a):
    return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)

def rmse(p, a):
    return float(np.sqrt(np.mean((p - a) ** 2)))


def baseline(v, w):
    """b[t] = sum_k w[k] v[t - P*(k+1)], valid for t >= len(w)*P."""
    L = len(w)
    T = v.shape[0]
    b = np.full_like(v, np.nan)
    b[L * P:] = 0.0
    for k in range(1, L + 1):
        b[L * P:] += w[k - 1] * v[L * P - P * k: T - P * k]
    return b, L * P


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    yr = index.year.values
    Lmax = 8
    start = Lmax * P                                   # common eval start, all variants
    tr = (yr <= 2018); tr[:start] = False
    te = (yr == 2019)

    def evalw(w, label):
        b, _ = baseline(v, w)
        a_tr, p_tr = v[tr].ravel(), b[tr].ravel()
        a_te, p_te = v[te].ravel(), b[te].ravel()
        return dict(label=label, tr_rmse=rmse(p_tr, a_tr),
                    te_rmse=rmse(p_te, a_te), te_r2=r2(p_te, a_te), w=w)

    results = []
    # flat 4-week (current baseline)
    results.append(evalw(np.full(4, 0.25), "flat 4-week (current)"))

    # EWMA sweep over decay alpha, window L=8
    for alpha in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        w = alpha ** np.arange(Lmax)
        w = w / w.sum()
        results.append(evalw(w, f"EWMA L=8 alpha={alpha}"))

    # free OLS weights (L=8), fit on 2017-18 subsample, sum-to-one not enforced
    rng = np.random.default_rng(0)
    sub = rng.choice(N, 600, replace=False)
    X = np.stack([v[start - P * k: T - P * k, sub] for k in range(1, Lmax + 1)], -1)
    Ytgt = v[start:, sub]
    yrt = yr[start:]
    m_tr = yrt <= 2018
    Xtr = X[m_tr].reshape(-1, Lmax); ytr = Ytgt[m_tr].reshape(-1)
    w_ols, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    results.append(evalw(w_ols, "OLS free weights L=8"))

    # pick best EWMA by TRAIN rmse
    ewma = [r for r in results if r["label"].startswith("EWMA")]
    best = min(ewma, key=lambda r: r["tr_rmse"])

    print(f"{'model':<26}{'train RMSE':>12}{'test RMSE':>12}{'test R2':>10}")
    print("-" * 60)
    for r in results:
        star = "  *" if r is best else ""
        print(f"{r['label']:<26}{r['tr_rmse']:>12.2f}{r['te_rmse']:>12.2f}"
              f"{r['te_r2']:>10.4f}{star}")
    print("\n(* = EWMA selected by train RMSE)")
    print("\nFor reference, the seasonal-residual AR (note 01) reached test RMSE 38.62.")
    print("\nweights, best EWMA:", np.round(best["w"], 3))
    print("weights, OLS free :", np.round(results[-1]["w"], 3))


if __name__ == "__main__":
    main()

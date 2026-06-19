"""Pooled AR at H=4 with a smooth FOURIER seasonal instead of 52 free week-of-year
means. Few harmonics (2K coefficients) -> low-variance deseasonalization on short
data, which the noisy per-week means could not provide.

For each sensor we fit a seasonal curve S_i(w) = intercept + sum_k a_k sin(2pi k w/52)
 + b_k cos(2pi k w/52) by least squares on 2017-18, deseasonalize, run the SAME pooled
direct AR on lags {4..7}, and re-add the smooth seasonal.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week

P = 168; SY = 52; H = 4


def r2(p, a): return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
def rmse(p, a): return float(np.sqrt(np.mean((p - a) ** 2)))


def fourier_design(w, K):
    cols = [np.ones_like(w, float)]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * w / SY), np.cos(2 * np.pi * k * w / SY)]
    return np.column_stack(cols)


def pooled_ar_forecast(L, ds, seas_all, tt, tr_idx, lags):
    """Shared-coefficient direct AR on deseasonalized ds; re-add seasonal seas_all."""
    N = L.shape[1]
    def stack(weeks):
        return np.column_stack([np.ones(weeks.size * N)] +
                               [ds[weeks - l].ravel() for l in lags])
    beta, *_ = np.linalg.lstsq(stack(tr_idx), ds[tr_idx].ravel(), rcond=None)
    pred = (stack(tt) @ beta).reshape(len(tt), N) + seas_all[tt]
    return pred, beta


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape; W = T // P
    L = v[:W * P].reshape(W, P, N).mean(1)
    year = index[:W * P:P].year.values
    woy = np.arange(W) % SY
    wall = np.arange(W)
    tr = year <= 2018
    lags = [H, H + 1, H + 2, H + 3]
    tt = np.where((year == 2019) & (wall - H - 3 - SY >= 0))[0]
    trw = np.where(tr & (wall - H - 3 - SY >= 0))[0]
    a = L[tt]

    # rules: trailing and 1-param blend (targets to beat)
    trailing = np.mean([L[tt - l] for l in lags], 0)
    snaive = L[tt - SY]
    bt = np.where((year == 2018) & (wall - H - 3 - SY >= 0))[0]
    tb = np.mean([L[bt - l] for l in lags], 0); sb = L[bt - SY]
    d = (sb - tb).ravel(); wgt = float(np.clip(np.sum((L[bt] - tb).ravel() * d) / np.sum(d * d), 0, 1))
    blend = (1 - wgt) * trailing + wgt * snaive

    rows = [("trailing (rule)", trailing),
            ("blend trailing+annual (1-param)", blend)]

    # 52 free week-of-year means (the noisy baseline deseasonalization)
    seas52 = np.zeros((SY, N))
    for s in range(SY):
        m = tr & (woy == s); seas52[s] = L[m].mean(0) if m.any() else L[tr].mean(0)
    ds52 = L - seas52[woy]
    p52, _ = pooled_ar_forecast(L, ds52, seas52[woy], tt, trw, lags)
    rows.append(("pooled AR + 52 week-means", p52))

    # Fourier seasonal, K harmonics
    for K in (2, 3, 4, 6):
        Phi = fourier_design(wall, K)
        theta, *_ = np.linalg.lstsq(Phi[tr], L[tr], rcond=None)   # (2K+1, N)
        seasF = Phi @ theta                                       # (W, N) smooth seasonal
        dsF = L - seasF
        pF, beta = pooled_ar_forecast(L, dsF, seasF, tt, trw, lags)
        rows.append((f"pooled AR + Fourier K={K} ({2*K+1} coef)", pF))

    print(f"Weekly-level forecasting at H={H}, test 2019, all {N} sensors "
          f"(blend weight {wgt:.2f})\n")
    print(f"{'model':<38}{'R2':>8}{'RMSE':>9}")
    print("-" * 55)
    for name, p in rows:
        print(f"{name:<38}{r2(p.ravel(),a.ravel()):>8.4f}{rmse(p.ravel(),a.ravel()):>9.2f}")


if __name__ == "__main__":
    main()

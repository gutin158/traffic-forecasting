"""YoY-growth-adjusted past-year baseline.

Idea (user): bring in annual seasonality via the value one year ago, scaled by
the year-over-year growth ratio, then average 50/50 with the existing 4-week
baseline:

  growth      g_{i,t}   = l4w_avg_now / l4w_avg_one_year_ago
  yoy term    yoy_{i,t} = g_{i,t} * y_{i, t-Y}            (Y = 52 weeks)
  combined    b2_{i,t}  = 0.5 * b_{i,t} + 0.5 * yoy_{i,t}

All terms sit >= ~1 year before the target, so this is a valid one-week-ahead
forecast. Evaluated out-of-sample on 2019 (all 2,352 sensors), apples-to-apples
against the flat 4-week and EWMA baselines on the identical target set.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from run_gba import load_years, impute_hour_of_week
from eda import seasonal_baseline
from ewma_baseline import baseline as ewma_baseline

P = 168
Y = 52 * P                      # 1 year = 52 weeks = 8736 h (keeps day-of-week aligned)


def r2(p, a):
    return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)

def rmse(p, a):
    return float(np.sqrt(np.mean((p - a) ** 2)))


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    year = index.year.values

    b, _ = seasonal_baseline(v)                       # flat 4-week hour-of-week avg
    w = 0.8 ** np.arange(8); w /= w.sum()
    b_ewma, _ = ewma_baseline(v, w)                   # EWMA baseline (alpha=0.8, L=8)

    # trailing 672-h (4-week) mean over ALL hours -> robust scalar growth
    cs = np.cumsum(v, axis=0)
    m = np.full_like(v, np.nan)
    m[672:] = (cs[672:] - cs[:-672]) / 672.0

    vs = Y + 672                                       # first target with full history
    sl, slY = slice(vs, T), slice(vs - Y, T - Y)
    bt, btY = b[sl], b[slY]
    yY = v[slY]
    yt = v[sl]
    ymask = (year[sl] == 2019)

    def clip_ratio(num, den):
        g = np.where(den > 1.0, num / den, 1.0)
        return np.clip(g, 0.5, 2.0)

    g_cell = clip_ratio(bt, btY)                       # per-(sensor,hour-of-week) growth
    g_scalar = clip_ratio(m[sl], m[slY])               # robust per-sensor scalar growth

    cand = {
        "flat 4-week (current)":        bt,
        "EWMA a=0.8":                   b_ewma[sl],
        "YoY combo 50/50 (per-cell g)": 0.5 * bt + 0.5 * g_cell * yY,
        "YoY combo 50/50 (scalar g)":   0.5 * bt + 0.5 * g_scalar * yY,
        "YoY standalone (scalar g)":    g_scalar * yY,
        "EWMA + YoY 50/50 (scalar g)":  0.5 * b_ewma[sl] + 0.5 * g_scalar * yY,
    }

    a = yt[ymask]
    print(f"mean YoY growth factor on 2019 (scalar): {g_scalar[ymask].mean():.3f}")
    print(f"\n{'baseline':<32}{'test RMSE':>11}{'test R2':>10}{'vs flat':>10}")
    print("-" * 63)
    flat_rmse = rmse(bt[ymask], a)
    for name, pred in cand.items():
        rr = rmse(pred[ymask], a)
        delta = 100 * (1 - rr / flat_rmse)
        print(f"{name:<32}{rr:>11.2f}{r2(pred[ymask], a):>10.4f}{delta:>9.1f}%")

    # best-case: optimal blend weight of the YoY term, fit on 2018, tested on 2019
    tr = (year[sl] == 2018)
    d = g_scalar * yY - bt                 # (YoY term) - baseline
    lam = np.sum((yt - bt)[tr] * d[tr]) / np.sum((d * d)[tr])
    pred = bt + lam * d
    print(f"\noptimal blend weight on the YoY term (fit 2018): lambda* = {lam:.3f}")
    print(f"  -> best-case test RMSE {rmse(pred[ymask], a):.2f}  "
          f"R2 {r2(pred[ymask], a):.4f}  ({100*(1-rmse(pred[ymask],a)/flat_rmse):+.1f}% vs flat)")


if __name__ == "__main__":
    main()

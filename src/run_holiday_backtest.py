"""Rolling-origin backtest of the weekly baselines + explicit holiday model.

Models (all leakage-safe one-week-ahead):
  - flat baseline         : 4-week hour-of-week mean
  - EWMA                  : decaying weekly weights (alpha=0.8, L=8)
  - flat + holiday        : flat baseline x learned per-(holiday,hour) factor
  - EWMA + holiday        : EWMA baseline x learned factor

Holiday factors are learned ONLY from 2017-2018 (the only data available before a
2019 holiday occurs). Rolling origins step by 24h across 2019; metrics are pooled
over all (origin, horizon) pairs and bucketed into holiday vs ordinary targets.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week
from holiday_model import special_day_keys

ROOT = Path(__file__).resolve().parents[1]
P = 168
H = 168
STRIDE = 24


class WeeklyBaseline:
    """Predict each target hour as a fixed weighted sum of its weekly lags.
    The full baseline array is precomputed; b[s] uses only y at s-P*k (k>=1),
    so slicing b[o+1:o+1+H] is a valid forecast from data <= o (no leakage)."""
    def __init__(self, v, weights):
        self.L = len(weights)
        T, N = v.shape
        b = np.full_like(v, np.nan)
        b[self.L * P:] = 0.0
        for k in range(1, self.L + 1):
            b[self.L * P:] += weights[k - 1] * v[self.L * P - P * k: T - P * k]
        self.b = b

    def predict(self, o, h):
        return self.b[o + 1: o + 1 + h]


class HolidayAdjusted:
    """Multiply a base model's prediction by a learned holiday factor on special
    target days."""
    def __init__(self, base, keys, hod, factors):
        self.base, self.keys, self.hod, self.factors = base, keys, hod, factors

    def predict(self, o, h):
        p = self.base.predict(o, h).copy()
        for j, s in enumerate(range(o + 1, o + 1 + h)):
            k = self.keys[s]
            if k:
                f = self.factors.get((k, self.hod[s]))
                if f is not None:
                    p[j] = p[j] * f
        return p


def learn_factors(base, v, keys, hod, train_mask):
    sy, sb = defaultdict(float), defaultdict(float)
    finite = np.isfinite(base.b[:, 0])
    for t in np.where(train_mask & (keys != "") & finite)[0]:
        c = (keys[t], hod[t])
        sy[c] += v[t].sum()
        sb[c] += base.b[t].sum()
    return {c: sy[c] / sb[c] for c in sy if sb[c] > 0}


def _upd(acc, a, d2):
    acc["n"] += a.size; acc["sse"] += d2.sum()
    acc["sa"] += a.sum(); acc["sa2"] += float((a * a).sum())

def _metrics(acc):
    n = acc["n"]; mean = acc["sa"] / n
    sstot = acc["sa2"] - n * mean ** 2
    return float(np.sqrt(acc["sse"] / n)), float(1 - acc["sse"] / sstot)


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    year = index.year.values
    hod = index.hour.values
    keys = special_day_keys(index)

    flat = WeeklyBaseline(v, np.full(4, 0.25))
    w = 0.8 ** np.arange(8); w /= w.sum()
    ewma = WeeklyBaseline(v, w)

    train_mask = year <= 2018
    f_flat = learn_factors(flat, v, keys, hod, train_mask)
    f_ewma = learn_factors(ewma, v, keys, hod, train_mask)

    models = {
        "flat baseline":    flat,
        "EWMA a=0.8":       ewma,
        "flat + holiday":   HolidayAdjusted(flat, keys, hod, f_flat),
        "EWMA + holiday":   HolidayAdjusted(ewma, keys, hod, f_ewma),
    }

    # rolling origins: predict windows that start in 2019
    o0 = int(np.where(year >= 2019)[0][0]) - 1
    origins = list(range(o0, T - H, STRIDE))

    accs = {m: {bkt: dict(n=0.0, sse=0.0, sa=0.0, sa2=0.0)
                for bkt in ("all", "ord", "hol")} for m in models}
    for o in origins:
        a = v[o + 1: o + 1 + H]
        tk = keys[o + 1: o + 1 + H]
        hm = (tk != "")
        for name, m in models.items():
            d2 = (m.predict(o, H) - a) ** 2
            _upd(accs[name]["all"], a, d2)
            _upd(accs[name]["ord"], a[~hm], d2[~hm])
            _upd(accs[name]["hol"], a[hm], d2[hm])

    hol_frac = 100 * accs["flat baseline"]["hol"]["n"] / accs["flat baseline"]["all"]["n"]
    print(f"Rolling-origin backtest: {len(origins)} origins x H={H}, stride {STRIDE}h, "
          f"test year 2019 ({N} sensors)")
    print(f"holiday-window targets = {hol_frac:.1f}% of evaluated points\n")
    print(f"{'model':<20}{'RMSE all':>10}{'R2 all':>9}{'RMSE ord':>10}{'RMSE hol':>10}")
    print("-" * 59)
    out = {}
    for name in models:
        ra, r2a = _metrics(accs[name]["all"])
        ro, _ = _metrics(accs[name]["ord"])
        rh, _ = _metrics(accs[name]["hol"])
        print(f"{name:<20}{ra:>10.2f}{r2a:>9.4f}{ro:>10.2f}{rh:>10.2f}")
        out[name] = dict(rmse_all=ra, r2_all=r2a, rmse_ord=ro, rmse_hol=rh)
    (ROOT / "results" / "holiday_backtest.json").write_text(json.dumps(out, indent=2))
    print("\nsaved results/holiday_backtest.json")


if __name__ == "__main__":
    main()

"""Per-sensor (heterogeneous) holiday factors vs the pooled factor.

Decompose the holiday effect into:
  - pooled per-(holiday,hour-of-day) SHAPE  (stable; reshapes the daily profile)
  - per-(sensor,holiday) SCALE, shrunk toward 1  (captures corridor heterogeneity:
    recreational routes gain traffic, commute routes drop)

  y_hat = b * shape[key,hod] * scale[sensor, holiday_identity]

Factors learned on 2017-18, rolling-origin backtest on 2019.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week
from holiday_model import special_day_keys
from run_holiday_backtest import (WeeklyBaseline, HolidayAdjusted, learn_factors,
                                  _upd, _metrics, P, H, STRIDE)

ROOT = Path(__file__).resolve().parents[1]


def learn_scale(base, v, keys, hod, shape, train_mask, tau):
    ids = sorted(set(k[:-2] for k in keys if k))            # holiday identities
    id_idx = {x: j for j, x in enumerate(ids)}
    N = v.shape[1]
    num = np.zeros((N, len(ids))); den = np.zeros((N, len(ids))); cnt = np.zeros(len(ids))
    finite = np.isfinite(base.b[:, 0])
    for t in np.where(train_mask & (keys != "") & finite)[0]:
        j = id_idx[keys[t][:-2]]
        sh = shape.get((keys[t], hod[t]), 1.0)
        num[:, j] += v[t]
        den[:, j] += base.b[t] * sh
        cnt[j] += 1
    raw = np.where(den > 0, num / np.where(den > 0, den, 1), 1.0)
    n = cnt[None, :]
    scale = (n * raw + tau) / (n + tau)                     # shrink toward 1.0
    return id_idx, np.where(den > 0, scale, 1.0)


class HolidayPerSensor:
    def __init__(self, base, keys, hod, shape, id_idx, scale):
        self.base, self.keys, self.hod, self.shape = base, keys, hod, shape
        self.id_idx, self.scale = id_idx, scale

    def predict(self, o, h):
        p = self.base.predict(o, h).copy()
        for j, s in enumerate(range(o + 1, o + 1 + h)):
            k = self.keys[s]
            if k:
                sh = self.shape.get((k, self.hod[s]))
                if sh is not None:
                    p[j] = p[j] * sh * self.scale[:, self.id_idx[k[:-2]]]
        return p


def evaluate(models, v, keys, origins):
    accs = {m: {b: dict(n=0.0, sse=0.0, sa=0.0, sa2=0.0) for b in ("all", "ord", "hol")}
            for m in models}
    for o in origins:
        a = v[o + 1: o + 1 + H]
        hm = (keys[o + 1: o + 1 + H] != "")
        for name, m in models.items():
            d2 = (m.predict(o, H) - a) ** 2
            _upd(accs[name]["all"], a, d2)
            _upd(accs[name]["ord"], a[~hm], d2[~hm])
            _upd(accs[name]["hol"], a[hm], d2[hm])
    return accs


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    year = index.year.values; hod = index.hour.values
    keys = special_day_keys(index)
    train = year <= 2018

    w = 0.8 ** np.arange(8); w /= w.sum()
    ewma = WeeklyBaseline(v, w)
    shape = learn_factors(ewma, v, keys, hod, train)        # pooled per-(holiday,hour)

    models = {
        "EWMA (no holiday)": ewma,
        "EWMA + holiday pooled": HolidayAdjusted(ewma, keys, hod, shape),
    }
    for tau in (200, 50, 10):
        id_idx, scale = learn_scale(ewma, v, keys, hod, shape, train, tau)
        models[f"EWMA + holiday per-sensor (tau={tau})"] = \
            HolidayPerSensor(ewma, keys, hod, shape, id_idx, scale)

    o0 = int(np.where(year >= 2019)[0][0]) - 1
    origins = list(range(o0, T - H, STRIDE))
    accs = evaluate(models, v, keys, origins)

    print(f"Rolling backtest 2019, {len(origins)} origins x H={H}\n")
    print(f"{'model':<38}{'RMSE all':>10}{'R2 all':>9}{'RMSE hol':>10}")
    print("-" * 67)
    for name in models:
        ra, r2a = _metrics(accs[name]["all"]); rh, _ = _metrics(accs[name]["hol"])
        print(f"{name:<38}{ra:>10.2f}{r2a:>9.4f}{rh:>10.2f}")

    # interpret heterogeneity: per-sensor July-4 scale, tau=50
    id_idx, scale = learn_scale(ewma, v, keys, hod, shape, train, 50)
    meta = pd.read_csv(ROOT / "data" / "ca" / "ca_meta.csv").set_index("ID")
    for hol in ["Independence Day", "Thanksgiving Day"]:
        sc = scale[:, id_idx[hol]]
        print(f"\n{hol}: per-sensor scale  median={np.median(sc):.2f} "
              f"[p5={np.percentile(sc,5):.2f}, p95={np.percentile(sc,95):.2f}], "
              f"frac>1.05: {100*np.mean(sc>1.05):.0f}%  (corridors that GAIN traffic)")
        top = np.argsort(sc)[-3:][::-1]
        for i in top:
            sid = int(ids[i])
            fw = meta.loc[sid, "Fwy"] if sid in meta.index else "?"
            print(f"    gain x{sc[i]:.2f}  sensor {sid} (Fwy {fw})")


if __name__ == "__main__":
    main()

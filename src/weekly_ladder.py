"""The ladder: do TRAINED models beat parameter-free RULES on the weekly-level,
multi-week forecasting problem?

Object: per-sensor weekly level L_{i,w} (2017-2019). Forecast L_{i, target} at
horizon H weeks; only info up to origin = target-H is used (k>=H leakage rule).
Test targets = 2019 weeks. Metric: pooled R^2/RMSE over (sensor, target).

Rungs:
  R0 trailing mean   : mean of the 4 most recent observable weeks {H..H+3}   (rule)
  R1 seasonal-naive  : L[target-52]                                          (rule)
  R2 YoY-growth      : L[target-52] * (recent level / year-ago level)        (rule)
  R3 Holt (classical): per-sensor: deseasonalize by trained week-of-year mean,
                       fit Holt (level+trend exp. smoothing, alpha/beta) on history
                       up to origin, forecast H, re-add season.   (TRAINED, few params)
  R4 global GBM (ML) : HistGradientBoosting on the panel, one direct model per H,
                       features = observable lags + annual lag + week-of-year +
                       sensor level + lanes. Trained on 2017-18 targets.  (TRAINED ML)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import Holt

ROOT = Path(__file__).resolve().parents[1]
P = 168
S_PER_YEAR = 52
HORIZONS = [1, 2, 4, 8]
N_SUB = 120


def r2(p, a): return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
def rmse(p, a): return float(np.sqrt(np.mean((p - a) ** 2)))


def build_level():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape
    W = T // P
    L = v[:W * P].reshape(W, P, N).mean(axis=1)          # (weeks, sensors)
    wk_year = index[:W * P:P].year.values
    woy = (np.arange(W) % S_PER_YEAR)
    return L, wk_year, woy, ids


def main():
    L, wk_year, woy, ids = build_level()
    W, N = L.shape
    rng = np.random.default_rng(0)
    S = np.sort(rng.choice(N, N_SUB, replace=False))
    Ls = L[:, S]
    meta = pd.read_csv(ROOT / "data" / "ca" / "ca_meta.csv").set_index("ID")
    lanes = np.array([meta.loc[int(ids[i]), "Lanes"] if int(ids[i]) in meta.index else 0
                      for i in S], float)
    sens_mean = Ls[wk_year <= 2018].mean(0)              # trained per-sensor level

    # trained week-of-year seasonal means (2017-18) for deseasonalizing (R3)
    seas = np.zeros((S_PER_YEAR, len(S)))
    tr_mask = wk_year <= 2018
    for s in range(S_PER_YEAR):
        m = tr_mask & (woy == s)
        seas[s] = Ls[m].mean(0) if m.any() else Ls[tr_mask].mean(0)
    deseason = Ls - seas[woy]

    test_targets = np.where(wk_year == 2019)[0]

    results = {h: {} for h in HORIZONS}
    for H in HORIZONS:
        tt = test_targets[test_targets - H - 3 - S_PER_YEAR >= 0]   # need full history
        a = Ls[tt]                                                  # (n_tgt, nS) actual

        # ---- R0 trailing mean of weeks {H..H+3} ----
        trail = np.mean([Ls[tt - H - j] for j in range(4)], axis=0)
        # ---- R1 seasonal-naive ----
        snaive = Ls[tt - S_PER_YEAR]
        # ---- R2 YoY growth ----
        rec = np.mean([Ls[tt - H - j] for j in range(4)], axis=0)
        ago = np.mean([Ls[tt - H - j - S_PER_YEAR] for j in range(4)], axis=0)
        g = np.clip(np.where(ago > 1, rec / ago, 1.0), 0.5, 2.0)
        yoy = snaive * g

        results[H]["R0 trailing (rule)"] = trail
        results[H]["R1 seasonal-naive (rule)"] = snaive
        results[H]["R2 YoY-growth (rule)"] = yoy
        results[H]["_actual"] = a
        results[H]["_tt"] = tt

    # ---- R3 Holt (trained classical), expanding per origin, deseasonalized ----
    print("fitting Holt (trained classical) ...", flush=True)
    for H in HORIZONS:
        tt = results[H]["_tt"]
        pred = np.empty_like(results[H]["_actual"])
        for j, t in enumerate(tt):
            o = t - H
            for k, s in enumerate(range(len(S))):
                hist = deseason[: o + 1, s]
                try:
                    fit = Holt(hist, initialization_method="estimated").fit()
                    f = fit.forecast(H)[-1]
                except Exception:
                    f = hist[-1]
                pred[j, k] = f + seas[woy[t], s]
        results[H]["R3 Holt (trained classical)"] = pred
        print(f"  H={H} done", flush=True)

    # ---- R4 global GBM (trained ML), direct per-horizon ----
    print("training global GBM (trained ML) ...", flush=True)
    def feats(targets, H):
        cols = {
            "lag_H":  Ls[targets - H],
            "lag_H1": Ls[targets - H - 1],
            "lag_H2": Ls[targets - H - 2],
            "lag_H3": Ls[targets - H - 3],
            "annual": Ls[targets - S_PER_YEAR],
            "trend":  Ls[targets - H] - Ls[targets - H - 3],
            "woy_sin": np.sin(2 * np.pi * woy[targets] / S_PER_YEAR)[:, None] * np.ones((1, len(S))),
            "woy_cos": np.cos(2 * np.pi * woy[targets] / S_PER_YEAR)[:, None] * np.ones((1, len(S))),
            "sens_mean": np.ones((len(targets), 1)) * sens_mean[None, :],
            "lanes": np.ones((len(targets), 1)) * lanes[None, :],
        }
        X = np.stack([c.ravel() for c in cols.values()], axis=1)
        return X
    for H in HORIZONS:
        train_t = np.where((wk_year <= 2018) & (np.arange(W) - H - 3 - S_PER_YEAR >= 0))[0]
        Xtr, ytr = feats(train_t, H), Ls[train_t].ravel()
        gbm = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05,
                                            max_iter=400, l2_regularization=1.0)
        gbm.fit(Xtr, ytr)
        tt = results[H]["_tt"]
        pred = gbm.predict(feats(tt, H)).reshape(len(tt), len(S))
        results[H]["R4 global GBM (trained ML)"] = pred
        print(f"  H={H} done", flush=True)

    # ---- report ----
    rungs = ["R0 trailing (rule)", "R1 seasonal-naive (rule)", "R2 YoY-growth (rule)",
             "R3 Holt (trained classical)", "R4 global GBM (trained ML)"]
    print(f"\nWeekly-level forecast accuracy (test 2019, {len(S)} sensors)\n")
    print(f"{'model':<32}" + "".join(f"  H={h:<2}(R2/RMSE)".ljust(16) for h in HORIZONS))
    print("-" * (32 + 16 * len(HORIZONS)))
    for rg in rungs:
        row = f"{rg:<32}"
        for H in HORIZONS:
            a = results[H]["_actual"].ravel(); p = results[H][rg].ravel()
            row += f"{r2(p,a):.3f}/{rmse(p,a):.1f}".ljust(16)
        print(row)
    print(f"\n(R0-R2 = parameter-free rules; R3 = trained few-param classical; "
          f"R4 = trained global ML)")


if __name__ == "__main__":
    main()

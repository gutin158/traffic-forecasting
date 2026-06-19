"""Cross-regional co-movement of the weekly LEVEL: is there a forecastable common
California factor that single-region rules cannot use?

For each available region (GBA, GLA, SD) we build the network-mean weekly level
over 2017-2019, remove the annual cycle (per-region week-of-year mean), and ask of
the DEVIATIONS-from-seasonal (the remainder):
  1. do they co-move across regions? (pairwise correlation, common-factor PCA)
  2. is the common factor PERSISTENT? (its own autocorrelation) -- because
     co-movement only HELPS forecasting if the shared factor's recent value predicts
     its future. A strong-but-white common factor denoises but does not forecast.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import impute_hour_of_week  # reused imputation
P = 168; SY = 52
CA = Path(__file__).resolve().parents[1] / "data" / "ca"
YEARS = [2017, 2018, 2019]


def region_weekly_mean(region):
    """Network-mean weekly level for a region over YEARS, or None if not cached."""
    vals, idx = [], []
    for y in YEARS:
        f = CA / f"{region}_hourly_{y}.npz"
        if not f.exists():
            return None, None
        d = np.load(f, allow_pickle=True)
        vals.append(d["vals"]); idx.append(d["index"])
    v = np.concatenate(vals).astype(float)
    index = pd.to_datetime(np.concatenate(idx))
    # impute then collapse to weekly network mean
    import pandas as _pd
    df = _pd.DataFrame(v, index=index)
    df = df.mask(df == 0.0)
    v = df.ffill().bfill().values
    W = v.shape[0] // P
    L = v[:W*P].reshape(W, P, v.shape[1]).mean(1)       # (W, N) weekly level
    return L.mean(1), index[:W*P:P]                      # network-mean (W,)


def deseason(x, method="woy", K=3):
    w = np.arange(len(x))
    if method == "woy":
        woy = w % SY
        m = np.array([x[woy == s].mean() for s in range(SY)])
        return x - m[woy]
    # smooth Fourier seasonal (low-variance): removes shared annual cycle cleanly
    cols = [np.ones_like(w, float)]
    for k in range(1, K + 1):
        cols += [np.sin(2*np.pi*k*w/SY), np.cos(2*np.pi*k*w/SY)]
    Phi = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Phi, x, rcond=None)
    return x - Phi @ beta


def acf(x, lags):
    x = x - x.mean(); v = np.dot(x, x)
    return [float(np.dot(x[:-k], x[k:]) / v) for k in lags]


def main():
    raw = {}
    for r in ["gba", "gla", "sd"]:
        m, _ = region_weekly_mean(r)
        if m is not None:
            raw[r] = m; print(f"  {r.upper()}: {len(m)} weeks loaded")
        else:
            print(f"  {r.upper()}: NOT cached yet (skipping)")
    names = list(raw)
    if len(names) < 2:
        print("\nNeed >=2 regions. Download/process GLA, SD first.")
        return

    for method in ("woy", "fourier"):
        R = np.vstack([deseason(raw[n], method) for n in names])
        tag = "52 week-means" if method == "woy" else "smooth Fourier K=3"
        print(f"\n=== deseasonalization: {tag} ===")
        C = np.corrcoef(R)
        print("        " + "".join(f"{n.upper():>8}" for n in names))
        for i, n in enumerate(names):
            print(f"  {n.upper():<5}" + "".join(f"{C[i,j]:>8.2f}" for j in range(len(names))))
        Z = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
        s = np.linalg.svd(Z, compute_uv=False)
        vt = np.linalg.svd(Z, full_matrices=False)[2]
        print(f"  common-factor (PC1) var share: {s[0]**2/np.sum(s**2):.2f}   "
              f"PC1 persistence ACF lags1-6w: {[round(a,2) for a in acf(vt[0], range(1,7))]}")
    print("\nReading: if the GLA-SD correlation + persistence SURVIVE Fourier, it's a")
    print("genuine common factor; if they shrink, it was residual annual seasonality.")


if __name__ == "__main__":
    main()

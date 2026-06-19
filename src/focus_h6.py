"""6-week-horizon forecasting with HONEST metrics.

The weekly-level R^2 (~0.96) is flattered: the level's variance is ~94% annual
cycle, so R^2 vs the mean looks high at any horizon. We report instead:
  - weekly RMSE (veh/h, absolute -- grows with horizon)
  - skill vs seasonal-naive (how much better than a trivial annual copy)
  - reconstructed HOURLY-flow R^2/RMSE (apples-to-apples with 1-week experience)
and contrast H=1 vs H=6 for the trailing rule.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gba import load_years, impute_hour_of_week
from sklearn.ensemble import HistGradientBoostingRegressor

P = 168; SY = 52


def r2(p, a): return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
def rmse(p, a): return float(np.sqrt(np.mean((p - a) ** 2)))


def fourier(w, K):
    cols = [np.ones_like(w, float)]
    for k in range(1, K + 1):
        cols += [np.sin(2*np.pi*k*w/SY), np.cos(2*np.pi*k*w/SY)]
    return np.column_stack(cols)


def main():
    v, index, ids = load_years([2017, 2018, 2019])
    v = impute_hour_of_week(v, index)
    T, N = v.shape; W = T // P
    Vr = v[:W*P].reshape(W, P, N)
    L = Vr.mean(1)                                   # weekly level (W,N)
    year = index[:W*P:P].year.values
    wall = np.arange(W); woy = wall % SY
    tr = year <= 2018

    # multiplicative hourly shape S[phi, sensor] from training weeks
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Vr / L[:, None, :]
    S = np.nanmean(np.where(np.isfinite(ratio), ratio, np.nan)[tr], 0)   # (P,N)

    def eval_h(H, models_wanted=True):
        lags = [H, H+1, H+2, H+3]
        tt = np.where((year == 2019) & (wall - H - 3 - SY >= 0))[0]
        trw = np.where(tr & (wall - H - 3 - SY >= 0))[0]
        aL = L[tt]
        aH = Vr[tt]                                   # actual hourly (n,P,N)

        preds = {}
        preds["trailing"] = np.mean([L[tt-l] for l in lags], 0)
        preds["seasonal-naive"] = L[tt-SY]
        # 1-param blend (weight fit on 2018)
        bt = np.where((year == 2018) & (wall-H-3-SY >= 0))[0]
        tb = np.mean([L[bt-l] for l in lags], 0); sb = L[bt-SY]
        d = (sb-tb).ravel(); wgt = float(np.clip(np.sum((L[bt]-tb).ravel()*d)/np.sum(d*d),0,1))
        preds["blend"] = (1-wgt)*preds["trailing"] + wgt*preds["seasonal-naive"]
        # pooled AR + Fourier K=2
        Phi = fourier(wall, 2); th,*_ = np.linalg.lstsq(Phi[tr], L[tr], rcond=None)
        seasF = Phi@th; dsF = L - seasF
        def st(weeks): return np.column_stack([np.ones(weeks.size*N)]+[dsF[weeks-l].ravel() for l in lags])
        beta,*_ = np.linalg.lstsq(st(trw), dsF[trw].ravel(), rcond=None)
        preds["pooled AR+Fourier"] = (st(tt)@beta).reshape(len(tt),N)+seasF[tt]
        # GBM
        def feats(weeks):
            woy_t = woy[weeks]
            base=[L[weeks-l] for l in lags]+[L[weeks-SY], L[weeks-H]-L[weeks-H-3]]
            extra=[np.sin(2*np.pi*woy_t/SY)[:,None]*np.ones((1,N)),
                   np.cos(2*np.pi*woy_t/SY)[:,None]*np.ones((1,N)),
                   np.ones((len(weeks),1))*L[tr].mean(0)[None,:]]
            return np.stack([c.ravel() for c in base+extra],1)
        g=HistGradientBoostingRegressor(max_depth=6,learning_rate=0.05,max_iter=400,l2_regularization=1.0)
        g.fit(feats(trw), L[trw].ravel())
        preds["global GBM"]=g.predict(feats(tt)).reshape(len(tt),N)

        sse_sn = np.sum((preds["seasonal-naive"]-aL)**2)
        rows=[]
        for name,pL in preds.items():
            yhat = pL[:,None,:]*S[None,:,:]           # reconstruct hourly
            skill = 100*(1-np.sum((pL-aL)**2)/sse_sn)
            rows.append((name, rmse(pL.ravel(),aL.ravel()), r2(pL.ravel(),aL.ravel()),
                         skill, r2(yhat.ravel(),aH.ravel()), rmse(yhat.ravel(),aH.ravel())))
        return rows

    print("="*78)
    print("Reference -- TRAILING rule, H=1 vs H=6 (shows the real horizon effect):")
    for H in (1, 6):
        r = [x for x in eval_h(H) if x[0]=="trailing"][0]
        print(f"  H={H}: weekly R2={r[2]:.4f}  weekly RMSE={r[1]:.1f}   ||   "
              f"HOURLY-flow R2={r[4]:.4f}  hourly RMSE={r[5]:.1f}")
    print("="*78)

    print("\n6-WEEK-AHEAD forecasting, test 2019, all sensors:\n")
    print(f"{'model':<20}{'wkRMSE':>8}{'wkR2':>8}{'skill_vs_SN':>12}"
          f"{'hourlyR2':>10}{'hourlyRMSE':>11}")
    print("-"*69)
    for name,wr,wr2,sk,hr2,hrm in eval_h(6):
        print(f"{name:<20}{wr:>8.1f}{wr2:>8.4f}{sk:>11.1f}%{hr2:>10.4f}{hrm:>11.1f}")
    print("\nNote: weekly R2 is flattered by the annual cycle in the denominator;")
    print("hourly-flow R2 and skill-vs-seasonal-naive are the honest difficulty measures.")


if __name__ == "__main__":
    main()

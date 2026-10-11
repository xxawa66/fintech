"""LH032 第六步：双因子倾斜 —— 一条轴赚 profile，一条轴赚 α。

单因子结论（四折，stage2 池内重排）：
  log_amt φ<0（往小成交额）：Δprofile +0.09~0.12（++++，LOFO 4/4），Δα 小幅正，
                             ΔIC +0.005~0.009，ΔT +0.0005  ⇒ Δfinal +0.018
  vol60   φ>0（往高波动）  ：Δα +0.054~0.086，Δprofile −0.03~−0.07，ΔIC −0.003~−0.011
                             ⇒ Δfinal ≈ +0.001（增益被 IC 拖住）

两条轴的作用量互补：一个给 profile，一个给 α。本步做二维网格，看是否可加。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "long_horizon"
sys.path.insert(0, str(CACHE))
sys.path.insert(0, str(ROOT / "research" / "LH023"))
sys.path.insert(0, str(ROOT))

import _probe_factors as _pf                                        # noqa: E402
if "mom20" not in _pf.FACTORS:
    _pf.FACTORS.append("mom20")
from _probe_e_attr import load_factors, top_sets, wide_of           # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide  # noqa: E402
from _probe_dgtw import decompose                                   # noqa: E402

KQ = 0.0022778298112255263
ANN = 252
NB = 10
FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
PHI_AMT = [-0.10, -0.075, -0.05, -0.04, -0.02, -0.01, 0.0]
PHI_VOL = [0.0, 0.01, 0.02, 0.04, 0.075]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _rank(a):
    o = np.argsort(a, kind="mergesort")
    r = np.empty(a.size)
    r[o] = np.arange(a.size, dtype=float)
    s = a[o]
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and s[j] == s[i]:
            j += 1
        if j - i > 1:
            r[o[i:j]] = (i + j - 1) / 2.0
        i = j
    return r


def pct_rank_2d(P):
    out = np.full(P.shape, np.nan)
    for t in range(P.shape[0]):
        x = P[t]
        m = np.isfinite(x)
        n = int(m.sum())
        if n < 2:
            continue
        out[t, m] = _rank(x[m]) / (n - 1)
    return out


def two_stage(pr_lh, pr_h1, K):
    out = np.full(pr_lh.shape, np.nan)
    for t in range(pr_lh.shape[0]):
        a, b = pr_lh[t], pr_h1[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        bb = np.where(cand & np.isfinite(b), b, np.nan)
        v = bb[np.isfinite(bb)]
        lo, hi = v.min(), v.max()
        within = np.where(np.isfinite(bb), (bb - lo) / (hi - lo), 0.5) if hi > lo else 0.5
        out[t] = np.where(cand, 1.0 + within, np.where(m, a, np.nan))
    return out


def tilt2(P0, pr_a, pr_b, pa, pb):
    D = np.where(np.isfinite(pr_a), (pr_a - 0.5) * pa, 0.0) \
        + np.where(np.isfinite(pr_b), (pr_b - 0.5) * pb, 0.0)
    P = P0 + D
    iscand = np.isfinite(P0) & (P0 >= 1.0)
    P = np.where(iscand, P, P - 1.0)
    P[~np.isfinite(P0)] = np.nan
    return P


def fold_block(fold, fdf):
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    ms = ["H01", "H05_F1T2", "H05_F2T2"]
    df = df.drop(columns=[c for c in ms if c in df.columns]).merge(
        pred[["ts_code", "trade_date"] + ms], on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in ms}
    pct = {m: pct_rank_2d(raw[m]) for m in ms}
    mix = np.nansum(np.stack([w * pct[m] for m, w in WEIGHTS.items()]), axis=0)
    P0 = two_stage(pct_rank_2d(mix), pct["H01"], 40.0)
    PA = pct_rank_2d(W["log_amt"])
    PB = pct_rank_2d(W["vol60"])

    rows = []

    def evaluate(tag, pa, pb, P):
        Pb, _ = fast_band(P, panel.LU, KQ)
        r = score(Pb, panel)
        S = top_sets(Pb, panel)
        d_vol = decompose(W, panel, S, ["vol60"], [NB], fold, tag)
        d_amt = decompose(W, panel, S, ["log_amt"], [NB], fold, tag)
        T = r["mean_turnover"]
        return dict(fold=fold, phi_amt=pa, phi_vol=pb, tag=tag,
                    ic=r["ic_mean"], excess=r["annual_excess"], turnover=T,
                    final=r["final_score"],
                    alpha_vol=float(d_vol["alpha"].mean()) * ANN,
                    prof_vol=float(d_vol["profile"].mean()) * ANN,
                    alpha_amt=float(d_amt["alpha"].mean()) * ANN,
                    prof_amt=float(d_amt["profile"].mean()) * ANN,
                    noprofile=0.4 * r["ic_mean"] + 0.3 * float(d_vol["alpha"].mean()) * ANN
                    + 0.3 * (1 - T))

    rows.append(evaluate("DEPLOY", 0.0, 0.0, P0))
    for pa in PHI_AMT:
        for pb in PHI_VOL:
            if pa == 0.0 and pb == 0.0:
                continue
            rows.append(evaluate(f"a{pa:+.3f}|v{pb:+.3f}", pa, pb,
                                 tilt2(P0, PA, PB, pa, pb)))
    log(f"  {fold} done")
    return pd.DataFrame(rows)


def main() -> int:
    log("读因子面板")
    fdf = load_factors(20210101, 20241231)
    parts = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        parts.append(fold_block(fold, fdf))
    M = pd.concat(parts, ignore_index=True)
    M.to_csv(ROOT / "experiments" / "LH032" / "dual_tilt_grid.csv", index=False)

    F = FOLDS
    b = M[M.tag == "DEPLOY"].set_index("fold").reindex(F)
    rows = []
    for (pa, pb), g in M[M.tag != "DEPLOY"].groupby(["phi_amt", "phi_vol"]):
        g = g.set_index("fold").reindex(F)
        dE = g["excess"] - b["excess"]
        dA = g["alpha_vol"] - b["alpha_vol"]
        dP = g["prof_vol"] - b["prof_vol"]
        dPa = g["prof_amt"] - b["prof_amt"]
        dI = g["ic"] - b["ic"]
        dT = g["turnover"] - b["turnover"]
        dF = g["final"] - b["final"]
        rows.append(dict(phi_amt=pa, phi_vol=pb,
                         dE=dE.mean(), dE_sign="".join("+" if v > 0 else "-" for v in dE),
                         dE_min=dE.min(),
                         dalpha=dA.mean(), dA_sign="".join("+" if v > 0 else "-" for v in dA),
                         dprof_vol=dP.mean(), dprof_amt=dPa.mean(),
                         dP_sign="".join("+" if v > 0 else "-" for v in dPa),
                         dIC=dI.mean(), dT=dT.mean(), dfinal=dF.mean(),
                         dF_sign="".join("+" if v > 0 else "-" for v in dF),
                         dF_min=dF.min()))
    R = pd.DataFrame(rows).set_index(["phi_amt", "phi_vol"])
    R.to_csv(ROOT / "experiments" / "LH032" / "dual_tilt_summary.csv")
    pd.set_option("display.width", 250)
    print("\n" + "=" * 130)
    print("双因子倾斜：log_amt（φ_amt<0 拿 profile）× vol60（φ_vol>0 拿 α），Δ 相对部署点")
    print("=" * 130)
    print(R.round(5).to_string())
    print("\n部署点（四折均值）：")
    print(b[["ic", "excess", "turnover", "final", "alpha_vol", "prof_vol"]].mean().round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LH032 第五步：把倾斜从「阶段2 池内重排」挪到「阶段1 候选池」。

LH032 第四步的 tilt 是 P += φ·(prank−0.5)，作用在两阶段打分之上 ⇒ 它直接拿因子分
去覆盖 H01 的池内次序，破坏了 α 的来源。LOFO 显示：
    Δprofile 4/4 可外推；但 Δfinal 只有 3/4，且 confirm2024（最近一折）增益衰减到≈0。

本步对照两种施加位置：
  A 池内重排（stage2）：P = P0 + φ·(prank−0.5)                  ← 改池内次序，伤 α
  B 候选池倾斜（stage1）：候选池 = (mix + φ·(prank−0.5)) top40%，池内仍严格按 H01
                                                                ← 只改「谁有机会被选」
  C 两阶段同时倾斜（对照，等同把 φ 加倍）

若 B 能在保住 Δprofile 的同时显著减小 Δα 损失 ⇒ 得到「α 中性、profile 可赚」的配置。
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
FACTORS = ["log_amt", "vol60"]
PHIS = [-0.40, -0.20, -0.15, -0.10, -0.075, -0.05, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04]


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
    """候选池 = pr_lh top K%；池内按 pr_h1 归一化重排。"""
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


def tilt_stage2(P0, pr_f, phi):
    """A：两阶段打分之上加性倾斜（改池内次序）。NaN 取中性。"""
    D = np.where(np.isfinite(pr_f), (pr_f - 0.5) * phi, 0.0)
    P = P0 + D
    iscand = np.isfinite(P0) & (P0 >= 1.0)
    P = np.where(iscand, P, P - 1.0)
    P[~np.isfinite(P0)] = np.nan
    return P


def tilt_stage1(pr_mix, pr_h1, pr_f, phi, K):
    """B：候选池用混合后的分位，池内仍严格按 H01。"""
    D = np.where(np.isfinite(pr_f), (pr_f - 0.5) * phi, 0.0)
    a = pr_mix + D
    return two_stage(a, pr_h1, K)


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
    pr_mix = pct_rank_2d(mix)
    P0 = two_stage(pr_mix, pct["H01"], 40.0)
    PR = {f: pct_rank_2d(W[f]) for f in FACTORS}

    rows = []

    def evaluate(tag, f, mode, phi, P):
        Pb, _ = fast_band(P, panel.LU, KQ)
        r = score(Pb, panel)
        S = top_sets(Pb, panel)
        d_vol = decompose(W, panel, S, ["vol60"], [NB], fold, tag)
        d_f = decompose(W, panel, S, [f], [NB], fold, tag)
        T = r["mean_turnover"]
        return dict(fold=fold, factor=f, mode=mode, phi=phi, tag=tag,
                    ic=r["ic_mean"], excess=r["annual_excess"], turnover=T,
                    final=r["final_score"],
                    alpha_vol=float(d_vol["alpha"].mean()) * ANN,
                    prof_vol=float(d_vol["profile"].mean()) * ANN,
                    alpha_f=float(d_f["alpha"].mean()) * ANN,
                    prof_f=float(d_f["profile"].mean()) * ANN,
                    noprofile=0.4 * r["ic_mean"] + 0.3 * float(d_vol["alpha"].mean()) * ANN
                    + 0.3 * (1 - T))

    rows.append(evaluate("DEPLOY", "log_amt", "base", 0.0, P0))
    for f in FACTORS:
        for phi in PHIS:
            rows.append(evaluate(f"{f}|s2|{phi:+.3f}", f, "stage2", phi,
                                 tilt_stage2(P0, PR[f], phi)))
            rows.append(evaluate(f"{f}|s1|{phi:+.3f}", f, "stage1", phi,
                                 tilt_stage1(pr_mix, pct["H01"], PR[f], phi, 40.0)))
        log(f"  {fold} {f} done")
    return pd.DataFrame(rows)


def main() -> int:
    log("读因子面板")
    fdf = load_factors(20210101, 20241231)
    parts = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        parts.append(fold_block(fold, fdf))
    M = pd.concat(parts, ignore_index=True)
    M.to_csv(ROOT / "experiments" / "LH032" / "stage_tilt_grid.csv", index=False)

    F = FOLDS
    b = M[M.tag == "DEPLOY"].set_index("fold").reindex(F)
    print("\n部署点（四折均值）：")
    print(b[["ic", "excess", "turnover", "final", "alpha_vol", "prof_vol"]].mean().round(5).to_string())

    rows = []
    for (f, mode, phi), g in M[M.tag != "DEPLOY"].groupby(["factor", "mode", "phi"]):
        g = g.set_index("fold").reindex(F)
        dE = g["excess"] - b["excess"]
        dA = g["alpha_vol"] - b["alpha_vol"]
        dP = g["prof_f"] - b["prof_f"]
        dI = g["ic"] - b["ic"]
        dT = g["turnover"] - b["turnover"]
        dF = g["final"] - b["final"]
        rows.append(dict(factor=f, mode=mode, phi=phi,
                         dE=dE.mean(), dE_sign="".join("+" if v > 0 else "-" for v in dE),
                         dE_min=dE.min(),
                         dalpha=dA.mean(), dA_sign="".join("+" if v > 0 else "-" for v in dA),
                         dprof=dP.mean(), dP_min=dP.min(),
                         dP_sign="".join("+" if v > 0 else "-" for v in dP),
                         dIC=dI.mean(), dT=dT.mean(), dfinal=dF.mean(),
                         dF_sign="".join("+" if v > 0 else "-" for v in dF),
                         dF_min=dF.min()))
    R = pd.DataFrame(rows).set_index(["factor", "mode", "phi"])
    R.to_csv(ROOT / "experiments" / "LH032" / "stage_tilt_summary.csv")
    pd.set_option("display.width", 260)
    print("\n" + "=" * 140)
    print("倾斜位置对照：stage1 = 只倾斜候选池（池内仍按 H01）；stage2 = 池内重排（覆盖 H01 次序）")
    print("=" * 140)
    print(R.round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

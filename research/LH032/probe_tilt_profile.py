"""LH032 第三步：把「可外推的 profile」真的做进选股里，看 ΔE 能拿多少、要付什么代价。

LH032 第二步已证：层溢价剖面在半年粒度上**可外推**（follow ≈ oracle，11/11 正，
reverse 显著负）。但那是「任意控制层暴露」的理想上界，不涉及选股。

本步把它落到真实管道：在两阶段打分之上做组内倾斜

    P_t = P0_t + φ · ( pct_rank(f)_t − 0.5 )

φ>0 往 f 高分位层倾斜，φ<0 往低分位。φ<1 保证不破坏候选池边界。

对每个因子同时报两套分解：
  - vol60 十层（与 LH020/LH029 可比）：α_vol / prof_vol
  - 因子 f 十层（本次倾斜真正发生作用的那一维）：α_f / prof_f
E 与 IC / T 与分解无关，唯一。
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
sys.path.insert(0, str(ROOT / "research" / "LH023"))   # 必须优先于 CACHE
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
FACTORS = ["vol60", "log_amt", "mom60", "rev1", "log_price", "mom20"]
PHI = [-0.30, -0.15, -0.075, 0.075, 0.15, 0.30]
COSTS = [0.0010, 0.0025, 0.0040]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _rank(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a, kind="mergesort")
    r = np.empty(a.size, dtype=np.float64)
    r[o] = np.arange(a.size, dtype=np.float64)
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


def pct_rank_2d(P: np.ndarray) -> np.ndarray:
    out = np.full(P.shape, np.nan)
    for t in range(P.shape[0]):
        x = P[t]
        m = np.isfinite(x)
        n = int(m.sum())
        if n < 2:
            continue
        out[t, m] = _rank(x[m]) / (n - 1)
    return out


def two_stage(pr_lh: np.ndarray, pr_h1: np.ndarray, K: float) -> np.ndarray:
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
        within = np.where(np.isfinite(bb), (bb - lo) / (hi - lo), 0.5) if hi > lo else \
            np.where(np.isfinite(bb), 0.5, 0.5)
        out[t] = np.where(cand, 1.0 + within, np.where(m, a, np.nan))
    return out


def tilt(P0: np.ndarray, pr_f: np.ndarray, phi: float) -> np.ndarray:
    """组内重排：候选池内加同一单调函数，池边界不变（|φ|<1）。

    因子 NaN（停牌 / 滚动窗口不足）处必须取中性 0，否则 (NaN-0.5)*φ=NaN 会
    把整只股票的分数打掉，可选集崩溃 ⇒ 换手与 IC 同时爆炸（踩过）。
    """
    D = np.where(np.isfinite(pr_f), (pr_f - 0.5) * phi, 0.0)
    P = P0 + D
    iscand = np.isfinite(P0) & (P0 >= 1.0)
    P = np.where(iscand, P, P - 1.0)
    P[~np.isfinite(P0)] = np.nan
    return P


def fold_block(fold: str, fdf: pd.DataFrame) -> pd.DataFrame:
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
    PR = {f: pct_rank_2d(W[f]) for f in FACTORS}

    rows = []

    def evaluate(tag, f, phi, P):
        Pb, _ = fast_band(P, panel.LU, KQ)      # band 留存 = 换手控制器，不可省
        r = score(Pb, panel)
        S = top_sets(Pb, panel)
        d_vol = decompose(W, panel, S, ["vol60"], [NB], fold, tag)
        d_f = decompose(W, panel, S, [f], [NB], fold, tag)
        T = r["mean_turnover"]
        ann_turn = 504.0 * T / (2.0 - T)
        rec = dict(fold=fold, factor=f, phi=phi, tag=tag,
                   ic=r["ic_mean"], excess=r["annual_excess"], turnover=T,
                   final=r["final_score"],
                   alpha_vol=float(d_vol["alpha"].mean()) * ANN,
                   prof_vol=float(d_vol["profile"].mean()) * ANN,
                   alpha_f=float(d_f["alpha"].mean()) * ANN,
                   prof_f=float(d_f["profile"].mean()) * ANN,
                   noprofile=0.4 * r["ic_mean"] + 0.3 * float(d_vol["alpha"].mean()) * ANN
                   + 0.3 * (1 - T),
                   ann_turn=ann_turn)
        for c in COSTS:
            rec[f"netE_{c:.4f}"] = rec["excess"] - ann_turn * c
        return rec

    rows.append(evaluate("DEPLOY", "vol60", 0.0, P0))
    for f in FACTORS:
        for phi in PHI:
            P = tilt(P0, PR[f], phi)
            rows.append(evaluate(f"{f}|{phi:+.3f}", f, phi, P))
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
    M.to_csv(ROOT / "experiments" / "LH032" / "tilt_profile_grid.csv", index=False)
    log(f"网格 {M.shape} 已存")

    F = FOLDS
    b = M[M.tag == "DEPLOY"].set_index("fold").reindex(F)
    rows = []
    for (f, phi), g in M[M.tag != "DEPLOY"].groupby(["factor", "phi"]):
        g = g.set_index("fold").reindex(F)
        dE = g["excess"] - b["excess"]
        dA = g["alpha_vol"] - b["alpha_vol"]
        dP = g["prof_f"] - b["prof_f"]
        dI = g["ic"] - b["ic"]
        dT = g["turnover"] - b["turnover"]
        dF = g["final"] - b["final"]
        rec = dict(factor=f, phi=phi, dE=dE.mean(), dE_t=dE.mean() / (dE.std() / 2),
                   dE_sign="".join("+" if v > 0 else "-" for v in dE),
                   dalpha=dA.mean(), dprof_f=dP.mean(),
                   dprof_t=dP.mean() / (dP.std() / 2) if dP.std() > 0 else np.nan,
                   dprof_sign="".join("+" if v > 0 else "-" for v in dP),
                   dIC=dI.mean(), dT=dT.mean(), dfinal=dF.mean(),
                   dfinal_t=dF.mean() / (dF.std() / 2) if dF.std() > 0 else np.nan)
        rows.append(rec)
    R = pd.DataFrame(rows).set_index(["factor", "phi"])
    R.to_csv(ROOT / "experiments" / "LH032" / "tilt_profile_summary.csv")

    pd.set_option("display.width", 250)
    print("\n" + "=" * 130)
    print("多因子倾斜：Δ 相对部署点（四折均值，dE_t 为配对 t；α 用 vol60 十层口径，prof_f 用该因子十层口径）")
    print("=" * 130)
    print(R.round(5).to_string())
    print("\n部署点绝对值（四折均值）：")
    print(b[["ic", "excess", "turnover", "final", "alpha_vol", "prof_vol"]].mean().round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

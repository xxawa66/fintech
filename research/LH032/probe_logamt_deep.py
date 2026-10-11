"""LH032 第四步：log_amt 倾斜的细扫 + 逐月定位。

四折网格里 log_amt 往小成交额（φ<0）倾斜是唯一「α 中性、profile 大涨」的轴：
    ΔE +0.054 / Δprofile +0.095~0.106（t=5.8, ++++）/ ΔT +0.0008 / Δfinal +0.018
但 confirm2024 是唯一负折（Δα −0.039，而 Δprofile 仍 +0.052）。

log_amt = 20 日成交额均值对数 = 流动性/小盘代理，2024 年是小微盘事件年。
本脚本回答：confirm2024 的亏损是集中在个别月份的事件风险，还是全年系统性损伤？

输出：
  1. φ 细扫（含很小幅度）四折指标
  2. 逐月 Δexcess 序列（定位亏损月份）
  3. 名单在 log_amt 十层的暴露变化
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
from _probe_dgtw import decompose, stratify                         # noqa: E402

KQ = 0.0022778298112255263
ANN = 252
NB = 10
FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
F = "log_amt"
PHIS = [-0.40, -0.30, -0.20, -0.15, -0.10, -0.075, -0.05, -0.04, -0.03, -0.02, -0.01]


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


def tilt(P0, pr_f, phi):
    D = np.where(np.isfinite(pr_f), (pr_f - 0.5) * phi, 0.0)
    P = P0 + D
    iscand = np.isfinite(P0) & (P0 >= 1.0)
    P = np.where(iscand, P, P - 1.0)
    P[~np.isfinite(P0)] = np.nan
    return P


def exposure(panel, S, X):
    """名单在 X 十层的占比（四折均值口径）。"""
    acc = np.zeros(NB)
    cnt = 0
    for t in range(panel.T):
        ok = panel.valid_top[t]
        if ok.sum() < 200 or not S[t].any():
            continue
        b = stratify(X[t], NB)
        m = ok & (b >= 0)
        sel = S[t] & m
        if sel.sum() < 20:
            continue
        for c in range(NB):
            acc[c] += (sel & (b == c)).sum() / sel.sum()
        cnt += 1
    return acc / max(cnt, 1)


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
    PR = pct_rank_2d(W[F])

    rows, daily = [], []

    def evaluate(tag, phi, P):
        Pb, _ = fast_band(P, panel.LU, KQ)
        r = score(Pb, panel)
        S = top_sets(Pb, panel)
        d_vol = decompose(W, panel, S, ["vol60"], [NB], fold, tag)
        d_f = decompose(W, panel, S, [F], [NB], fold, tag)
        ex = d_vol.set_index("trade_date")["excess"]
        daily.append(pd.DataFrame({"fold": fold, "phi": phi, "date": ex.index,
                                   "excess": ex.to_numpy(),
                                   "alpha": d_vol.set_index("trade_date")["alpha"].to_numpy(),
                                   "profile": d_f.set_index("trade_date")["profile"].to_numpy()}))
        T = r["mean_turnover"]
        expo = exposure(panel, S, W[F])
        rec = dict(fold=fold, phi=phi, tag=tag, ic=r["ic_mean"], excess=r["annual_excess"],
                   turnover=T, final=r["final_score"],
                   alpha_vol=float(d_vol["alpha"].mean()) * ANN,
                   prof_vol=float(d_vol["profile"].mean()) * ANN,
                   alpha_f=float(d_f["alpha"].mean()) * ANN,
                   prof_f=float(d_f["profile"].mean()) * ANN,
                   noprofile=0.4 * r["ic_mean"] + 0.3 * float(d_vol["alpha"].mean()) * ANN + 0.3 * (1 - T))
        rec.update({f"E_L{k}": expo[k] for k in range(NB)})
        return rec

    rows.append(evaluate("DEPLOY", 0.0, P0))
    for phi in PHIS:
        rows.append(evaluate(f"{F}|{phi:+.3f}", phi, tilt(P0, PR, phi)))
    log(f"  {fold} done")
    return pd.DataFrame(rows), pd.concat(daily, ignore_index=True)


def main() -> int:
    log("读因子面板")
    fdf = load_factors(20210101, 20241231)
    parts, dparts = [], []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        a, b = fold_block(fold, fdf)
        parts.append(a)
        dparts.append(b)
    M = pd.concat(parts, ignore_index=True)
    D = pd.concat(dparts, ignore_index=True)
    M.to_csv(ROOT / "experiments" / "LH032" / "logamt_phi_grid.csv", index=False)
    D.to_csv(ROOT / "experiments" / "LH032" / "logamt_daily.csv", index=False)

    F4 = FOLDS
    b = M[M.tag == "DEPLOY"].set_index("fold").reindex(F4)
    print("\n" + "=" * 128)
    print(f"{F} 倾斜 φ 细扫：Δ 相对部署点（四折；α/prof_vol 用 vol60 口径，prof_f 用 log_amt 口径）")
    print("=" * 128)
    rows = []
    for phi in PHIS:
        g = M[np.isclose(M.phi, phi)].set_index("fold").reindex(F4)
        dE = g["excess"] - b["excess"]
        dA = g["alpha_vol"] - b["alpha_vol"]
        dP = g["prof_f"] - b["prof_f"]
        dI = g["ic"] - b["ic"]
        dT = g["turnover"] - b["turnover"]
        dF = g["final"] - b["final"]
        rows.append(dict(phi=phi,
                         dE=dE.mean(), dE_sign="".join("+" if v > 0 else "-" for v in dE),
                         dE_min=dE.min(),
                         dalpha=dA.mean(), dalpha_sign="".join("+" if v > 0 else "-" for v in dA),
                         dprof=dP.mean(), dprof_sign="".join("+" if v > 0 else "-" for v in dP),
                         dprof_min=dP.min(),
                         dIC=dI.mean(), dT=dT.mean(), dfinal=dF.mean(),
                         dfinal_sign="".join("+" if v > 0 else "-" for v in dF)))
    R = pd.DataFrame(rows).set_index("phi")
    print(R.round(5).to_string())

    # ---- 逐月定位 ----
    D["month"] = D["date"].astype(str).str[:6]
    base = D[np.isclose(D.phi, 0.0)].set_index(["fold", "date"])
    print("\n" + "=" * 128)
    print("逐月 Δexcess（相对部署点，年化口径×12；定位 confirm2024 的亏损月份）")
    print("=" * 128)
    for phi in [-0.30, -0.15, -0.05]:
        g = D[np.isclose(D.phi, phi)].copy()
        g = g.set_index(["fold", "date"])
        g["dE"] = g["excess"] - base["excess"]
        g["dA"] = g["alpha"] - base["alpha"]
        g = g.reset_index()
        g["month"] = g["date"].astype(str).str[:6]
        mth = g.groupby(["fold", "month"])[["dE", "dA"]].mean()
        for fold in FOLDS:
            s = mth.loc[fold]
            neg = s[s["dE"] < 0]
            print(f"\n[φ={phi:+.2f} | {fold}] 负月份 {len(neg)}/{len(s)}，"
                  f"最差 {s['dE'].idxmin()} {s['dE'].min():+.4f}，最好 {s['dE'].idxmax()} {s['dE'].max():+.4f}")
            if fold == "confirm2024":
                print("   " + "  ".join(f"{m[-2:]}:{v*252:+.3f}" for m, v in s["dE"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LH031-d：官方 band 编码把预测向量切成三段，其中**只有一段进交易**。把每段的排序自由度扫出来。

结构（读 fast_band + official_eval 得出）
---------------------------------------
提交的原始分 P0 经过官方 band 编码后变成三段，且段间顺序被**强制**：

    band 段   (~10%)  在榜股，保留全市场分位        -> 最终向量的最上面
    nonpool 段(~60%)  候选池之外，保留全市场分位      -> 中间
    drop 段   (~30%)  池内但未在榜，整体下移 1−q*    -> 最下面

交易（年化超额 / 换手 / α / profile）只吃 band 段。而 IC 吃**整条向量**。
⇒ **只要 band 段的成员集合不变，另外两段的排序、以及 band 段内部的排序，
   对 E / α / profile / T 全部逐位无影响，只改 IC。**

本脚本把三段各自的排序信号当成独立自由度扫：
    top_sig    band 段内部次序（成员集合恒为 H01 池内 top 25%）
    rest_sig   drop 段内部次序
    np_sig     nonpool 段内部次序
部署点 = (H01, H01, mix)。

判据：四折配对 ΔIC / t / 逐折符号；硬自检 = E/T/α/profile 相对部署点逐位为 0。
全程只用 2021-2024。
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
_OUT = _REPO / "experiments" / "LH031"
_OUT.mkdir(parents=True, exist_ok=True)

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from itertools import product  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = _REPO
CACHE = REPO / "outputs" / "long_horizon"
OUT = _OUT
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/LH023"))

from _probe_dgtw import decompose, pct_rank_2d, stratify  # noqa: E402
from _probe_e_attr import load_factors, top_sets, wide_of  # noqa: E402
from _probe_root_cause import (  # noqa: E402
    Panel,
    fast_band,
    holding_stats,
    load_fold,
    score,
    to_wide,
)

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KQ = 0.0022778298112255263
K_STAGE1 = 40.0
ANN = 252
NBIN = 10
LAYER = "vol60"
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
F_TOP = 0.25                      # pool 内 H01 序的前 25% = 全池 top10% = band 段

# 三段各自的候选排序信号
SIGS = ["H01", "mix", "H05_F1T2", "H05_F2T2"]
GRID_TOP = ["H01", "mix", "H05_F1T2", "H05_F2T2"]
GRID_REST = ["H01", "mix", "H05_F2T2"]
GRID_NP = ["mix", "H05_F1T2"]

ARCHIVE = {
    "wf2021": dict(ic=0.081156, excess=0.257336, turnover=0.022824,
                   final=0.402816, alpha=0.256086, profile=-0.009986),
    "wf2022": dict(ic=0.097536, excess=0.207012, turnover=0.016528,
                   final=0.396160, alpha=0.150050, profile=0.051173),
    "wf2023": dict(ic=0.069012, excess=0.138517, turnover=0.012252,
                   final=0.365484, alpha=0.140308, profile=-0.006093),
    "confirm2024": dict(ic=0.097487, excess=0.230244, turnover=0.021913,
                        final=0.401494, alpha=0.209000, profile=0.018148),
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _norm(v: np.ndarray) -> np.ndarray:
    """压到 [0,1]，缺失置 0.5。"""
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.where(np.isfinite(v), 0.5, 0.5)
    return np.where(np.isfinite(v), (v - lo) / (hi - lo), 0.5)


def build(pr_mix, pr_h1, sig_top, sig_rest, sig_np, K, f):
    out = np.full(pr_mix.shape, np.nan)
    Tn = pr_mix.shape[0]
    for t in range(Tn):
        a, b = pr_mix[t], pr_h1[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        # 成员集合：恒由 H01 的池内分位决定（与部署点一致）
        wh = _norm(np.where(cand, b, np.nan))
        top = cand & np.isfinite(wh) & (wh >= 1.0 - f)
        rest = cand & (~top)
        v = np.where(top, 2.0 + _norm(np.where(top, sig_top[t], np.nan)), np.nan)
        v = np.where(rest, 1.0 + _norm(np.where(rest, sig_rest[t], np.nan)), v)
        out[t] = np.where(cand, v, np.where(m, sig_np[t], np.nan))
    return out


def base_P(pr_mix, pr_h1, K):
    """部署点：pool=1+within(H01)，nonpool=prank(mix)。"""
    out = np.full(pr_mix.shape, np.nan)
    for t in range(pr_mix.shape[0]):
        a, b = pr_mix[t], pr_h1[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        wh = _norm(np.where(cand, b, np.nan))
        out[t] = np.where(cand, 1.0 + wh, np.where(m, a, np.nan))
    return out


def fold_block(fold: str, fdf: pd.DataFrame) -> list[dict]:
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    need = ["H01", "H05_F1T2", "H05_F2T2"]
    df = df.drop(columns=[c for c in need if c in df.columns]).merge(
        pred[["ts_code", "trade_date"] + need], on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    vol = W[LAYER]
    Lbin = np.vstack([stratify(vol[t], NBIN) for t in range(panel.T)])
    pct = {m: pct_rank_2d(to_wide(panel, df[m].to_numpy(dtype="float64"))) for m in need}
    mix = np.nansum(np.stack([w * pct[m] for m, w in WEIGHTS.items()]), axis=0)
    pr_mix = pct_rank_2d(mix)
    S = {"H01": pct["H01"], "mix": pr_mix,
         "H05_F1T2": pct["H05_F1T2"], "H05_F2T2": pct["H05_F2T2"]}

    def evaluate(tag: str, P: np.ndarray) -> dict:
        r = score(P, panel)
        Sp = top_sets(P, panel)
        d = decompose(W, panel, Sp, [LAYER], [NBIN], fold, tag)
        alpha = float(d["alpha"].mean()) * ANN
        prof = float(d["profile"].mean()) * ANN
        hs = holding_stats(r["_S"])
        c = composition(r["_S"], Lbin, panel.valid_turn)
        T = r["mean_turnover"]
        return dict(fold=fold, tag=tag, ic=r["ic_mean"], excess=r["annual_excess"],
                    turnover=T, final=r["final_score"], alpha=alpha, profile=prof,
                    hold_days=hs["avg_hold_days"], list_size=hs["list_size"],
                    **{f"L{k}": c[k] for k in range(NBIN)})

    def composition(Sp, LB, valid_turn):
        out = {}
        for k in range(NBIN):
            num = den = 0.0
            for t in range(Sp.shape[0]):
                idx = Sp[t][Sp[t] >= 0]
                if idx.size == 0:
                    continue
                m = valid_turn[t][idx] & (LB[t][idx] >= 0)
                if m.sum() == 0:
                    continue
                num += (LB[t][idx][m] == k).sum()
                den += m.sum()
            out[k] = num / den if den > 0 else np.nan
        return out

    rows = []
    P0 = base_P(pr_mix, pct["H01"], K_STAGE1)
    rows.append(evaluate("DEPLOY", fast_band(P0, panel.LU, KQ)[0]))
    r = rows[-1]
    log(f"  {fold} DEPLOY          IC {r['ic']:.5f} E {r['excess']:.5f} "
        f"T {r['turnover']:.5f} F {r['final']:.5f} a {r['alpha']:.5f} p {r['profile']:+.5f}")

    for gt, gr, gn in product(GRID_TOP, GRID_REST, GRID_NP):
        tag = f"{gt}|{gr}|{gn}"
        Pb = build(pr_mix, pct["H01"], S[gt], S[gr], S[gn], K_STAGE1, F_TOP)
        rec = evaluate(tag, fast_band(Pb, panel.LU, KQ)[0])
        rows.append(rec)
        log(f"  {fold} {tag:28s} IC {rec['ic']:.5f} E {rec['excess']:.5f} "
            f"T {rec['turnover']:.5f} F {rec['final']:.5f} "
            f"dIC {rec['ic']-r['ic']:+.5f} dE {rec['excess']-r['excess']:+.2e} "
            f"dT {rec['turnover']-r['turnover']:+.2e}")
    return rows


def main() -> int:
    fdf = load_factors(20210101, 20241231)
    allrows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        allrows.extend(fold_block(fold, fdf))
    M = pd.DataFrame(allrows).drop_duplicates(["fold", "tag"], keep="last")
    M.to_csv(OUT / "free_ic_grid.csv", index=False)

    F = FOLDS
    print()
    print("=" * 96)
    print("自检 1：DEPLOY 逐位复现归档 TS_K40")
    print("=" * 96)
    bad = 0
    for fold in F:
        r = M[(M.fold == fold) & (M.tag == "DEPLOY")].iloc[0]
        a = ARCHIVE[fold]
        for k in ["ic", "excess", "turnover", "final", "alpha", "profile"]:
            if abs(float(r[k]) - a[k]) > 2e-5:
                print(f"  MISMATCH {fold} {k}: {float(r[k]):.6f} vs {a[k]:.6f}")
                bad += 1
    print(f"  {'PASS' if bad == 0 else f'FAIL ({bad})'}")

    print()
    print("=" * 96)
    print("自检 2：三段重排后，E / α / profile / T / 构成 / 持股 是否逐位不变")
    print("=" * 96)
    keys = ["excess", "alpha", "profile", "turnover", "hold_days", "list_size"] + \
           [f"L{k}" for k in range(NBIN)]
    worst = {}
    for fold in F:
        b = M[(M.fold == fold) & (M.tag == "DEPLOY")].iloc[0]
        sub = M[(M.fold == fold) & (M.tag != "DEPLOY")]
        for k in keys:
            dd = float((sub[k].astype(float) - float(b[k])).abs().max())
            worst[k] = max(worst.get(k, 0.0), dd)
    for k in keys:
        print(f"  {'OK ' if worst[k] < 1e-9 else 'DIFF'} {k:11s} 四折最大绝对偏差 = {worst[k]:.3e}")
    nz = [k for k in keys if worst[k] >= 1e-9]
    print(f"  ⇒ {'成立：只有 IC 会变' if not nz else f'不成立，变了: {nz}'}")

    print()
    print("=" * 96)
    print("四折 ΔIC（= Δfinal/0.4）")
    print("=" * 96)
    piv = M.pivot_table(index="tag", columns="fold", values="ic")[F]
    d = piv.sub(piv.loc["DEPLOY"], axis=1)
    tab = pd.DataFrame({"dIC": d.mean(axis=1), "min": d.min(axis=1),
                        "t": d.mean(axis=1) / (d.std(axis=1) / 2),
                        "sign": ["".join("+" if v > 0 else "-" for v in d.loc[i])
                                 for i in piv.index],
                        "dfinal": 0.4 * d.mean(axis=1)})
    print(tab.drop(index="DEPLOY").sort_values("dIC", ascending=False).round(6).to_string())
    print()
    print(f"产物 {OUT/'free_ic_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

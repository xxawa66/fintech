"""LH031-c：两阶段编码的 IC 代价 —— 能不能把「选对人」和「排好序」部分解耦。

背景（LH031-a 量出来的）
----------------------
官方 IC = 逐日**全池** Spearman(pred, y)。而 TS_K40 的编码是

    pool    (top 40% by mix) -> 1 + within(H01)   ∈ [1, 2]
    nonpool (bottom 60%)     -> prank(mix)        ∈ [0, 1]

问题：整个向量里只有 **top 10% 的排序会进交易**，但 IC 要的是**全池**排序。
而纯 mix 信号的全池 IC 是 **0.10649**，TS_K40 只有 **0.08630** —— 两阶段编码
为了让 top10% 被 H01 选出，把池内 40% 的股票从 mix 序改成了 H01 序，
**为此付了 ΔIC = −0.020（= 0.008 分）**。

⇒ 有一笔 0.008 分压在编码里。本脚本试两条把它拿回来的路：

  θ 轴   池内排序 = (1−θ)·within(H01) + θ·within(mix)
         θ=0 部署点（H01 选人），θ=1 纯 mix（IC 最高但选人变）
  B 轴   池内**分层**：H01 序的前 f 分位（= 全池 top10%）置顶 unchanged，
         池内其余 30% 改回 mix 序。⇒ 选中的集合不变，只有补位流会变。

判据：四折配对 Δ / t / 逐折符号；官方 final = 0.4·IC + 0.3·E + 0.3·(1−T)。
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
THETA = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TOPFRAC = [0.15, 0.25, 0.35]

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


def _within(v: np.ndarray) -> np.ndarray:
    """把 v 线性压到 [0,1]（缺失置 0.5），与 two_stage 的 within 一致。"""
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.where(np.isfinite(v), 0.5, 0.5)
    return np.where(np.isfinite(v), (v - lo) / (hi - lo), 0.5)


def build_theta(pr_mix, pr_h1, K, theta):
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
        wh = _within(np.where(cand, b, np.nan))
        wm = _within(np.where(cand, a, np.nan))
        out[t] = np.where(cand, 1.0 + (1 - theta) * wh + theta * wm, np.where(m, a, np.nan))
    return out


def build_lsplit(pr_mix, pr_h1, K, f):
    """池内分层：H01 序的前 f 分位置顶（值 ∈[2,3]），池内其余按 mix 序（∈[1,2]）。"""
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
        wh = _within(np.where(cand, b, np.nan))
        top = cand & (wh >= 1.0 - f) & np.isfinite(wh)
        rest = cand & (~top)
        v = np.where(top, 2.0 + np.clip((wh - (1.0 - f)) / f, 0.0, 1.0), np.nan)
        wm = _within(np.where(rest, a, np.nan))
        v = np.where(rest, 1.0 + wm, v)
        out[t] = np.where(cand, v, np.where(m, a, np.nan))
    return out


def fold_block(fold: str, fdf: pd.DataFrame) -> list[dict]:
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    need = [m for m in WEIGHTS if m in pred.columns]
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

    def evaluate(tag: str, P: np.ndarray) -> dict:
        r = score(P, panel)
        S = top_sets(P, panel)
        d = decompose(W, panel, S, [LAYER], [NBIN], fold, tag)
        alpha = float(d["alpha"].mean()) * ANN
        prof = float(d["profile"].mean()) * ANN
        hs = holding_stats(r["_S"])
        c = composition(r["_S"], Lbin, panel.valid_turn)
        T = r["mean_turnover"]
        return dict(fold=fold, tag=tag, ic=r["ic_mean"], excess=r["annual_excess"],
                    turnover=T, final=r["final_score"], alpha=alpha, profile=prof,
                    hold_days=hs["avg_hold_days"], list_size=hs["list_size"],
                    **{f"L{k}": c[k] for k in range(NBIN)})

    def composition(S, LB, valid_turn):
        out = {}
        for k in range(NBIN):
            num = den = 0.0
            for t in range(S.shape[0]):
                idx = S[t][S[t] >= 0]
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
    variants = [("theta|0", build_theta(pr_mix, pct["H01"], K_STAGE1, 0.0))]
    for th in THETA:
        if th == 0.0:
            continue
        variants.append((f"theta|{th:g}", build_theta(pr_mix, pct["H01"], K_STAGE1, th)))
    for f in TOPFRAC:
        variants.append((f"lsplit|{f:g}", build_lsplit(pr_mix, pct["H01"], K_STAGE1, f)))
    for tag, P0 in variants:
        P, _ = fast_band(P0, panel.LU, KQ)
        r = evaluate(tag, P)
        rows.append(r)
        log(f"  {fold} {tag:12s} IC {r['ic']:.5f} E {r['excess']:.5f} "
            f"T {r['turnover']:.5f} F {r['final']:.5f} a {r['alpha']:.5f} "
            f"p {r['profile']:+.5f} hold {r['hold_days']:.0f}")
    return rows


def main() -> int:
    fdf = load_factors(20210101, 20241231)
    allrows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        allrows.extend(fold_block(fold, fdf))
    M = pd.DataFrame(allrows)
    M.to_csv(OUT / "encoding_ic_grid.csv", index=False)

    F = FOLDS
    print()
    print("=" * 92)
    print("自检：theta|0 必须逐位复现归档 TS_K40（四折 × 6 项）")
    print("=" * 92)
    bad = 0
    for fold in F:
        r = M[(M.fold == fold) & (M.tag == "theta|0")].iloc[0]
        a = ARCHIVE[fold]
        for k in ["ic", "excess", "turnover", "final", "alpha", "profile"]:
            if abs(float(r[k]) - a[k]) > 2e-5:
                print(f"  MISMATCH {fold} {k}: {float(r[k]):.6f} vs {a[k]:.6f}")
                bad += 1
    print(f"  {'PASS' if bad == 0 else f'FAIL ({bad})'}")

    print()
    print("=" * 92)
    print("四折均值")
    print("=" * 92)
    G = M.groupby("tag")[["ic", "excess", "turnover", "final", "alpha", "profile",
                          "hold_days"]].mean()
    G["noprofile"] = 0.4 * G["ic"] + 0.3 * G["alpha"] + 0.3 * (1 - G["turnover"])
    print(G.round(5).to_string())

    print()
    print("=" * 92)
    print("配对 Δ vs theta|0（四折）")
    print("=" * 92)
    cc = ["ic", "excess", "turnover", "final", "alpha", "profile"]
    piv = {k: M.pivot_table(index="tag", columns="fold", values=k)[F] for k in cc}
    b0 = {k: piv[k].loc["theta|0"] for k in cc}
    tab = []
    for tag in G.index:
        if tag == "theta|0":
            continue
        rec = {"tag": tag}
        for k in cc:
            d = piv[k].loc[tag] - b0[k]
            rec[f"d{k}"] = d.mean()
            rec[f"t_{k}"] = d.mean() / (d.std() / 2)
            rec[f"s_{k}"] = "".join("+" if v > 0 else "-" for v in d)
        tab.append(rec)
    T = pd.DataFrame(tab).set_index("tag")
    show = T[["dic", "t_ic", "s_ic", "dexcess", "t_excess", "dturnover", "t_turnover",
              "dfinal", "s_final", "dalpha", "s_alpha", "dprofile"]]
    print(show.round(5).to_string())
    print()
    print(f"产物 {OUT/'encoding_ic_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

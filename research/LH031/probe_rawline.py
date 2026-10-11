"""LH031-e：明线 / 暗线分离 —— 把 top10% 之外铺成「原始预测」。

结构（LH031-d 实测确认）
-----------------------
官方 band 编码把提交的预测向量切成三段，段间顺序被强制：

    band 段    ~10%   在榜股，保留全池分位        -> 最上面   ← **交易只吃这一段**
    drop 段    ~30%   池内未在榜，整体下移 1−q*    -> 最下面
    nonpool 段 ~60%   候选池之外，保留全池分位      -> 中间

实测（confirm2024，逐位）：
  只要 band 段保持 H01 序 ⇒ E / α / profile / T / 持股天数 **全部逐位 0 变化**；
  drop 段与 nonpool 段换成任何信号，暗线都不动，只改 IC。

⇒ 用户原则可执行：**band 段（暗线，决定组合）不动；band 段之外一律铺原始预测（明线）。**
本脚本扫「原始预测」该取哪一个。

支持集坑：直接用别的模型的百分位会因 NaN 覆盖不同把股票排出可选集（实测 dE −0.0089），
因此所有候选信号都用 `where(isfinite(raw), raw, mix_raw)` 补齐到与 mix 完全相同的支持集。

判据：四折配对 ΔIC / t / 逐折符号，硬自检 dE = dT = dα = dprofile = 0。
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
F_TOP = 0.25
MODELS = ["H01", "H05_F1T2", "H05_F2T2"]
GRID_REST = ["H01", "mix", "H05_F1T2", "H05_F2T2", "avg3", "rawmix"]
GRID_NP = ["mix", "H05_F1T2", "H05_F2T2", "avg3", "rawmix"]

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
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.where(np.isfinite(v), 0.5, 0.5)
    return np.where(np.isfinite(v), (v - lo) / (hi - lo), 0.5)


def build(pr_mix, pr_h1, sig_top, sig_rest, sig_np, K, f):
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
        wh = _norm(np.where(cand, b, np.nan))          # 成员集合恒由 H01 池内分位决定
        top = cand & np.isfinite(wh) & (wh >= 1.0 - f)
        rest = cand & (~top)
        v = np.where(top, 2.0 + _norm(np.where(top, sig_top[t], np.nan)), np.nan)
        v = np.where(rest, 1.0 + _norm(np.where(rest, sig_rest[t], np.nan)), v)
        out[t] = np.where(cand, v, np.where(m, sig_np[t], np.nan))
    return out


def base_P(pr_mix, pr_h1, K):
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
        out[t] = np.where(cand, 1.0 + _norm(np.where(cand, b, np.nan)),
                          np.where(m, a, np.nan))
    return out


def fold_block(fold: str, fdf: pd.DataFrame) -> list[dict]:
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    df = df.drop(columns=[c for c in MODELS if c in df.columns]).merge(
        pred[["ts_code", "trade_date"] + MODELS], on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    vol = W[LAYER]
    Lbin = np.vstack([stratify(vol[t], NBIN) for t in range(panel.T)])
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in MODELS}
    # 部署定义：mix = Σ w·pct_rank(raw)（**分位级**融合，不可改成原始值级）
    pct = {m: pct_rank_2d(raw[m]) for m in MODELS}
    mix = np.nansum(np.stack([w * pct[m] for m, w in WEIGHTS.items()]), axis=0)
    pr_mix = pct_rank_2d(mix)

    # 支持集对齐：必须取**交集**（不是并集）。并集会把 mix 缺失、但别的模型有值的股票
    # 放进可选集，n_elig 一变 n_top 就变，暗线被改动（实测 dE −0.0089）。
    def safe(x):
        return np.where(np.isfinite(pr_mix), np.where(np.isfinite(x), x, pr_mix), np.nan)

    S = {m: safe(pct[m]) for m in MODELS}
    S["mix"] = pr_mix
    S["avg3"] = np.mean(np.stack([S[m] for m in MODELS]), axis=0)
    mix_raw = np.nansum(np.stack([w * raw[m] for m, w in WEIGHTS.items()]), axis=0)
    S["rawmix"] = safe(pct_rank_2d(np.where(np.isfinite(mix_raw), mix_raw, np.nan)))

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
    r0 = rows[-1]
    log(f"  {fold} DEPLOY        IC {r0['ic']:.5f} E {r0['excess']:.5f} "
        f"T {r0['turnover']:.5f} F {r0['final']:.5f}")

    for gr, gn in product(GRID_REST, GRID_NP):
        tag = f"{gr}|{gn}"
        Pb = build(pr_mix, pct["H01"], pct["H01"], S[gr], S[gn], K_STAGE1, F_TOP)
        rec = evaluate(tag, fast_band(Pb, panel.LU, KQ)[0])
        rows.append(rec)
        if abs(rec["excess"] - r0["excess"]) > 1e-9 or abs(rec["turnover"] - r0["turnover"]) > 1e-9:
            log(f"  {fold} {tag:22s} ** 暗线被改动 dE={rec['excess']-r0['excess']:+.5f} "
                f"dT={rec['turnover']-r0['turnover']:+.5f}")
    log(f"  {fold} {len(GRID_REST)*len(GRID_NP)} 变体完成")
    return rows


def main() -> int:
    fdf = load_factors(20210101, 20241231)
    allrows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        allrows.extend(fold_block(fold, fdf))
    M = pd.DataFrame(allrows).drop_duplicates(["fold", "tag"], keep="last")
    M.to_csv(OUT / "rawline_grid.csv", index=False)

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
    print("自检 2：暗线（E / α / profile / T / 持股 / 构成）是否逐位不变")
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

    print()
    print("=" * 96)
    print("四折 ΔIC（暗线不动 ⇒ Δfinal = 0.4·ΔIC）")
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
    print(f"产物 {OUT/'rawline_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

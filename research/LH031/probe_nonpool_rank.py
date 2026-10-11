"""LH031-b：非候选区（bottom 60%）排序信号整条轴 —— 一条**只改 IC** 的零成本通道。

动机
----
官方 IC = 逐日**全池** Spearman(pred, y)，只剔 y 缺失；而年化超额 / 换手只吃 top 10%。
⇒ 预测向量里 ~90% 的股票不进交易，却参与 IC。

TS_K40 的两阶段编码：

    pool    (top 40% by mix) -> 1 + within(H01)   ∈ [1, 2]
    nonpool (bottom 60%)     -> prank(mix)        ∈ [0, 1]

因为 pool 的值恒 ≥ 1 > nonpool 的值恒 ≤ 1，**nonpool 永远进不了 top 10%**。
`fast_band` 只在「全池排位」上取 top10%（在榜股保留全池分位、落选下移 1−q），
而在榜股的全池分位只由「有多少 nonpool 排在它下面」决定 —— 与 nonpool 的**内部次序**无关。

⇒ **改动 nonpool 区的排序信号，E / α / profile / T 必须逐位不变，只有 IC 变。**
本脚本把这条轴整条扫一遍，并把「逐位不变」写成硬自检。

诊断（LH031-a）已给出：nonpool 区内 mix 的块内 IC 只有 0.1033，
而 H05_F1T2 = 0.1096、H05_F2T2 = 0.1092。当前 60% 的宇宙是用一个更弱的信号排的。

判据
----
四折 CV（2021-2024）配对 ΔIC / t / 逐折符号。因为 ΔE = ΔT = 0 是恒等式，
这里 Δfinal = 0.4·ΔIC，所以只需要看 IC。
多重比较防御：只认四折同号 + 幅度排名在四折上一致者。
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
ALL_MODELS = ["H01", "H05_F2T1", "H10_F2T1", "H20_F2T1", "H30_F2T1",
              "H05_F1T2", "H10_F1T2", "H20_F1T2", "H30_F1T2",
              "H05_F2T2", "H10_F2T2", "H20_F2T2", "H30_F2T2"]

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


def two_stage_np(pr_mix: np.ndarray, pr_h1: np.ndarray, sig_np: np.ndarray,
                 K: float) -> np.ndarray:
    """two_stage 的变体：nonpool 区改用 sig_np 排序。pool 区一字不改。"""
    out = np.full(pr_mix.shape, np.nan)
    for t in range(pr_mix.shape[0]):
        a, b, s = pr_mix[t], pr_h1[t], sig_np[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        bb = np.where(cand & np.isfinite(b), b, np.nan)
        v = bb[cand & np.isfinite(b)]
        lo, hi = v.min(), v.max()
        if hi > lo:
            within = np.where(np.isfinite(bb), (bb - lo) / (hi - lo), 0.5)
        else:
            within = np.where(np.isfinite(bb), 0.5, 0.5)
        # nonpool：用 s 的全池分位（∈[0,1]，恒 < pool 的 ≥1）
        out[t] = np.where(cand, 1.0 + within, np.where(m, s, np.nan))
    return out


def fold_block(fold: str, fdf: pd.DataFrame) -> list[dict]:
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    need = [m for m in ALL_MODELS if m in pred.columns]
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

    # ---- 候选的非候选区信号 ----
    sigs: dict[str, np.ndarray] = {"mix(base)": pr_mix}
    for m in need:
        sigs[m] = pct[m]
    sigs["avg13"] = np.nanmean(np.stack([pct[m] for m in need]), axis=0)
    sigs["F1T2+F2T2"] = 0.5 * pct["H05_F1T2"] + 0.5 * pct["H05_F2T2"]
    sigs["F2T2fam"] = np.nanmean(np.stack([pct[f"H{h:02d}_F2T2"] for h in (5, 10, 20, 30)]), axis=0)
    sigs["F1T2fam"] = np.nanmean(np.stack([pct[f"H{h:02d}_F1T2"] for h in (5, 10, 20, 30)]), axis=0)
    sigs["noH01"] = np.nanmean(np.stack([pct[m] for m in need if m != "H01"]), axis=0)

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
    base = None
    for name, s in sigs.items():
        P0 = two_stage_np(pr_mix, pct["H01"], s, K_STAGE1)
        P, _ = fast_band(P0, panel.LU, KQ)
        rec = evaluate(name, P)
        rows.append(rec)
        if base is None:
            base = rec
            log(f"  {fold} {name:12s} IC {rec['ic']:.5f} E {rec['excess']:.5f} "
                f"T {rec['turnover']:.5f} F {rec['final']:.5f} a {rec['alpha']:.5f} "
                f"p {rec['profile']:+.5f} (BASE)")
        else:
            log(f"  {fold} {name:12s} IC {rec['ic']:.5f} E {rec['excess']:.5f} "
                f"T {rec['turnover']:.5f} F {rec['final']:.5f} a {rec['alpha']:.5f} "
                f"p {rec['profile']:+.5f}  dIC {rec['ic']-base['ic']:+.5f} "
                f"dE {rec['excess']-base['excess']:+.2e} dT {rec['turnover']-base['turnover']:+.2e}")
    return rows


def main() -> int:
    fdf = load_factors(20210101, 20241231)
    allrows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        allrows.extend(fold_block(fold, fdf))
    M = pd.DataFrame(allrows)
    M.to_csv(OUT / "nonpool_rank_grid.csv", index=False)

    F = FOLDS
    print()
    print("=" * 92)
    print("自检 1：基线 vs 归档 TS_K40（四折 × 6 项）")
    print("=" * 92)
    bad = 0
    for fold in F:
        r = M[(M.fold == fold) & (M.tag == "mix(base)")].iloc[0]
        a = ARCHIVE[fold]
        for k, kk in [("ic", "ic"), ("excess", "excess"), ("turnover", "turnover"),
                      ("final", "final"), ("alpha", "alpha"), ("profile", "profile")]:
            if abs(float(r[k]) - a[kk]) > 2e-5:
                print(f"  MISMATCH {fold} {k}: {float(r[k]):.6f} vs {a[kk]:.6f}")
                bad += 1
    print(f"  逐位一致：{'PASS' if bad == 0 else f'FAIL ({bad})'}")

    print()
    print("=" * 92)
    print("自检 2：改动 nonpool 排序后，E / α / profile / T / 名单构成是否逐位不变")
    print("=" * 92)
    keys = ["excess", "alpha", "profile", "turnover", "hold_days", "list_size"] + \
           [f"L{k}" for k in range(NBIN)]
    worst = {}
    for fold in F:
        b = M[(M.fold == fold) & (M.tag == "mix(base)")].iloc[0]
        sub = M[(M.fold == fold) & (M.tag != "mix(base)")]
        for k in keys:
            dd = (sub[k].astype(float) - float(b[k])).abs().max()
            worst[k] = max(worst.get(k, 0.0), float(dd))
    for k in keys:
        flag = "OK " if worst[k] < 1e-9 else "DIFF"
        print(f"  {flag} {k:11s} 四折最大绝对偏差 = {worst[k]:.3e}")
    nz = [k for k in keys if worst[k] >= 1e-9]
    print(f"  ⇒ {'零成本通道成立：只有 IC 会变' if not nz else f'不成立，以下项也变了: {nz}'}")

    print()
    print("=" * 92)
    print("四折 IC 与 ΔIC")
    print("=" * 92)
    piv = M.pivot_table(index="tag", columns="fold", values="ic").reindex(columns=F)
    base_ic = piv.loc["mix(base)"]
    d = piv.sub(base_ic, axis=1)
    tab = pd.DataFrame({"ICmean": piv.mean(axis=1), "dICmean": d.mean(axis=1),
                        "dICmin": d.min(axis=1),
                        "t": d.mean(axis=1) / (d.std(axis=1) / 2),
                        "sign": ["".join("+" if v > 0 else "-" for v in d.loc[i]) for i in piv.index]})
    print(tab.sort_values("dICmean", ascending=False).round(6).to_string())
    print()
    print(f"Δfinal = 0.4·ΔIC ⇒ 最佳者 Δfinal = "
          f"{0.4 * tab['dICmean'].max():+.5f}")

    order = ["mix(base)"] + ALL_MODELS + ["avg13", "F1T2+F2T2", "F2T2fam", "F1T2fam", "noH01"]
    print()
    print("=== 逐折 IC ===")
    print(piv.reindex([o for o in order if o in piv.index]).round(5).to_string())
    print()
    print(f"产物 {OUT/'nonpool_rank_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

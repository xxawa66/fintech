"""LH031-a：非候选区（bottom 60%）的排序质量诊断。

动机
----
官方 IC = 逐日**全池** Spearman(pred, y)，只剔 y 缺失；而超额 / 换手只吃 top 10%。
⇒ 预测向量里 ~90% 的股票**不进交易，却参与 IC**。

当前 TS_K40 的两阶段编码：

    pool    (top 40% by mix) -> 1 + within(H01)   ∈ [1, 2]
    nonpool (bottom 60%)     -> prank(mix)        ∈ [0, ~0.6]

因为 pool 的值恒 > nonpool 的值，nonpool **永远不可能**进入 top 10%，
也永远不会成为留仓股 ⇒ **对 nonpool 区做任何保序/换序，E / α / profile / T 全部逐位不变**，
只有 IC 变。这是一条零成本通道。

本脚本只做诊断：在 nonpool 区里，现有排序（mix）的 IC 是多少，
换成其它可用信号（反转类因子、模型分）能到多少。不改动任何东西。
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

import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = _REPO
CACHE = REPO / "outputs" / "long_horizon"
OUT = _OUT
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/LH023"))

from _probe_dgtw import pct_rank_2d  # noqa: E402
from _probe_e_attr import load_factors, wide_of  # noqa: E402
from _probe_root_cause import Panel, load_fold, to_wide  # noqa: E402

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
K_STAGE1 = 40.0
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
NEG_FACTORS = ["vol60", "mom20", "mom60", "log_amt", "rev1", "log_price"]
LAM = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wide_all(df: pd.DataFrame, panel: Panel, cols: list[str]) -> dict[str, np.ndarray]:
    """wide_of 的通用版（wide_of 只摊 FACTORS，这里要额外因子）。"""
    sub = df[(df.trade_date >= panel.dates[0]) & (df.trade_date <= panel.dates[-1])]
    m = sub.merge(pd.DataFrame({"trade_date": panel.dates}), on="trade_date", how="inner")
    di = {d: i for i, d in enumerate(panel.dates)}
    ci = {c: i for i, c in enumerate(panel.codes)}
    ti = m["trade_date"].map(di).to_numpy()
    si = m["ts_code"].map(ci).to_numpy()
    ok = ~np.isnan(ti) & ~np.isnan(si)
    ti, si = ti[ok].astype(int), si[ok].astype(int)
    out = {}
    for c in cols:
        W = np.full((panel.T, panel.N), np.nan)
        W[ti, si] = m[c].to_numpy(dtype="float64")[ok]
        out[c] = W
    return out


def cand_mask(pr_mix: np.ndarray, K: float) -> np.ndarray:
    """复刻 two_stage 的候选池边界（逐行，与 LH029 逐位一致）。"""
    m = np.isfinite(pr_mix)
    out = np.zeros_like(m)
    for t in range(pr_mix.shape[0]):
        a = pr_mix[t]
        mm = m[t]
        if mm.sum() < 200:
            continue
        thr = np.nanquantile(a[mm], 1.0 - K / 100.0)
        out[t] = mm & (a >= thr)
    return out.astype(bool)


def block_ic(sig: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    """逐日块内 Spearman 的均值（块内 rank，只剔 y 缺失）。"""
    vals = []
    for t in range(sig.shape[0]):
        mm = m[t] & np.isfinite(sig[t]) & np.isfinite(y[t])
        if mm.sum() < 30:
            continue
        a = rankdata(sig[t][mm])
        b = rankdata(y[t][mm])
        if a.std() == 0 or b.std() == 0:
            continue
        vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals)) if vals else np.nan


def main() -> int:
    dw = pd.read_csv(REPO / "experiments/LH019_deploy_weights.csv")
    row = dw[np.isclose(dw.lam, 0.5)].iloc[0]
    WEIGHTS.update({kv.split(":")[0]: float(kv.split(":")[1])
                    for kv in str(row["w"]).split(";")})
    log(f"部署权重 {WEIGHTS}")
    fdf = load_factors(20210101, 20241231)

    rows = []
    for fold in FOLDS:
        df = load_fold(fold)
        pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
        pred["ts_code"] = pred["ts_code"].astype(str)
        pred["trade_date"] = pred["trade_date"].astype("int64")
        extra = [m for m in WEIGHTS if m in pred.columns and m not in df.columns]
        if extra:
            df = df.drop(columns=[c for c in extra if c in df.columns]).merge(
                pred[["ts_code", "trade_date"] + extra], on=["ts_code", "trade_date"], how="left")
            df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
        panel = Panel(df)
        W = wide_all(fdf, panel, ["vol60"] + NEG_FACTORS)
        raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in WEIGHTS}
        mix = np.nansum(np.stack([WEIGHTS[m] * pct_rank_2d(raw[m]) for m in WEIGHTS]), axis=0)
        pr_mix = pct_rank_2d(mix)
        CM = cand_mask(pr_mix, K_STAGE1)
        NP = (~CM) & np.isfinite(pr_mix)
        Y = panel.Y
        V = np.isfinite(Y)

        sigs = {"mix": pr_mix}
        for m in WEIGHTS:
            sigs[m] = raw[m]
        for f in NEG_FACTORS:
            sigs["-" + f] = np.where(np.isfinite(W[f]), -W[f], np.nan)
        z = np.nanmean(np.stack([pct_rank_2d(sigs["-" + f]) for f in NEG_FACTORS]), axis=0)
        sigs["Z6"] = z
        for lam in LAM:
            if lam == 0.0:
                continue
            sigs[f"blend{lam:g}"] = (1 - lam) * pr_mix + lam * z

        for name, s in sigs.items():
            ic_np = block_ic(s, Y, NP & V)
            ic_all = block_ic(s, Y, np.isfinite(pr_mix) & V)
            rows.append(dict(fold=fold, signal=name, ic_nonpool=ic_np, ic_full=ic_all))
        log(f"  {fold} done  nonpool n={int((NP&V)[0].sum())}")

    M = pd.DataFrame(rows)
    M.to_csv(OUT / "nonpool_ic.csv", index=False)
    piv = M.pivot_table(index="signal", columns="fold", values="ic_nonpool")
    piv = piv[FOLDS]
    piv["mean"] = piv.mean(axis=1)
    order = ["mix", "H01", "H05_F1T2", "H05_F2T2"] + ["-" + f for f in NEG_FACTORS] + \
            ["Z6"] + [f"blend{l:g}" for l in LAM if l > 0]
    print()
    print("=== nonpool 区（bottom 60%）逐日块内 IC ===")
    print(piv.reindex([o for o in order if o in piv.index]).round(5).to_string())
    print()
    print("=== 全池 IC（对照）===")
    pf = M.pivot_table(index="signal", columns="fold", values="ic_full")[FOLDS]
    pf["mean"] = pf.mean(axis=1)
    print(pf.reindex([o for o in order if o in pf.index]).round(5).to_string())
    print()
    print(f"产物 {OUT/'nonpool_ic.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

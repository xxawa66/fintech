"""LH019：按「只用 IC 选型 + Active Share 预算」重新优化出最终模型。

方法论（LH018 定下的规矩）：
- IC 是唯一可估、可外推的量（一个折 ~110 万对日截面观测，日间近似独立）
  ⇒ **目标函数只用 IC**，不用 final、不用 E。
- E 一年只有约 1 个独立观测 ⇒ 不可估 ⇒ **不进目标函数**，改用
  「与基线名单的偏离度（Active Share）」当**风险预算约束**。
- 权重用 **留一折外（LOFO）** 求：在另外 3 个折上优化，在留出折上评估
  ⇒ 报出的 IC 是真正的样本外 IC，不含权重调优的乐观偏差。

目标（可解析、快）：
    对每个交易日 t，令 a_k = 信号 k 的截面百分位排名，b = y 的截面排名。
    S(w) = mean_t Pearson( Σ_k w_k a_k , b )
         = mean_t (w·u_t) / ( sqrt(w' C_t w) · s_b(t) )
    其中 u_t = 各信号与 b 的离差叉积，C_t = 各信号离差的 Gram 矩阵。
    ⇒ 逐日只需预计算 u_t(13), C_t(13×13), s_b(t)，优化瞬间完成。

约束：w ≥ 0, Σw = 1, w[H01] ≥ λ（λ = 换篮子预算，λ 越大越贴基线）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "outputs/long_horizon"))

from _probe_root_cause import (  # noqa: E402
    ANN, FOLD_WINDOWS, Panel, daily_topsets, fast_band, load_fold, score, to_wide,
)

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KQ_STAR = 0.0022778298112255263
STUDY = "LH003"

# LH003 候选池；实际使用 = 四个折的**交集**（否则 LOFO 无法一致）
CANDIDATES = [
    "H01",
    "H05_F2T1", "H10_F2T1", "H20_F2T1", "H30_F2T1",
    "H05_F1T2", "H10_F1T2", "H20_F1T2", "H30_F1T2",
    "H05_F2T2", "H10_F2T2", "H20_F2T2", "H30_F2T2",
    "H05_MEANRANK", "H10_MEANRANK", "H20_MEANRANK", "H30_MEANRANK",
]


def available() -> list[str]:
    sets = []
    for f in FOLDS:
        p = pd.read_parquet(ROOT / "outputs/long_horizon" / STUDY / f
                            / "raw_predictions.parquet")
        sets.append(set(c for c in p.columns if c not in ("ts_code", "trade_date")))
    common = set.intersection(*sets)
    return [c for c in CANDIDATES if c in common]


def pct_rank(P: np.ndarray) -> np.ndarray:
    out = np.full_like(P, np.nan)
    for t in range(P.shape[0]):
        m = np.isfinite(P[t])
        if m.any():
            out[t, m] = rankdata(P[t][m]) / int(m.sum())
    return out


# --------------------------------------------------------------------------
# 1) 预计算：每折的逐日充分统计量 + 宽矩阵
# --------------------------------------------------------------------------
FOLD_DATA: dict[str, dict] = {}


def prepare(fold: str, sigs: list[str]) -> dict:
    if fold in FOLD_DATA:
        return FOLD_DATA[fold]
    df = load_fold(fold)                       # 只有 5 个模型，需要补读
    need = [c for c in sigs if c not in df.columns]
    if need:
        pred = pd.read_parquet(
            ROOT / "outputs/long_horizon" / STUDY / fold / "raw_predictions.parquet")
        pred["ts_code"] = pred["ts_code"].astype(str)
        pred["trade_date"] = pred["trade_date"].astype("int64")
        df = df.merge(pred[["ts_code", "trade_date"] + need],
                      on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        df = df.sort_values(["trade_date", "ts_code"],
                            kind="stable").reset_index(drop=True)
    panel = Panel(df)
    wide = {s: pct_rank(to_wide(panel, df[s].to_numpy(dtype="float64")))
            for s in sigs}
    K = len(sigs)
    us, Cs, sbs = [], [], []
    for t in range(panel.T):
        m = panel.valid_ic[t]
        if m.sum() < 30:
            continue
        A = np.column_stack([wide[s][t][m] for s in sigs])   # (n, K)
        ok = np.isfinite(A).all(axis=1)
        if ok.sum() < 30:
            continue
        A = A[ok]
        b = rankdata(panel.Y[t][m][ok]).astype("float64")
        Ac = A - A.mean(axis=0)
        bc = b - b.mean()
        us.append(Ac.T @ bc)
        Cs.append(Ac.T @ Ac)
        sbs.append(float(np.sqrt(bc @ bc)))
    d = dict(panel=panel, wide=wide,
             u=np.asarray(us), C=np.asarray(Cs), sb=np.asarray(sbs))
    FOLD_DATA[fold] = d
    return d


# --------------------------------------------------------------------------
# 2) 在给定折集合上求 IC 最优权重
# --------------------------------------------------------------------------
def solve_weights(folds: list[str], sigs: list[str], lam: float,
                  n_start: int = 8, seed: int = 0) -> np.ndarray:
    K = len(sigs)
    U = np.vstack([prepare(f, sigs)["u"] for f in folds])
    C = np.vstack([prepare(f, sigs)["C"] for f in folds]).reshape(-1, K, K)
    SB = np.concatenate([prepare(f, sigs)["sb"] for f in folds])
    i0 = sigs.index("H01")

    def neg(w: np.ndarray) -> float:
        num = U @ w
        den = np.sqrt(np.einsum("i,kij,j->k", w, C, w)) * SB
        den = np.maximum(den, 1e-12)
        return -float(np.mean(num / den))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "ineq", "fun": lambda w: w[i0] - lam}]
    bnds = [(0.0, 1.0)] * K
    rng = np.random.default_rng(seed)
    best, bestv = None, np.inf
    starts = [np.full(K, 1.0 / K)]
    starts.append(np.where(np.arange(K) == i0, 1.0, 0.0))
    for _ in range(n_start):
        x = rng.dirichlet(np.ones(K))
        x[i0] = max(x[i0], lam)
        starts.append(x / x.sum())
    for x0 in starts:
        try:
            r = minimize(neg, x0, method="SLSQP", bounds=bnds,
                         constraints=cons, options=dict(maxiter=300, ftol=1e-10))
        except Exception:
            continue
        if r.success and np.isfinite(r.fun) and r.fun < bestv:
            best, bestv = r.x, r.fun
    if best is None:
        best = np.full(K, 1.0 / K)
    best = np.clip(best, 0, None)
    return best / best.sum()


# --------------------------------------------------------------------------
# 3) 在留出折上按官方口径评估
# --------------------------------------------------------------------------
def evaluate(fold: str, sigs: list[str], w: np.ndarray) -> dict:
    d = prepare(fold, sigs)
    panel, wide = d["panel"], d["wide"]
    P = np.zeros_like(wide[sigs[0]])
    for k, s in enumerate(sigs):
        if w[k] > 0:
            P = P + w[k] * wide[s]
    Pb, _ = fast_band(P, panel.LU, KQ_STAR)
    r = score(Pb, panel)
    # 与基线名单的重合度
    Sb = daily_topsets(fast_band(wide["H01"], panel.LU, KQ_STAR)[0], panel)
    S = daily_topsets(Pb, panel)
    inter = (S & Sb).sum(1).astype(float)
    sz = S.sum(1).astype(float)
    share = float(np.nanmean(np.where(sz > 0, inter / np.maximum(sz, 1), np.nan)))
    return dict(fold=fold, ic=r["ic_mean"], excess=r["annual_excess"],
                turnover=r["mean_turnover"], final=r["final_score"],
                overlap_with_base=share, active_share=1 - share,
                w=";".join(f"{s}:{v:.4f}" for s, v in zip(sigs, w) if v > 1e-4))


def lofo(sigs: list[str], lam: float) -> pd.DataFrame:
    rows = []
    for held in FOLDS:
        train = [f for f in FOLDS if f != held]
        w = solve_weights(train, sigs, lam)
        rows.append(evaluate(held, sigs, w))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    avail = available()
    print("四折共有信号:", avail, flush=True)
    for f in FOLDS:
        prepare(f, avail)
        print(f"  {f} 预计算完成", flush=True)

    out = []
    # 基线：w = 纯 H01
    for f in FOLDS:
        w = np.zeros(len(avail)); w[avail.index("H01")] = 1.0
        out.append(dict(tag="baseline_H01", lam=1.0, **evaluate(f, avail, w)))
    # 等权融合（LH017 方案）：H01 + H05_F2T2 + H10_F2T2
    mix3 = ["H01", "H05_F2T2", "H10_F2T2"]
    w = np.array([1 / 3 if s in mix3 else 0.0 for s in avail])
    for f in FOLDS:
        out.append(dict(tag="LH017_equal_mix3", lam=1 / 3, **evaluate(f, avail, w)))
    # IC 最优（LOFO），不同换篮子预算
    for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        df = lofo(avail, lam)
        df["tag"] = f"ICopt_lam{lam:.1f}"
        df["lam"] = lam
        out.extend(df.to_dict("records"))
        print(f"lam={lam:.1f}", df[["ic", "active_share", "final"]].mean().round(4).to_dict(), flush=True)

    m = pd.DataFrame(out)
    m.to_csv("experiments/LH019_icopt_lofo.csv", index=False)
    print("\n=== LOFO 汇总（四折均值）===")
    print(m.groupby("tag")[["ic", "excess", "turnover", "final",
                            "active_share", "overlap_with_base"]].mean().round(4).to_string())

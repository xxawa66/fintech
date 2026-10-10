"""LH020 第四步：两阶段模型（粗筛 + 精排）。

分层归因给出的机制
------------------
按 vol60 十层把 E 拆开后（年化）：

    tag        α(四折均值)  α(测试期)   profile(四折)  profile(测试期)
    H01          0.1844       0.1911       +0.0004         −0.0113
    H05_F1T2     0.1109       0.1053       +0.0812         +0.0172
    H05_F2T2     0.1198       0.0668       +0.0900         −0.0014
    LH019        0.1437       0.0871       +0.0735         +0.0154

- H01 的 α 四折 0.184 → 测试期 0.191（误差 +0.007），profile ≈ 0：
  **H01 不做风格押注，E 全部来自「在同类股票里挑到更好的」**。
- 长周期家族的 α 比 H01 低 0.04~0.07，CV 上多出来的 E 全部来自 profile
  （低波动轮廓 +0.075~0.090），而 profile 在测试期塌到 ≈ 0。
- α 与 IC 是可外推的两项；profile 不是。

⇒ 长周期模型的 IC 增益是真的，但它在**头部同层内选优**上不如 H01；
  它的「选股」实际上做的是「买低波动」。

设计：两阶段
------------
阶段 1（粗筛）：用长周期融合信号决定谁进候选池（top K%），它决定 bottom (100−K)%
                的排序 ⇒ 保留已被两次证实的 IC 增益。
阶段 2（精排）：候选池内用 H01 排序，最终 top 10% 全部来自 H01 的排序
                ⇒ 拿到 H01 的 α 与 ≈0 的因子轮廓。

K=10 退化为纯长周期，K=100 退化为纯 H01。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TESTDIR = ROOT.parent.parent / "test_y_2025_2026"
CACHE = ROOT / "outputs/long_horizon"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CACHE))

from _probe_dgtw import decompose, pct_rank_2d  # noqa: E402
from _probe_e_attr import load_factors, top_sets, wide_of  # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide  # noqa: E402

KQ = 0.0022778298112255263
ANN = 252
KS = [10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 100.0]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def two_stage(pr_lh: np.ndarray, pr_h1: np.ndarray, K: float) -> np.ndarray:
    """候选池 = 长周期 top K%；候选池内按 H01 重排，整体落在 [1,2]；非候选落在 [0,1]。"""
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
        # 候选池内 H01 的分位
        bb = np.where(cand & np.isfinite(b), b, np.nan)
        v = bb[cand & np.isfinite(b)]
        lo, hi = v.min(), v.max()
        within = np.full(a.shape, np.nan)
        if hi > lo:
            within = np.where(np.isfinite(bb), (bb - lo) / (hi - lo), 0.5)
        else:
            within = np.where(np.isfinite(bb), 0.5, 0.5)
        out[t] = np.where(cand, 1.0 + within, np.where(m, a, np.nan))
    return out


def run_fold(fold, fdf, weights, with_alpha=True):
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    extra = [m for m in ["H01", "H05_F1T2", "H05_F2T2"] if m in pred.columns and m not in df.columns]
    if extra:
        df = df.drop(columns=[c for c in extra if c in df.columns]).merge(
            pred[["ts_code", "trade_date"] + extra], on=["ts_code", "trade_date"], how="left")
        df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in ["H01", "H05_F1T2", "H05_F2T2"]}
    pct = {m: pct_rank_2d(P) for m, P in raw.items()}
    mix = np.nansum(np.stack([w * pct[m] for m, w in weights.items()]), axis=0)

    variants = {"H01": raw["H01"], "LH019": mix}
    for K in KS:
        if K == 100.0:
            continue
        variants[f"TS_K{K:g}"] = two_stage(pct_rank_2d(mix), pct["H01"], K)

    rows, dailies = [], []
    for tag, P0 in variants.items():
        P, _ = fast_band(P0, panel.LU, KQ)
        r = score(P, panel)
        S = top_sets(P, panel)
        row = dict(fold=fold, tag=tag, ic=r["ic_mean"], excess=r["annual_excess"],
                   turnover=r["mean_turnover"], final=r["final_score"])
        if with_alpha:
            d = decompose(W, panel, S, ["vol60"], [10], fold, tag)
            dailies.append(d)
            row["alpha"] = float(d["alpha"].mean()) * ANN
            row["profile"] = float(d["profile"].mean()) * ANN
        rows.append(row)
        log(f"  {fold} {tag:12s} IC {r['ic_mean']:.4f} E {r['annual_excess']:.4f} "
            f"T {r['mean_turnover']:.4f} F {r['final_score']:.4f}"
            + (f" α {row['alpha']:.4f} prof {row['profile']:.4f}" if with_alpha else ""))
    return pd.DataFrame(rows), pd.concat(dailies, ignore_index=True) if dailies else None


if __name__ == "__main__":
    dw = pd.read_csv(ROOT / "experiments/LH019_deploy_weights.csv")
    weights = {kv.split(":")[0]: float(kv.split(":")[1])
               for kv in str(dw[dw.lam == 0.5].iloc[0]["w"]).split(";")}
    fdf = load_factors(20210101, 20241231)
    allm, alld = [], []
    for fold in ["wf2021", "wf2022", "wf2023", "confirm2024"]:
        log(f"=== {fold} ===")
        m, d = run_fold(fold, fdf, weights)
        allm.append(m); alld.append(d)
    M = pd.concat(allm, ignore_index=True)
    D = pd.concat(alld, ignore_index=True)
    M.to_csv(ROOT / "experiments/LH020_twostage_folds.csv", index=False)
    D.to_csv(ROOT / "experiments/LH020_twostage_daily.csv", index=False)

    F = ["wf2021", "wf2022", "wf2023", "confirm2024"]
    pd.set_option("display.width", 320)
    print("\n=== 四折均值（按 IC 降序）===")
    g = M.groupby("tag")[["ic", "excess", "turnover", "final", "alpha", "profile"]].mean()
    g["adj_score"] = 0.4 * g["ic"] + 0.3 * (g["alpha"] + g["profile"]) + 0.3 * (1 - g["turnover"])
    g["adj_noprof"] = 0.4 * g["ic"] + 0.3 * g["alpha"] + 0.3 * (1 - g["turnover"])
    print(g.sort_values("adj_noprof", ascending=False).round(4).to_string())
    print("\n=== final 逐折 ===")
    p = M.pivot_table(index="tag", columns="fold", values="final")[F]
    print(p.round(4).assign(mean=p.mean(axis=1).round(4), worst=p.min(axis=1).round(4)).to_string())
    print("\n=== α 逐折 ===")
    q = M.pivot_table(index="tag", columns="fold", values="alpha")[F]
    print(q.round(4).assign(mean=q.mean(axis=1).round(4), sd=q.std(axis=1).round(4)).to_string())
    print("\n=== 换手 逐折 ===")
    t = M.pivot_table(index="tag", columns="fold", values="turnover")[F]
    print(t.round(4).assign(mean=t.mean(axis=1).round(4)).to_string())
    log("完成")

"""LH028 —— IC 的评价域含不可交易股票：「涨停股排序」是一个零成本自由度？

观察（读 `_probe_root_cause.Panel` 的三套有效集）

    valid_ic   = ~isnan(Y)                 官方 IC 域：**含当日涨停**
    valid_top  = (~LU) & (~isnan(Y))       官方超额域：剔涨停
    valid_turn = ~LU                       官方换手域：剔涨停

而 `_top_indices` 只在 valid_top / valid_turn 内部取 argpartition —— top 集合完全由
**非涨停股之间的相对次序**决定，与涨停股排在哪里无关。

⇒ 把当日涨停股的打分改成任意值，**E 与 T 一分钱都不动，但官方 IC 会变**。
这是一个此前从未打开过的自由度，且它的成本严格为零。

本脚本的目的**不是用这个自由度刷分**，而是把它量化清楚：
  A. 官方 IC 里有多少来自不可交易股票？（把 IC 拆成可交易域 / 不可交易域）
  B. 涨停股真实的次日收益处在什么位置 ⇒ 「诚实的排序」应该放哪？
  C. 极端 Beenchmark：把涨停股全部压到底 / 全部顶到顶，IC 各是多少？
     这两个数就是这条轴的容量上下界 —— 若诚实排序恰好落在下界附近且容量可观，
     说明官方 IC 系统性高估了可投资能力。

判据不变：只用 2021–2024 四个 CV 折 + 四折配对 Δ/t + 逐折符号。
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = (
    _P(_os.environ["FINTECH_ROOT"])
    if _os.environ.get("FINTECH_ROOT")
    else next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
)
_LH028_DIR = _P(_os.environ.get("LH028_DIR", _REPO / "experiments" / "LH028"))
_LH028_DIR.mkdir(parents=True, exist_ok=True)

import os  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

REPO = _REPO
CACHE = REPO / "outputs/long_horizon"
OUT = _LH028_DIR
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/LH023"))
sys.path.insert(0, str(REPO / "research/LH025"))

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
import probe_band_freeze as L25  # noqa: E402

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KQ = 0.0022778298112255263
K_STAGE1 = 40.0
ANN = 252
NBIN = 10
LAYER = "vol60"
IC_MIN_VALID = 30


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
def daily_ic_subset(P: np.ndarray, panel: Panel, mask_extra: np.ndarray | None = None):
    """逐日 IC；mask_extra 为 None 时 = 官方口径（valid_ic）。"""
    T = panel.T
    out = np.full(T, np.nan)
    n_all = np.zeros(T)
    n_inv = np.zeros(T)
    for t in range(T):
        m = panel.valid_ic[t] if mask_extra is None else (panel.valid_ic[t] & mask_extra[t])
        if int(m.sum()) < IC_MIN_VALID:
            continue
        rp = rankdata(P[t][m])
        ry = rankdata(panel.Y[t][m])
        if rp.std() == 0 or ry.std() == 0:
            continue
        out[t] = float(np.corrcoef(rp, ry)[0, 1])
        n_all[t] = int(m.sum())
        n_inv[t] = int((panel.valid_ic[t] & ~panel.LU[t]).sum())
    return out, n_all, n_inv


def mean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def evaluate(tag: str, Pb: np.ndarray, panel: Panel, W: dict) -> dict:
    r = score(Pb, panel)
    ic_off = r["ic_mean"]
    ic_inv, n_all, n_inv = daily_ic_subset(Pb, panel, ~panel.LU)
    lu = panel.LU & panel.valid_ic
    hs = holding_stats(r["_S"])
    S = top_sets(Pb, panel)
    d = decompose(W, panel, S, [LAYER], [NBIN], "x", tag)
    return dict(tag=tag, ic_official=ic_off, ic_investable=mean(ic_inv),
                excess=r["annual_excess"], turnover=r["mean_turnover"],
                final=r["final_score"],
                alpha=float(d["alpha"].mean()) * ANN,
                profile=float(d["profile"].mean()) * ANN,
                hold_days=hs["avg_hold_days"],
                share_investable=float(np.nanmean(n_inv / np.maximum(n_all, 1))),
                lu_count=float(np.nanmean((lu).sum(axis=1))),
                lu_ret=float(np.nanmean(np.where(lu, panel.Y, np.nan))),
                mkt_ret=float(np.nanmean(np.where(panel.valid_ic, panel.Y, np.nan))),
                inv_ret=float(np.nanmean(np.where(panel.valid_ic & ~panel.LU, panel.Y, np.nan))),
                final_max=0.4 * ic_off + 0.3 * r["annual_excess"]
                + 0.3 * (1 - r["mean_turnover"]))


def main() -> int:
    t0 = time.time()
    dw = pd.read_csv(REPO / "experiments/LH019_deploy_weights.csv")
    row = dw[np.isclose(dw.lam, 0.5)].iloc[0]
    weights = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in str(row["w"]).split(";")}
    log(f"部署权重 {weights}")

    fdf = load_factors(20210101, 20241231)
    rows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        df = load_fold(fold)
        pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
        pred["ts_code"] = pred["ts_code"].astype(str)
        pred["trade_date"] = pred["trade_date"].astype("int64")
        extra = [m for m in weights if m in pred.columns and m not in df.columns]
        if extra:
            df = df.drop(columns=[c for c in extra if c in df.columns]).merge(
                pred[["ts_code", "trade_date"] + extra], on=["ts_code", "trade_date"], how="left")
            df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
        panel = Panel(df)
        W = wide_of(fdf, panel)
        raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in weights}
        pct = {m: pct_rank_2d(P) for m, P in raw.items()}
        mix = np.nansum(np.stack([w * pct[m] for m, w in weights.items()]), axis=0)
        P0 = L25.two_stage(pct_rank_2d(mix), pct["H01"], K_STAGE1)
        Pb, _ = fast_band(P0, panel.LU, KQ)

        lu = panel.LU.copy()
        variants = {"V_official": Pb}
        lo = np.nanmin(Pb) - 1.0
        hi = np.nanmax(Pb) + 1.0
        for name, val in [("LU_to_bottom", lo), ("LU_to_top", hi)]:
            Pv = Pb.copy()
            Pv[lu] = val
            variants[name] = Pv
        # 诚实排序：涨停股既然明日收益已知偏低/偏高，按其**真实期望**位置插回
        # 用「涨停股次日收益在非涨停分布中的分位」作为其诚实的横截面位置
        Pv = Pb.copy()
        for t in range(panel.T):
            m = lu[t]
            if not m.any():
                continue
            Pv[t, m] = np.nanmedian(np.where(panel.valid_ic[t] & ~lu[t], Pb[t], np.nan))
        variants["LU_to_median"] = Pv

        for tag, Px in variants.items():
            r = evaluate(tag, Px, panel, W)
            r["fold"] = fold
            rows.append(r)
            log(f"  {fold:12s} {tag:14s} IC_off {r['ic_official']:.5f} "
                f"IC_inv {r['ic_investable']:.5f} E {r['excess']:.5f} T {r['turnover']:.5f} "
                f"F {r['final']:.5f}")

    M = pd.DataFrame(rows)
    M.to_csv(OUT / "lu_rank_grid.csv", index=False)

    pd.set_option("display.width", 400)
    cols = ["ic_official", "ic_investable", "excess", "turnover", "final", "alpha", "profile"]
    print("\n=== 四折均值 ===")
    g = M.groupby("tag")[cols + ["hold_days", "share_investable", "lu_count",
                                 "lu_ret", "inv_ret", "mkt_ret"]].mean()
    print(g.round(5).to_string())

    print("\n=== 配对 vs V_official（Δ / t / 逐折符号）===")
    b = M[M.tag == "V_official"].set_index("fold")
    for tag in [t for t in M.tag.unique() if t != "V_official"]:
        o = M[M.tag == tag].set_index("fold").reindex(FOLDS)
        for k in ["ic_official", "excess", "turnover", "final"]:
            d = (o[k] - b[k]).to_numpy()
            sd = float(np.std(d, ddof=1))
            print(f"  {tag:14s} {k:13s} Δ={d.mean():+.6f} "
                  f"t={d.mean()/(sd/2) if sd>0 else float('nan'):+6.2f} "
                  f"{''.join('+' if v>0 else '-' for v in d)}")

    print(f"\n用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

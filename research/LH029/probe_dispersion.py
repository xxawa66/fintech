"""LH029-b：α 到底由什么决定？—— 检验「单位预测技能在高离散层的收益回报更大」这条机理。

LH029 主探针发现：把 TS_K40 的打分向**高波**方向小幅倾斜（phi=-0.1），
在不额外付出换手的前提下把 α 从 0.1889 抬到 0.2506（Δ=+0.0618，t=4.85，四折 `++++`），
代价是 IC 掉 0.0054、profile 掉 0.0564。

这个 α 涨幅是两种东西之一：
  (a) 模型在高波层里**真的选得更好**（更高的层内 IC）⇒ 能力，应当稳定；
  (b) 高波层的**横截面离散度更大**，同样的排序 skill 换算成收益单位时被放大 ⇒ 算术，必然稳定。

DGTW 的 α 定义是 Σ_bins w_b · (层内选中收益 − 层内均值)，量纲是收益率，
所以 (b) 的存在几乎是必然的。本脚本把它量化：逐层算
  · 层内 Spearman IC（单位 skill）
  · 层内收益横截面标准差 σ_b（换算倍数）
  · 层内 top20% 的收益优势（可实现 DFT amortized 上限）
若 IC_b 近似跨层平坦、而 σ_b 单调放大 ⇒ α 的绝大部分可由 (b) 解释
⇒ 「向高波倾斜必然买到 α」，与 regime 无关，是结构性而非运气。

只用 2021-2024 四个 CV 折。
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
_OUT = _REPO / "experiments" / "LH029"
_OUT.mkdir(parents=True, exist_ok=True)
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = _REPO
CACHE = REPO / "outputs" / "long_horizon"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/LH023"))
from _probe_dgtw import stratify  # noqa: E402
from _probe_e_attr import load_factors, wide_of  # noqa: E402
from _probe_root_cause import Panel, load_fold, to_wide  # noqa: E402

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
NBIN = 10
ANN = 252
LAYER = "vol60"
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}


def main() -> int:
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
        W = wide_of(fdf, panel)
        vol = W[LAYER]
        mixw = np.nansum(np.stack([w * to_wide(panel, df[m].to_numpy(float))
                                   for m, w in WEIGHTS.items()]), axis=0)
        for src, Pmat in [("H01", to_wide(panel, df["H01"].to_numpy(float))),
                          ("mix", mixw)]:
            for t in range(panel.T):
                ok = panel.valid_top[t]
                if ok.sum() < 100:
                    continue
                b = stratify(vol[t], NBIN)
                vb = ok & np.isfinite(vol[t]) & (b >= 0)
                if vb.sum() < 200:
                    continue
                y = panel.Y[t]
                p = Pmat[t]
                mkt = float(np.mean(y[ok]))
                for k in range(NBIN):
                    mb = vb & (b == k)
                    pm = mb & np.isfinite(p) & np.isfinite(y)
                    if pm.sum() < 30:
                        continue
                    ic = spearmanr(p[pm], y[pm]).statistic
                    sd = float(np.nanstd(y[mb]))
                    # 层内 top20% 相对层内均值的日超额
                    yy = y[pm]
                    pp = p[pm]
                    thr = np.nanquantile(pp, 0.8)
                    adv = float(np.mean(yy[pp >= thr]) - np.mean(yy))
                    rows.append(dict(fold=fold, src=src, layer=k, ic=ic, sd=sd, adv=adv,
                                     mkt_rel=float(np.mean(y[mb]) - mkt)))
    Dt = pd.DataFrame(rows)
    Dt.to_csv(_OUT / "within_layer_skill.csv", index=False)

    for src in ["H01", "mix"]:
        S = Dt[Dt.src == src]
        g = S.groupby("layer")[["ic", "sd", "adv", "mkt_rel"]].mean()
        g["sd×252"] = g["sd"] * np.sqrt(ANN)
        g["adv年化"] = g["adv"] * ANN
        g["mkt_rel年化"] = g["mkt_rel"] * ANN
        g["adv/ic (每单位skill的日收益)"] = g["adv"] / g["ic"]
        print(f"\n=== {src}：逐层 skill / 离散度 / 可实现优势（四折日频均值）===")
        print(g[["ic", "sd×252", "adv年化", "mkt_rel年化", "adv/ic (每单位skill的日收益)"]].round(4).to_string())
        # 相关性
        print(f"  corr(层号, sd)      = {np.corrcoef(g.index, g['sd'])[0,1]:+.3f}")
        print(f"  corr(层号, IC)      = {np.corrcoef(g.index, g['ic'])[0,1]:+.3f}")
        print(f"  corr(层号, adv)     = {np.corrcoef(g.index, g['adv'])[0,1]:+.3f}")
        print(f"  L9/L0: sd {g['sd'].iloc[-1]/g['sd'].iloc[0]:.2f}倍 | "
              f"IC {g['ic'].iloc[-1]/g['ic'].iloc[0]:+.2f}倍 | "
              f"adv {g['adv'].iloc[-1]/g['adv'].iloc[0]:+.2f}倍")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

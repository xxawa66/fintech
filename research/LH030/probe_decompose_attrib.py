"""LH030 —— 为什么 φ=−0.1 把 α 抬上去的同时把 profile 砸下来？

把 DGTW 分解写成可归因的形式。官方 `decompose` 的定义是（逐日）：

    mkt   = mean(y[valid_top])
    lev_k = mean(y[层k内valid]) − mkt           层 k 的「层均 − 市场」
    adv_k = mean(y[层k内被选中]) − mean(y[层k内valid])   层 k 内可实现优势
    wb_k  = 选中集合里层 k 的占比
    wv_k  = 有效全集里层 k 的占比

    profile = Σ_k (wb_k − wv_k) · lev_k
    alpha   = Σ_k  wb_k        · adv_k
    excess  = alpha + profile  （恒等式，非拟合）

关键结构事实：**lev_k 与 φ 完全无关**——层划分只由当天 vol60 决定，层均值只由当天收益决定，
φ 不改变任何一个股票的收益，只改变「谁被选中」。于是：

    Δprofile = Σ_k Δwb_k · lev_k                       ⇒ 100% 构成效应，无残差
    Δalpha   = Σ_k Δwb_k · adv_k  +  Σ_k wb'_k · Δadv_k ⇒ 构成效应 + 选择效应

本脚本把两项都量出来，回答「profile 下降是不是 α 上升的代价」——答案是：
两者由**同一个** Δwb 驱动，乘的是两个不同的 per-layer 价格，
而这两个价格在高波层的符号天然相反（lev 巨负、adv 巨正）。
"""
from __future__ import annotations

import os as _os
import sys
import time
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

OUT = _REPO / "experiments" / "LH030"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "research/LH023"))
sys.path.insert(0, str(_REPO / "research/LH029"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _probe_dgtw import pct_rank_2d, stratify  # noqa: E402
from _probe_e_attr import load_factors, top_sets, wide_of  # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide  # noqa: E402

import probe_regime_tilt as R  # noqa: E402

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
PHIS = [0.0, -0.1, -0.3, 0.1]
ANN = 252
NBIN = 10
LAYER = "vol60"
KQ = R.KQ
K_STAGE1 = R.K_STAGE1
WEIGHTS = R.WEIGHTS
CACHE = R.CACHE


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def layer_attrib(panel, vol, S):
    """逐日 × 逐层的 (wb, wv, lev, adv)——把 decompose 摊开，便于归因。"""
    rows = []
    for t in range(panel.T):
        ok = panel.valid_top[t]
        if ok.sum() < 100:
            continue
        y = panel.Y[t]
        if not S[t].any():
            continue
        sel = ok & S[t].astype(bool)
        if sel.sum() < 20:
            continue
        mkt = float(np.mean(y[ok]))
        b = stratify(vol[t], NBIN)
        fin = np.isfinite(vol[t])
        valid = ok & fin
        sel2 = sel & fin
        if valid.sum() < 200 or sel2.sum() < 20:
            continue
        codes = np.unique(b[valid])
        codes = codes[codes >= 0]
        nB = int(sel2.sum())
        nV = int(valid.sum())
        d = int(panel.dates[t])
        for c in codes:
            vb = valid & (b == c)
            sb = sel2 & (b == c)
            if not sb.any():
                continue
            mb = float(np.mean(y[vb]))
            rows.append((d, int(c), sb.sum() / nB, vb.sum() / nV,
                         mb - mkt, float(np.mean(y[sb])) - mb))
    return pd.DataFrame(rows, columns=["trade_date", "layer", "wb", "wv", "lev", "adv"])


def fold_arrays(fold, fdf):
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
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in WEIGHTS}
    pct = {m: pct_rank_2d(P) for m, P in raw.items()}
    mix = np.nansum(np.stack([w * pct[m] for m, w in WEIGHTS.items()]), axis=0)
    P0 = R.two_stage(pct_rank_2d(mix), pct["H01"], K_STAGE1)
    plv = pct_rank_2d(np.where(np.isfinite(vol), -vol, np.nan))
    plv = np.where(np.isfinite(plv), plv, 0.5)
    return panel, vol, P0, plv


def pair_attr(A, B):
    """A=基准 DataFrame，B=对照 DataFrame。返回逐日 Δ 的字典。"""
    m = A.merge(B, on=["trade_date", "layer"], suffixes=("_b", "_o"))
    m["dw"] = m.wb_o - m.wb_b
    m["prof_b"] = (m.wb_b - m.wv_b) * m.lev_b
    m["prof_o"] = (m.wb_o - m.wv_o) * m.lev_b          # lev 与 φ 无关：共用 lev_b
    m["alpha_b"] = m.wb_b * m.adv_b
    m["alpha_o"] = m.wb_o * m.adv_o
    m["comp_a"] = m.dw * m.adv_b                       # 构成效应
    m["sel_a"] = m.wb_o * (m.adv_o - m.adv_b)          # 选择效应
    m["comp_p"] = m.dw * m.lev_b
    g = m.groupby("trade_date")[
        ["prof_b", "prof_o", "alpha_b", "alpha_o", "comp_a", "sel_a", "comp_p"]].sum()
    return g.mean() * ANN, len(g)


def main() -> int:
    t0 = time.time()
    WEIGHTS.update(R.load_weights())
    log(f"部署权重 {WEIGHTS}")
    fdf = load_factors(20210101, 20241231)

    rows_price, rows_attr, rows_w = [], [], []
    for fold in FOLDS:
        log(f"--- {fold} ---")
        panel, vol, P0, plv = fold_arrays(fold, fdf)
        tab = {}
        for phi in PHIS:
            Pt, _ = fast_band(R.tilt(P0, plv, np.full(panel.T, phi)), panel.LU, KQ)
            S = top_sets(Pt, panel)
            tab[phi] = layer_attrib(panel, vol, S)
            r = score(Pt, panel)
            log(f"  phi={phi:+.2f}  IC {r['ic_mean']:.5f} E {r['annual_excess']:.5f} "
                f"alpha {tab[phi].pipe(lambda d: (d.wb*d.adv).groupby(d.trade_date).sum().mean()*ANN):.5f} "
                f"profile {tab[phi].pipe(lambda d: ((d.wb-d.wv)*d.lev).groupby(d.trade_date).sum().mean()*ANN):+.5f}")

        # per-layer 价格表（lev / adv 与 φ 无关，只用基准算）
        base = tab[0.0]
        g = base.groupby("layer").agg(wv=("wv", "mean"), lev=("lev", "mean"))
        for phi, tabp in tab.items():
            gw = tabp.groupby("layer").agg(w=("wb", "mean"), adv=("adv", "mean"))
            for k in range(NBIN):
                rows_price.append(dict(fold=fold, phi=phi, layer=k,
                                       w=float(gw.w.get(k, np.nan)),
                                       wv=float(gw.wv.get(k, np.nan)) if "wv" in gw else float(g.wv.get(k, np.nan)),
                                       lev_ann=float(g.lev.get(k, np.nan)) * ANN,
                                       adv_ann=float(gw.adv.get(k, np.nan)) * ANN))
                rows_w.append(dict(fold=fold, phi=phi, layer=k,
                                   w=float(gw.w.get(k, np.nan)),
                                   adv_ann=float(gw.adv.get(k, np.nan)) * ANN))

        for phi in PHIS:
            if phi == 0.0:
                continue
            res, n = pair_attr(base, tab[phi])
            rows_attr.append(dict(fold=fold, phi=phi, n_days=n,
                                  alpha_base=res.alpha_b, alpha_phi=res.alpha_o,
                                  dAlpha=res.alpha_o - res.alpha_b,
                                  dAlpha_comp=res.comp_a, dAlpha_sel=res.sel_a,
                                  prof_base=res.prof_b, prof_phi=res.prof_o,
                                  dProf=res.prof_o - res.prof_b, dProf_comp=res.comp_p,
                                  resid_check=res.prof_o - res.prof_b - res.comp_p))

    P = pd.DataFrame(rows_price)
    A = pd.DataFrame(rows_attr)
    P.to_csv(OUT / "layer_price.csv", index=False)
    A.to_csv(OUT / "attribution.csv", index=False)

    print("\n" + "=" * 92)
    print("per-layer 价格表（四折均值；lev/adv 为日均值 ×252）")
    print("=" * 92)
    g = P.groupby(["phi", "layer"])[["w", "wv", "lev_ann", "adv_ann"]].mean()
    t = g.loc[0.0].copy()
    t.columns = ["w(φ=0)", "wv(=市场)", "lev 层均−市场", "adv 层内优势"]
    for phi in PHIS[1:]:
        t[f"w(φ={phi:+g})"] = g.loc[phi, "w"]
    print(t.round(4).to_string())

    print("\n" + "=" * 92)
    print("归因（四折配对；单位为年化）")
    print("=" * 92)
    print(A.groupby("phi")[["alpha_base", "alpha_phi", "dAlpha", "dAlpha_comp", "dAlpha_sel",
                            "prof_base", "prof_phi", "dProf", "dProf_comp",
                            "resid_check"]].mean().round(5).to_string())
    print("\n( resid_check 应恒等于 0：Δprofile 100% 由 layer 占比变化解释，无任何残差 )")

    print("\n" + "=" * 92)
    print("构成效应 vs 选择效应（Δα 拆解，四折均值）")
    print("=" * 92)
    for phi in PHIS[1:]:
        s = A[A.phi == phi]
        d = s.dAlpha.mean()
        print(f"  φ={phi:+g}:  Δα {d:+.5f} = 构成 {s.dAlpha_comp.mean():+.5f} "
              f"({100*s.dAlpha_comp.mean()/d:5.1f}%) + 选择 {s.dAlpha_sel.mean():+.5f} "
              f"({100*s.dAlpha_sel.mean()/d:5.1f}%)")

    log(f"done {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

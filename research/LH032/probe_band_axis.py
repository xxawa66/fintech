"""LH032 第八步：把「band 段信号」参数化，与倾斜做二维网格。

背景（为什么必须做这一步）
  LH031 曾断言「band 段不能动」，依据是 `experiments/LH031/free_ic_grid.csv`，
  而那份表当时只有 confirm2024 一折。补跑四折后结论反转：band 段内部次序换
  H05_F2T2 得到 ΔIC +0.00860 / ΔE +0.02371 / ΔT −0.00290 / Δfinal +0.01142
  （四折同号，最小折 +0.00573）。
  ⇒ 「M8 最优」是在一个被错误截断的搜索空间里选出来的，必须重做。

本步的参数化点
  两段编码 `two_stage(pr_mix, s_band, K)`：
    候选池 = mix 分位 top K%；池内按 **s_band** 重排，值域 [1, 2]；池外 = pr_mix ∈ [0,1]。
  第二参数 s_band 就是 band 段信号，部署点固定为 pct(H01)。本脚本把它放开成
  六个候选信号，与倾斜 (φ_amt, φ_vol) 做二维网格。

注意（本步只看两段框架）
  三段框架（铺 drop 段明线）会引入 top/rest 分段，倾斜的作用机制随之改变，
  不可与两段框架直接混比。因此本步只在两段框架内扫描；与 LH031 明线的叠加
  留到下一步单独验证可加性。

自检（必须逐位通过，否则结果不可信）
  1. (H01, 0, 0)          == DEPLOY   IC 0.08630 E 0.20828 T 0.01838 F 0.39149
  2. (H01, -0.100, +0.040) == M2      Δfinal +0.02220
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "long_horizon"
sys.path.insert(0, str(CACHE))
sys.path.insert(0, str(ROOT / "research" / "LH023"))
sys.path.insert(0, str(ROOT))

import _probe_factors as _pf                                        # noqa: E402
if "mom20" not in _pf.FACTORS:
    _pf.FACTORS.append("mom20")
from _probe_e_attr import load_factors, top_sets, wide_of           # noqa: E402
from _probe_root_cause import (Panel, fast_band, holding_stats,      # noqa: E402
                               load_fold, score, to_wide)
from _probe_dgtw import decompose, pct_rank_2d, stratify             # noqa: E402

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KQ = 0.0022778298112255263
ANN = 252
NB = 10
K = 40.0
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
MODELS = list(WEIGHTS)

# band 段信号候选（含部署点 H01）
GRID_BAND = ["H01", "H05_F1T2", "H05_F2T2", "mix", "rawmix", "avg3"]
# 倾斜候选（含零倾斜）
GRID_TILT = [(0.0, 0.0), (-0.075, 0.040), (-0.100, 0.040), (-0.125, 0.075)]
# 额外的模型列（raw_predictions 里还有 H10/H20/H30 系列，默认不参与，供扩展网格用）
EXTRA = []
# 派生信号：由若干已有信号等权平均而成
DERIVED = {}

OUT = ROOT / "experiments" / "LH032" / "band_axis_grid.csv"
# 池外段（明线）信号名：None = pr_mix（原样）；"mix" = 归一化对照；"rawmix" = M1 明线
OUTER = None


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pct_rank_2d(P):
    return _pct(P)


def _pct(P):
    out = np.full(P.shape, np.nan)
    for t in range(P.shape[0]):
        x = P[t]
        m = np.isfinite(x)
        n = int(m.sum())
        if n < 2:
            continue
        o = np.argsort(x[m], kind="mergesort")
        r = np.empty(n)
        r[o] = np.arange(n, dtype=float)
        s = x[m][o]
        i = 0
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            if j - i > 1:
                r[o[i:j]] = (i + j - 1) / 2.0
            i = j
        out[t, m] = r / (n - 1)
    return out


def two_stage(pr_mix, s_band, K_, outer=None):
    """候选池 = pr_mix top K%；池内按 s_band 重排（值域 [1,2]）；
    池外 = pr_mix（outer=None）或指定信号归一化到 [0,1)（明线自由度）。"""
    out = np.full(pr_mix.shape, np.nan)
    _so = None if outer is None else globals().get("_S_CUR", {}).get(outer)
    for t in range(pr_mix.shape[0]):
        a, b = pr_mix[t], s_band[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K_ / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        bb = np.where(cand & np.isfinite(b), b, np.nan)
        v = bb[np.isfinite(bb)]
        lo, hi = v.min(), v.max()
        within = np.where(np.isfinite(bb), (bb - lo) / (hi - lo), 0.5) if hi > lo else 0.5
        if _so is None:
            outv = a
        else:
            c = _so[t]
            cc = np.where(m & ~cand & np.isfinite(c), c, np.nan)
            vv = cc[np.isfinite(cc)]
            if vv.size == 0:
                outv = np.full(a.shape, 0.5)
            else:
                lo2, hi2 = vv.min(), vv.max()
                outv = np.where(np.isfinite(cc), (cc - lo2) / (hi2 - lo2), 0.5) \
                    if hi2 > lo2 else np.full(a.shape, 0.5)
        out[t] = np.where(cand, 1.0 + within, np.where(m, outv, np.nan))
    return out


def tilt2(P0, pr_a, pr_b, pa, pb):
    """池内加位移；**因子 NaN 必须取中性**，否则整只股票被打掉、可选集崩溃。"""
    D = np.where(np.isfinite(pr_a), (pr_a - 0.5) * pa, 0.0) \
        + np.where(np.isfinite(pr_b), (pr_b - 0.5) * pb, 0.0)
    P = P0 + D
    iscand = np.isfinite(P0) & (P0 >= 1.0)
    P = np.where(iscand, P, P - 1.0)
    P[~np.isfinite(P0)] = np.nan
    return P


def stratify_of(W, key, panel):
    """按因子分十层，返回 (T,N) 层号矩阵（-1 表示无值）。"""
    return np.vstack([stratify(W[key][t], NB) for t in range(panel.T)])


def fold_block(fold: str, fdf: pd.DataFrame) -> pd.DataFrame:
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    allm = MODELS + [c for c in EXTRA if c not in MODELS]
    df = df.drop(columns=[c for c in allm if c in df.columns]).merge(
        pred[["ts_code", "trade_date"] + allm], on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in allm}
    pct = {m: _pct(raw[m]) for m in allm}
    mix = np.nansum(np.stack([w * pct[m] for m, w in WEIGHTS.items()]), axis=0)
    pr_mix = _pct(mix)

    # 支持集对齐：必须取**交集**（并集会改变可选集 ⇒ 漏进暗线）
    def safe(x):
        return np.where(np.isfinite(pr_mix), np.where(np.isfinite(x), x, pr_mix), np.nan)

    S = {m: safe(pct[m]) for m in allm}
    S["mix"] = pr_mix
    S["avg3"] = np.mean(np.stack([S[m] for m in MODELS]), axis=0)
    mix_raw = np.nansum(np.stack([w * raw[m] for m, w in WEIGHTS.items()]), axis=0)
    S["rawmix"] = safe(_pct(np.where(np.isfinite(mix_raw), mix_raw, np.nan)))
    for name, parts in DERIVED.items():
        parts = [p for p in parts if p in S]
        if parts:
            S[name] = np.mean(np.stack([S[p] for p in parts]), axis=0)

    global _S_CUR
    _S_CUR = S

    PA = _pct(W["log_amt"])
    PV = _pct(W["vol60"])
    Lbin_amt = stratify_of(W, "log_amt", panel)
    Lbin_vol = stratify_of(W, "vol60", panel)

    def composition(Sp, LB):
        out = {}
        for k in range(NB):
            num = den = 0.0
            for t in range(Sp.shape[0]):
                idx = np.flatnonzero(Sp[t])
                if idx.size == 0:
                    continue
                m = panel.valid_turn[t][idx] & (LB[t][idx] >= 0)
                if m.sum() == 0:
                    continue
                num += (LB[t][idx][m] == k).sum()
                den += m.sum()
            out[k] = num / den if den > 0 else np.nan
        return out

    def evaluate(tag: str, P: np.ndarray) -> dict:
        Pb, _ = fast_band(P, panel.LU, KQ)
        r = score(Pb, panel)
        Sp = top_sets(Pb, panel)
        d_vol = decompose(W, panel, Sp, ["vol60"], [NB], fold, tag)
        d_amt = decompose(W, panel, Sp, ["log_amt"], [NB], fold, tag)
        hs = holding_stats(r["_S"])
        c_amt = composition(Sp, Lbin_amt)
        c_vol = composition(Sp, Lbin_vol)
        T = r["mean_turnover"]
        return dict(fold=fold, tag=tag,
                    band=tag.split("|")[0], phi_amt=float(tag.split("|")[1]),
                    phi_vol=float(tag.split("|")[2]),
                    ic=r["ic_mean"], excess=r["annual_excess"], turnover=T,
                    final=r["final_score"],
                    alpha_vol=float(d_vol["alpha"].mean()) * ANN,
                    prof_vol=float(d_vol["profile"].mean()) * ANN,
                    alpha_amt=float(d_amt["alpha"].mean()) * ANN,
                    prof_amt=float(d_amt["profile"].mean()) * ANN,
                    hold_days=hs["avg_hold_days"], list_size=hs["list_size"],
                    **{f"E_L{k}": c_amt[k] for k in range(NB)},
                    **{f"V_L{k}": c_vol[k] for k in range(NB)})

    rows = []
    for sb in GRID_BAND:
        P0 = two_stage(pr_mix, S[sb], K, OUTER)
        for pa, pv in GRID_TILT:
            tag = f"{sb}|{pa:+.3f}|{pv:+.3f}"
            P = P0 if (pa == 0.0 and pv == 0.0) else tilt2(P0, PA, PV, pa, pv)
            rows.append(evaluate(tag, P))
        log(f"  {fold} band={sb} 完成")
    return pd.DataFrame(rows)


def main() -> int:
    log("读因子面板")
    fdf = load_factors(20210101, 20241231)
    parts = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        parts.append(fold_block(fold, fdf))
    M = pd.concat(parts, ignore_index=True)
    M.to_csv(OUT, index=False)
    log(f"写出 {OUT}  行数 {len(M)}")

    F = FOLDS
    b = M[(M.band == "H01") & (M.phi_amt == 0.0) & (M.phi_vol == 0.0)]
    b = b.set_index("fold").reindex(F)
    log("")
    log("=== 自检 1：band=H01, 无倾斜 应逐位复现 DEPLOY ===")
    log("    归档: IC 0.08630  E 0.20828  T 0.01838  final 0.39149")
    log(f"    实测: IC {b['ic'].mean():.5f}  E {b['excess'].mean():.5f}  "
        f"T {b['turnover'].mean():.5f}  final {b['final'].mean():.5f}")
    m2 = M[(M.band == "H01") & (np.isclose(M.phi_amt, -0.100)) & (np.isclose(M.phi_vol, 0.040))]
    m2 = m2.set_index("fold").reindex(F)
    dF = m2["final"] - b["final"]
    dE = m2["excess"] - b["excess"]
    log("")
    log("=== 自检 2：band=H01, (-0.100,+0.040) 应复现 M2 ===")
    log(f"    归档 Δfinal +0.02220 / ΔE +0.06940")
    log(f"    实测 Δfinal {dF.mean():+.5f} / ΔE {dE.mean():+.5f}   逐折 "
        + " ".join(f"{v:+.5f}" for v in dF))

    log("")
    log("=== 主表：四折均值 ===")
    rows = []
    for (sb, pa, pv), g in M.groupby(["band", "phi_amt", "phi_vol"]):
        g = g.set_index("fold").reindex(F)
        dI = g["ic"] - b["ic"]
        dE = g["excess"] - b["excess"]
        dT = g["turnover"] - b["turnover"]
        dF = g["final"] - b["final"]
        dA = g["alpha_vol"] - b["alpha_vol"]
        dP = g["prof_amt"] - b["prof_amt"]
        rows.append(dict(band=sb, a=pa, v=pv,
                         ic=g["ic"].mean(), excess=g["excess"].mean(),
                         turnover=g["turnover"].mean(), final=g["final"].mean(),
                         alpha_vol=g["alpha_vol"].mean(), prof_amt=g["prof_amt"].mean(),
                         hold=g["hold_days"].mean(), size=g["list_size"].mean(),
                         dIC=dI.mean(), dE=dE.mean(), dE_min=dE.min(),
                         dT=dT.mean(), dalpha=dA.mean(), dprof=dP.mean(),
                         dfinal=dF.mean(), df_min=dF.min(),
                         sign="".join("+" if x > 0 else "-" for x in dF),
                         t=dF.mean() / (dF.std() / 2) if dF.std() > 0 else np.inf))
    R = pd.DataFrame(rows).sort_values("dfinal", ascending=False)
    pd.set_option("display.width", 320)
    print()
    print(R.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

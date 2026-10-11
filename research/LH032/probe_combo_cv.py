"""LH032 第七步：双因子倾斜（暗线） × LH031 明线分离（drop 段）能否叠加。

背景
  暗线（组合/收益/换手/α/profile）由「候选池内 top f 的股票集合」决定；
  明线（被官方 IC 度量的提交向量排序）中只有 drop 段（池内落选的 30%）可自由重排。

要回答的问题
  倾斜改变了 top 段成员（暗线），因此 drop 段成员也随之改变。
  叠加后：暗线是否与「单独倾斜」逐位相同？ΔIC 是否等于两者之和？

两种构造（必须区分）
  COMBO_A：先倾斜（定成员），再把落选段按 rawmix 重排  ← 正确顺序
  COMBO_B：先按 rawmix 重排，再整体加倾斜位移      ← 测试期脚本当前用的顺序
  两者在数学上不等价：B 的 top 段被抬起固定值，倾斜位移 |D|≤0.07 不足以跨越
  top/rest 之间 0.25 的间隔 ⇒ 成员可能不动。本脚本实测判定。
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
from _probe_dgtw import decompose, pct_rank_2d                      # noqa: E402

KQ = 0.0022778298112255263
ANN = 252
NB = 10
FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
MODELS = list(WEIGHTS)
K = 40.0
F = 0.25
PA, PV = -0.100, 0.040


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _norm(v):
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.where(np.isfinite(v), 0.5, 0.5)
    return np.where(np.isfinite(v), (v - lo) / (hi - lo), 0.5)


def base_P(pr_mix, pr_h1, K_):
    out = np.full(pr_mix.shape, np.nan)
    for t in range(pr_mix.shape[0]):
        a, b = pr_mix[t], pr_h1[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K_ / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        out[t] = np.where(cand, 1.0 + _norm(np.where(cand, b, np.nan)),
                          np.where(m, a, np.nan))
    return out


def build_rest(pr_mix, pr_h1, sig_rest, K_, f):
    """top 段由 H01 序决定，其余（drop 段）按 sig_rest 排。"""
    out = np.full(pr_mix.shape, np.nan)
    for t in range(pr_mix.shape[0]):
        a, b = pr_mix[t], pr_h1[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K_ / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        wh = _norm(np.where(cand, b, np.nan))
        top = cand & np.isfinite(wh) & (wh >= 1.0 - f)
        rest = cand & (~top)
        v = np.where(top, 2.0 + _norm(np.where(top, sig_rest[t] * 0 + wh, np.nan)) * f, np.nan)
        v = np.where(rest, 1.0 + _norm(np.where(rest, sig_rest[t], np.nan)) * (1.0 - f), v)
        out[t] = np.where(cand, v, np.where(m, a, np.nan))
    return out


def tilt2(P0, pa_pr, pv_pr, pa, pv):
    D = np.where(np.isfinite(pa_pr), (pa_pr - 0.5) * pa, 0.0) \
        + np.where(np.isfinite(pv_pr), (pv_pr - 0.5) * pv, 0.0)
    P = P0 + D
    iscand = np.isfinite(P0) & (P0 >= 1.0)
    P = np.where(iscand, P, P - 1.0)
    P[~np.isfinite(P0)] = np.nan
    return P


def combo_a(pr_mix, pr_h1, sig_rest, P_tilt, K_, f):
    """先倾斜（成员由 P_tilt 定），再把落选段按 sig_rest 重排。

    top 段的内部次序保持与 P_tilt 一致（决定涨停股滑出后的补位顺序）。
    """
    out = np.full(pr_mix.shape, np.nan)
    for t in range(pr_mix.shape[0]):
        a, b = pr_mix[t], pr_h1[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K_ / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        s = np.where(cand, P_tilt[t], np.nan)          # 倾斜后的池内分数
        thr_s = np.nanquantile(s[np.isfinite(s)], 1.0 - f)
        top = cand & np.isfinite(s) & (s >= thr_s)
        rest = cand & (~top)
        v = np.where(top, 2.0 + _norm(np.where(top, s, np.nan)) * f, np.nan)
        v = np.where(rest, 1.0 + _norm(np.where(rest, sig_rest[t], np.nan)) * (1.0 - f), v)
        out[t] = np.where(cand, v, np.where(m, a, np.nan))
    return out


def jaccard_vs(A, B, panel):
    """逐日 top set 与参考集 A 的平均 Jaccard 距离（0=完全一致）。

    注意：top_sets 返回的是 (T, N) **布尔矩阵**，不能用 `A[t][A[t] >= 0]` 取成员
    （bool 上该筛选恒为真，会拿到整行 ⇒ 距离恒为 0）。必须 flatnonzero。
    """
    tot, n = 0.0, 0
    for t in range(panel.T):
        s1 = set(np.flatnonzero(A[t]).tolist())
        s2 = set(np.flatnonzero(B[t]).tolist())
        u = len(s1 | s2)
        if u == 0:
            continue
        tot += 1.0 - len(s1 & s2) / u
        n += 1
    return tot / max(n, 1)


def fold_block(fold: str, fdf: pd.DataFrame) -> pd.DataFrame:
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    df = df.drop(columns=[c for c in MODELS if c in df.columns]).merge(
        pred[["ts_code", "trade_date"] + MODELS], on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in MODELS}
    pct = {m: pct_rank_2d(raw[m]) for m in MODELS}
    mix = np.nansum(np.stack([w * pct[m] for m, w in WEIGHTS.items()]), axis=0)
    pr_mix = pct_rank_2d(mix)
    P0 = base_P(pr_mix, pct["H01"], K)

    PA_pr = pct_rank_2d(W["log_amt"])
    PV_pr = pct_rank_2d(W["vol60"])

    # LH031 明线信号：原始值级融合的百分位，支持集取**交集**（并集会漏进暗线）
    mix_raw = np.nansum(np.stack([w * raw[m] for m, w in WEIGHTS.items()]), axis=0)
    S_rawmix = np.where(np.isfinite(pr_mix),
                        np.where(np.isfinite(mix_raw),
                                 pct_rank_2d(np.where(np.isfinite(mix_raw), mix_raw, np.nan)),
                                 pr_mix), np.nan)

    P_tilt = tilt2(P0, PA_pr, PV_pr, PA, PV)
    CAND = {
        "DEPLOY": P0,
        "TILT": P_tilt,
        "RAWMIX": build_rest(pr_mix, pct["H01"], S_rawmix, K, F),
        "COMBO_A": combo_a(pr_mix, pct["H01"], S_rawmix, P_tilt, K, F),
        "COMBO_B": tilt2(build_rest(pr_mix, pct["H01"], S_rawmix, K, F), PA_pr, PV_pr, PA, PV),
    }

    rows = []
    Sref = None
    for tag, P in CAND.items():
        Pb, _ = fast_band(P, panel.LU, KQ)
        r = score(Pb, panel)
        S = top_sets(Pb, panel)
        if tag == "TILT":
            Sref = S
        d_vol = decompose(W, panel, S, ["vol60"], [NB], fold, tag)
        d_amt = decompose(W, panel, S, ["log_amt"], [NB], fold, tag)
        hs = holding_stats(r["_S"])
        rows.append(dict(fold=fold, tag=tag,
                         ic=r["ic_mean"], excess=r["annual_excess"],
                         turnover=r["mean_turnover"], final=r["final_score"],
                         alpha_vol=float(d_vol["alpha"].mean()) * ANN,
                         prof_vol=float(d_vol["profile"].mean()) * ANN,
                         alpha_amt=float(d_amt["alpha"].mean()) * ANN,
                         prof_amt=float(d_amt["profile"].mean()) * ANN,
                         hold_days=hs["avg_hold_days"], list_size=hs["list_size"],
                         jac_vs_tilt=np.nan))
    M = pd.DataFrame(rows)
    # 参考集一致性（相对 TILT）
    jac = {}
    for tag, P in CAND.items():
        Pb, _ = fast_band(P, panel.LU, KQ)
        jac[tag] = jaccard_vs(Sref, top_sets(Pb, panel), panel)
    M["jac_vs_tilt"] = M["tag"].map(jac)
    log(f"  {fold} done  " + "  ".join(f"{k}={v:.4f}" for k, v in jac.items()))
    return M


def main() -> int:
    log("读因子面板")
    fdf = load_factors(20210101, 20241231)
    parts = [fold_block(f, fdf) for f in FOLDS]
    M = pd.concat(parts, ignore_index=True)
    M.to_csv(ROOT / "experiments" / "LH032" / "combo_cv_grid.csv", index=False)

    FL = FOLDS
    b = M[M.tag == "DEPLOY"].set_index("fold").reindex(FL)
    print("\n" + "=" * 118)
    print(f"组合验证：倾斜 (φ_amt={PA}, φ_vol={PV}) × LH031 drop 段 rawmix，Δ 相对部署点")
    print("=" * 118)
    print(f"{'变体':10s} {'ΔIC':>9s} {'ΔE':>9s} {'Δα_vol':>9s} {'Δprof_amt':>10s} "
          f"{'ΔT':>9s} {'Δfinal':>9s} {'jac_vs_TILT':>12s} {'符号':>6s}")
    out = {}
    for tag in ["TILT", "RAWMIX", "COMBO_A", "COMBO_B"]:
        g = M[M.tag == tag].set_index("fold").reindex(FL)
        dI = g["ic"] - b["ic"]
        dE = g["excess"] - b["excess"]
        dA = g["alpha_vol"] - b["alpha_vol"]
        dP = g["prof_amt"] - b["prof_amt"]
        dT = g["turnover"] - b["turnover"]
        dF = g["final"] - b["final"]
        out[tag] = dict(dIC=dI, dE=dE, dA=dA, dP=dP, dT=dT, dF=dF,
                        maxdE=float(dE.abs().max()), maxdT=float(dT.abs().max()))
        print(f"{tag:10s} {dI.mean():+9.5f} {dE.mean():+9.5f} {dA.mean():+9.5f} "
              f"{dP.mean():+10.5f} {dT.mean():+9.5f} {dF.mean():+9.5f} "
              f"{g['jac_vs_tilt'].mean():12.5f} {''.join('+' if v > 0 else '-' for v in dF):>6s}")

    print("\n--- 暗线是否与单独倾斜逐位相同（四折最大绝对偏差）---")
    for tag in ["COMBO_A", "COMBO_B"]:
        g = M[M.tag == tag].set_index("fold").reindex(FL)
        t = M[M.tag == "TILT"].set_index("fold").reindex(FL)
        dE = (g["excess"] - t["excess"]).abs().max()
        dT = (g["turnover"] - t["turnover"]).abs().max()
        dA = (g["alpha_vol"] - t["alpha_vol"]).abs().max()
        dH = (g["hold_days"] - t["hold_days"]).abs().max()
        print(f"  {tag}: max|ΔE| {dE:.3e}  max|ΔT| {dT:.3e}  max|Δα| {dA:.3e}  max|Δhold| {dH:.3e}")

    print("\n--- ΔIC 可加性检验 ---")
    s = out["TILT"]["dIC"] + out["RAWMIX"]["dIC"]
    for tag in ["COMBO_A", "COMBO_B"]:
        d = out[tag]["dIC"] - s
        print(f"  {tag}: ΔIC {out[tag]['dIC'].mean():+.6f}  vs TILT+RAWMIX {s.mean():+.6f}  "
              f"差 {d.mean():+.6f} (max|.| {d.abs().max():.3e})")

    print("\n--- 部署点（四折均值）---")
    print(b[["ic", "excess", "turnover", "final", "alpha_vol", "prof_vol"]].mean().round(6).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

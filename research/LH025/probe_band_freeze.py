"""LH025 —— band 的「留存规则」是不是最后一个自由度？

背景
----
LH024 §2 的层内化探针给了两条实测事实：

  1. beta=1（纯层内排序）时，**单日** top10% 在 vol60 十层上基本均匀（L0 0.104 / L9 0.088）；
  2. 但同一套信号过 band 之后，名单变成 L0 0.158 / L9 0.029。

⇒ 名单的层倾斜 100% 由 **band 的冻结动力学** 生成，与单日排序无关。由此推出一个此前
从未动过的自由度：此前所有优化动的都是「谁进名单」（信号 / 权重 / 特征 / 标签），
**没人动过「名单怎么留存」**——band 的留存规则本身。

本项目 band 的留存规则是「绝对全池分位下限」：在榜股票当日分位跌破 q*=0.00228 即出局。
单日看极宽松（全市场最后 0.23%），但 250 个交易日累积的风险很大，且**这个累积风险
随个股名次波动率递增** ⇒ 规则在事实上系统性挑选「名次稳定」的股票 ⇒ 低波倾斜。
本脚本把层内化从**预测端**搬到 **band 判定端**，并同时试掉 §7.5 里登记的另一个未试
方向（换判定形式：名次恶化幅度）。

六个臂
------
  V0_base     原 fast_band（绝对全池分位下限 q*）——基线，必须复现归档 TS_K40 四折
  V1_lkeep    留存判定层内化：留仓条件改为「层内分位 >= q」，补仓排序仍用全池分位
  V2_lall     留存 + 补仓全层内化：判定与补仓都用层内分位 ⇒ 名单构成按构造被压平
  V3_quota    分层配额：每层独立跑 band，配额 = n_top/10 ⇒ 名单层构成恒等于全池
  V4_relent   相对触发（入榜基准）：出局条件改为「今日分位 < 入榜分位 − Δ」，
              取消绝对下限 ⇒ 出局风险不再随名次波动率累积
  V5_reldod   相对触发（日间）：出局条件改为「今日分位 < 昨日分位 − Δ」

判据（沿用 LH023/LH024）
------------------------
  可外推口径 = 0.4·IC + 0.3·α + 0.3·(1−T)（不含 profile）
  对比一律走「同等换手」+ 四折配对 Δ/t + 逐折符号。
  **全程只用 2021–2024 四个 CV 折，不读 2025 以后数据。**

自检
----
V0_base 的四折 IC/E/T/final/α/profile 必须与 experiments/LH020_twostage_folds.csv
的 TS_K40 行逐位一致（归档：IC 0.086298 / E 0.208277 / T 0.018379 / F 0.391489 /
α 0.188861 / profile 0.013311）。
"""
from __future__ import annotations

# --- LH025 archival shim: 按脚本位置反查仓库根，与当前工作目录无关 ---
import os as _os
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = (
    _P(_os.environ["FINTECH_ROOT"])
    if _os.environ.get("FINTECH_ROOT")
    else next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
)
_LH025_DIR = _P(_os.environ.get("LH025_DIR", _REPO / "experiments" / "LH025"))
_LH025_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------------

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = _REPO
CACHE = REPO / "outputs/long_horizon"
OUT = _LH025_DIR
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
TOP_MIN_VALID = 100
# SHIFT 必须为 0：band 的编码（在榜保留全池分位、落选下移 1−q*）本身就是官方三项指标的
# 一部分——把在榜股整体上移会改变涨停股与在榜股的相对次序，实测 IC 差 −0.0021。
SHIFT = 0.0

# 网格：核心区间是「低换手端」，因为基线 T 已贴在下限附近
Q_GRID = [0.0005, 0.001, KQ, 0.005, 0.01, 0.02, 0.04, 0.08]
REL_ENTRY_GRID = [0.6, 0.8, 0.9, 0.95, 1.0]
REL_DOD_GRID = [0.6, 0.8, 0.9, 0.95, 1.0]

ARCHIVE = {  # experiments/LH020_twostage_folds.csv 的 TS_K40 行
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


# --------------------------------------------------------------------------
# 两阶段打分（TS_K40 部署口径）
# --------------------------------------------------------------------------
def two_stage(pr_lh: np.ndarray, pr_h1: np.ndarray, K: float) -> np.ndarray:
    """候选池 = 长周期 mix top K%；池内按 H01 重排，整体落在 [1,2]；非候选落在 [0,1]。"""
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
        bb = np.where(cand & np.isfinite(b), b, np.nan)
        v = bb[cand & np.isfinite(b)]
        lo, hi = v.min(), v.max()
        if hi > lo:
            within = np.where(np.isfinite(bb), (bb - lo) / (hi - lo), 0.5)
        else:
            within = np.where(np.isfinite(bb), 0.5, 0.5)
        out[t] = np.where(cand, 1.0 + within, np.where(m, a, np.nan))
    return out


def load_weights() -> dict:
    dw = pd.read_csv(REPO / "experiments/LH019_deploy_weights.csv")
    row = dw[np.isclose(dw.lam, 0.5)].iloc[0]
    return {kv.split(":")[0]: float(kv.split(":")[1]) for kv in str(row["w"]).split(";")}


# --------------------------------------------------------------------------
# band 的通用实现：只换留存规则，其余与官方一致
# --------------------------------------------------------------------------
def layer_pct(P: np.ndarray, elig: np.ndarray, Lbin: np.ndarray, nbin: int,
              min_n: int = 20) -> np.ndarray:
    """逐日、逐层在**可交易集合内**做百分位；层号 <0（因子缺失）单独成一组。"""
    T, N = P.shape
    LP = np.full((T, N), np.nan)
    for t in range(T):
        m = elig[t]
        b = Lbin[t]
        for k in list(range(nbin)) + [-1]:
            mk = m & (b == k)
            if int(mk.sum()) >= min_n:
                LP[t, mk] = rankdata(P[t][mk]) / int(mk.sum())
    return LP


def band_generic(P: np.ndarray, LU: np.ndarray, Lbin: np.ndarray, mode: str,
                 q: float, rel: float) -> tuple[np.ndarray, np.ndarray]:
    """mode ∈ {pool, lkeep, lall, quota, relent, reldod}。

    返回 (编码后的打分宽矩阵, 逐日在榜布尔矩阵)。编码规则与官方 band 同构：
    在榜股保留全池分位（整体上移 SHIFT 以消除边界交叠），落选股下移 1−q*。
    """
    T, N = P.shape
    nbin = NBIN
    elig = (~LU) & np.isfinite(P)
    full = np.isfinite(P)
    rank_elig = np.full((T, N), np.nan)
    rank_full = np.full((T, N), np.nan)
    for t in range(T):
        m = elig[t]
        if m.any():
            rank_elig[t, m] = rankdata(P[t][m]) / int(m.sum())
        m = full[t]
        if m.any():
            rank_full[t, m] = rankdata(P[t][m]) / int(m.sum())
    need_layer = mode in ("lkeep", "lall", "quota")
    LP = layer_pct(P, elig, Lbin, nbin) if need_layer else None

    out = rank_full.copy()
    delta = 1.0 - KQ + 1e-9
    prev: list[int] = []
    entry: dict[int, float] = {}
    prevrow = None
    tops = []
    for t in range(T):
        row = rank_elig[t]
        ok = ~np.isnan(row)
        n_elig = int(ok.sum())
        if n_elig < TOP_MIN_VALID:
            prev, entry, prevrow = [], {}, row
            tops.append(np.zeros(N, dtype=bool))
            continue
        n_top = max(n_elig // 10, 1)

        if mode == "pool":
            keep = [i for i in prev if ok[i] and row[i] >= q]
            key = np.where(ok, row, -np.inf)
        elif mode == "lkeep":
            lp = LP[t]
            keep = [i for i in prev if ok[i] and np.isfinite(lp[i]) and lp[i] >= q]
            key = np.where(ok, row, -np.inf)
        elif mode == "lall":
            lp = LP[t]
            keep = [i for i in prev if ok[i] and np.isfinite(lp[i]) and lp[i] >= q]
            key = np.where(ok & np.isfinite(lp), lp, -np.inf)
        elif mode == "quota":
            lp = LP[t]
            b = Lbin[t]
            quota = n_top // nbin
            keep = []
            for k in range(nbin):
                prevk = [i for i in prev if b[i] == k and ok[i]
                         and np.isfinite(lp[i]) and lp[i] >= q]
                if len(prevk) > quota:
                    prevk = sorted(prevk, key=lambda i: -lp[i])[:quota]
                if len(prevk) < quota:
                    chosen = set(prevk)
                    cand = np.flatnonzero(ok & (b == k) & np.isfinite(lp))
                    if cand.size:
                        for j in cand[np.argsort(-lp[cand])]:
                            if len(prevk) >= quota:
                                break
                            j = int(j)
                            if j not in chosen:
                                prevk.append(j)
                                chosen.add(j)
                keep.extend(prevk)
            key = np.where(ok, row, -np.inf)
        elif mode == "relent":
            keep = [i for i in prev if ok[i] and row[i] >= entry.get(i, row[i]) - rel]
            key = np.where(ok, row, -np.inf)
        elif mode == "reldod":
            if prevrow is None:
                keep = [i for i in prev if ok[i]]
            else:
                keep = [i for i in prev if ok[i]
                        and (not np.isfinite(prevrow[i]) or row[i] >= prevrow[i] - rel)]
            key = np.where(ok, row, -np.inf)
        else:
            raise ValueError(mode)

        if len(keep) > n_top:
            keep = sorted(keep, key=lambda i: -key[i])[:n_top]
        if len(keep) < n_top:
            chosen = set(keep)
            for idx in np.argsort(-key):
                if len(keep) >= n_top:
                    break
                j = int(idx)
                if key[j] == -np.inf:
                    break
                if j not in chosen:
                    keep.append(j)
                    chosen.add(j)

        kset = np.zeros(N, dtype=bool)
        kset[keep] = True
        tops.append(kset)
        for j in keep:
            entry.setdefault(j, row[j])
        entry = {j: v for j, v in entry.items() if kset[j]}
        drop = ok & ~kset
        out[t, keep] = rank_full[t, keep] + SHIFT
        out[t, drop] = rank_full[t, drop] - delta
        prev, prevrow = keep, row
    return out, np.array(tops)


MODES = {
    "pool": [dict(q=KQ, rel=0.0)],
    "lkeep": [dict(q=q, rel=0.0) for q in Q_GRID],
    "lall": [dict(q=q, rel=0.0) for q in Q_GRID],
    "quota": [dict(q=q, rel=0.0) for q in Q_GRID],
    "relent": [dict(q=0.0, rel=r) for r in REL_ENTRY_GRID],
    "reldod": [dict(q=0.0, rel=r) for r in REL_DOD_GRID],
}


# --------------------------------------------------------------------------
# 逐折评估
# --------------------------------------------------------------------------
def composition(S: np.ndarray, Lbin: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """在榜名单（剔涨停口径的 top 集合）在 vol60 十层上的占比，逐日平均。"""
    T, N = S.shape
    acc = np.zeros(NBIN)
    tot = 0.0
    for t in range(T):
        m = S[t] & valid[t] & (Lbin[t] >= 0)
        if m.sum() < 20:
            continue
        b = Lbin[t][m]
        cnt = np.bincount(b, minlength=NBIN).astype(float)
        acc += cnt / cnt.sum()
        tot += 1.0
    return acc / tot if tot else np.full(NBIN, np.nan)


def fold_block(fold: str, fdf: pd.DataFrame, weights: dict) -> tuple[list[dict], list[dict]]:
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
    vol = W[LAYER]
    Lbin = np.vstack([stratify(vol[t], NBIN) for t in range(panel.T)])
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in weights}
    pct = {m: pct_rank_2d(P) for m, P in raw.items()}
    mix = np.nansum(np.stack([w * pct[m] for m, w in weights.items()]), axis=0)
    P0 = two_stage(pct_rank_2d(mix), pct["H01"], K_STAGE1)

    rows, comps = [], []
    n_nolayer = int((Lbin[0] < 0).sum())
    log(f"  {fold}: T={panel.T} N={panel.N} 无层因子个股={n_nolayer}")

    def evaluate(tag: str, P: np.ndarray) -> dict:
        r = score(P, panel)
        S = top_sets(P, panel)
        d = decompose(W, panel, S, [LAYER], [NBIN], fold, tag)
        alpha = float(d["alpha"].mean()) * ANN
        prof = float(d["profile"].mean()) * ANN
        hs = holding_stats(r["_S"])
        c = composition(r["_S"], Lbin, panel.valid_turn)
        return dict(fold=fold, mode=tag.split("|")[0], param=float(tag.split("|")[1]), tag=tag,
                    ic=r["ic_mean"], excess=r["annual_excess"], turnover=r["mean_turnover"],
                    final=r["final_score"], alpha=alpha, profile=prof,
                    noprofile=0.4 * r["ic_mean"] + 0.3 * alpha + 0.3 * (1 - r["mean_turnover"]),
                    hold_days=hs["avg_hold_days"], list_size=hs["list_size"],
                    daily_replace=hs["daily_replace"], **{f"L{k}": c[k] for k in range(NBIN)})

    # V0：官方 fast_band（自检用）
    P_base, _ = fast_band(P0, panel.LU, KQ)
    rows.append(evaluate(f"V0_base|{KQ:g}", P_base))
    log(f"  {fold} V0_base      IC {rows[-1]['ic']:.4f} E {rows[-1]['excess']:.4f} "
        f"T {rows[-1]['turnover']:.4f} F {rows[-1]['final']:.4f} "
        f"a {rows[-1]['alpha']:.4f} p {rows[-1]['profile']:+.4f} "
        f"np {rows[-1]['noprofile']:.4f}")

    for mode, params in MODES.items():
        for prm in params:
            tag = f"{mode}|{prm['q'] if mode not in ('relent', 'reldod') else prm['rel']:g}"
            P, _ = band_generic(P0, panel.LU, Lbin, mode, prm["q"], prm["rel"])
            rows.append(evaluate(tag, P))
            r = rows[-1]
            log(f"  {fold} {tag:14s} IC {r['ic']:.4f} E {r['excess']:.4f} "
                f"T {r['turnover']:.4f} F {r['final']:.4f} "
                f"a {r['alpha']:.4f} p {r['profile']:+.4f} np {r['noprofile']:.4f} "
                f"L0 {r['L0']:.3f} L9 {r['L9']:.3f} hold {r['hold_days']:.0f}")
    return rows, comps


def paired(base: pd.Series, other: pd.Series, metric: str) -> dict:
    d = (other - base).to_numpy()
    sd = float(np.std(d, ddof=1))
    return dict(metric=metric, delta=float(np.mean(d)),
                t=float(np.mean(d) / (sd / 2)) if sd > 0 else np.nan,
                signs="".join("+" if v > 0 else "-" for v in d))


def main() -> int:
    t0 = time.time()
    weights = load_weights()
    log(f"部署权重 {weights}")
    fdf = load_factors(20210101, 20241231)
    rows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        r, _ = fold_block(fold, fdf, weights)
        rows.extend(r)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "band_freeze_grid.csv", index=False)

    pd.set_option("display.width", 400)
    F = FOLDS
    cols = ["ic", "excess", "turnover", "final", "alpha", "profile", "noprofile",
            "hold_days", "L0", "L9"]

    # ---- 自检 ----
    print("\n=== 自检：V0_base vs 归档 TS_K40 ===")
    bad = 0
    for fold in F:
        got = M[(M.fold == fold) & (M["mode"] == "V0_base")].iloc[0]
        for k, v in ARCHIVE[fold].items():
            if abs(float(got[k]) - v) > 1e-5:
                bad += 1
                print(f"  FAIL {fold} {k}: {float(got[k]):.6f} vs {v:.6f}")
    print(f"  四折 × 6 项 = 24 项，不一致 {bad} 项 -> {'PASS' if bad == 0 else 'FAIL'}")

    print("\n=== 自检：generic(pool,q*) vs 官方 fast_band ===")
    for fold in F:
        a = M[(M.fold == fold) & (M["mode"] == "V0_base")].iloc[0]
        b = M[(M.fold == fold) & (M["mode"] == "pool")].iloc[0]
        print(f"  {fold}: dIC {b.ic - a.ic:+.2e} dE {b.excess - a.excess:+.2e} "
              f"dT {b.turnover - a.turnover:+.2e} dAlpha {b.alpha - a.alpha:+.2e}")

    # ---- 四折均值 ----
    print("\n=== 四折均值（按可外推口径降序）===")
    g = M.groupby(["mode", "param"])[cols].mean()
    print(g.sort_values("noprofile", ascending=False).round(4).to_string())

    # ---- 同等换手：每个 mode 选 T 最接近基线的参数点 ----
    base_t = float(M[M["mode"] == "V0_base"]["turnover"].mean())
    print(f"\n=== 同等换手对照（基线四折均换手 T={base_t:.5f}）===")
    picks = {}
    for mode in [m for m in MODES if m != "pool"]:
        sub = M[M["mode"] == mode]
        gg = sub.groupby("param")["turnover"].mean().sort_index()
        i = int(np.argmin(np.abs(gg.to_numpy() - base_t)))
        picks[mode] = float(gg.index[i])
        print(f"  {mode:8s} 取 param={gg.index[i]:g}（T={gg.iloc[i]:.5f}）")

    print("\n=== 配对 Δ vs V0_base（同等换手点，n=4）===")
    out_rows = []
    base = M[M["mode"] == "V0_base"].set_index("fold")
    param_f = M["param"].astype(float)
    for mode, p in picks.items():
        oth = M[(M["mode"] == mode) & np.isclose(param_f, p)].set_index("fold")
        oth = oth.reindex(F)
        for metric in ["ic", "alpha", "profile", "excess", "turnover", "final", "noprofile"]:
            r = paired(base[metric], oth[metric], metric)
            r["mode"] = mode
            r["param"] = p
            out_rows.append(r)
    P = pd.DataFrame(out_rows)[["mode", "param", "metric", "delta", "t", "signs"]]
    P.to_csv(OUT / "band_freeze_paired.csv", index=False)
    print(P.round(6).to_string(index=False))

    print("\n=== 逐折明细（同等换手点）===")
    for metric in ["turnover", "alpha", "noprofile", "profile"]:
        tab = pd.DataFrame({("V0_base", 0.0): base[metric]})
        for mode, p in picks.items():
            oth = M[(M["mode"] == mode) & np.isclose(param_f, p)].set_index("fold").reindex(F)
            tab[(mode, p)] = oth[metric]
        tab.loc["mean"] = tab.mean()
        print(f"\n-- {metric} --")
        print(tab.round(4).to_string())

    # ---- 名单层构成 ----
    print("\n=== 名单层构成（十层占比，四折均值）===")
    cc = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
    comp = M.groupby(["mode", "param"])[cc].mean()
    show = pd.concat([
        M[M["mode"] == "V0_base"].groupby("mode")[cc].mean(),
        comp.loc[[(m, p) for m, p in picks.items()]],
    ])
    print(show.round(3).to_string())

    log(f"完成，用时 {time.time() - t0:.0f}s  输出 {OUT / 'band_freeze_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

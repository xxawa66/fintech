"""LH032 第一步：profile 到底能不能外推？把它拆成「可外推的一半」和「不可外推的一半」。

profile 的恒等式（DGTW，非拟合）：

    profile = Σ_b ( w_b^B − w_b^V ) · ( m_b − mkt )

    暴露差 (w_b^B − w_b^V)  ← 名单落在哪些层，只由选股规则决定，**可外推**
    层溢价 (m_b − mkt)      ← 市场给的钱，**LH020 只证明 vol60 的层溢价断档**

⇒ 「profile 不可外推」是 vol60 这一个因子的性质，不是 profile 这个量的性质。
   本探针回答两个问题：

  Q1 同向性：层内判别力 IC_b 与层溢价 prem_b 在各层之间是否同向？
     vol60 上是反的（高波层判别力强 3.59×，但低波溢价为正）⇒ 倾斜时 α 与 profile 互相抵消。
     若存在 corr(IC_b, prem_b) > 0 的因子，则倾斜可让 α 与 profile 同向上升。

  Q2 稳定性：层溢价剖面 prem_b 在时间上是否稳定？
     按半年窗口切段，看剖面形状的段间相关、spread(L0−L9) 序列是否有趋势/断档。

同时输出部署名单（TS_K40）在各因子上的暴露差与 profile 分层来源。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs/long_horizon"
sys.path.insert(0, str(CACHE))
sys.path.insert(0, str(ROOT / "research" / "LH023"))   # 必须优先于 CACHE
sys.path.insert(0, str(ROOT))

import _probe_factors as _pf                                        # noqa: E402
if "mom20" not in _pf.FACTORS:                                      # wide_of 只摊 FACTORS
    _pf.FACTORS.append("mom20")
from _probe_e_attr import load_factors, top_sets, wide_of          # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, to_wide  # noqa: E402

KQ = 0.0022778298112255263
ANN = 252
NB = 10
FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
FACTORS = ["vol60", "log_amt", "mom60", "rev1", "log_price", "mom20"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
def _rank(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a, kind="mergesort")
    r = np.empty(a.size, dtype=np.float64)
    r[o] = np.arange(a.size, dtype=np.float64)
    # 并列取平均秩
    s = a[o]
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and s[j] == s[i]:
            j += 1
        if j - i > 1:
            r[o[i:j]] = (i + j - 1) / 2.0
        i = j
    return r


def spear(a: np.ndarray, b: np.ndarray) -> float:
    """a、b 的 Spearman（内部 rank + Pearson）。样本 <10 返回 NaN。"""
    if a.size < 10:
        return np.nan
    ra = _rank(a)
    rb = _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def pct_rank_2d(P: np.ndarray) -> np.ndarray:
    out = np.full(P.shape, np.nan)
    for t in range(P.shape[0]):
        x = P[t]
        m = np.isfinite(x)
        n = int(m.sum())
        if n < 2:
            continue
        out[t, m] = _rank(x[m]) / (n - 1)
    return out


def strata(x: np.ndarray, nb: int) -> np.ndarray:
    """当日按 x 分位切 nb 层，NaN→-1。"""
    out = np.full(x.shape, -1)
    m = np.isfinite(x)
    if int(m.sum()) < nb * 20:
        return out
    q = np.quantile(x[m], np.linspace(0, 1, nb + 1)[1:-1])
    out[m] = np.searchsorted(q, x[m], side="right")
    return out


# --------------------------------------------------------------------------
def fold_data(fold: str, fdf: pd.DataFrame):
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    ms = ["H01", "H05_F1T2", "H05_F2T2"]
    df = df.drop(columns=[c for c in ms if c in df.columns]).merge(
        pred[["ts_code", "trade_date"] + ms], on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in ms}
    pct = {m: pct_rank_2d(raw[m]) for m in ms}
    mix = np.nansum(np.stack([w * pct[m] for m, w in WEIGHTS.items()]), axis=0)
    # TS_K40：候选池 = mix top40%，池内按 H01 重排
    pr_mix = pct_rank_2d(mix)
    pool = pr_mix >= 0.60
    p2 = np.where(pool, pct["H01"], np.where(np.isfinite(pr_mix), pr_mix - 2.0, np.nan))
    P = np.where(np.isfinite(pr_mix), p2, pr_mix)
    Pb, _ = fast_band(P, panel.LU, KQ)
    S = top_sets(Pb, panel)
    return panel, W, {"H01": raw["H01"], "mix": mix, "pr_mix": pr_mix}, S


def window_id(date: int) -> str:
    d = str(int(date))
    return d[:4] + ("H1" if int(d[4:6]) <= 6 else "H2")


def scan(fold: str, fdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, W, SIG, S = fold_data(fold, fdf)
    Y = panel.Y
    rows = []     # 逐日 × 因子 × 层
    for f in FACTORS:
        X = W[f]
        for t in range(panel.T):
            ok = panel.valid_top[t]
            if ok.sum() < 200:
                continue
            x = X[t]
            m = ok & np.isfinite(x)
            if m.sum() < NB * 20:
                continue
            yv = Y[t][m]
            b = strata(x[m], NB)
            mkt = float(np.mean(yv))
            sel = S[t] & m
            nb_tot = int(sel.sum())
            nv_tot = int(m.sum())
            for c in range(NB):
                s = b == c
                ns = int(s.sum())
                if ns < 10:
                    continue
                yy = yv[s]
                ic = spear(yy, SIG["H01"][t][m][s])
                prem = float(np.mean(yy)) - mkt
                wB = (sel[m][s].sum() / nb_tot) if nb_tot else np.nan
                wV = ns / nv_tot
                rows.append(dict(fold=fold, window=window_id(panel.dates[t]), factor=f,
                                 layer=c, n=ns, ic=ic, prem=prem,
                                 wB=wB, wV=wV, n_top=nb_tot, n_valid=nv_tot))
    D = pd.DataFrame(rows)
    return D, pd.DataFrame({"fold": fold, "date": panel.dates})


# --------------------------------------------------------------------------
def main() -> int:
    log("读因子面板")
    fdf = load_factors(20210101, 20241231)
    parts = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        D, _ = scan(fold, fdf)
        parts.append(D)
    A = pd.concat(parts, ignore_index=True)
    A.to_csv(ROOT / "experiments" / "LH032" / "layer_daily.csv", index=False)
    log(f"逐日层表 {A.shape}")

    # ---- Q1 同向性：IC_b 与 prem_b ----
    g = A.groupby(["fold", "factor", "layer"])[["ic", "prem", "wB", "wV"]].mean()
    g = g.reset_index()
    g["prem_a"] = g["prem"] * ANN
    g["dw"] = g["wB"] - g["wV"]

    rows = []
    for (fold, f), sub in g.groupby(["fold", "factor"]):
        s = sub.sort_values("layer")
        ok = s[["ic", "prem_a"]].notna().all(axis=1)
        v = s[ok]
        if len(v) < 6:
            continue
        r = float(np.corrcoef(v["ic"], v["prem_a"])[0, 1])
        # 加权：暴露差可及的部分（dw 与 prem 的乘积 = 现有 profile）
        prof = float((v["dw"] * v["prem_a"]).sum())
        rows.append(dict(fold=fold, factor=f, corr=r,
                         ic_spread=float(v["ic"].iloc[-1] - v["ic"].iloc[0]),
                         prem_spread=float(v["prem_a"].iloc[-1] - v["prem_a"].iloc[0]),
                         profile=prof,
                         ic_mean=float(v["ic"].mean())))
    C = pd.DataFrame(rows)
    C.to_csv(ROOT / "experiments" / "LH032" / "layer_corr.csv", index=False)

    print("\n" + "=" * 96)
    print("Q1 同向性：corr(层内 IC_b, 层年化溢价_b)  ——  >0 ⇒ 倾斜可让 α 与 profile 同向")
    print("=" * 96)
    piv = C.pivot_table(index="factor", columns="fold", values="corr")[FOLDS]
    piv["mean"] = piv.mean(axis=1)
    piv["sign"] = piv[FOLDS].apply(lambda r: "".join("+" if v > 0 else "-" for v in r), axis=1)
    sp = C.pivot_table(index="factor", columns="fold", values="prem_spread")[FOLDS]
    icsp = C.pivot_table(index="factor", columns="fold", values="ic_spread")[FOLDS]
    out = piv.copy()
    out["prem_spread(L9-L0)"] = sp.mean(axis=1)
    out["ic_spread(L9-L0)"] = icsp.mean(axis=1)
    print(out.round(4).to_string())

    # ---- Q2 稳定性：层溢价剖面在半年窗口间的一致性 ----
    gw = A.groupby(["window", "factor", "layer"])["prem"].mean().mul(ANN).reset_index()
    win = sorted(gw["window"].unique())
    rows = []
    for f, sub in gw.groupby("factor"):
        M = sub.pivot_table(index="window", columns="layer", values="prem").reindex(win)
        M = M.dropna()
        if len(M) < 4:
            continue
        # 剖面形状稳定性：窗口间两两相关（去掉各窗口均值后的形状相关）
        Z = M.sub(M.mean(axis=1), axis=0)
        cm = Z.T.corr()
        iu = np.triu_indices(len(M), 1)
        shape_r = float(np.nanmean(cm.values[iu]))
        # 幅度稳定性：spread 序列
        sp_ser = (M[9] - M[0]).reindex(win)
        mu, sd = float(sp_ser.mean()), float(sp_ser.std())
        # 趋势：对窗口序号回归的斜率 t
        k = np.arange(len(sp_ser), dtype=float)
        kk = k - k.mean()
        yy = sp_ser.values - np.nanmean(sp_ser.values)
        den = float((kk * kk).sum())
        slope = float((kk * yy).sum() / den) if den > 0 else np.nan
        resid = yy - slope * kk
        se = float(np.sqrt((resid**2).sum() / max(len(yy) - 2, 1)) / np.sqrt(den)) if den > 0 else np.nan
        rows.append(dict(factor=f, n_win=len(M), shape_r=shape_r,
                         spread_mean=mu, spread_sd=sd,
                         spread_snr=abs(mu) / sd if sd > 0 else np.nan,
                         slope_t=slope / se if se and se > 0 else np.nan,
                         spread_series=" ".join(f"{v:+.3f}" for v in sp_ser.values)))
    ST = pd.DataFrame(rows)
    if ST.empty:
        print("\n[Q2] 窗口数不足，跳过（需四折）")
        return 0
    ST = ST.set_index("factor")
    ST.to_csv(ROOT / "experiments" / "LH032" / "layer_stability.csv")
    print("\n" + "=" * 96)
    print(f"Q2 稳定性：层溢价剖面（半年窗口 {win[0]}~{win[-1]}，spread = L9 − L0，年化）")
    print("=" * 96)
    print(ST.round(4).to_string())

    # ---- 部署名单的 profile 来源分解 ----
    print("\n" + "=" * 96)
    print("部署名单（TS_K40）的层暴露差与 profile 分层来源（四折年化）")
    print("=" * 96)
    for f in FACTORS:
        s = g[g.factor == f].groupby("layer")[["dw", "prem_a"]].mean()
        prof = float((s["dw"] * s["prem_a"]).sum())
        top = s.assign(contr=s["dw"] * s["prem_a"]).sort_values("contr", key=abs, ascending=False)
        print(f"\n[{f}]  profile = {prof:+.4f}")
        print("  layer   dw(暴露差)   prem(年化)   contr")
        for c, r in top.iterrows():
            print(f"   L{c:<2d}  {r['dw']:+.5f}    {r['prem_a']:+.5f}   {r['contr']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

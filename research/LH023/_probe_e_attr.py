"""LH020：E 的因子归因 —— 找出 2025–2026 那 0.076 到底丢在哪个因子上。

对 CV 四折（2021–2024，可用于选型）与测试期（仅用于机制诊断，不用于选型）
分别算：
  - 每日因子收益（log_amt / vol60 / mom60 / rev1 / log_price 的 30% 多空）
  - 每个模型名单的因子暴露与日超额
  - 日超额对因子收益的时间序列回归 ⇒ α（选股）与 Σβ·f̄（因子贡献）
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

from _probe_factors import FACTORS  # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide  # noqa: E402

KQ = 0.0022778298112255263
ANN = 252


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
def load_factors(start: int, end: int) -> pd.DataFrame:
    log(f"读因子面板 {start}~{end}")
    df = pd.read_parquet(CACHE / "_probe_factors_full.parquet",
                         filters=[("trade_date", ">=", start), ("trade_date", "<=", end)])
    for c in ["y_ret_1d", "close", "rev1", "log_price", "log_amt", "vol60", "mom20", "mom60"]:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    return df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def wide_of(df: pd.DataFrame, panel: Panel) -> dict[str, np.ndarray]:
    """把因子列摊成与 panel 同序的宽矩阵。df 必须覆盖 panel 的全部 (date, code)。"""
    sub = df[(df.trade_date >= panel.dates[0]) & (df.trade_date <= panel.dates[-1])]
    m = sub.merge(pd.DataFrame({"trade_date": panel.dates}), on="trade_date", how="inner")
    out = {}
    key = m["trade_date"].to_numpy() * 1000000
    # 用 (date, code) → (t, n) 映射
    di = {d: i for i, d in enumerate(panel.dates)}
    ci = {c: i for i, c in enumerate(panel.codes)}
    ti = m["trade_date"].map(di).to_numpy()
    si = m["ts_code"].map(ci).to_numpy()
    ok = ~np.isnan(ti) & ~np.isnan(si)
    ti, si = ti[ok].astype(int), si[ok].astype(int)
    for c in FACTORS:
        W = np.full((panel.T, panel.N), np.nan)
        W[ti, si] = m[c].to_numpy(dtype="float64")[ok]
        out[c] = W
    return out


def factor_returns(panel: Panel, W: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for t in range(panel.T):
        ok = panel.valid_top[t]
        n = int(ok.sum())
        if n < 100:
            continue
        y = panel.Y[t]
        row = {"trade_date": int(panel.dates[t]), "market_ret": float(np.mean(y[ok])),
               "dispersion": float(np.nanstd(y[ok]))}
        cut = max(int(n * 0.3), 10)
        for k in FACTORS:
            x = W[k][t]
            m = ok & np.isfinite(x)
            if m.sum() < 100:
                row[k] = np.nan
                continue
            xs, ys = x[m], y[m]
            nn = len(xs)
            c = max(int(nn * 0.3), 10)
            lo = np.argpartition(xs, c - 1)[:c]
            hi = np.argpartition(-xs, c - 1)[:c]
            row[k] = float(np.mean(ys[hi]) - np.mean(ys[lo]))
        rows.append(row)
    return pd.DataFrame(rows)


def pct_rank_row(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    n = int(m.sum())
    if n < 2:
        return out
    v = x[m]
    order = np.argsort(v, kind="mergesort")
    rk = np.empty(n)
    rk[order] = np.arange(n, dtype=float)
    srt = v[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and srt[j] == srt[i]:
            j += 1
        rk[order[i:j]] = (i + j - 1) / 2.0
        i = j
    out[m] = rk / (n - 1)
    return out


def basket_stats(panel: Panel, W: dict[str, np.ndarray], S: np.ndarray, tag: str) -> pd.DataFrame:
    rows = []
    for t in range(panel.T):
        ok = panel.valid_top[t]
        if ok.sum() < 100 or not S[t].any():
            continue
        sel = ok & S[t]
        if sel.sum() == 0:
            continue
        y = panel.Y[t]
        row = {"trade_date": int(panel.dates[t]), "tag": tag,
               "basket_ret": float(np.mean(y[sel])),
               "market_ret": float(np.mean(y[ok])),
               "n_sel": int(sel.sum())}
        row["excess"] = row["basket_ret"] - row["market_ret"]
        for k in FACTORS:
            pr = pct_rank_row(W[k][t])
            row[f"exp_{k}"] = float(np.nanmean(pr[sel]) - 0.5) if np.isfinite(pr[sel]).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def top_sets(P: np.ndarray, panel: Panel) -> np.ndarray:
    """官方超额口径的逐日 top 集合（剔涨停 + y 缺失，取 len//10）。"""
    S = np.zeros((panel.T, panel.N), dtype=bool)
    for t in range(panel.T):
        ok = panel.valid_top[t]
        n = int(ok.sum())
        if n < 100:
            continue
        k = max(n // 10, 1)
        v = np.where(ok, P[t], -np.inf)
        idx = np.argpartition(-v, k - 1)[:k]
        S[t, idx] = True
    return S


def regress(daily: pd.DataFrame, fr: pd.DataFrame, label: str) -> pd.DataFrame:
    fr = fr.set_index("trade_date")
    out = []
    for tag, g in daily.groupby("tag"):
        g = g.set_index("trade_date").drop(columns=["market_ret"], errors="ignore")
        j = g.join(fr[["market_ret"] + FACTORS], how="inner")
        y = j["excess"].to_numpy(dtype="float64")
        X = j[FACTORS].to_numpy(dtype="float64")
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        y, X = y[m], X[m]
        if len(y) < 60:
            continue
        A = np.hstack([np.ones((len(y), 1)), X])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
        fmean = X.mean(axis=0)
        contrib = coef[1:] * fmean * ANN
        row = dict(label=label, tag=tag, n_days=len(y),
                   ann_excess=float(y.mean()) * ANN,
                   alpha_ann=float(coef[0]) * ANN, r2=float(r2),
                   factor_total=float(contrib.sum()))
        row.update({f"beta_{k}": float(b) for k, b in zip(FACTORS, coef[1:])})
        row.update({f"contrib_{k}": float(c) for k, c in zip(FACTORS, contrib)})
        row.update({f"exp_{k}": float(g[f"exp_{k}"].mean()) for k in FACTORS})
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def build_cv(fold: str, fdf: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_fold(fold)
    # load_fold 只保留 MODELS 子集，补上 F1T2 系列
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
    fr = factor_returns(panel, W)
    daily = []
    variants = {}
    # 单模型
    for m in ["H01", "H05_F1T2", "H05_F2T2"]:
        if m not in df.columns:
            continue
        P, _ = fast_band(to_wide(panel, df[m].to_numpy(dtype="float64")), panel.LU, KQ)
        variants[m] = P
    # LH019 部署权重组合
    pct = {m: pct_rank_2d(to_wide(panel, df[m].to_numpy(dtype="float64"))) for m in weights}
    mix = np.nansum(np.stack([w * pct[m] for m, w in weights.items()]), axis=0)
    variants["LH019_lam0.5"] = fast_band(mix, panel.LU, KQ)[0]
    for tag, P in variants.items():
        daily.append(basket_stats(panel, W, top_sets(P, panel), tag))
        r = score(P, panel)
        log(f"  {fold} {tag}: IC {r['ic_mean']:.4f} E {r['annual_excess']:.4f} T {r['mean_turnover']:.4f} final {r['final_score']:.4f}")
    return pd.concat(daily, ignore_index=True), fr


def pct_rank_2d(P: np.ndarray) -> np.ndarray:
    return np.vstack([pct_rank_row(P[t]) for t in range(P.shape[0])])


def build_test(fdf_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = {
        "H01": "test_pred_baseline_H01_band.csv",
        "H05_F1T2": "test_pred_single_H05_F1T2_band.csv",
        "H05_F2T2": "test_pred_single_H05_F2T2_band.csv",
        "LH019_lam0.5": "test_pred_LH019_lam0.5_band.csv",
    }
    frames = []
    for tag, fn in files.items():
        p = TESTDIR / "lh019_test_run" / fn
        if not p.exists():
            log(f"  缺 {fn}")
            continue
        d = pd.read_csv(p, dtype={"ts_code": "str", "trade_date": "int64"})
        d["tag"] = tag
        frames.append(d[["ts_code", "trade_date", "y_ret_1d", "flag_limit_up", "pred", "tag"]])
    allp = pd.concat(frames, ignore_index=True)
    meta = allp.drop_duplicates(["ts_code", "trade_date"])[
        ["ts_code", "trade_date", "y_ret_1d", "flag_limit_up"]]
    panel = Panel(meta)
    W = wide_of(fdf_test, panel)
    fr = factor_returns(panel, W)
    daily = []
    for tag, d in allp.groupby("tag"):
        P = to_wide(panel, d.sort_values(["trade_date", "ts_code"], kind="mergesort")["pred"].to_numpy())
        daily.append(basket_stats(panel, W, top_sets(P, panel), tag))
        r = score(P, panel)
        log(f"  测试期 {tag}: IC {r['ic_mean']:.4f} E {r['annual_excess']:.4f} T {r['mean_turnover']:.4f} final {r['final_score']:.4f}")
    return pd.concat(daily, ignore_index=True), fr


# --------------------------------------------------------------------------
if __name__ == "__main__":
    dw = pd.read_csv(ROOT / "experiments/LH019_deploy_weights.csv")
    w50 = dw[dw.lam == 0.5].iloc[0]["w"]
    weights = {}
    for kv in str(w50).split(";"):
        k, v = kv.split(":")
        weights[k] = float(v)
    log(f"部署权重 λ=0.5: {weights}")

    fdf = load_factors(20210101, 20241231)
    fdf_te = load_factors(20250101, 20260630)

    all_daily, all_fr, all_reg = [], [], []
    for fold in ["wf2021", "wf2022", "wf2023", "confirm2024"]:
        log(f"=== {fold} ===")
        d, f = build_cv(fold, fdf, weights)
        d["label"] = fold
        f["label"] = fold
        all_daily.append(d)
        all_fr.append(f)
        all_reg.append(regress(d, f, fold))

    log("=== 测试期 ===")
    d, f = build_test(fdf_te)
    d["label"] = "test2025_2026"
    f["label"] = "test2025_2026"
    all_daily.append(d); all_fr.append(f); all_reg.append(regress(d, f, "test2025_2026"))

    pd.concat(all_daily, ignore_index=True).to_csv(ROOT / "experiments/LH020_daily_excess.csv", index=False)
    pd.concat(all_fr, ignore_index=True).to_csv(ROOT / "experiments/LH020_factor_returns.csv", index=False)
    reg = pd.concat(all_reg, ignore_index=True)
    reg.to_csv(ROOT / "experiments/LH020_factor_attribution.csv", index=False)

    pd.set_option("display.width", 400)
    print("\n=== 因子收益（年化均值，×252）===")
    fr = pd.concat(all_fr, ignore_index=True)
    g = fr.groupby("label")[["market_ret"] + FACTORS].mean() * ANN
    g["n_days"] = fr.groupby("label").size()
    g["dispersion"] = fr.groupby("label")["dispersion"].mean()
    print(g.round(4).to_string())

    print("\n=== 名单因子暴露（相对全市场分位，0 为中性）===")
    dd = pd.concat(all_daily, ignore_index=True)
    e = dd.groupby(["label", "tag"])[[f"exp_{k}" for k in FACTORS]].mean()
    print(e.round(3).to_string())

    print("\n=== 归因：α vs 因子贡献（年化）===")
    cols = ["ann_excess", "alpha_ann", "factor_total", "r2"] + \
           [f"contrib_{k}" for k in FACTORS] + [f"beta_{k}" for k in FACTORS]
    print(reg.set_index(["label", "tag"])[cols].round(4).to_string())
    log("完成")

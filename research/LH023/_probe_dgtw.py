"""LH020 第三步：分层归因 —— 把 E 拆成「组内选股 α」+「因子轮廓贡献」。

为什么不用时间序列回归：因子收益高度共线，β 符号在折间翻转，α 不可信。
改用 DGTW 式的横截面分层，逐日把名单超额精确拆成两项：

    excess_t = Σ_b w_b^B · ( 名单在层 b 的收益 − 层 b 的收益 )      ← 组内选股 α
             + Σ_b ( w_b^B − w_b^V ) · ( 层 b 的收益 − 市场收益 )   ← 因子轮廓贡献

第二项完全由「名单落在哪些层」决定，会随因子溢价翻转而翻转；
第一项只依赖「在同类股票里是否挑到更好的」，不依赖溢价是否延续。

分层变量：vol60（十层）单独一次；vol60(5) × log_amt(4) 联合一次。
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

from _probe_e_attr import load_factors, top_sets, wide_of  # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide  # noqa: E402

KQ = 0.0022778298112255263
ANN = 252


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pct_rank_row(x):
    out = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    n = int(m.sum())
    if n < 2:
        return out
    v = x[m]
    o = np.argsort(v, kind="mergesort")
    rk = np.empty(n); rk[o] = np.arange(n, dtype=float)
    s = v[o]
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        rk[o[i:j]] = (i + j - 1) / 2.0
        i = j
    out[m] = rk / (n - 1)
    return out


def pct_rank_2d(P):
    return np.vstack([pct_rank_row(P[t]) for t in range(P.shape[0])])


def stratify(x: np.ndarray, nbin: int) -> np.ndarray:
    """把当日截面按 x 分成 nbin 层（按分位切），返回层号，NaN 为 -1。"""
    out = np.full(x.shape, -1)
    m = np.isfinite(x)
    if m.sum() < nbin * 20:
        return out
    q = np.quantile(x[m], np.linspace(0, 1, nbin + 1)[1:-1])
    out[m] = np.searchsorted(q, x[m], side="right")
    return out


def decompose(W, panel, S, keys, nbins, label, tag):
    """keys/nbins: [(factor_name, n_bin), ...] ⇒ 多维分层。"""
    rows = []
    for t in range(panel.T):
        ok = panel.valid_top[t]
        if ok.sum() < 100 or not S[t].any():
            continue
        sel = ok & S[t]
        y = panel.Y[t]
        mkt = float(np.mean(y[ok]))
        # 联合层号
        code = np.zeros(panel.N, dtype=np.int64)
        valid = ok.copy()
        step = 1
        ok_bin = True
        for k, nb in zip(keys, nbins):
            b = stratify(W[k][t], nb)
            ok_bin &= np.isfinite(W[k][t])
            code = code + np.where(b >= 0, b, 0) * step
            step *= nb
        valid = ok & ok_bin
        sel2 = sel & ok_bin
        if valid.sum() < 200 or sel2.sum() < 20:
            continue
        codes = np.unique(code[valid])
        alpha = 0.0
        prof = 0.0
        nB = int(sel2.sum())
        nV = int(valid.sum())
        for c in codes:
            vb = valid & (code == c)
            sb = sel2 & (code == c)
            if not sb.any():
                continue
            mb = float(np.mean(y[vb]))
            wb = sb.sum() / nB
            wv = vb.sum() / nV
            alpha += wb * (float(np.mean(y[sb])) - mb)
            prof += (wb - wv) * (mb - mkt)
        rows.append(dict(label=label, tag=tag, trade_date=int(panel.dates[t]),
                         excess=float(np.mean(y[sel2])) - mkt, alpha=alpha, profile=prof))
    return pd.DataFrame(rows)


def cv_run(fold, fdf, weights, specs):
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
    variants = {"H01": raw["H01"], "H05_F1T2": raw["H05_F1T2"],
                "H05_F2T2": raw["H05_F2T2"], "LH019": mix}
    out = []
    for tag, P0 in variants.items():
        P, _ = fast_band(P0, panel.LU, KQ)
        S = top_sets(P, panel)
        for name, keys, nbins in specs:
            out.append(decompose(W, panel, S, keys, nbins, f"{fold}|{name}", tag))
    return pd.concat(out, ignore_index=True)


def test_run(fdf, specs):
    files = {"H01": "test_pred_baseline_H01_band.csv",
             "H05_F1T2": "test_pred_single_H05_F1T2_band.csv",
             "H05_F2T2": "test_pred_single_H05_F2T2_band.csv",
             "LH019": "test_pred_LH019_lam0.5_band.csv"}
    frames = []
    for tag, fn in files.items():
        d = pd.read_csv(TESTDIR / "lh019_test_run" / fn,
                        dtype={"ts_code": "str", "trade_date": "int64"})
        d["tag"] = tag
        frames.append(d[["ts_code", "trade_date", "y_ret_1d", "flag_limit_up", "pred", "tag"]])
    allp = pd.concat(frames, ignore_index=True)
    meta = allp.drop_duplicates(["ts_code", "trade_date"])[["ts_code", "trade_date", "y_ret_1d", "flag_limit_up"]]
    panel = Panel(meta)
    W = wide_of(fdf, panel)
    out = []
    for tag, d in allp.groupby("tag"):
        P = to_wide(panel, d.sort_values(["trade_date", "ts_code"], kind="mergesort")["pred"].to_numpy())
        S = top_sets(P, panel)
        for name, keys, nbins in specs:
            out.append(decompose(W, panel, S, keys, nbins, f"test|{name}", tag))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    dw = pd.read_csv(ROOT / "experiments/LH019_deploy_weights.csv")
    weights = {kv.split(":")[0]: float(kv.split(":")[1])
               for kv in str(dw[dw.lam == 0.5].iloc[0]["w"]).split(";")}
    specs = [("vol60x10", ["vol60"], [10]),
             ("vol5xamt4", ["vol60", "log_amt"], [5, 4])]
    fdf = load_factors(20210101, 20241231)
    fdf_te = load_factors(20250101, 20260630)

    parts = []
    for fold in ["wf2021", "wf2022", "wf2023", "confirm2024"]:
        log(f"=== {fold} ===")
        parts.append(cv_run(fold, fdf, weights, specs))
    log("=== 测试期 ===")
    parts.append(test_run(fdf_te, specs))
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(ROOT / "experiments/LH020_dgtw_daily.csv", index=False)

    g = D.groupby(["label", "tag"])[["excess", "alpha", "profile"]].mean().mul(ANN)
    g["n_days"] = D.groupby(["label", "tag"]).size()
    pd.set_option("display.width", 300)
    for name in ["vol60x10", "vol5xamt4"]:
        print(f"\n### 分层方案 {name}（年化，四折 + 测试期）")
        sub = g[[name in i for i in g.index.get_level_values("label")]]
        p = sub.reset_index()
        p["fold"] = p["label"].str.split("|").str[0]
        for k in ["excess", "alpha", "profile"]:
            t = p.pivot_table(index="tag", columns="fold", values=k)[
                ["wf2021", "wf2022", "wf2023", "confirm2024", "test"]]
            print(f"[{k}]")
            print(t.round(4).assign(cv_mean=t[["wf2021", "wf2022", "wf2023", "confirm2024"]].mean(axis=1).round(4),
                                    cv_sd=t[["wf2021", "wf2022", "wf2023", "confirm2024"]].std(axis=1).round(4)).to_string())
            print()
    log("完成")

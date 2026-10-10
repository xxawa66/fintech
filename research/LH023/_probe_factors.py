"""LH020：把「年化超额 E」拆成因子暴露 + 选股残差。

动机（LH018/LH019 结论）
----------------------
ΔIC 外推成功两次（+0.0291→+0.0301、+0.0258→+0.0279），ΔE 每次整段反转
（+0.0346→−0.1105、+0.0349→−0.0764）。四折 ΔE 的配对 std 只有 0.011，
实测偏离 −0.076 ~ −0.110 ⇒ 不是日度抽样噪声，是「折间共享某个因子」的
遗漏变量问题：存在一个因子 F，2021–2024 对模型篮子有利、2025–2026 不利，
而模型篮子对 F 有稳定暴露。

本脚本做的事
------------
1. 建 2018-01-01 ~ 2026-06-08 的价格面板（close / amount / vol / flag / y）。
2. 用**只用历史**的数据算 5 个风格因子：
   log_amt（规模/流动性）、vol60（波动）、mom60（动量）、rev1（短期反转）、
   log_price（价格水平）。
3. 逐日算因子收益（因子模仿组合的收益 = 因子分位组合多空收益）。
4. 对每个模型的每日 top10% 名单，算：
   - 名单的因子暴露（横截面 z 分位均值）
   - 名单日超额 d_t 对因子收益的回归 ⇒ α（选股）与 Σβ·f（因子贡献）
5. 对 CV 四折与测试期分别做，比较「因子收益是否翻转」「暴露是否稳定」。

产出 experiments/LH020_*.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TESTDIR = ROOT.parent.parent / "test_y_2025_2026"      # D:/github-fintech/test_y_2025_2026
CACHE = ROOT / "outputs/long_horizon"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CACHE))

FACTORS = ["log_amt", "vol60", "mom60", "rev1", "log_price"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. 价格面板
# --------------------------------------------------------------------------
def build_price_panel() -> pd.DataFrame:
    cache = CACHE / "_probe_price_full.parquet"
    if cache.exists():
        log(f"复用价格面板缓存 {cache}")
        return pd.read_parquet(cache)

    usecols = ["ts_code", "trade_date", "close", "vol", "amount",
               "flag_limit_up", "flag_limit_down"]
    dtypes = {"ts_code": "str", "trade_date": "int64", "close": "float64",
              "vol": "float64", "amount": "float64",
              "flag_limit_up": "int8", "flag_limit_down": "int8"}

    log("读 训练集.csv ...")
    parts = []
    for chunk in pd.read_csv(ROOT / "data/raw/训练集.csv", usecols=usecols + ["y_ret_1d"],
                             chunksize=2_000_000, dtype=dtypes):
        parts.append(chunk)
    tr = pd.concat(parts, ignore_index=True)
    del parts
    log(f"  训练集 {tr.shape}")

    log("读 测试集_X.csv ...")
    te = pd.read_csv(ROOT / "data/raw/测试集_X.csv", usecols=usecols, dtype=dtypes)
    y = pd.read_csv(TESTDIR / "测试集_Y.csv", dtype={"ts_code": "str", "trade_date": "int64",
                                                    "y_ret_1d": "float64"})
    te = te.merge(y, on=["ts_code", "trade_date"], how="left")
    log(f"  测试集 {te.shape}  Y 缺失率 {te.y_ret_1d.isna().mean():.4f}")

    df = pd.concat([tr, te], ignore_index=True)
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    log(f"  合并 {df.shape}  {df.trade_date.min()} ~ {df.trade_date.max()}")
    df.to_parquet(cache)
    return df


# --------------------------------------------------------------------------
# 2. 因子（只用历史）
# --------------------------------------------------------------------------
def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    cache = CACHE / "_probe_factors_full.parquet"
    if cache.exists():
        log(f"复用因子缓存 {cache}")
        return pd.read_parquet(cache)

    g = df.groupby("ts_code", sort=False)
    close = df["close"].to_numpy(dtype="float64")
    # 当日收益（t-1 → t），已知于 t 收盘
    prev = g["close"].shift(1).to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((prev > 0) & np.isfinite(prev) & np.isfinite(close),
                     close / prev - 1.0, np.nan)
    df = df.assign(_r=r)

    out = {"ts_code": df["ts_code"].to_numpy(), "trade_date": df["trade_date"].to_numpy(),
           "y_ret_1d": df["y_ret_1d"].to_numpy(dtype="float64"),
           "flag_limit_up": df["flag_limit_up"].to_numpy(),
           "close": close}

    # rev1：当日收益
    out["rev1"] = r

    # log_price
    out["log_price"] = np.log(np.where(close > 0, close, np.nan))

    # log_amt：过去 20 个有效交易日的成交额均值取对数
    amt = df["amount"].to_numpy(dtype="float64")
    amt_log = np.log(np.where(amt > 0, amt, np.nan))
    s = pd.Series(amt_log)
    out["log_amt"] = s.groupby(df["ts_code"], sort=False).apply(
        lambda x: x.rolling(20, min_periods=10).mean()).droplevel(0).reindex(df.index).to_numpy()

    # vol60：过去 60 日收益标准差
    rs = pd.Series(r)
    out["vol60"] = rs.groupby(df["ts_code"], sort=False).apply(
        lambda x: x.rolling(60, min_periods=30).std()).droplevel(0).reindex(df.index).to_numpy()

    # mom20 / mom60
    for w in (20, 60):
        p = g["close"].shift(w).to_numpy(dtype="float64")
        out[f"mom{w}"] = np.where((p > 0) & np.isfinite(p) & np.isfinite(close),
                                  close / p - 1.0, np.nan)

    fdf = pd.DataFrame(out)
    log(f"因子面板 {fdf.shape}；缺失率 " +
        ", ".join(f"{k}={fdf[k].isna().mean():.3f}" for k in FACTORS))
    fdf.to_parquet(cache)
    return fdf


# --------------------------------------------------------------------------
# 3. 每日工具
# --------------------------------------------------------------------------
def pct_rank(x: np.ndarray) -> np.ndarray:
    """逐行（每日）百分位排名，NaN 保持 NaN。"""
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        m = np.isfinite(row)
        n = int(m.sum())
        if n < 2:
            continue
        order = np.argsort(row[m], kind="mergesort")
        rk = np.empty(n)
        rk[order] = np.arange(n)
        # 并列取平均
        v = row[m]
        srt = v[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and srt[j] == srt[i]:
                j += 1
            rk[order[i:j]] = (i + j - 1) / 2.0
            i = j
        out[t, m] = rk / (n - 1)
    return out


def factor_returns(day: pd.DataFrame) -> tuple[dict[str, float], float]:
    """当日各因子的模仿组合收益（前 30% − 后 30% 等权）与全市场等权收益。"""
    y = day["y_ret_1d"].to_numpy()
    ok = np.isfinite(y) & (day["flag_limit_up"].to_numpy() == 0)
    mkt = float(np.mean(y[ok])) if ok.sum() >= 30 else np.nan
    fr = {}
    for k in FACTORS:
        x = day[k].to_numpy()
        m = ok & np.isfinite(x)
        if m.sum() < 100:
            fr[k] = np.nan
            continue
        xs, ys = x[m], y[m]
        n = len(xs)
        cut = max(int(n * 0.3), 10)
        lo = np.argpartition(xs, cut - 1)[:cut]
        hi = np.argpartition(-xs, cut - 1)[:cut]
        fr[k] = float(np.mean(ys[hi]) - np.mean(ys[lo]))
    return fr, mkt


def basket_daily(day: pd.DataFrame, in_basket: np.ndarray) -> tuple[float, float]:
    """当日名单的等权收益与市场收益（口径同官方：剔涨停 + y 缺失）。"""
    y = day["y_ret_1d"].to_numpy()
    lu = day["flag_limit_up"].to_numpy()
    ok = np.isfinite(y) & (lu == 0)
    if ok.sum() < 100 or in_basket.sum() == 0:
        return np.nan, np.nan
    sel = ok & in_basket
    if sel.sum() == 0:
        return np.nan, float(np.mean(y[ok]))
    return float(np.mean(y[sel])), float(np.mean(y[ok]))


def exposure(day: pd.DataFrame, in_basket: np.ndarray) -> dict[str, float]:
    """名单相对全市场的因子暴露（用当日全市场分位，0.5 为中性）。"""
    out = {}
    for k in FACTORS:
        x = day[k].to_numpy()
        m = np.isfinite(x)
        if m.sum() < 100:
            out[k] = np.nan
            continue
        xs = x[m]
        n = len(xs)
        order = np.argsort(xs, kind="mergesort")
        rk = np.empty(n)
        rk[order] = np.arange(n)
        pr = np.full(x.shape, np.nan)
        pr[m] = rk / (n - 1)
        sel = in_basket & m
        if sel.sum() == 0:
            out[k] = np.nan
        else:
            out[k] = float(np.nanmean(pr[sel]) - 0.5)
    return out


def top_set(day: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """官方 top 集合：剔涨停 + y 缺失后取 pred 最高的 len//10 只。"""
    y = day["y_ret_1d"].to_numpy()
    lu = day["flag_limit_up"].to_numpy()
    ok = np.isfinite(y) & (lu == 0)
    n = int(ok.sum()) // 10
    if n <= 0:
        return np.zeros(len(y), dtype=bool)
    score = np.where(ok & np.isfinite(pred), pred, -np.inf)
    idx = np.argpartition(-score, n - 1)[:n]
    s = np.zeros(len(y), dtype=bool)
    s[idx] = True
    return s & ok


# --------------------------------------------------------------------------
# 4. 主流程：对一段日期区间做归因
# --------------------------------------------------------------------------
def attribute(fdf: pd.DataFrame, pred_by_day: dict, label: str) -> pd.DataFrame:
    """pred_by_day: {tag: DataFrame(index=(date,code) 对齐 fdf, 列 pred)}"""
    dates = np.sort(fdf["trade_date"].unique())
    codes = np.sort(fdf["ts_code"].unique())
    fdf = fdf.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    idx = pd.MultiIndex.from_arrays([fdf["trade_date"], fdf["ts_code"]])

    rows = []
    # 预先按天切块
    bounds = np.searchsorted(fdf["trade_date"].to_numpy(), dates)
    bounds = np.append(bounds, len(fdf))

    for di, d in enumerate(dates):
        sl = slice(bounds[di], bounds[di + 1])
        day = fdf.iloc[sl]
        fr, mkt = factor_returns(day)
        if not np.isfinite(mkt):
            continue
        for tag, pdf in pred_by_day.items():
            pred = pdf["pred"].to_numpy()[sl] if pdf is not None else None
            if pred is None:
                continue
            S = top_set(day, pred)
            br, bm = basket_daily(day, S)
            if not np.isfinite(br):
                continue
            ex = exposure(day, S)
            row = dict(label=label, trade_date=int(d), tag=tag,
                       basket_ret=br, market_ret=bm, excess=br - bm)
            row.update({f"exp_{k}": ex[k] for k in FACTORS})
            row.update({f"fr_{k}": fr[k] for k in FACTORS})
            rows.append(row)
    return pd.DataFrame(rows)


def regress(day_df: pd.DataFrame) -> pd.DataFrame:
    """d_t ~ 1 + 因子收益，返回 α（年化）、β、因子贡献（年化）。"""
    out = []
    for (label, tag), g in day_df.groupby(["label", "tag"]):
        y = g["excess"].to_numpy(dtype="float64")
        cols = [f"fr_{k}" for k in FACTORS]
        X = g[cols].to_numpy(dtype="float64")
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        y, X = y[m], X[m]
        if len(y) < 60:
            continue
        A = np.hstack([np.ones((len(y), 1)), X])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
        fmean = X.mean(axis=0)
        contrib = coef[1:] * fmean
        out.append(dict(label=label, tag=tag, n_days=len(y),
                        ann_excess=float(y.mean()) * 252,
                        alpha_ann=float(coef[0]) * 252,
                        r2=r2,
                        **{f"beta_{k}": float(b) for k, b in zip(FACTORS, coef[1:])},
                        **{f"contrib_{k}": float(c) * 252 for k, c in zip(FACTORS, contrib)}))
    return pd.DataFrame(out)


if __name__ == "__main__":
    log("=== 建价格面板 ===")
    df = build_price_panel()
    log("=== 建因子 ===")
    fdf = build_factors(df)
    fdf.to_parquet(CACHE / "_probe_factors_full.parquet")
    log("完成")

"""根因诊断探针（LH017）：只用 ≤20241231 的 CV 折数据，不接触测试期。

目的：回答「为什么长周期模型在 2021–2024 折上更好、却无法外推」。

核心假设链
----------
H1. 换手被 band 压到下限后，top 1/10 名单每天只换 ~4 只 / 456 只
    ⇒ 名单平均持有期 ~100 个交易日 ⇒ E 本质是「持有约 5 个月的组合」的收益，
    而不是「日频选股能力」。
H2. 因此 E 的年度值由市场状态（横截面分散度、风格轮动）主导，
    跨年 std 与模型间差异同量级 ⇒ 4 个年折的均值区分不了模型。
H3. IC（全池 Spearman）与 E（只吃 top 1/10）可分离 ⇒ 可以分别优化。

产出（均写入 experiments/LH017_*）
---------------------------------
- LH017_fold_metrics.csv      : 折 × 模型 × keep_q 的官方三项 + 名单持有期
- LH017_halfyear_variance.csv : E/IC 按半年窗口切分后的方差结构
- LH017_random_basket.csv     : 随机固定名单的持有超额分布（E 中真 alpha 占比）
- LH017_market_state.csv      : 各折市场状态（分散度 / 完美预测上限 E）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.clean_data import valid_quote  # noqa: E402
from src.evaluation.turnover import band_scores  # noqa: E402

CACHE = ROOT / "outputs/long_horizon"
PRICE_COLUMNS = ["open", "high", "low", "close"]

# 四个 walk-forward 折（validation.folds 里的 fold1=wf2023、fold2=confirm2024）
FOLD_WINDOWS = {
    "wf2021": (20210101, 20211231),
    "wf2022": (20220101, 20221231),
    "wf2023": (20230101, 20231231),
    "confirm2024": (20240101, 20241231),
}
FOLD_STUDY = {  # 每个折去哪个研究目录取预测
    "wf2021": "LH003", "wf2022": "LH003",
    "wf2023": "LH003", "confirm2024": "LH003",
}
MODELS = ["H01", "H05_F2T2", "H10_F2T2", "H20_F2T2", "H30_F2T2"]
IC_MIN_VALID, TOP_MIN_VALID, ANN = 30, 100, 252


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_fold(fold: str) -> pd.DataFrame:
    """读（或复用缓存）某个折的面板：标签 + flag + 全部模型预测，按 (date, code) 排序。"""
    cache = CACHE / f"_probe_rc_{fold}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    start, end = FOLD_WINDOWS[fold]
    usecols = ["ts_code", "trade_date", "flag_limit_up", "y_ret_1d"] + PRICE_COLUMNS
    dtypes = {"ts_code": "str", "trade_date": "int64",
              "flag_limit_up": "int64", "y_ret_1d": "float64",
              **{c: "float64" for c in PRICE_COLUMNS}}
    parts = []
    for chunk in pd.read_csv(ROOT / "data/raw/训练集.csv", usecols=usecols,
                             chunksize=1_000_000, dtype=dtypes):
        mask = (chunk["trade_date"] >= start) & (chunk["trade_date"] <= end)
        if mask.any():
            parts.append(chunk.loc[mask])
    panel = pd.concat(parts, ignore_index=True)

    pred = pd.read_parquet(CACHE / FOLD_STUDY[fold] / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    use = [m for m in MODELS if m in pred.columns]
    df = panel.merge(pred[["ts_code", "trade_date"] + use],
                     on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    df.to_parquet(cache)
    return df


class Panel:
    """把 long 面板摊成 (T × N) 宽矩阵，供向量化评估。"""

    def __init__(self, df: pd.DataFrame):
        self.dates = np.sort(df["trade_date"].unique())
        self.codes = np.sort(df["ts_code"].unique())
        self.T, self.N = len(self.dates), len(self.codes)
        di = {d: i for i, d in enumerate(self.dates)}
        ci = {c: i for i, c in enumerate(self.codes)}
        ti = df["trade_date"].map(di).to_numpy()
        si = df["ts_code"].map(ci).to_numpy()
        self.ti, self.si = ti, si
        self.Y = np.full((self.T, self.N), np.nan)
        self.LU = np.zeros((self.T, self.N), dtype=bool)
        self.Y[ti, si] = df["y_ret_1d"].to_numpy(dtype="float64")
        self.LU[ti, si] = df["flag_limit_up"].to_numpy(dtype="int64") == 1
        self.valid_ic = ~np.isnan(self.Y)                       # 官方 IC：只剔 y 缺失
        self.valid_top = (~self.LU) & (~np.isnan(self.Y))       # 官方超额：剔涨停 + y 缺失
        self.valid_turn = ~self.LU                              # 官方换手：只剔涨停


def to_wide(panel: Panel, values: np.ndarray) -> np.ndarray:
    """把一个与 panel 同序（按 date, code 排序）的长向量摊成 (T × N)。"""
    out = np.full((panel.T, panel.N), np.nan)
    out[panel.ti, panel.si] = np.asarray(values, dtype="float64")
    return out


# --------------------------------------------------------------------------
# 官方口径的向量化评估
# --------------------------------------------------------------------------
def _top_indices(p: np.ndarray, mask: np.ndarray, t: int):
    """当日 top 1/10 的下标（mask 决定有效集合，与官方一致）。"""
    ok = mask[t]
    n_elig = int(ok.sum())
    if n_elig < TOP_MIN_VALID:
        return None, 0
    n_top = max(n_elig // 10, 1)
    v = np.where(ok, p, -np.inf)
    idx = np.argpartition(-v, n_top - 1)[:n_top]
    return idx, n_top


def daily_ic(P: np.ndarray, panel: Panel) -> np.ndarray:
    out = np.full(panel.T, np.nan)
    for t in range(panel.T):
        m = panel.valid_ic[t]
        if m.sum() < IC_MIN_VALID:
            continue
        rp = rankdata(P[t][m])
        ry = rankdata(panel.Y[t][m])
        if rp.std() == 0 or ry.std() == 0:
            continue
        out[t] = float(np.corrcoef(rp, ry)[0, 1])
    return out


def daily_excess(P: np.ndarray, panel: Panel):
    """返回 (top_ret, market_ret, n_top) 逐日序列。"""
    top_ret = np.full(panel.T, np.nan)
    mkt = np.full(panel.T, np.nan)
    n_top = np.zeros(panel.T, dtype=int)
    for t in range(panel.T):
        idx, n = _top_indices(P[t], panel.valid_top, t)
        if idx is None:
            continue
        top_ret[t] = float(panel.Y[t][idx].mean())
        mkt[t] = float(panel.Y[t][panel.valid_top[t]].mean())
        n_top[t] = n
    return top_ret, mkt, n_top


def daily_topsets(P: np.ndarray, panel: Panel) -> np.ndarray:
    """官方换手口径的逐日 top 集合布尔矩阵（只剔涨停）。"""
    S = np.zeros((panel.T, panel.N), dtype=bool)
    for t in range(panel.T):
        idx, _ = _top_indices(P[t], panel.valid_turn, t)
        if idx is None:
            continue
        S[t, idx] = True
    return S


def turnover_from_sets(S: np.ndarray) -> np.ndarray:
    """Jaccard 距离；空集合日之后的第一个交易日不计（官方同口径）。"""
    T = S.shape[0]
    out = np.full(T, np.nan)
    prev = None
    for t in range(T):
        if not S[t].any():
            prev = None
            continue
        if prev is not None and prev.any():
            inter = int((S[t] & prev).sum())
            union = int((S[t] | prev).sum())
            out[t] = 1.0 - inter / union if union else np.nan
        prev = S[t]
    return out


def score(P: np.ndarray, panel: Panel) -> dict:
    ic = daily_ic(P, panel)
    top_ret, mkt, _ = daily_excess(P, panel)
    S = daily_topsets(P, panel)
    tu = turnover_from_sets(S)
    exc = top_ret - mkt
    ok_ic = ~np.isnan(ic); ok_ex = ~np.isnan(exc); ok_tu = ~np.isnan(tu)
    ic_mean = float(np.mean(ic[ok_ic]))
    ann_excess = float(np.mean(exc[ok_ex])) * ANN
    mean_tu = float(np.mean(tu[ok_tu]))
    return {
        "ic_mean": ic_mean,
        "annual_excess": ann_excess,
        "mean_turnover": mean_tu,
        "final_score": ic_mean * 0.4 + ann_excess * 0.3 + (1 - mean_tu) * 0.3,
        "_ic": ic, "_exc": exc, "_tu": tu, "_S": S,
    }


# --------------------------------------------------------------------------
# 名单持有期
# --------------------------------------------------------------------------
def holding_stats(S: np.ndarray) -> dict:
    """从逐日 top 集合追踪：日均替换数、平均持有期、名单规模。"""
    T = S.shape[0]
    reps, sizes = [], []
    for t in range(1, T):
        if not S[t].any() or not S[t - 1].any():
            continue
        reps.append(int((S[t] & ~S[t - 1]).sum()))
        sizes.append(int(S[t].sum()))
    reps = np.asarray(reps, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    n = float(np.median(sizes))
    r = float(np.mean(reps))
    return {
        "list_size": n,
        "daily_replace": r,
        "avg_hold_days": n / r if r > 0 else np.nan,
        "turnover_theory": 2 * r / (n + r) if (n + r) > 0 else np.nan,
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def fast_band(P: np.ndarray, LU: np.ndarray, keep_q: float) -> np.ndarray:
    """留仓带的宽矩阵实现：与 ``turnover.band_scores`` 同逻辑，但跳过 pivot。

    逐位自检见 ``--selfcheck``（在 confirm2024 / H01 上与官方实现比排序一致性）。
    只输出排序编码（留仓股保持全市场分位，落选候选下移 ``1 - keep_q``），
    官方三项指标只依赖排序，因此等价。
    """
    T, N = P.shape
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
    out = rank_full.copy()
    delta = 1.0 - keep_q + 1e-9
    prev: list[int] = []
    tops: list[np.ndarray] = []
    for t in range(T):
        row = rank_elig[t]
        ok = ~np.isnan(row)
        n_elig = int(ok.sum())
        if n_elig < TOP_MIN_VALID:
            prev = []
            tops.append(np.zeros(N, dtype=bool))
            continue
        n_top = max(n_elig // 10, 1)
        keep = [i for i in prev if ok[i] and row[i] >= keep_q]
        if len(keep) > n_top:
            keep = sorted(keep, key=lambda i: -row[i])[:n_top]
        if len(keep) < n_top:
            chosen = set(keep)
            for idx in np.argsort(-np.where(ok, row, -np.inf)):
                if len(keep) >= n_top:
                    break
                j = int(idx)
                if j not in chosen:
                    keep.append(j)
                    chosen.add(j)
        kset = np.zeros(N, dtype=bool)
        kset[keep] = True
        tops.append(kset)
        drop = np.flatnonzero(ok & ~kset)
        out[t, drop] = rank_full[t, drop] - delta
        prev = keep
    return out, np.array(tops)


def band_wide(df: pd.DataFrame, panel: Panel, model: str, keep_q: float) -> np.ndarray:
    """模型 pred → 留仓带编码 → 宽矩阵。df 必须已按 (trade_date, ts_code) 排序。"""
    p = df[model].to_numpy(dtype="float64")
    good = np.isfinite(p)
    if not good.all():
        p = np.where(good, p, np.nanmin(p[good]) - 1.0)
    P = to_wide(panel, p)
    return fast_band(P, panel.LU, keep_q)[0]


def selfcheck() -> None:
    """fast_band 与官方 band_scores 的一致性自检（排序 + 指标）。"""
    df = load_fold("confirm2024")
    panel = Panel(df)
    for model in ("H01", "H20_F2T2"):
        if model not in df.columns:
            continue
        p = df[model].to_numpy(dtype="float64")
        good = np.isfinite(p)
        p = np.where(good, p, np.nanmin(p[good]) - 1.0)
        scored = df[["ts_code", "trade_date", "flag_limit_up"]].assign(pred=p)
        for kq in (0.0022778298112255263, 0.20):
            ref = to_wide(panel, band_scores(scored, kq).to_numpy(dtype="float64"))
            mine, _ = fast_band(to_wide(panel, p), panel.LU, kq)
            a, b = score(ref, panel), score(mine, panel)
            diffs = {k: abs(a[k] - b[k]) for k in
                     ("ic_mean", "annual_excess", "mean_turnover", "final_score")}
            worst = max(diffs.values())
            print(f"  {model} keep_q={kq}: 最大指标差 {worst:.3e} -> "
                  f"{'PASS' if worst < 1e-9 else 'FAIL'} {diffs}")


def raw_wide(df: pd.DataFrame, panel: Panel, model: str) -> np.ndarray:
    p = df[model].to_numpy(dtype="float64")
    good = np.isfinite(p)
    if not good.all():
        p = np.where(good, p, np.nanmin(p[good]) - 1.0)
    return to_wide(panel, p)


def run_fold_metrics(keep_qs=(0.0022778298112255263, 0.10, 0.20)) -> pd.DataFrame:
    rows = []
    for fold in FOLD_WINDOWS:
        df = load_fold(fold)
        panel = Panel(df)
        log(f"{fold}: {panel.T} 天 × {panel.N} 只")
        for model in MODELS:
            if model not in df.columns:
                continue
            P_raw = raw_wide(df, panel, model)
            r = score(P_raw, panel)
            h = holding_stats(r["_S"])
            rows.append({"fold": fold, "model": model, "keep_q": 0.0, "arm": "raw",
                         "ic_mean": r["ic_mean"], "annual_excess": r["annual_excess"],
                         "mean_turnover": r["mean_turnover"],
                         "final_score": r["final_score"], **h})
            log(f"  {model:10s} raw     IC {r['ic_mean']:+.4f} E {r['annual_excess']:+.4f} "
                f"T {r['mean_turnover']:.4f} | 日换 {h['daily_replace']:.1f} 持有 {h['avg_hold_days']:.0f}d")
            for kq in keep_qs:
                P = band_wide(df, panel, model, kq)
                r = score(P, panel)
                h = holding_stats(r["_S"])
                rows.append({"fold": fold, "model": model, "keep_q": kq, "arm": "band",
                             "ic_mean": r["ic_mean"], "annual_excess": r["annual_excess"],
                             "mean_turnover": r["mean_turnover"],
                             "final_score": r["final_score"], **h})
                log(f"  {model:10s} kq{kq:<7.4f} IC {r['ic_mean']:+.4f} E {r['annual_excess']:+.4f} "
                    f"T {r['mean_turnover']:.4f} F {r['final_score']:.4f} | "
                    f"日换 {h['daily_replace']:.1f} 持有 {h['avg_hold_days']:.0f}d")
    return pd.DataFrame(rows)


def run_halfyear(keep_q: float = 0.0022778298112255263) -> pd.DataFrame:
    """把每个折切成 2/4/6 段，看 E 与 IC 的窗口间方差 → 年折 E 的标准误。"""
    rows = []
    for fold in FOLD_WINDOWS:
        df = load_fold(fold)
        panel = Panel(df)
        for model in MODELS:
            if model not in df.columns:
                continue
            P = band_wide(df, panel, model, keep_q)
            r = score(P, panel)
            for k in (1, 2, 4, 6):
                bounds = np.linspace(0, panel.T, k + 1).astype(int)
                ics, exs = [], []
                for b in range(k):
                    sl = slice(bounds[b], bounds[b + 1])
                    ic = np.nanmean(r["_ic"][sl]); ex = np.nanmean(r["_exc"][sl]) * ANN
                    ics.append(ic); exs.append(ex)
                ics = np.asarray(ics); exs = np.asarray(exs)
                rows.append({
                    "fold": fold, "model": model, "n_windows": k,
                    "ic_mean": float(np.mean(ics)),
                    "ic_std_across_windows": float(np.std(ics, ddof=1)) if k > 1 else np.nan,
                    "excess_mean": float(np.mean(exs)),
                    "excess_std_across_windows": float(np.std(exs, ddof=1)) if k > 1 else np.nan,
                })
    return pd.DataFrame(rows)


def run_random_basket(n_draw: int = 600, seed: int = 7) -> pd.DataFrame:
    """随机固定名单的年度持有超额分布 → 实际 E 落在哪个分位（真 alpha 有多少）。

    同时给出「事后最优固定名单」（作弊上界）与「事后最差」作为 E 的可达范围参照。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fold in FOLD_WINDOWS:
        df = load_fold(fold)
        panel = Panel(df)
        daily_mkt = np.array([
            panel.Y[t][panel.valid_top[t]].mean() if panel.valid_top[t].sum() >= TOP_MIN_VALID
            else np.nan for t in range(panel.T)])
        ok = ~np.isnan(daily_mkt)
        mkt = daily_mkt[ok]
        Yv = np.where(panel.valid_top, panel.Y, np.nan)[ok]      # (T', N)
        n_top = int(np.median([int(panel.valid_turn[t].sum()) // 10 for t in range(panel.T)]))
        # 每只股票在「可交易日」上的日均收益（用于排序找最优/最差固定名单）
        per_stock = np.nanmean(Yv, axis=0)                        # (N,)
        best = np.argsort(-per_stock)[:n_top]
        worst = np.argsort(per_stock)[:n_top]
        best_e = float(np.nanmean(np.nanmean(Yv[:, best], axis=1) - mkt)) * ANN
        worst_e = float(np.nanmean(np.nanmean(Yv[:, worst], axis=1) - mkt)) * ANN
        draws = np.empty(n_draw)
        for b in range(n_draw):
            pick = rng.choice(panel.N, size=n_top, replace=False)
            draws[b] = float(np.nanmean(np.nanmean(Yv[:, pick], axis=1) - mkt)) * ANN
        rows.append({
            "fold": fold, "n_top": n_top,
            "rand_basket_mean": float(draws.mean()),
            "rand_basket_std": float(draws.std(ddof=1)),
            "rand_basket_p05": float(np.percentile(draws, 5)),
            "rand_basket_p95": float(np.percentile(draws, 95)),
            "oracle_best_excess": best_e,
            "oracle_worst_excess": worst_e,
        })
        log(f"{fold}: 随机固定名单 {draws.mean():+.4f} ± {draws.std(ddof=1):.4f} "
            f"| 事后最优 {best_e:+.4f} / 事后最差 {worst_e:+.4f} (n={n_top})")
    return pd.DataFrame(rows)


def run_market_state() -> pd.DataFrame:
    """各折市场状态：横截面分散度 + 完美预测能达到的 E 上限。"""
    rows = []
    for fold in FOLD_WINDOWS:
        df = load_fold(fold)
        panel = Panel(df)
        disp, perfect = [], []
        for t in range(panel.T):
            m = panel.valid_top[t]
            if m.sum() < TOP_MIN_VALID:
                continue
            y = panel.Y[t][m]
            n = max(m.sum() // 10, 1)
            disp.append(float(y.std(ddof=1)))
            ys = np.sort(y)[::-1]
            perfect.append(float(ys[:n].mean() - y.mean()))
        rows.append({
            "fold": fold,
            "cross_section_disp": float(np.mean(disp)),
            "perfect_excess_ann": float(np.mean(perfect)) * ANN,
            "market_ann": float(np.nanmean([
                panel.Y[t][panel.valid_top[t]].mean() if panel.valid_top[t].sum() >= TOP_MIN_VALID
                else np.nan for t in range(panel.T)])) * ANN,
        })
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    out = ROOT / "experiments"

    log("=== 1/4 折 × 模型 × keep_q ===")
    m = run_fold_metrics()
    m.to_csv(out / "LH017_fold_metrics.csv", index=False)

    log("=== 2/4 半年/季度窗口方差 ===")
    h = run_halfyear()
    h.to_csv(out / "LH017_window_variance.csv", index=False)

    log("=== 3/4 随机名单 bootstrap ===")
    b = run_random_basket()
    b.to_csv(out / "LH017_random_basket.csv", index=False)

    log("=== 4/4 市场状态 ===")
    s = run_market_state()
    s.to_csv(out / "LH017_market_state.csv", index=False)

    log(f"完成，用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

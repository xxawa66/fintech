"""多周期前瞻标签与训练边界规则（1 日版边界规则的自然推广）。

因果口径：标签 ``y_ret_{h}d = close(t+h)/close(t) - 1`` 只作训练监督目标；
特征全部只使用 t 日及以前的信息（见 ``src/features``），预测时不涉及任何未来
数据。训练窗口末尾 h 个交易日的标签因使用验证期价格而必须剔除，与
``src/evaluation/validation.py`` 对 ``y_ret_1d`` 的边界规则同理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_name(horizon: int) -> str:
    return f"y_ret_{horizon}d"


def forward_labels(raw: pd.DataFrame, horizons) -> pd.DataFrame:
    """在 (ts_code, trade_date) 排序的平衡面板上计算全部前瞻标签。

    面板必须完全平衡：每只股票的交易日序列完全一致（数据集为 4650 只 ×
    全部交易日的平衡面板，清洗后按 KEY 稳定排序，见 ``clean_history``）。
    用全局日历平移实现“第 h 个后续交易日”；close(t) 或 close(t+h) 缺失
    （停牌/未上市/面板末端）时标签为 NaN。返回与 raw 等长等序的 DataFrame。
    """
    n_stocks = int(raw["ts_code"].nunique())
    n_days = int(raw["trade_date"].nunique())
    if len(raw) != n_stocks * n_days:
        raise ValueError(f"面板不平衡：{len(raw):,} 行 != {n_stocks} 只 × {n_days} 日")
    dates = raw["trade_date"].to_numpy().reshape(n_stocks, n_days)
    if not (dates == dates[0]).all():
        raise ValueError("各股票交易日历不一致，无法按全局日历平移。")
    close = raw["close"].to_numpy(dtype="float64").reshape(n_stocks, n_days)
    out = {}
    for h in horizons:
        if not 0 < h < n_days:
            raise ValueError(f"horizon {h} 超出面板范围（交易日 {n_days} 个）")
        fwd = np.full((n_stocks, n_days), np.nan)
        fwd[:, : n_days - h] = close[:, h:]
        out[label_name(h)] = (fwd / close - 1.0).reshape(-1)
    return pd.DataFrame(out, index=raw.index)


def verify_against_official(computed: np.ndarray, official: pd.Series,
                            trade_dates: np.ndarray, tol: float = 1e-12) -> dict:
    """h=1 重算标签与官方 y_ret_1d 逐位核对（排除面板最后一天）。

    面板最后一天（20241231）的官方标签由测试期首日收盘价计算，训练面板内
    无该价格，重算值必然缺失，故排除。其余行上 NaN 模式与数值都必须一致。
    """
    last = trade_dates.max()
    keep = trade_dates != last
    c = np.asarray(computed, dtype="float64")[keep]
    o = official.to_numpy(dtype="float64")[keep]
    nan_mismatch = int((np.isnan(c) != np.isnan(o)).sum())
    both = ~np.isnan(c) & ~np.isnan(o)
    max_diff = float(np.max(np.abs(c[both] - o[both]))) if both.any() else 0.0
    if nan_mismatch or max_diff > tol:
        raise ValueError(
            f"标签口径不一致：NaN 模式差异 {nan_mismatch} 行，最大数值差 {max_diff:.3e}")
    return {"checked_rows": int(keep.sum()), "nan_mismatch": nan_mismatch,
            "max_abs_diff": max_diff}


def training_label_mask(train: pd.DataFrame, label_col: str, dates_sorted: np.ndarray,
                        horizon: int, train_end: int) -> np.ndarray:
    """horizon 日标签的监督样本掩码：标签有限 + 当日报价有效 + 不跨验证边界。

    第 horizon 个后续交易日的日期 > train_end 意味着标签使用了验证期价格，
    这些样本不得进入训练。当日报价有效（quote_valid）与 1 日版
    ``training_arrays`` 的过滤一致。

    窗口类标签（``forward_window_labels``）在 t 日使用第 t..t+h-1 个交易日，
    最后一个被用到的交易日 t+h-1 的一日收益需要 close(t+h)，因此边界规则与
    ``y_ret_{h}d`` **完全相同**，本函数可直接复用。
    """
    y = train[label_col].to_numpy()
    pos = np.searchsorted(dates_sorted, train["trade_date"].to_numpy(), side="left")
    fwd = pos + horizon
    ok = fwd < len(dates_sorted)
    fwd_date = np.where(ok, dates_sorted[np.minimum(fwd, len(dates_sorted) - 1)], -1)
    return (np.isfinite(y) & train["quote_valid"].to_numpy()
            & ok & (fwd_date <= train_end))


# ---------------------------------------------------------------------------
# 窗口一致性标签：衡量「整段窗口里是否一直在前列」，而不是终点单日爆发。
# ``y_ret_{h}d = close(t+h)/close(t)-1`` 只看端点，一只前 29 天平平、最后一天
# 涨停的股票能拿到很高的标签，而「选进去就掉队」的股票在端点上可能看不出来。
# 下列标签把 h 日窗口内的**逐日截面排名**聚合起来（全部只作训练监督目标，
# 不进入特征；预测时仍只用 t 日及以前信息）：
#
# - ``y_meanrank_{h}d``：窗口内每日截面百分位排名的均值（越接近 1 越好）；
# - ``y_topfrac_{h}d`` ：窗口内「当日进入截面 top 10%」的天数占比（∈[0,1]）；
# - ``y_p25rank_{h}d`` ：窗口内每日截面百分位排名的 25 分位（惩罚掉队日）；
# - ``y_avgret_{h}d``  ：窗口内每日收益的算术均值（数值口径对照）。
#
# 窗口内允许有停牌（一日收益缺失），要求至少 ``min_frac`` 比例的交易日可观测，
# 否则标签为 NaN。
# ---------------------------------------------------------------------------

WINDOW_KINDS = ("meanrank", "topfrac", "p25rank", "avgret")


def window_label_name(kind: str, horizon: int) -> str:
    return f"y_{kind}_{horizon}d"


def _forward_cummean(values: np.ndarray, horizon: int,
                     min_periods: int) -> np.ndarray:
    """沿时间轴的前瞻窗口均值（cumsum 向量化，缺失忽略，末尾按可用天数）。"""
    n_stocks, n_days = values.shape
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    csum = np.zeros((n_stocks, n_days + 1))
    ccnt = np.zeros((n_stocks, n_days + 1))
    csum[:, 1:] = np.cumsum(filled, axis=1)
    ccnt[:, 1:] = np.cumsum(valid.astype("float64"), axis=1)
    idx = np.arange(n_days)
    end = np.minimum(idx + horizon, n_days)
    total = csum[:, end] - csum[:, idx]
    count = ccnt[:, end] - ccnt[:, idx]
    out = np.full((n_stocks, n_days), np.nan)
    ok = count >= min_periods
    out[ok] = total[ok] / count[ok]
    return out


def _forward_nanquantile(values: np.ndarray, horizon: int, min_periods: int,
                         q: float, rows_per_chunk: int = 256) -> np.ndarray:
    """沿时间轴的前瞻窗口分位数（滑窗视图 + 行分块，控制峰值内存）。"""
    import warnings

    n_stocks, n_days = values.shape
    out = np.full((n_stocks, n_days), np.nan)
    pad = np.full((n_stocks, horizon - 1), np.nan)
    for start in range(0, n_stocks, rows_per_chunk):
        block = values[start:start + rows_per_chunk]
        padded = np.concatenate([block, np.full((block.shape[0], horizon - 1), np.nan)],
                               axis=1)
        windows = np.lib.stride_tricks.sliding_window_view(padded, horizon, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[start:start + rows_per_chunk] = np.nanquantile(windows, q, axis=2)
    valid = np.isfinite(values)
    ccnt = np.zeros((n_stocks, n_days + 1))
    ccnt[:, 1:] = np.cumsum(valid.astype("float64"), axis=1)
    idx = np.arange(n_days)
    end = np.minimum(idx + horizon, n_days)
    out[ccnt[:, end] - ccnt[:, idx] < min_periods] = np.nan
    return out


def _forward_reduce(values: np.ndarray, horizon: int, min_periods: int,
                    reduce: str) -> np.ndarray:
    """沿时间轴（axis=1）对每个 t 取 t..t+h-1 的前瞻窗口聚合。

    ``reduce`` ∈ {mean, frac_top, p25}；缺失（NaN）不动用，窗口内有效天数
    少于 ``min_periods`` 时输出 NaN。末尾不足 h 天的窗口按实际可用天数计算。
    """
    if reduce in ("mean", "frac_top"):
        return _forward_cummean(values, horizon, min_periods)
    if reduce == "p25":
        return _forward_nanquantile(values, horizon, min_periods, 0.25)
    raise ValueError(f"未知聚合方式 {reduce}")


def _cross_section_rank_pct(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """逐日截面百分位排名（(0,1]，并列取平均秩）；缺失保持 NaN。"""
    frame = pd.DataFrame({"d": dates, "v": values})
    return frame.groupby("d", sort=False, observed=True)["v"].rank(
        pct=True, method="average").to_numpy(dtype="float64")


def forward_window_labels(raw: pd.DataFrame, horizons, kinds=("meanrank", "topfrac",
                                                              "p25rank", "avgret"),
                          min_frac: float = 0.5,
                          top_q: float = 0.9) -> pd.DataFrame:
    """在平衡面板上计算窗口一致性标签（返回与 raw 等长等序的 DataFrame）。

    面板须与 ``forward_labels`` 同样完全平衡（4650 只 × 全部交易日）。
    ``kinds`` 用于只计算需要的种类（每多一列约 60 MB，全量 16 列约 1 GB）。
    """
    unknown = [k for k in kinds if k not in WINDOW_KINDS]
    if unknown:
        raise ValueError(f"未知窗口标签 {unknown}，可选 {WINDOW_KINDS}")
    n_stocks = int(raw["ts_code"].nunique())
    n_days = int(raw["trade_date"].nunique())
    if len(raw) != n_stocks * n_days:
        raise ValueError(f"面板不平衡：{len(raw):,} 行 != {n_stocks} 只 × {n_days} 日")
    dates = raw["trade_date"].to_numpy()
    r1 = raw["y_ret_1d"].to_numpy(dtype="float64")
    need_rank = any(k in kinds for k in ("meanrank", "topfrac", "p25rank"))
    mat_pct = mat_top = None
    if need_rank:
        pct = _cross_section_rank_pct(r1, dates)
        mat_pct = pct.reshape(n_stocks, n_days)
        if "topfrac" in kinds:
            in_top = np.where(np.isfinite(pct), (pct >= top_q).astype("float64"), np.nan)
            mat_top = in_top.reshape(n_stocks, n_days)
        pct = None
    mat_ret = r1.reshape(n_stocks, n_days) if "avgret" in kinds else None
    out: dict[str, np.ndarray] = {}
    for h in horizons:
        if not 0 < h < n_days:
            raise ValueError(f"horizon {h} 超出面板范围（交易日 {n_days} 个）")
        mp = max(2, int(np.ceil(h * min_frac)))
        if "meanrank" in kinds:
            out[window_label_name("meanrank", h)] = _forward_reduce(
                mat_pct, h, mp, "mean").reshape(-1)
        if "topfrac" in kinds:
            out[window_label_name("topfrac", h)] = _forward_reduce(
                mat_top, h, mp, "frac_top").reshape(-1)
        if "p25rank" in kinds:
            out[window_label_name("p25rank", h)] = _forward_reduce(
                mat_pct, h, mp, "p25").reshape(-1)
        if "avgret" in kinds:
            out[window_label_name("avgret", h)] = _forward_reduce(
                mat_ret, h, mp, "mean").reshape(-1)
    return pd.DataFrame(out, index=raw.index)

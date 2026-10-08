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
    """
    y = train[label_col].to_numpy()
    pos = np.searchsorted(dates_sorted, train["trade_date"].to_numpy(), side="left")
    fwd = pos + horizon
    ok = fwd < len(dates_sorted)
    fwd_date = np.where(ok, dates_sorted[np.minimum(fwd, len(dates_sorted) - 1)], -1)
    return (np.isfinite(y) & train["quote_valid"].to_numpy()
            & ok & (fwd_date <= train_end))

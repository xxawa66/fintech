"""Long-horizon incremental features (strictly backward-looking).

V1 的 40 个特征窗口最长 60 个交易日，对 5–30 日的前瞻预测来说偏短。本模块
在 V1 之上补一层**长周期专属**特征（120/250 个交易日量级），全部只使用 t 日
及以前的收盘/成交量信息，与 ``src/features/build_features.py`` 同一约定：

- 逐股在日历行序上计算（未上市/停牌的 NaN 价格行算作行，不跳行）；
- 滚动窗口一律 ``min_periods=w``，即窗口内有效观测不足则为 NaN；
- 除数用 ``safe_ratio`` 保护，非正值不参与；
- 最后对若干长周期特征做**逐日截面**百分位排名（同样只用到 t 日截面）。

命名统一加 ``lh_`` 前缀，避免与 V1 的 40 个特征冲突。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.load_data import KEYS, X_COLUMNS
from src.features.cross_section_features import percentile_rank
from src.features.price_features import safe_ratio
from src.data.clean_data import valid_quote

CANDLES = ["close", "high", "low", "open", "vol", "amount"]


def long_feature_names(settings: dict) -> list[str]:
    """按配置枚举长周期特征名，并做数量/去重自检。"""
    names: list[str] = []
    names += [f"lh_ret_{w}" for w in settings["long_return_windows"]]
    names += [f"lh_ret_{w}_skip{s}" for w, s in settings["skip_windows"]]
    names += [f"lh_ma_bias_{w}" for w in settings["long_ma_windows"]]
    names += [f"lh_volatility_{w}" for w in settings["long_volatility_windows"]]
    names += [f"lh_price_position_{w}" for w in settings["long_position_windows"]]
    names += [f"lh_vol_ratio_{w}" for w in settings["long_volume_windows"]]
    names += [f"lh_amount_ratio_{w}" for w in settings["long_amount_windows"]]
    names += [f"lh_up_days_{w}" for w in settings["trend_windows"]]
    names += [f"lh_drawdown_{w}" for w in settings["drawdown_windows"]]
    names += [f"lh_skew_{w}" for w in settings["skew_windows"]]
    names += [f"rank_{source}" for source in settings["rank_sources"]]
    if len(names) != settings["expected_count"] or len(set(names)) != len(names):
        raise ValueError(f"长周期特征数量或命名不符：{len(names)} vs {settings['expected_count']}")
    return names


def _stock_long_features(stock: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    close, high, low = stock["close"], stock["high"], stock["low"]
    vol, amount = stock["vol"], stock["amount"]
    ret_1 = safe_ratio(close, close.shift(1)) - 1
    values: dict[str, pd.Series] = {}

    for w in settings["long_return_windows"]:
        values[f"lh_ret_{w}"] = safe_ratio(close, close.shift(w)) - 1
    # 跳月动量：跳过最近 s 日，取更早一段的累计收益（避开短期反转污染）
    for w, s in settings["skip_windows"]:
        values[f"lh_ret_{w}_skip{s}"] = safe_ratio(close.shift(s), close.shift(w)) - 1
    for w in settings["long_ma_windows"]:
        values[f"lh_ma_bias_{w}"] = safe_ratio(close, close.rolling(w, min_periods=w).mean()) - 1
    for w in settings["long_volatility_windows"]:
        values[f"lh_volatility_{w}"] = ret_1.rolling(w, min_periods=w).std(ddof=1)
    for w in settings["long_position_windows"]:
        minimum = close.rolling(w, min_periods=w).min()
        maximum = close.rolling(w, min_periods=w).max()
        values[f"lh_price_position_{w}"] = safe_ratio(close - minimum, maximum - minimum)
    for w in settings["long_volume_windows"]:
        values[f"lh_vol_ratio_{w}"] = safe_ratio(vol, vol.rolling(w, min_periods=w).mean())
    for w in settings["long_amount_windows"]:
        values[f"lh_amount_ratio_{w}"] = safe_ratio(amount, amount.rolling(w, min_periods=w).mean())
    for w in settings["trend_windows"]:
        values[f"lh_up_days_{w}"] = (ret_1 > 0).rolling(w, min_periods=w).mean()
    for w in settings["drawdown_windows"]:
        values[f"lh_drawdown_{w}"] = safe_ratio(close, close.rolling(w, min_periods=w).max()) - 1
    for w in settings["skew_windows"]:
        values[f"lh_skew_{w}"] = ret_1.rolling(w, min_periods=w).skew()
    return values


def build_long_features(x: pd.DataFrame, settings: dict,
                        progress: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """输入清洗后的 KEYS + X_COLUMNS，返回 KEYS + 长周期特征。"""
    if set(x.columns) != set(KEYS + X_COLUMNS):
        raise ValueError("长周期特征输入必须恰好是 keys 与原始 X，不含任何标签。")
    if x.duplicated(KEYS).any():
        raise ValueError("长周期特征输入键不唯一。")
    frame = x.sort_values(KEYS, kind="stable").reset_index(drop=True)
    names = long_feature_names(settings)
    positions = {name: i for i, name in enumerate(names)}
    matrix = np.full((len(frame), len(names)), np.nan, dtype="float32")
    groups = frame.groupby("ts_code", sort=False, observed=True)
    total = frame["ts_code"].nunique()
    for done, (_, stock) in enumerate(groups, 1):
        rows = stock.index.to_numpy()
        for name, series in _stock_long_features(stock, settings).items():
            matrix[rows, positions[name]] = series.to_numpy(dtype="float32")
        if progress and (done % 1000 == 0 or done == total):
            progress(f"long-horizon feature stocks {done}/{total}")
    eligible = valid_quote(frame)
    for source in settings["rank_sources"]:
        rank = percentile_rank(pd.Series(matrix[:, positions[source]], index=frame.index),
                               frame["trade_date"], eligible)
        matrix[:, positions[f"rank_{source}"]] = rank.to_numpy(dtype="float32")
    matrix[np.isinf(matrix)] = np.nan
    result = pd.concat([frame[KEYS], pd.DataFrame(matrix, columns=names, copy=False)], axis=1)
    return result, names

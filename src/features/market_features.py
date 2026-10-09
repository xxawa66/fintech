"""S007 市场状态特征：日度横截面聚合 + 滚动市场体制统计（只广播，不新增控制器层）。

设计约束（与 AGENTS.md、docs/validation.py 一致）：

- 只使用赛题允许字段（open/high/low/close/vol/amount/flag_limit_up/flag_limit_down）
  及其派生的 V1 特征；全部统计只使用 t 日及以前信息。
- 横截面聚合在每个交易日内完成（当日全体可成交股票），rolling 统计只向过去取
  ``window`` 个交易日，因此对任意验证终点都是因果的。
- 市场特征在当日对所有股票取同一值（广播）；树模型可以在市场特征上先分裂、
  再在个股特征上分裂，等价于显式的 regime 上下文，无需新增 controller 嵌套。
- 市场收益用 V1 的 ``ret_1``（close_t/close_{t-1}-1，向后收益）的横截面统计，
  不使用 ``y_ret_1d``（那是 t+1 的前向收益，用作特征即泄漏）。

特征清单（11 个基础 + 3 个与个股动量/反转的交互，交互在研究脚本中于合并后计算）：

- ``mkt_ret_mean`` / ``mkt_ret_median`` / ``mkt_ret_std``：当日全市场（可成交）
  ret_1 的均值 / 中位数 / 横截面离散度（std, ddof=1）。
- ``mkt_adv_ratio``：当日上涨股票比例（ret_1 > 0）。
- ``mkt_limit_up_ratio`` / ``mkt_limit_down_ratio``：当日涨停 / 跌停股票比例
  （分母为可成交股票）。
- ``mkt_amount_chg``：全市场成交额对数变化 ln(Σamount_t / Σamount_{t-1})。
- ``mkt_ret_mean_20``：市场日收益的 20 日均值（市场趋势）。
- ``mkt_ret_vol_20``：市场日收益的 20 日标准差（市场波动体制）。
- ``mkt_breadth_20``：上涨比例的 20 日均值（市场宽度体制）。
- ``mkt_mom_20``：20 日市场累积收益（按日复利合成）。

rolling 统一 ``min_periods=window``：2018 年前 19 个交易日为 NaN，由 LightGBM
的缺失值处理吸收，不影响 2021–2024 各折。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.clean_data import valid_quote
from src.data.load_data import KEYS

MARKET_WINDOW = 20
MARKET_BASE = [
    "mkt_ret_mean", "mkt_ret_median", "mkt_ret_std", "mkt_adv_ratio",
    "mkt_limit_up_ratio", "mkt_limit_down_ratio", "mkt_amount_chg",
    "mkt_ret_mean_20", "mkt_ret_vol_20", "mkt_breadth_20", "mkt_mom_20",
]
MARKET_INTERACTION = ["mkt_x_mom_vol20", "mkt_x_mom_breadth", "mkt_x_rev_mktmom"]
MARKET_ALL = MARKET_BASE + MARKET_INTERACTION


def market_daily_table(features: pd.DataFrame, raw: pd.DataFrame,
                       window: int = MARKET_WINDOW) -> pd.DataFrame:
    """从 V1 特征帧与对齐的原始行计算日度市场表（每交易日一行，按日期升序）。

    ``features``：KEYS + V1 40 特征（需含 ``ret_1``），行序与 ``raw`` 完全一致
    （调用方用 ``features[KEYS].equals(raw[KEYS])`` 保证）。停牌等不可成交行
    （四价不全或非正）不参与任何聚合。
    """
    if len(features) != len(raw) or not features[KEYS].equals(raw[KEYS]):
        raise ValueError("Market features require raw rows aligned with the feature frame.")
    valid = valid_quote(raw).to_numpy()
    ret1 = features["ret_1"].to_numpy(dtype="float64")
    daily = pd.DataFrame({
        "trade_date": features["trade_date"].to_numpy(),
        "ret_1": np.where(valid, ret1, np.nan),
        "limit_up": np.where(valid, raw["flag_limit_up"].to_numpy(), 0).astype("float64"),
        "limit_down": np.where(valid, raw["flag_limit_down"].to_numpy(), 0).astype("float64"),
        "amount": np.where(valid, raw["amount"].to_numpy(), np.nan),
    })
    grouped = daily.groupby("trade_date", sort=True, observed=True)
    table = pd.DataFrame({
        "mkt_ret_mean": grouped["ret_1"].mean(),
        "mkt_ret_median": grouped["ret_1"].median(),
        "mkt_ret_std": grouped["ret_1"].std(ddof=1),
        "mkt_adv_ratio": grouped["ret_1"].apply(
            lambda s: s.gt(0).sum() / s.notna().sum() if s.notna().any() else np.nan),
        "mkt_limit_up_ratio": grouped["limit_up"].mean(),
        "mkt_limit_down_ratio": grouped["limit_down"].mean(),
        "market_amount": grouped["amount"].sum(min_count=1),
    }).reset_index()
    table["mkt_amount_chg"] = np.log(table["market_amount"] / table["market_amount"].shift(1))
    roll = table["mkt_ret_mean"].rolling(window, min_periods=window)
    table["mkt_ret_mean_20"] = roll.mean()
    table["mkt_ret_vol_20"] = roll.std(ddof=1)
    table["mkt_breadth_20"] = table["mkt_adv_ratio"].rolling(window, min_periods=window).mean()
    table["mkt_mom_20"] = np.expm1(
        np.log1p(table["mkt_ret_mean"]).rolling(window, min_periods=window).sum())
    table = table.drop(columns=["market_amount"])
    for name in MARKET_BASE:
        table[name] = table[name].astype("float32")
    if not np.isinf(table[MARKET_BASE].to_numpy()).any():
        return table
    raise ValueError("Market feature table contains infinity.")


def market_interactions(dataset: pd.DataFrame) -> pd.DataFrame:
    """在市场表广播合并后的数据帧上计算三个显式交互特征（返回新列帧）。

    - ``mkt_x_mom_vol20``：个股 20 日动量 × 市场波动体制（高波动下动量的含义不同）。
    - ``mkt_x_mom_breadth``：个股 20 日动量 × 市场宽度体制。
    - ``mkt_x_rev_mktmom``：个股 1 日反转 × 市场 20 日动量体制。
    """
    if not set(MARKET_BASE) <= set(dataset.columns):
        raise ValueError("Market base features must be merged before interactions.")
    interaction = pd.DataFrame(index=dataset.index)
    interaction["mkt_x_mom_vol20"] = dataset["ret_20"] * dataset["mkt_ret_vol_20"]
    interaction["mkt_x_mom_breadth"] = dataset["ret_20"] * dataset["mkt_breadth_20"]
    interaction["mkt_x_rev_mktmom"] = dataset["ret_1"] * dataset["mkt_mom_20"]
    for name in MARKET_INTERACTION:
        interaction[name] = interaction[name].astype("float32")
    if np.isinf(interaction.to_numpy()).any():
        raise ValueError("Market interaction features contain infinity.")
    return interaction

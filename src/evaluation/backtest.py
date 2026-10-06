"""Top 组合分析（成员 B）。

对固定验证预测做 Top 组合诊断：Top 10%（官方口径）、Top 20%、Bottom 10% 的
逐日等权收益、市场基准收益、超额与多空价差；以及累计净值、月度超额和
最大回撤。配合 ``notebooks/04_result_visualization.ipynb`` 出报告图表。

口径约定（与 ``src/evaluation/official_eval.py`` 的官方 Top 指标逐字一致）：

- 过滤：``flag_limit_up == 0`` 且 ``y_ret_1d`` 非缺失；有效样本 < 100 的交易日跳过；
- 排序：``pred`` 降序，``n_top = max(有效数 // 10, 1)``；
- ``top10_ret`` 为 Top 组等权日收益，``market_ret`` 为同一过滤后全体等权日收益，
  ``top10_excess = top10_ret - market_ret``；
- 年化：日均 × ``ANNUALIZATION_DAYS``（252）。

扩展口径（非官方、仅诊断）：Top 20%（前 1/5）、Bottom 10%（后 1/10）、
``top_bottom_spread = top10_ret - bottom10_ret``。Bottom 组使用与 Top 组相同的
过滤条件，仅方向不同。

CLI 示例::

    python -m src.evaluation.backtest \
        --pred outputs/predictions/E001_baseline_repeat/valid_2024.csv \
        --labels outputs/metrics/E001_baseline_repeat/validation_labels.csv \
        --daily-out outputs/metrics/E001_baseline_repeat/top_analysis_daily.csv \
        --monthly-out outputs/metrics/E001_baseline_repeat/top_analysis_monthly.csv \
        --summary-out outputs/metrics/E001_baseline_repeat/top_analysis_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.official_eval import ANNUALIZATION_DAYS, TOP_MIN_VALID

ROOT = Path(__file__).resolve().parents[2]


def load_scored(pred_path: str | Path, labels_path: str | Path,
                start: int | None = None, end: int | None = None) -> pd.DataFrame:
    """合并预测与标签为评分帧（inner join，键唯一性校验）。"""
    pred = pd.read_csv(pred_path, dtype={"trade_date": int})
    labels = pd.read_csv(labels_path, dtype={"trade_date": int})
    if pred.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("预测文件存在重复键")
    if labels.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("标签文件存在重复键")
    df = pred.merge(labels, on=["ts_code", "trade_date"], how="inner",
                    validate="one_to_one")
    if start is not None:
        df = df[df["trade_date"] >= start]
    if end is not None:
        df = df[df["trade_date"] <= end]
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def daily_portfolio_returns(scored: pd.DataFrame) -> pd.DataFrame:
    """逐日组合收益表。

    返回列：``trade_date, n_valid, top10_ret, top20_ret, bottom10_ret,
    market_ret, top10_excess, top20_excess, top_bottom_spread``。
    官方过滤条件下有效样本 < 100 的交易日整行剔除（与官方跳过日一致）。
    """
    rows: list[dict] = []
    for m_date, group in scored.groupby("trade_date"):
        valid = group[
            (group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()
        ].sort_values("pred", ascending=False).reset_index(drop=True)
        if len(valid) < TOP_MIN_VALID:
            continue
        n10 = max(len(valid) // 10, 1)
        n20 = max(len(valid) // 5, 1)
        top10_ret = float(valid["y_ret_1d"].iloc[:n10].mean())
        top20_ret = float(valid["y_ret_1d"].iloc[:n20].mean())
        bottom10_ret = float(valid["y_ret_1d"].iloc[-n10:].mean())
        market_ret = float(valid["y_ret_1d"].mean())
        rows.append({
            "trade_date": int(m_date),
            "n_valid": int(len(valid)),
            "top10_ret": top10_ret,
            "top20_ret": top20_ret,
            "bottom10_ret": bottom10_ret,
            "market_ret": market_ret,
            "top10_excess": top10_ret - market_ret,
            "top20_excess": top20_ret - market_ret,
            "top_bottom_spread": top10_ret - bottom10_ret,
        })
    return pd.DataFrame(rows)


def monthly_excess(daily: pd.DataFrame) -> pd.DataFrame:
    """月度汇总：日均超额年化、日超额为正比例、Top 组与市场月度年化收益。"""
    frame = daily.copy()
    frame["month"] = frame["trade_date"].astype(str).str[:6]
    rows = []
    for month, group in frame.groupby("month"):
        rows.append({
            "month": month,
            "n_days": len(group),
            "excess_annual": group["top10_excess"].mean() * ANNUALIZATION_DAYS,
            "excess_pos_ratio": (group["top10_excess"] > 0).mean(),
            "top10_annual": group["top10_ret"].mean() * ANNUALIZATION_DAYS,
            "market_annual": group["market_ret"].mean() * ANNUALIZATION_DAYS,
            "spread_annual": group["top_bottom_spread"].mean() * ANNUALIZATION_DAYS,
        })
    return pd.DataFrame(rows)


def _max_drawdown(cum: pd.Series) -> float:
    """累计净值序列的最大回撤（正值表示回撤幅度）。"""
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return float(-drawdown.min())


def summarize(daily: pd.DataFrame) -> dict:
    """汇总指标：年化收益、年化超额、正超额天数比例、累计与回撤。"""
    cum_top10 = (1.0 + daily["top10_ret"]).cumprod()
    cum_market = (1.0 + daily["market_ret"]).cumprod()
    cum_bottom10 = (1.0 + daily["bottom10_ret"]).cumprod()
    rel = cum_top10 / cum_market
    return {
        "n_days": int(len(daily)),
        "annualization_days": ANNUALIZATION_DAYS,
        "top10_annual_ret": float(daily["top10_ret"].mean() * ANNUALIZATION_DAYS),
        "top20_annual_ret": float(daily["top20_ret"].mean() * ANNUALIZATION_DAYS),
        "bottom10_annual_ret": float(daily["bottom10_ret"].mean() * ANNUALIZATION_DAYS),
        "market_annual_ret": float(daily["market_ret"].mean() * ANNUALIZATION_DAYS),
        "annual_excess": float(daily["top10_excess"].mean() * ANNUALIZATION_DAYS),
        "top20_annual_excess": float(daily["top20_excess"].mean() * ANNUALIZATION_DAYS),
        "top_bottom_annual_spread": float(
            daily["top_bottom_spread"].mean() * ANNUALIZATION_DAYS),
        "excess_positive_ratio": float((daily["top10_excess"] > 0).mean()),
        "cum_top10_return": float(cum_top10.iloc[-1] - 1.0),
        "cum_market_return": float(cum_market.iloc[-1] - 1.0),
        "cum_bottom10_return": float(cum_bottom10.iloc[-1] - 1.0),
        "cum_relative_return": float(rel.iloc[-1] - 1.0),
        "top10_max_drawdown": _max_drawdown(cum_top10),
        "relative_max_drawdown": _max_drawdown(rel),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Top 组合分析（官方口径 Top10%）")
    parser.add_argument("--pred", required=True, help="预测 CSV（ts_code,trade_date,pred）")
    parser.add_argument("--labels", required=True,
                        help="标签 CSV（ts_code,trade_date,y_ret_1d,flag_limit_up）")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--daily-out", default=None)
    parser.add_argument("--monthly-out", default=None)
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args(argv)

    scored = load_scored(args.pred, args.labels, args.start, args.end)
    daily = daily_portfolio_returns(scored)
    if daily.empty:
        raise SystemExit("有效交易日为 0：检查预测/标签合并与日期区间")

    summary = summarize(daily)
    monthly = monthly_excess(daily)

    print(f"交易日 {summary['n_days']}（有效样本 >= {TOP_MIN_VALID} 的日）")
    print("\n===== Top 组合分析汇总（官方口径 Top10%）=====")
    for key, value in summary.items():
        if isinstance(value, int):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}")
    print("\n===== 月度超额（年化）=====")
    print(monthly.to_string(index=False))

    for out, obj in ((args.daily_out, daily), (args.monthly_out, monthly)):
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            obj.to_csv(out, index=False)
            print(f"\n已写入 {out}")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"已写入 {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

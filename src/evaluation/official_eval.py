"""本地验证评分适配器：复用根目录官方 ``evaluate.py`` 的指标口径。

官方指标定义（与根目录 ``evaluate.py`` 完全一致，不得修改）：

1. Rank IC（权重 40%）：逐交易日 ``spearmanr(pred, y_ret_1d)``，仅剔除 y 缺失行
   （不剔除涨停）；当日有效样本 < 30 时跳过该日。
   ``ic_std`` 为样本标准差（ddof=1），``icir = ic_mean / ic_std``（std<=0 时取 0）。
2. Top 组超额收益（权重 30%）：逐日剔除涨停与 y 缺失；有效样本 < 100 跳过；
   按 pred 降序取前 1/10（``n_top = max(len // 10, 1)``）等权平均收益减全市场均值，
   日均超额 × 252 年化；``top1_annual_ret`` 为 Top 组日均绝对收益 × 252。
3. 预测换手率（权重 30%）：逐日仅剔除涨停（不要求 y 非缺失）；有效样本 < 100 时
   该日不计换手并将前一日集合重置为空；Top 集合的逐日 Jaccard 距离
   ``1 - |交| / |并|`` 取平均。
4. ``final_score = 0.4 * ic_mean + 0.3 * annual_excess + 0.3 * (1 - mean_turnover)``。

与官方版本的唯一差异是数据入口：官方读测试集_Y / 测试集_X；本模块接受本地验证
预测（``ts_code, trade_date, pred``）与标签来源（训练集切分或自带标签列的验证文件），
合并方式与官方相同（键 inner merge）。

用法（在仓库根目录运行）::

    python -m src.evaluation.official_eval --pred outputs/predictions/e000_valid.csv \
        --label-source data/raw/训练集.csv --start 20240101 --end 20241231 \
        --daily-out outputs/metrics/e000_daily.csv --metrics-out outputs/metrics/e000_metrics.json

若预测文件本身含 ``y_ret_1d`` 和 ``flag_limit_up`` 列，可省略 ``--label-source``。

口径核对::

    python -m src.evaluation.official_eval --selfcheck

在合成数据上同时运行根目录官方评分器与本模块，逐项断言 8 个官方指标一致。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KEY_COLUMNS = ["ts_code", "trade_date"]
OFFICIAL_METRICS = [
    "ic_mean", "ic_std", "icir", "ic_positive_ratio",
    "annual_excess", "top1_annual_ret", "mean_turnover", "final_score",
]

# 与官方 evaluate.py 相同的阈值
IC_MIN_VALID = 30
TOP_MIN_VALID = 100
ANNUALIZATION_DAYS = 252


def evaluate_frame(df: pd.DataFrame) -> dict:
    """对已合并的验证数据计算官方全部指标。

    ``df`` 必须含列 ``ts_code, trade_date, pred, y_ret_1d, flag_limit_up``，
    且已按官方方式完成键合并（inner）。逐日循环逻辑逐行对应根目录 ``evaluate.py``。
    """
    ic_list: list[float] = []
    excess_list: list[float] = []
    top1_ret_list: list[float] = []
    turnover_list: list[float] = []
    prev_set: set | None = None

    # groupby('trade_date') 默认按日期升序，与官方一致
    for _, m_group in df.groupby("trade_date"):
        # --- 1. Rank IC：仅剔除 y 缺失，<30 跳过 ---
        valid_ic = m_group.dropna(subset=["y_ret_1d"])
        if len(valid_ic) >= IC_MIN_VALID:
            ic, _ = spearmanr(valid_ic["pred"], valid_ic["y_ret_1d"])
            ic_list.append(ic)

        # --- 2. Top 超额：剔除涨停与 y 缺失，<100 跳过 ---
        valid_top = m_group[
            (m_group["flag_limit_up"] == 0) & m_group["y_ret_1d"].notna()
        ].copy()
        if len(valid_top) >= TOP_MIN_VALID:
            valid_top = valid_top.sort_values("pred", ascending=False).reset_index(drop=True)
            n_top = max(len(valid_top) // 10, 1)
            top1_ret = valid_top["y_ret_1d"].iloc[:n_top].mean()
            market_ret = valid_top["y_ret_1d"].mean()
            top1_ret_list.append(top1_ret)
            excess_list.append(top1_ret - market_ret)

        # --- 3. 换手率：仅剔除涨停（不要求 y），<100 重置前日集合 ---
        valid_turn = m_group[m_group["flag_limit_up"] == 0].copy()
        if len(valid_turn) < TOP_MIN_VALID:
            prev_set = None
            continue
        valid_turn = valid_turn.sort_values("pred", ascending=False)
        n_top = max(len(valid_turn) // 10, 1)
        curr_set = set(valid_turn["ts_code"].iloc[:n_top])
        if prev_set is not None and len(prev_set) > 0:
            intersection = len(curr_set & prev_set)
            union = len(curr_set | prev_set)
            turnover_list.append(1.0 - intersection / union)
        prev_set = curr_set

    if not ic_list or not excess_list or not turnover_list:
        raise ValueError(
            f"有效交易日不足：ic {len(ic_list)} 天 / top {len(excess_list)} 天 / "
            f"turnover {len(turnover_list)} 天。请检查日期范围与预测覆盖。"
        )

    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list, ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_positive_ratio = np.mean(np.array(ic_list) > 0)
    annual_excess = np.mean(excess_list) * ANNUALIZATION_DAYS
    top1_annual_ret = np.mean(top1_ret_list) * ANNUALIZATION_DAYS
    mean_turnover = np.mean(turnover_list)
    final_score = ic_mean * 0.4 + annual_excess * 0.3 + (1 - mean_turnover) * 0.3

    return {
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "icir": float(icir),
        "ic_positive_ratio": float(ic_positive_ratio),
        "annual_excess": float(annual_excess),
        "top1_annual_ret": float(top1_annual_ret),
        "mean_turnover": float(mean_turnover),
        "final_score": float(final_score),
        "n_days_ic": len(ic_list),
        "n_days_top": len(excess_list),
        "n_days_turnover": len(turnover_list),
    }


def daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """逐日指标表：ic / top 超额 / 换手率 / 样本数，供归因分析使用。

    列定义与官方口径一致：``ic`` 当日 Rank IC（有效样本 <30 为 NaN）；
    ``top_excess`` 当日 Top 组超额（有效样本 <100 为 NaN）；``turnover``
    当日 Top 集合与前一日（跳过日重置后为 NaN）的 Jaccard 距离。
    """
    rows = []
    prev_set: set | None = None
    for m_date, m_group in df.groupby("trade_date"):
        valid_ic = m_group.dropna(subset=["y_ret_1d"])
        ic = np.nan
        if len(valid_ic) >= IC_MIN_VALID:
            ic, _ = spearmanr(valid_ic["pred"], valid_ic["y_ret_1d"])
            ic = float(ic)

        valid_top = m_group[
            (m_group["flag_limit_up"] == 0) & m_group["y_ret_1d"].notna()
        ].copy()
        top_excess = np.nan
        top1_ret = np.nan
        market_ret = np.nan
        if len(valid_top) >= TOP_MIN_VALID:
            valid_top = valid_top.sort_values("pred", ascending=False).reset_index(drop=True)
            n_top = max(len(valid_top) // 10, 1)
            top1_ret = float(valid_top["y_ret_1d"].iloc[:n_top].mean())
            market_ret = float(valid_top["y_ret_1d"].mean())
            top_excess = top1_ret - market_ret

        valid_turn = m_group[m_group["flag_limit_up"] == 0].copy()
        turnover = np.nan
        if len(valid_turn) < TOP_MIN_VALID:
            prev_set = None
        else:
            valid_turn = valid_turn.sort_values("pred", ascending=False)
            n_top = max(len(valid_turn) // 10, 1)
            curr_set = set(valid_turn["ts_code"].iloc[:n_top])
            if prev_set is not None and len(prev_set) > 0:
                turnover = float(1.0 - len(curr_set & prev_set) / len(curr_set | prev_set))
            prev_set = curr_set

        rows.append({
            "trade_date": m_date,
            "ic": ic,
            "top_excess": top_excess,
            "top1_ret": top1_ret,
            "market_ret": market_ret,
            "turnover": turnover,
            "n_valid_ic": int(len(valid_ic)),
            "n_valid_top": int(len(valid_top)),
        })
    return pd.DataFrame(rows)


def load_validation_frame(
    pred_path: str | Path,
    label_source: str | Path | None = None,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """读取预测文件并按官方方式合并标签，返回 evaluate_frame / daily_metrics 的输入。

    预测文件必须含 ``ts_code, trade_date, pred``；若同时含 ``y_ret_1d`` 和
    ``flag_limit_up`` 则视为自带标签。否则从 ``label_source``（如训练集 CSV，
    需含同名列）按键 inner merge，可选 ``start`` / ``end``（YYYYMMDD 整数，
    闭区间）限定验证日期范围。合并覆盖率会打印到 stderr 供核对。
    """
    pred_path = Path(pred_path)
    df_pred = pd.read_csv(pred_path)
    missing = {"ts_code", "trade_date", "pred"} - set(df_pred.columns)
    if missing:
        raise ValueError(f"预测文件 {pred_path} 缺少列: {sorted(missing)}")

    has_labels = {"y_ret_1d", "flag_limit_up"} <= set(df_pred.columns)
    if not has_labels:
        if label_source is None:
            raise ValueError("预测文件不含标签列，必须提供 --label-source")
        df_lab = pd.read_csv(
            label_source,
            usecols=["ts_code", "trade_date", "y_ret_1d", "flag_limit_up"],
        )
        if start is not None:
            df_lab = df_lab[df_lab["trade_date"] >= start]
        if end is not None:
            df_lab = df_lab[df_lab["trade_date"] <= end]
        df = df_pred.merge(df_lab, on=KEY_COLUMNS, how="inner")
        n_pred, n_lab = len(df_pred), len(df_lab)
    else:
        df = df_pred
        if start is not None:
            df = df[df["trade_date"] >= start]
        if end is not None:
            df = df[df["trade_date"] <= end]
        n_pred, n_lab = len(df_pred), len(df)

    n_missing = len(df_pred) - (df_pred.merge(df[KEY_COLUMNS].drop_duplicates(),
                                              on=KEY_COLUMNS, how="inner").shape[0])
    print(
        f"预测行数 {n_pred:,}，标签行数 {n_lab:,}，合并后 {len(df):,}，"
        f"交易日数 {df['trade_date'].nunique()}，预测中无标签的行 {n_missing:,}",
        file=sys.stderr,
    )
    if df["trade_date"].nunique() == 0:
        raise ValueError("合并后为空：请检查日期范围与键匹配。")
    return df


def _load_official_evaluator(repo_root: Path):
    """以模块方式加载根目录官方 evaluate.py（保持官方文件原样，不导入安装）。"""
    spec = importlib.util.spec_from_file_location("official_evaluate", repo_root / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selfcheck(repo_root: Path) -> dict:
    """在合成数据上核对本模块与官方 evaluate.py 的口径一致性。

    构造含涨停、y 缺失、有效样本不足（触发换手重置）的平衡面板，
    分别用官方评分器与本模块评分，逐项断言 8 个官方指标在 1e-12 内一致。
    """
    rng = np.random.default_rng(7)
    n_stocks, n_days = 120, 40
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = 20240100 + np.arange(1, n_days + 1)  # 20240101..20240209 样式整数

    recs = []
    for d in dates:
        limit_up = rng.random(n_stocks) < 0.05
        # 第 10、25 个交易日大面积涨停，触发官方"有效样本<100 → 前日集合重置"分支
        if d in (dates[10], dates[25]):
            limit_up = rng.random(n_stocks) < 0.9
        y_missing = rng.random(n_stocks) < 0.03
        y = rng.normal(0, 0.02, n_stocks)
        pred = 0.3 * y + rng.normal(0, 0.05, n_stocks)  # 弱正相关，IC 为正
        for i in range(n_stocks):
            recs.append({
                "ts_code": stocks[i],
                "trade_date": int(d),
                "flag_limit_up": int(limit_up[i]),
                "y_ret_1d": np.nan if y_missing[i] else y[i],
                "pred": pred[i],
            })
    df = pd.DataFrame(recs)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 官方评分器要求的文件名与列
        df[["ts_code", "trade_date", "y_ret_1d"]].to_csv(td / "测试集_Y.csv", index=False)
        df[["ts_code", "trade_date", "flag_limit_up"]].to_csv(td / "测试集_X.csv", index=False)
        df[["ts_code", "trade_date", "pred"]].to_csv(td / "submission.csv", index=False)

        official = _load_official_evaluator(repo_root).evaluate(str(td / "submission.csv"), str(td))
        mine = evaluate_frame(df.copy())

    diffs = {k: abs(official[k] - mine[k]) for k in OFFICIAL_METRICS}
    for k, v in diffs.items():
        status = "OK" if v < 1e-12 else "MISMATCH"
        print(f"  {k:20s} official={official[k]: .10f}  adapter={mine[k]: .10f}  diff={v:.2e}  {status}")
    bad = [k for k, v in diffs.items() if v >= 1e-12]
    if bad:
        raise AssertionError(f"口径不一致: {bad}")
    print("selfcheck PASS：8 项官方指标与官方评分器完全一致。")
    return mine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地验证评分（官方口径适配）")
    parser.add_argument("--pred", help="预测 CSV 路径（ts_code, trade_date, pred）")
    parser.add_argument("--label-source", help="标签来源 CSV（含 y_ret_1d, flag_limit_up）")
    parser.add_argument("--start", type=int, help="验证起始日期 YYYYMMDD（闭区间）")
    parser.add_argument("--end", type=int, help="验证结束日期 YYYYMMDD（闭区间）")
    parser.add_argument("--daily-out", help="逐日指标 CSV 输出路径")
    parser.add_argument("--metrics-out", help="汇总指标 JSON 输出路径")
    parser.add_argument("--selfcheck", action="store_true",
                        help="合成数据口径核对，通过后退出")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]

    if args.selfcheck:
        selfcheck(repo_root)
        return 0
    if not args.pred:
        parser.error("需要 --pred 或 --selfcheck")

    df = load_validation_frame(args.pred, args.label_source, args.start, args.end)
    metrics = evaluate_frame(df)
    daily = daily_metrics(df)

    print("\n===== 评分结果（官方口径） =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}" if isinstance(v, int) else f"  {k}: {v:.6f}")

    if args.daily_out:
        out = Path(args.daily_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out, index=False)
        print(f"逐日指标已写入 {out}")
    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"汇总指标已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

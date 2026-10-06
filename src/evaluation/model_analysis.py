"""通用模型评分分析模板（成员 B）。

每次模型/策略优化产出固定验证预测后，一条命令完成三项官方指标的
**单独分析**（IC / Top 组合 / 换手，含换手来源诊断）与**加权分析**
（``final_score = 0.4·IC + 0.3·Excess + 0.3·(1−换手)`` 的拆解与参照归因），
可选叠加已锁定的留仓带换手层（``evaluation.band.keep_q``，0 为关闭）。

口径全部复用既有模块，不重新实现评分：

- 官方 8 项指标与逐日指标：``src/evaluation/official_eval.py``；
- Top 组合（Top10/20、Bottom10、月度超额、回撤）：``src/evaluation/backtest.py``；
- 留仓带换手层：``src/evaluation/turnover.py``；
- 评分权重：``configs/project.yaml`` 的 ``evaluation.weights``（加载时与
  官方口径 0.4/0.3/0.3 断言一致）。

输出（``--out-dir``）::

    analysis_summary.json    官方 8 项 + 三分项单独分析 + 加权拆解 + 参照归因（+ band 层）
    REPORT.md                人类可读报告，可直接作比赛报告素材
    ic_daily.csv / ic_monthly.csv / ic_quarterly.csv
    top_daily.csv / top_monthly.csv / top_summary.json
    turnover_daily.csv / turnover_diagnosis.json
    band_summary.json        仅当叠加留仓带层时

CLI 示例::

    python -m src.evaluation.model_analysis \
        --pred outputs/predictions/<exp>/valid_2024.csv \
        --labels outputs/metrics/<exp>/validation_labels.csv \
        --out-dir outputs/metrics/<exp>/analysis \
        --name <exp> \
        --ref-pred outputs/predictions/E001_baseline_repeat/valid_2024.csv \
        --ref-labels outputs/metrics/E001_baseline_repeat/validation_labels.csv

``--ref-*`` 提供参照实验（基线或 Champion）时，加权分析额外输出逐指标
Δ 与三项权重的归因拆分（``Δfinal = 0.4·ΔIC + 0.3·ΔExcess − 0.3·Δ换手``）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    daily_portfolio_returns,
    load_scored,
    monthly_excess,
    summarize,
)
from src.evaluation.official_eval import (
    OFFICIAL_METRICS,
    TOP_MIN_VALID,
    daily_metrics,
    evaluate_frame,
)
from src.evaluation.turnover import _wide_rank, evaluate_band
from src.utils.project import load_config

ROOT = Path(__file__).resolve().parents[2]

# 官方评分权重（configs/project.yaml 的 evaluation.weights 必须与此一致）
OFFICIAL_WEIGHTS = {"rank_ic": 0.4, "annual_excess": 0.3, "stability": 0.3}
TOP_FRACTION = 10  # 官方取前 1/10（与 turnover.py 一致）


# ---------------------------------------------------------------------------
# 评分权重与加权拆解
# ---------------------------------------------------------------------------

def load_weights(config_path: str | Path | None = None) -> dict:
    """读取 configs/project.yaml 的 evaluation.weights，并与官方口径断言一致。"""
    cfg, _ = load_config(config_path)
    weights = dict(cfg.get("evaluation", {}).get("weights", {}))
    for key, value in OFFICIAL_WEIGHTS.items():
        assert abs(weights.get(key, -1.0) - value) < 1e-12, (
            f"configs 权重 {key}={weights.get(key)} 与官方口径 {value} 不一致")
    return weights


def load_band_keep_q(config_path: str | Path | None = None) -> float:
    """读取 evaluation.band.keep_q（未配置时返回 0，即不叠加留仓带层）。"""
    cfg, _ = load_config(config_path)
    return float(cfg.get("evaluation", {}).get("band", {}).get("keep_q", 0.0))


def score_decomposition(metrics: dict, weights: dict | None = None) -> dict:
    """把官方 8 项中的 3 项拆成加权得分。

    返回 ``ic_term / excess_term / stability_term / final_recomputed``，并断言
    重算结果与官方 ``final_score`` 逐位一致（口径单一来源校验）。
    """
    w = weights or OFFICIAL_WEIGHTS
    ic_term = w["rank_ic"] * metrics["ic_mean"]
    excess_term = w["annual_excess"] * metrics["annual_excess"]
    stability_term = w["stability"] * (1.0 - metrics["mean_turnover"])
    final_recomputed = ic_term + excess_term + stability_term
    assert abs(final_recomputed - metrics["final_score"]) < 1e-9, (
        f"加权拆解 {final_recomputed!r} 与官方 final_score "
        f"{metrics['final_score']!r} 不一致（权重或口径有误）")
    return {
        "ic_term": ic_term,
        "excess_term": excess_term,
        "stability_term": stability_term,
        "final_recomputed": final_recomputed,
    }


def reference_attribution(metrics: dict, ref_metrics: dict,
                          weights: dict | None = None) -> dict:
    """相对参照实验的逐指标 Δ 与三项权重归因。

    ``delta_final = w_ic·ΔIC + w_excess·ΔExcess − w_stab·Δ换手``（由
    ``score_decomposition`` 的恒等关系保证，这里逐项显式给出）。
    """
    w = weights or OFFICIAL_WEIGHTS
    d_ic = metrics["ic_mean"] - ref_metrics["ic_mean"]
    d_excess = metrics["annual_excess"] - ref_metrics["annual_excess"]
    d_turnover = metrics["mean_turnover"] - ref_metrics["mean_turnover"]
    d_final = metrics["final_score"] - ref_metrics["final_score"]
    return {
        "delta_ic": d_ic,
        "delta_annual_excess": d_excess,
        "delta_turnover": d_turnover,
        "delta_final": d_final,
        "attrib_ic_term": w["rank_ic"] * d_ic,
        "attrib_excess_term": w["annual_excess"] * d_excess,
        "attrib_stability_term": -w["stability"] * d_turnover,
        "ic_retention": metrics["ic_mean"] / ref_metrics["ic_mean"]
        if ref_metrics["ic_mean"] != 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# ① IC 单独分析
# ---------------------------------------------------------------------------

def ic_analysis(scored: pd.DataFrame, metrics: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """IC 时间结构：月/季汇总、头部月份贡献、上下半年对比。

    返回 (汇总 dict, 月度表, 季度表)。贡献口径：``top3_share`` = 最好 3 个月
    的日均 IC 之和 / 全部日均 IC 之和；``retained_mean_ratio`` = 剔除最好 3 个月
    后剩余日期的日均 IC / 全部日均 IC。
    """
    daily = daily_metrics(scored)[["trade_date", "ic"]].dropna().reset_index(drop=True)
    frame = daily.copy()
    frame["month"] = frame["trade_date"].astype(str).str[:6]
    frame["quarter"] = frame["trade_date"].astype(str).str[:4] + "Q" + (
        (frame["trade_date"].astype(str).str[4:6].astype(int) - 1) // 3 + 1).astype(str)

    def _grp(g: pd.DataFrame) -> dict:
        return {
            "ic_mean": float(g["ic"].mean()),
            "ic_std": float(g["ic"].std(ddof=1)),
            "n_days": int(len(g)),
            "icir": float(g["ic"].mean() / g["ic"].std(ddof=1)),
            "positive_ratio": float((g["ic"] > 0).mean()),
        }

    monthly = frame.groupby("month").apply(_grp, include_groups=False).apply(
        pd.Series).reset_index()
    quarterly = frame.groupby("quarter").apply(_grp, include_groups=False).apply(
        pd.Series).reset_index()

    # Keep YYYYMM as the index so removing the strongest months matches dates.
    monthly_means = monthly.set_index("month")["ic_mean"]
    top3_months = monthly_means.nlargest(3)
    top3_mask = frame["month"].isin(top3_months.index)
    total_mean = float(daily["ic"].mean())
    h1 = frame[frame["month"].str[4:6].astype(int) <= 6]
    h2 = frame[frame["month"].str[4:6].astype(int) > 6]

    summary = {
        "ic_mean": metrics["ic_mean"],
        "ic_std": metrics["ic_std"],
        "icir": metrics["icir"],
        "ic_positive_ratio": metrics["ic_positive_ratio"],
        "n_days": int(len(daily)),
        "n_months_positive": int((monthly_means > 0).sum()),
        "n_months": int(len(monthly)),
        "monthly_ic_cv": float(monthly_means.std(ddof=1) / monthly_means.mean())
        if monthly_means.mean() != 0 else np.nan,
        "best_month": str(top3_months.index[0]),
        "best_month_ic": float(top3_months.iloc[0]),
        "worst_month": str(monthly_means.idxmin()),
        "worst_month_ic": float(monthly_means.min()),
        "top3_share": float(top3_months.sum() / monthly_means.sum())
        if monthly_means.sum() != 0 else np.nan,
        "retained_mean_ratio": float(frame.loc[~top3_mask, "ic"].mean() / total_mean)
        if total_mean != 0 else np.nan,
        "h1_icir": float(h1["ic"].mean() / h1["ic"].std(ddof=1)),
        "h2_icir": float(h2["ic"].mean() / h2["ic"].std(ddof=1)),
    }
    return summary, monthly, quarterly


# ---------------------------------------------------------------------------
# ② Top 组合单独分析（复用 backtest.py）
# ---------------------------------------------------------------------------

def top_analysis(scored: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Top 组合分析：官方口径汇总 + 月度超额 + 逐日明细。"""
    daily = daily_portfolio_returns(scored)
    if daily.empty:
        raise ValueError("有效交易日为 0：检查预测/标签合并")
    return summarize(daily), monthly_excess(daily), daily


# ---------------------------------------------------------------------------
# ③ 换手单独分析（官方口径 + 来源诊断）
# ---------------------------------------------------------------------------

def _top_membership(scored: pd.DataFrame) -> tuple[list[set], pd.DataFrame]:
    """按官方换手组口径重建逐日 Top 1/10 集合与候选排名分位。

    候选 = 当日 ``flag_limit_up == 0`` 的股票（不要求标签非缺失）；
    有效样本 < 100 的交易日跳过并**重置**前日集合（与官方评分器一致），
    该日换手为 NaN。返回 (逐日 Top 集合列表, 逐日候选分位宽表)。
    分位口径复用 ``turnover._wide_rank``（eligible 模式）。
    """
    wide = _wide_rank(scored, eligible_only=True)
    vals = wide.to_numpy(dtype=float)
    members: list[set] = []
    for t in range(len(wide.index)):
        row = vals[t]
        ok = ~np.isnan(row)
        if int(ok.sum()) < TOP_MIN_VALID:
            members.append(set())
            continue
        n_top = max(int(ok.sum()) // TOP_FRACTION, 1)
        idx = np.argsort(-np.where(ok, row, -np.inf))[:n_top]
        members.append(set(int(i) for i in idx))
    return members, wide


def turnover_analysis(scored: pd.DataFrame, metrics: dict) -> tuple[dict, pd.DataFrame]:
    """换手单独分析：官方均值 + 来源诊断。

    诊断口径：``exit_rank_q`` = 被换出股票在**昨日**候选分位均值；
    ``entry_rank_q`` = 新进股票在**今日**候选分位均值；两者贴近视同一切线
    说明换手来自边界抖动。``mean_abs_rank_change`` = 全体候选股相邻两日
    分位的平均绝对变化（整体排序抖动）。连续在榜统计只对"在榜过的股票"计算。
    """
    daily = daily_metrics(scored)[["trade_date", "turnover"]].reset_index(drop=True)
    members, wide = _top_membership(scored)
    vals = wide.to_numpy(dtype=float)

    exits_q, entries_q, abs_changes = [], [], []
    for t in range(1, len(members)):
        prev, cur = members[t - 1], members[t]
        if not cur:  # 当日有效样本 < 100：换手 NaN，跳过诊断
            continue
        exited = prev - cur
        entered = cur - prev
        if exited:
            exits_q.append(float(np.nanmean([vals[t - 1, c] for c in exited])))
        if entered:
            entries_q.append(float(np.nanmean([vals[t, c] for c in entered])))
        both = np.flatnonzero(~np.isnan(vals[t - 1]) & ~np.isnan(vals[t]))
        if len(both):
            abs_changes.append(float(np.nanmean(
                np.abs(vals[t, both] - vals[t - 1, both]))))

    # 连续在榜：逐股连续段统计
    run_max, run_lens, cum_days = 0, [], {}
    for j in range(vals.shape[1]):
        best = cur_len = 0
        for t in range(len(members)):
            if j in members[t]:
                cur_len += 1
                cum_days[j] = cum_days.get(j, 0) + 1
            else:
                best = max(best, cur_len)
                cur_len = 0
        best = max(best, cur_len)
        if best > 0:
            run_max = max(run_max, best)
            run_lens.append(best)

    summary = {
        "mean_turnover": metrics["mean_turnover"],
        "n_turnover_days": int(daily["turnover"].notna().sum()),
        "top_size": int(np.mean([len(m) for m in members if m])) if any(members) else 0,
        "n_ever_in_top": len(cum_days),
        "max_consecutive_days": run_max,
        "mean_consecutive_days": float(np.mean(run_lens)) if run_lens else 0.0,
        "max_cumulative_days": max(cum_days.values()) if cum_days else 0,
        "exit_rank_quantile": float(np.mean(exits_q)) if exits_q else np.nan,
        "entry_rank_quantile": float(np.mean(entries_q)) if entries_q else np.nan,
        "mean_abs_rank_change": float(np.mean(abs_changes)) if abs_changes else np.nan,
    }
    return summary, daily


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 6) -> str:
    if isinstance(value, str):  # 已预格式化的值（如 "12 / 12"）原样通过
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "NaN"
    return f"{float(value):.{digits}f}"


def _md_table(rows: list[tuple[str, object]], digits: int = 6) -> str:
    lines = ["| 指标 | 值 |", "| --- | --- |"]
    lines += [f"| {k} | {_fmt(v, digits)} |" for k, v in rows]
    return "\n".join(lines)


def analyze_model(pred: str | Path, labels: str | Path, out_dir: str | Path,
                  name: str | None = None,
                  ref_pred: str | Path | None = None,
                  ref_labels: str | Path | None = None,
                  band_keep_q: float | None = None,
                  config_path: str | Path | None = None) -> dict:
    """完整分析一条验证预测，产物写入 ``out_dir``，返回总表 dict。"""
    weights = load_weights(config_path)
    if band_keep_q is None:
        band_keep_q = load_band_keep_q(config_path)

    scored = load_scored(pred, labels)
    metrics = evaluate_frame(scored)
    decomposition = score_decomposition(metrics, weights)
    ic_summary, ic_monthly, ic_quarterly = ic_analysis(scored, metrics)
    top_summary, top_monthly, top_daily = top_analysis(scored)
    to_summary, to_daily = turnover_analysis(scored, metrics)

    summary: dict = {
        "name": name or Path(pred).parent.name,
        "weights": weights,
        "official_metrics": {k: metrics[k] for k in OFFICIAL_METRICS},
        "decomposition": decomposition,
        "ic_analysis": ic_summary,
        "top_analysis": top_summary,
        "turnover_analysis": to_summary,
    }

    band_block = None
    if band_keep_q and band_keep_q > 0.0:
        band_metrics, _band_daily = evaluate_band(scored, band_keep_q)
        band_block = {
            "keep_q": band_keep_q,
            "metrics": {k: band_metrics[k] for k in OFFICIAL_METRICS},
            "decomposition": score_decomposition(band_metrics, weights),
            "delta_vs_raw": reference_attribution(band_metrics, metrics, weights),
        }
        summary["band_layer"] = band_block

    if ref_pred is not None and ref_labels is not None:
        ref_metrics = evaluate_frame(load_scored(ref_pred, ref_labels))
        summary["reference"] = {
            "official_metrics": {k: ref_metrics[k] for k in OFFICIAL_METRICS},
            "attribution_raw": reference_attribution(metrics, ref_metrics, weights),
        }
        if band_block is not None:
            summary["reference"]["attribution_band"] = reference_attribution(
                band_block["metrics"], ref_metrics, weights)

    # ---- 落盘 ----
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trade_date": daily_metrics(scored)["trade_date"],
                  "ic": daily_metrics(scored)["ic"]}).to_csv(
        out / "ic_daily.csv", index=False)
    ic_monthly.to_csv(out / "ic_monthly.csv", index=False)
    ic_quarterly.to_csv(out / "ic_quarterly.csv", index=False)
    top_daily.to_csv(out / "top_daily.csv", index=False)
    top_monthly.to_csv(out / "top_monthly.csv", index=False)
    to_daily.to_csv(out / "turnover_daily.csv", index=False)
    with open(out / "top_summary.json", "w", encoding="utf-8") as f:
        json.dump(top_summary, f, ensure_ascii=False, indent=2)
    with open(out / "turnover_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(to_summary, f, ensure_ascii=False, indent=2)
    if band_block is not None:
        with open(out / "band_summary.json", "w", encoding="utf-8") as f:
            json.dump(band_block, f, ensure_ascii=False, indent=2)
    with open(out / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    (out / "REPORT.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _render_report(s: dict) -> str:
    """人类可读报告（比赛报告素材）。"""
    om, dec = s["official_metrics"], s["decomposition"]
    ic, tp, to = s["ic_analysis"], s["top_analysis"], s["turnover_analysis"]
    lines = [
        f"# 模型评分分析报告：{s['name']}",
        "",
        f"评分权重 `final = {s['weights']['rank_ic']}·IC + "
        f"{s['weights']['annual_excess']}·Excess + "
        f"{s['weights']['stability']}·(1−Turnover)`（官方口径）。",
        "",
        "## 官方综合指标与加权拆解",
        "",
        _md_table([
            ("ic_mean", om["ic_mean"]), ("ic_std", om["ic_std"]),
            ("icir", om["icir"]), ("ic_positive_ratio", om["ic_positive_ratio"]),
            ("annual_excess", om["annual_excess"]),
            ("top1_annual_ret", om["top1_annual_ret"]),
            ("mean_turnover", om["mean_turnover"]),
            ("final_score", om["final_score"]),
            ("— IC 项", dec["ic_term"]), ("— 超额项", dec["excess_term"]),
            ("— 稳定性项", dec["stability_term"]),
            ("— 重算校验", dec["final_recomputed"]),
        ]),
        "",
        "## ① IC 单独分析",
        "",
        _md_table([
            ("月均 IC 为正月数", f"{ic['n_months_positive']} / {ic['n_months']}"),
            ("月均 IC 变异系数", ic["monthly_ic_cv"]),
            ("最好月份", f"{ic['best_month']}（{ic['best_month_ic']:.4f}）"),
            ("最差月份", f"{ic['worst_month']}（{ic['worst_month_ic']:.4f}）"),
            ("最好 3 个月贡献占比", ic["top3_share"]),
            ("剔除后日均 IC 保留", ic["retained_mean_ratio"]),
            ("上半年 ICIR", ic["h1_icir"]), ("下半年 ICIR", ic["h2_icir"]),
        ]),
        "",
        "## ② Top 组合单独分析",
        "",
        _md_table([
            ("Top10 年化", tp["top10_annual_ret"]),
            ("市场年化", tp["market_annual_ret"]),
            ("年化超额", tp["annual_excess"]),
            ("Top20 年化超额", tp["top20_annual_excess"]),
            ("Bottom10 年化", tp["bottom10_annual_ret"]),
            ("多空年化价差", tp["top_bottom_annual_spread"]),
            ("正超额天数比例", tp["excess_positive_ratio"]),
            ("相对最大回撤", tp["relative_max_drawdown"]),
        ]),
        "",
        "## ③ 换手单独分析（含来源诊断）",
        "",
        _md_table([
            ("日均换手", to["mean_turnover"]),
            ("Top 组规模", to["top_size"]),
            ("在榜过的股票数", to["n_ever_in_top"]),
            ("最长连续在榜（天）", to["max_consecutive_days"]),
            ("平均连续在榜（天）", to["mean_consecutive_days"]),
            ("全年累计在榜最多（天）", to["max_cumulative_days"]),
            ("换出股票昨日分位", to["exit_rank_quantile"]),
            ("新进股票今日分位", to["entry_rank_quantile"]),
            ("全市场日均 |分位变化|", to["mean_abs_rank_change"]),
        ]),
        "",
        "换出/新进分位同时贴近同一阈值（0.9）说明换手主要来自切线边界抖动。",
    ]
    band = s.get("band_layer")
    if band is not None:
        bm, bd = band["metrics"], band["decomposition"]
        dv = band["delta_vs_raw"]
        lines += [
            "",
            f"## ④ 留仓带换手层（keep_q={band['keep_q']:.2f}，团队锁定配置）",
            "",
            _md_table([
                ("final_score", bm["final_score"]),
                ("ic_mean", bm["ic_mean"]), ("annual_excess", bm["annual_excess"]),
                ("mean_turnover", bm["mean_turnover"]),
                ("— IC 项", bd["ic_term"]), ("— 超额项", bd["excess_term"]),
                ("— 稳定性项", bd["stability_term"]),
                ("Δfinal（band − 原始）", dv["delta_final"]),
                ("ΔIC", dv["delta_ic"]), ("ΔExcess", dv["delta_annual_excess"]),
                ("Δ换手", dv["delta_turnover"]),
                ("IC 保留率", dv["ic_retention"]),
            ]),
        ]
    ref = s.get("reference")
    if ref is not None:
        rom = ref["official_metrics"]
        at = ref["attribution_raw"]
        lines += [
            "",
            "## ⑤ 参照归因（相对参照实验，当前预测层）",
            "",
            f"参照 final_score = {_fmt(rom['final_score'])}。",
            "",
            _md_table([
                ("Δfinal", at["delta_final"]),
                ("ΔIC × 0.4", at["attrib_ic_term"]),
                ("ΔExcess × 0.3", at["attrib_excess_term"]),
                ("−Δ换手 × 0.3", at["attrib_stability_term"]),
                ("三项归因合计", at["attrib_ic_term"] + at["attrib_excess_term"]
                 + at["attrib_stability_term"]),
                ("IC 保留率", at["ic_retention"]),
            ]),
            "",
            "归因合计与 Δfinal 之差为 0（加权恒等式），可据此判断提分来自"
            "预测质量（IC/超额）还是评分结构（低换手）。",
        ]
        if "attribution_band" in ref:
            ab = ref["attribution_band"]
            lines += [
                "",
                "留仓带层相对参照："
                f"Δfinal {_fmt(ab['delta_final'])}（ΔIC {_fmt(ab['delta_ic'])}，"
                f"ΔExcess {_fmt(ab['delta_annual_excess'])}，"
                f"Δ换手 {_fmt(ab['delta_turnover'])}，"
                f"IC 保留率 {_fmt(ab['ic_retention'])}）。",
            ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="通用模型评分分析模板（IC/Top/换手 单独+加权）")
    parser.add_argument("--pred", required=True, help="预测 CSV（ts_code,trade_date,pred）")
    parser.add_argument("--labels", required=True,
                        help="标签 CSV（ts_code,trade_date,y_ret_1d,flag_limit_up）")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default=None, help="报告标题名（默认取预测目录名）")
    parser.add_argument("--ref-pred", default=None, help="参照实验预测 CSV（可选）")
    parser.add_argument("--ref-labels", default=None, help="参照实验标签 CSV（可选）")
    parser.add_argument("--band-keep-q", type=float, default=None,
                        help="叠加留仓带层的 keep_q（默认取 configs 的 evaluation.band.keep_q；"
                             "0 或负数表示不叠加）")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/project.yaml）")
    args = parser.parse_args(argv)

    if (args.ref_pred is None) != (args.ref_labels is None):
        parser.error("--ref-pred 与 --ref-labels 必须同时提供")

    summary = analyze_model(
        args.pred, args.labels, args.out_dir, name=args.name,
        ref_pred=args.ref_pred, ref_labels=args.ref_labels,
        band_keep_q=args.band_keep_q, config_path=args.config)

    om = summary["official_metrics"]
    print(f"===== {summary['name']}：官方综合指标 =====")
    for key in OFFICIAL_METRICS:
        print(f"  {key}: {om[key]:.10f}")
    dec = summary["decomposition"]
    print(f"  拆解：IC {dec['ic_term']:.6f} + 超额 {dec['excess_term']:.6f} + "
          f"稳定 {dec['stability_term']:.6f} = {dec['final_recomputed']:.10f}")
    if "band_layer" in summary:
        bm = summary["band_layer"]["metrics"]
        print(f"  留仓带(keep_q={summary['band_layer']['keep_q']}): "
              f"final {bm['final_score']:.10f}（换手 {bm['mean_turnover']:.6f}）")
    if "reference" in summary:
        at = summary["reference"]["attribution_raw"]
        print(f"  相对参照：Δfinal {at['delta_final']:+.6f} "
              f"（IC {at['attrib_ic_term']:+.6f} / 超额 {at['attrib_excess_term']:+.6f} / "
              f"稳定 {at['attrib_stability_term']:+.6f}）")
    print(f"\n已写入 {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Publish actual S009 findings, bounded next-step advice and immutable handoff."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from src.evaluation.paired_block_bootstrap import block_indices, interval, mean_draws
from src.models.alpha_research import relative, save_csv, write_text
from src.models.optuna_tuning import provenance, read_json, save_json
from src.models.target_research import table_markdown
from src.utils.experiments import read_records
from src.utils.project import ROOT, sha256, timestamp
from src.utils.research_cache import contained_path

MODEL_COLORS = {"T030": "#53778d", "Y1": "#ba9253", "Y4": "#579b72", "Market": "#af6178", "T033": "#8c70a8"}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def diagnostic_summary(cells, tables, root):
    boundary, ties, holdings = [], [], []
    for cell in cells:
        ids = {k: cell[k] for k in ["model", "year", "layer"]}
        def part(name):
            f = tables[name]
            return f[(f.model == cell["model"]) & (f.year == cell["year"]) & (f.layer == cell["layer"])]
        b, t, h = part("boundary"), part("ties"), part("holdings")
        ci = cell["boundary_ci"]
        boundary.append({**ids, "gap_annualized": b.gap_5_10_minus_10_15.mean()*252,
            "gap_ci_lower": ci["lower"]*252 if ci["lower"] is not None else np.nan,
            "gap_ci_upper": ci["upper"]*252 if ci["upper"] is not None else np.nan,
            "precision10": b.precision10.mean(), "inference_recall20": b.inference_recall20.mean(),
            "inference_recall30": b.inference_recall30.mean(), "top20_ic": b.top20_ic.mean(),
            "remaining80_ic": b.remaining80_ic.mean(), "monotone_days_fraction": b.bins_monotone.mean()})
        ties.append({**ids, "days": len(t), "duplicate_count_mean": t.duplicate_count.mean(),
            "largest_tie_mean": t.largest_tie.mean(), "tied_stock_count_mean": t.tied_stock_count.mean(),
            "boundary_crossed_days": int(t.boundary_crossed.sum()), "boundary_crossed_fraction": t.boundary_crossed.mean(),
            "boundary_tie_n_mean": t.boundary_tie_n.mean(), "zero_score_n_mean": t.zero_score_n.mean(),
            "zero_rank_mean": t.zero_rank_mean.mean(), "fallback_zero_n_mean": t.fallback_zero_n.mean(),
            "fallback_turn_top_mean": t.fallback_in_turn_top.mean(), "fallback_return_top_mean": t.fallback_in_return_top.mean(),
            "return_turn_jaccard_mean": t.return_turn_jaccard.mean()})
        holdings.append({**ids, **cell["spells"], "entered_n_mean_ex_first": h.loc[~h.first_day, "entered_n"].mean(),
            "retained_n_mean_ex_first": h.loc[~h.first_day, "retained_n"].mean(),
            "forced_ineligible_exits": h.forced_ineligible_exits.sum(), "eligible_exits": h.eligible_exits.sum(),
            "below_q_exits": h.below_q_exits.sum(), "mean_selected_raw_rank": h.mean_raw_rank.mean(),
            "raw_top_overlap_mean": h.raw_top_overlap.mean(), "return_recovery_max_difference": h.return_recovery_difference.max()})
    tables["boundary_summary"], tables["ties_summary"], tables["holdings_summary"] = map(pd.DataFrame, [boundary, ties, holdings])


def overlaps(root, cells):
    rows = []
    for cell in cells:
        if cell["model"] == "T030":
            continue
        base_path = root / "cells/T030" / cell["fold"] / cell["layer"] / "actual_top_sets.npz"
        my_path = root / "cells" / cell["model"] / cell["fold"] / cell["layer"] / "actual_top_sets.npz"
        with np.load(base_path, allow_pickle=False) as base, np.load(my_path, allow_pickle=False) as mine:
            for pool in ["return_top", "turn_top"]:
                intersection = (base[pool] & mine[pool]).sum(axis=1)
                union = (base[pool] | mine[pool]).sum(axis=1)
                for day, value, overlap in zip(mine["dates"], intersection/union, intersection/base[pool].sum(axis=1)):
                    rows.append({"model": cell["model"], "year": cell["year"], "layer": cell["layer"],
                                 "pool": pool, "trade_date": int(day), "jaccard": value, "reference_top_retained": overlap})
    return pd.DataFrame(rows)


def advise(cfg, tables):
    s = cfg["top_tail_diagnostics"]["advance_rules"]
    raw_y4 = tables["boundary_summary"]
    raw_y4 = raw_y4[(raw_y4.model == "Y4") & (raw_y4.layer == "raw") & (raw_y4.year <= 2023)]
    weak = int((raw_y4.gap_ci_lower <= 0).sum())
    limited = int(((raw_y4.inference_recall20 < s["candidate20_recall_floor"])
                   | (raw_y4.inference_recall30 < s["candidate30_recall_floor"])).sum())
    states = tables["states"]
    past = states[(states.model == "Market") & (states.layer == "raw") & (states.year <= 2023)]
    evidence = []
    for (family, state), group in past.groupby(["family", "state"], sort=True):
        supported = group[group.days >= s["state_min_days"]]
        good = supported[(supported.delta_excess > 0) & (supported.delta_ic >= -.005)]
        evidence.append({"family": family, "state": state, "supported_years": len(supported),
            "positive_years": len(good), "positive_ci_years": int((good.excess_ci_lower > 0).sum()),
            "eligible_for_causal_gate_research": len(good) >= s["minimum_cv_years"]})
    table = pd.DataFrame(evidence)
    advance10 = weak >= s["minimum_cv_years"] or limited >= s["minimum_cv_years"]
    advance11 = bool(table.eligible_for_causal_gate_research.any())
    return {"stage": "diagnostic_only", "formal_recommendation": "retain_S003",
        "advance_S010": bool(advance10), "advance_S011": advance11,
        "Y4_weak_boundary_cv_years": weak, "Y4_limited_candidate_recall_cv_years": limited,
        "candidate20_recall_floor": s["candidate20_recall_floor"], "candidate30_recall_floor": s["candidate30_recall_floor"],
        "no_new_model_claim": True, "no_hidden_test_score_claim": True,
        "gate_evidence": table.to_dict(orient="records")}, table


def make_figures(cfg, root, study_id, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = contained_path(cfg["paths"]["figures"]) / study_id
    folder.mkdir(parents=True, exist_ok=True)
    figures = []
    def finish(fig, name):
        path = folder / f"{study_id}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        figures.append(path)
    models = cfg["top_tail_diagnostics"]["models"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    bins = tables["bins"]
    bins = bins[(bins.layer == "raw") & (bins.period_type == "year")]
    for year, ax in zip(range(2021, 2025), axes.flat):
        for model in models:
            f = bins[(bins.year == year) & (bins.model == model)].sort_values("lower")
            if len(f):
                ax.plot((f.lower+f.upper)*50, f.annual_excess, marker="o", ms=3, label=model, color=MODEL_COLORS[model])
        ax.axhline(0, color=".6", lw=.7); ax.set_title(str(year)); ax.set_xlabel("Predicted rank from top (%)")
        ax.set_ylabel("Annualised excess (disjoint bins)")
    axes.flat[0].legend(fontsize=8)
    finish(fig, "tail_curve")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for year, ax in zip(range(2021, 2025), axes.flat):
        b = tables["boundary_summary"]
        b = b[(b.layer == "raw") & (b.year == year)].set_index("model").reindex(models).dropna(subset=["gap_annualized"])
        x = np.arange(len(b)); ax.bar(x, b.gap_annualized, color=[MODEL_COLORS[m] for m in b.index])
        ax.vlines(x, b.gap_ci_lower, b.gap_ci_upper, colors=".25", lw=1.5)
        ax.set_xticks(x, b.index); ax.axhline(0, color=".6", lw=.7); ax.set_title(str(year))
        ax.set_ylabel("5-10% minus 10-15% annualised return")
    finish(fig, "boundary")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    raw = tables["ties_summary"][tables["ties_summary"].layer == "raw"]
    for model in models:
        f = raw[raw.model == model].sort_values("year")
        axes[0].plot(f.year, f.zero_rank_mean, marker="o", label=model, color=MODEL_COLORS[model])
        axes[1].plot(f.year, f.boundary_crossed_fraction*100, marker="o", label=model, color=MODEL_COLORS[model])
    axes[0].set_ylabel("Mean rank percentile of exact zero scores"); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Days with a tie across Top10 boundary (%)")
    for ax in axes:
        ax.set_xticks(range(2021, 2025)); ax.grid(alpha=.2); ax.set_xlabel("Year")
    axes[0].legend(fontsize=8); finish(fig, "ties")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cohort = tables["cohorts"]
    cohort = cohort[(cohort.layer == "band") & (cohort.pool == "return") & (cohort.period_type == "year")]
    for year, ax in zip(range(2021, 2025), axes.flat):
        pivot = cohort[cohort.year == year].pivot(index="model", columns="cohort", values="annual_excess_contribution").reindex(models)
        x = np.arange(len(pivot))
        for offset, group, color in [(-.25, "retained", "#579b72"), (0., "entered", "#ba9253"), (.25, "initial", "#86919b")]:
            if group in pivot:
                ax.bar(x+offset, pivot[group], width=.24, label=group, color=color)
        ax.set_xticks(x, models); ax.axhline(0, color=".5", lw=.7); ax.set_title(str(year))
        ax.set_ylabel("Contribution to official annual excess")
    axes.flat[0].legend(fontsize=8); finish(fig, "cohorts")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    state = tables["states"][tables["states"].model == "Market"].copy()
    state["label"] = state.family+":"+state.state
    for layer, ax in zip(["raw", "band"], axes):
        f = state[(state.layer == layer) & (state.days >= cfg["top_tail_diagnostics"]["advance_rules"]["state_min_days"])]
        matrix = f.pivot(index="label", columns="year", values="delta_excess").reindex(columns=range(2021, 2025))
        data = matrix.to_numpy()
        span = float(np.nanmax(abs(data))) if np.isfinite(data).any() else 1.
        im = ax.imshow(data, cmap="RdBu_r", vmin=-max(span, .01), vmax=max(span, .01), aspect="auto")
        ax.set_xticks(range(4), range(2021, 2025)); ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set_title(f"Market minus T030 annual excess: {layer}")
        for i, j in zip(*np.where(np.isfinite(data))):
            ax.text(j, i, f"{data[i,j]:+.3f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(data[i,j]) > span*.6 else ".15")
        fig.colorbar(im, ax=ax, shrink=.8)
    finish(fig, "states")
    return figures


def publish(cfg, root, meta, cells, tables, preparation, log):
    study_id = meta["study_id"]
    diagnostic_summary(cells, tables, root)
    tables["overlaps"] = overlaps(root, cells)
    decision, tables["gate_evidence"] = advise(cfg, tables)
    reports = contained_path(cfg["paths"]["research_reports"])
    experiments = contained_path(cfg["paths"]["experiment_log"]).parent
    prefix = f"top_tail_{study_id}"
    public_names = ["official", "monthly", "bins", "cumulative", "cohorts", "subgroups",
                    "boundary_summary", "ties_summary", "holdings_summary", "order_sensitivity", "bootstrap", "states", "gate_evidence"]
    for name, table in tables.items():
        save_csv(root / "tables" / f"{name}.csv", table)
        if name in public_names:
            save_csv(experiments / f"{prefix}_{name}.csv", table)
    figs = make_figures(cfg, root, study_id, tables)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    published_figs = []
    for path in figs:
        destination = reports / "figures" / path.name
        shutil.copyfile(path, destination); published_figs.append(destination)
    figures_pass = False  # Visual inspection is recorded separately after generation.
    off = tables["official"]
    b = tables["boundary_summary"]
    ties = tables["ties_summary"]
    cohort = tables["cohorts"]
    group = cohort[(cohort.layer == "band") & (cohort.pool == "return") & (cohort.period_type == "year")]
    weak_months = tables["monthly"]
    weak_months = weak_months[(weak_months.layer == "band") & (weak_months.model.isin(["T030", "Y4", "Market", "T033"]))]
    monthly_special = weak_months[weak_months.month.isin([202401, 202409])]
    native = [c["native"] for c in cells if c["layer"] == "raw" and c["native"]["status"] == "available"]
    source_max = float(off.source_max_difference.max())
    lines = [f"# {study_id}：头部边界、并列与留仓收益诊断", "", f"完成时间：{timestamp()}", "",
        f"执行源码先提交、推送 main `{meta['git']['commit']}`；输入 / 配置 / 源码摘要独立归档。",
        "本轮为零训练诊断：19 个模型年度 × raw / 原 q* band = 38 个主比较单元；"
        "Y1 的 2024 缺失，不补训。正式研究参照保留 S003，没有产生新提交模型或隐藏测试成绩。", "",
        "## 官方口径与来源验收", "",
        f"主比较落盘官方八指标最大差 {off.official_max_difference.max():.3e}；"
        f"与来源已发布指标最大差 {source_max:.3e}。完整键 / 有限值 / canonical 行序均核对。",
        f"11 个可用历史原生模型重新加载并推理；实际原生 / CSV 最大差 "
        f"{max([n['max_value_difference'] for n in native], default=0):.3e}。"
        "S007/T033 原模型未保存，无法补做其原生重载核对。",
        "raw → q* 只传入键、pred、涨停标志；全年冷启动连续递推。实际 Top 集合与值、"
        "19 次真实前缀重放已核对；完整顺序 / 并列与两种 CSV 解析的差异单独保留。", "",
        table_markdown(off, ["model", "year", "layer", "ic_mean", "annual_excess", "mean_turnover", "final_score"]), "",
        "## 头部收益与候选池上限", "",
        "分层在官方收益池中按实际评分排序；互斥区间和累计 Top 分开保存。真实事件用 y 平均百分位，"
        "精确并列保持同标签。推理候选池由 raw、t 日有效价格与非涨停形成，先构造候选，再用已有标签评召回；"
        "缺 y 不进入选择输入。召回低意味着精排无法找回池外股票，不是新模型已证实失败。", "",
        table_markdown(b[b.layer == "raw"], ["model", "year", "gap_annualized", "gap_ci_lower", "gap_ci_upper", "precision10", "inference_recall20", "inference_recall30"]), "",
        "区间是 20 日循环块、1000 次、seed=42 的配对 / 日期统计抽样；分层 gap 是年化辅助量。"
        "股票池内中位数为实际 pooled median；均值 / 超额以日等权。", "",
        "## 精确并列、零预测与行序", "",
        table_markdown(ties[ties.layer == "raw"], ["model", "year", "duplicate_count_mean", "largest_tie_mean", "boundary_crossed_days", "zero_rank_mean", "fallback_turn_top_mean", "fallback_return_top_mean"]), "",
        "duplicate_count 是 N−unique，与属于重复组的总人数不同。缺价 raw 0 与有效价格下真实输出 0 分开；"
        "band 编码后不再用最终 pred==0 当作缺价标记。逆序与固定日内置换只移动原文件行字节，不重新序列化数字。"
        "行序敏感性只用于描述，正式排序不按得分重新选择。", "",
        table_markdown(tables["order_sensitivity"].groupby(["model", "layer"], as_index=False).agg(
            max_abs_score_delta=("final_delta", lambda s: s.abs().max()), max_official_difference=("official_max_difference", "max")),
            ["model", "layer", "max_abs_score_delta", "max_official_difference"]), "",
        "## 留仓、换入和两种官方池", "",
        "H_turn 只删除涨停；H_ret 另删除缺 y。状态集合的可评分收益不冒充官方 Top 收益。"
        "下面把 H_ret 按是否在昨日 H_turn 分组，其加权贡献（包括首日 initial）精确相加为全年官方超额。", "",
        table_markdown(group, ["model", "year", "cohort", "n_mean", "mean_ret_mean", "annual_excess_contribution"]), "",
        table_markdown(tables["holdings_summary"][tables["holdings_summary"].layer == "band"],
            ["model", "year", "mean_spell_days", "entered_n_mean_ex_first", "mean_selected_raw_rank", "raw_top_overlap_mean", "forced_ineligible_exits", "eligible_exits"]), "",
        "首日单列；持仓段在年度末右删失，长度不是最终完整持有寿命。below_q 为实际换出者的已知阈值标记，"
        "resize_exits_minimum 只是规模缩小需要退出的下界，不伪装成旧控制器未保存的主动换仓操作日志。", "",
        "## 市场状态与流动性 / 波动切片", "",
        "波动 / 宽度阈值只取各外层年份之前的市场序列中位数，趋势阈值为 0。股票 amount / volatility_20 "
        "按当日已知 X 四分位分组，缺失单列。全部年份和三状态 family 的支持天数与区间保留，"
        "不足 60 日的状态不支持门控晋级。", "",
        table_markdown(tables["gate_evidence"], ["family", "state", "supported_years", "positive_years", "positive_ci_years", "eligible_for_causal_gate_research"]), "",
        "门控证据只统计 2021–2023 的 market raw 相对 T030，单年 / 单状态的条件差不证明因果关系；"
        "若进入 S011，必须用更早年度 OOF 学习当期权重，不能以本年表现配本年权重。", "",
        "## 多时期不确定性与预声明月份", "",
        table_markdown(tables["bootstrap"][(tables["bootstrap"].layer == "band") & (tables["bootstrap"].period.str.contains("CV|Historical"))],
            ["model", "period", "metric", "delta", "lower", "upper"]), "",
        "各年内成对抽相同日期，再等权平均年度；只抽原时间路径已算好的 IC / 超额 / 换手日统计，"
        "不在拼接块边界制造新的持仓或换手。95% 区间不校正既往择模和多轮候选筛选，"
        "2021–2024 已被反复使用，不能称为独立留出。", "",
        table_markdown(monthly_special, ["model", "month", "ic_mean", "annual_excess", "mean_turnover", "final_score"]), "",
        "全部月份 / 季度在公共切片表，未只保存这些切片。", "",
        "## 分流建议", "",
        f"- S010 头部学习：**{decision['advance_S010']}**。Y4 原始三折中，"
        f"头部边界 gap 下界不为正的年份 {decision['Y4_weak_boundary_cv_years']} 个，"
        f"因果候选召回不足的年份 {decision['Y4_limited_candidate_recall_cv_years']} 个。",
        f"- S011 有限门控：**{decision['advance_S011']}**。按上述支持天数和过去年度重复性核对。",
        "- 当前正式研究建议仍为 **retain_S003**；本轮没有训练新模型，诊断仅支持下一项有限实验。"
        "若第一阶段召回有限，应先比较全市场 B10/B20，再检验 R20/R30/L30 的精排是否补偿筛池损失。", "",
        "S009 输入 / 预测 / 原数据 / 官方附件保持不变，共享表原 718 条全部保持，没有重复追加来源或把切片当作官方实验。"
        "模型缺失、Y1 2024 缺失、条件置信区间及历史选择偏差明确保留；成员 B 独立复核待接续。"
        "本轮没有新增或运行测试套件，不启动 S010/S011。", "", "## 图表", ""]
    for path in published_figs:
        lines += [f"![{path.stem}](figures/{path.name})", ""]
    report_path = reports / f"top_tail_{study_id}.md"
    write_text(report_path, "\n".join(lines))
    decision_path = reports / f"top_tail_{study_id}_decision.json"
    save_json(root / "decision.json", clean_json(decision)); save_json(decision_path, clean_json(decision))
    if read_records(contained_path(cfg["paths"]["experiment_log"])) != meta["initial_records"]:
        raise ValueError("Diagnostic run changed the shared experiment log.")
    if provenance(cfg) != meta["data"]:
        raise ValueError("Original data or official attachments changed during diagnosis.")
    checked = {}
    for path, expected in meta["frozen_inputs"].items():
        if sha256(contained_path(path)) != expected:
            raise ValueError(f"Frozen input changed during diagnosis: {path}")
        checked[path] = expected
    runs = []
    for cell in cells:
        directory = root / "cells" / cell["model"] / cell["fold"] / cell["layer"]
        manifest = directory / "cell.json"
        if read_json(manifest) != cell:
            raise ValueError("Passed cell manifest changed.")
        for path, expected in cell["artifact_hashes"].items():
            if sha256(contained_path(path)) != expected:
                raise ValueError(f"Diagnostic artifact changed: {path}")
        runs.append({"model": cell["model"], "fold": cell["fold"], "layer": cell["layer"],
                     "manifest": relative(manifest), "sha256": sha256(manifest), "artifact_hashes": cell["artifact_hashes"]})
    context_files = [root / "market_daily.csv", root / "state_thresholds.csv", root / "context.json",
                     *[contained_path(e["path"]) for e in preparation["contexts"].values()]]
    public_files = [report_path, decision_path, *published_figs,
                    *sorted(experiments.glob(f"{prefix}_*.csv"))]
    audit = {"status": "passed", "study_id": study_id, "time": timestamp(), "implementation_commit": meta["git"]["commit"],
        "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
        "data_provenance": meta["data"], "frozen_inputs": checked, "completed_cases": len(cells),
        "new_model_fits": 0, "shared_records_before": len(meta["initial_records"]),
        "shared_records_after": len(meta["initial_records"]), "shared_records_unchanged": True,
        "max_official_difference": max(off.official_max_difference.max(), tables["order_sensitivity"].official_max_difference.max()),
        "max_source_difference": source_max, "native_models_reloaded": len(native),
        "max_native_prediction_difference": max([n["max_value_difference"] for n in native], default=0),
        "market_prefix_replay": preparation["market_prefix_replay"], "band_prefix_replays": 19,
        "known_missing": ["Y1 confirm2024", "S007/T033 original saved model files"],
        "visual_inspection_complete": figures_pass, "runs": runs,
        "context_artifacts": {relative(p): sha256(p) for p in context_files},
        "published_files": {relative(p): {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in public_files},
        "decision": clean_json(decision)}
    save_json(root / "audit.json", clean_json(audit))
    save_json(reports / f"top_tail_{study_id}_artifacts.json", clean_json(audit))
    log(f"published {report_path.name}; S010={decision['advance_S010']}; S011={decision['advance_S011']}; retain S003")

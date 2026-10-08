"""Publish S006's actual saved scores, selection, diagnostics and SHA index."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from src.evaluation.official_eval import OFFICIAL_METRICS
from src.evaluation.research_diagnostics import monthly_metrics
from src.models.alpha_research import quarter_metrics, relative, save_csv, table_markdown, write_text
from src.models.optuna_tuning import save_json
from src.utils.project import sha256, timestamp
from src.utils.research_cache import contained_path


def markdown(frame, columns):
    return "\n".join(table_markdown(frame, columns))


def periods(cfg, run):
    directory = contained_path(run["diagnostics"]) if "diagnostics" in run else contained_path(cfg["paths"]["metrics"]) / run["exp_id"]
    daily = pd.read_csv(directory / "daily_metrics.csv")
    return monthly_metrics(daily), quarter_metrics(daily)


def comparison(results, baseline):
    rows = []
    for r in results:
        changes = {"ic_component_change": .4*(r["mean_ic"]-baseline["mean_ic"]),
                   "excess_component_change": .3*(r["mean_excess"]-baseline["mean_excess"]),
                   "stability_component_change": -.3*(r["mean_turnover"]-baseline["mean_turnover"])}
        if abs(sum(changes.values())-(r["cv_mean"]-baseline["cv_mean"])) > 1e-10:
            raise ValueError("CV weighted attribution does not add up.")
        rows.append({"candidate": r["id"], "signal": r["signal"]["id"],
            "components": "+".join(r["signal"]["components"]), "weights": json.dumps(r["signal"]["weights"]),
            "controller": r["controller_id"], "controller_method": r["controller"]["method"],
            "alias_of": r["alias_of"], "passes_cv_gate": r["passes_cv_gate"],
            **{k: r[k] for k in ["cv_mean", "cv_std", "cv_worst", "mean_ic", "mean_excess", "mean_turnover"]},
            **r["scores"], "cv_delta_vs_S003": r["cv_mean"]-baseline["cv_mean"], **changes})
    table = pd.DataFrame(rows)
    values = table[["mean_ic", "mean_excess", "mean_turnover"]].to_numpy()*[1, 1, -1]
    table["pareto"] = [not ((values >= row-1e-12).all(axis=1) & (values > row+1e-12).any(axis=1)).any() for row in values]
    return table


def figures(cfg, meta, table, monthly, selection, confirmation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prefix = f"alpha_{meta['study_id']}"
    directory = contained_path(cfg["paths"]["figures"]) / meta["study_id"]
    directory.mkdir(parents=True, exist_ok=True)
    highest = table.sort_values("cv_mean", ascending=False).head(10)
    if "F000" not in set(highest.candidate):
        highest = pd.concat([highest, table[table.candidate == "F000"]])
    highest = highest.sort_values("cv_mean")
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    y = np.arange(len(highest))
    ax.barh(y-.16, highest.cv_mean, .32, label="Mean", color="#3679a8")
    ax.barh(y+.16, highest.cv_worst, .32, label="Worst fold", color="#80a8b8")
    labels = [f"{r.candidate}: {r.components}, {r.controller}" for r in highest.itertuples()]
    ax.set_yticks(y, labels); ax.set_xlabel("Official final score")
    ax.set_title("S006: top CV means and the frozen S003 reference")
    ax.legend(loc="lower right", frameon=False); ax.set_xlim(0, max(highest.cv_mean)*1.12)
    fig.savefig(directory / f"{prefix}_cv_scores.png", dpi=150); plt.close(fig)
    unique = table[table.alias_of.isna()]
    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    points = ax.scatter(unique.mean_ic, unique.mean_excess, c=unique.mean_turnover, cmap="viridis", s=48)
    frontier = unique[unique.pareto]
    ax.scatter(frontier.mean_ic, frontier.mean_excess, facecolors="none", edgecolors="red", s=90, linewidths=1)
    ids = ["F000", table.loc[table.cv_mean.idxmax(), "candidate"]]
    if selection["winner"]:
        ids.append(selection["winner"]["id"])
    for i, name in enumerate(dict.fromkeys(ids)):
        row = table[table.candidate == name].iloc[0]
        ax.annotate(name, (row.mean_ic, row.mean_excess), xytext=(8, (i*17)+6), textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "color": ".3"})
    fig.colorbar(points, ax=ax, label="Mean Jaccard turnover")
    ax.set_xlabel("Mean daily Rank IC"); ax.set_ylabel("Annual excess")
    ax.set_title("Complete schemes: red border = three-metric Pareto point"); ax.margins(.17)
    fig.savefig(directory / f"{prefix}_pareto.png", dpi=150); plt.close(fig)
    winner_id = selection["winner"]["id"] if selection["winner"] else table.loc[table.cv_mean.idxmax(), "candidate"]
    years = [2021, 2022, 2023] + ([2024] if confirmation else [])
    fig, axes = plt.subplots(len(years), 1, figsize=(10, 2.5*len(years)), constrained_layout=True)
    for ax, year in zip(axes, years):
        for label, candidate, color in [("S003", "F000", "#737373"), (winner_id, winner_id, "#276ca5")]:
            group = monthly[(monthly.candidate == candidate) & (monthly.year == year)].sort_values("month")
            ax.plot(np.arange(1, len(group)+1), group.final_score, label=label, color=color, marker="o", markersize=3)
        ax.set_title(str(year)); ax.set_xticks(range(1, 13)); ax.set_ylabel("Official final score"); ax.grid(alpha=.2)
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.2), ncol=2, frameon=False)
    axes[-1].set_xlabel("Month; statistics from full-year controller state")
    fig.savefig(directory / f"{prefix}_monthly.png", dpi=150); plt.close(fig)
    target = table[table.candidate == winner_id].iloc[0]
    values = [target.ic_component_change, target.excess_component_change, target.stability_component_change]
    labels = ["IC", "Annual excess", "Stability"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#3886aa", "#63a273", "#bb874e"])
    ax.axhline(0, color=".25", lw=.8)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.6f}", (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 5 if value >= 0 else -13), textcoords="offset points", ha="center")
    span = max(max(values)-min(values), .005)
    ax.set_ylim(min(0, min(values))-.22*span, max(0, max(values))+.22*span)
    ax.set_title(f"{winner_id} CV attribution vs S003; total {sum(values):+.6f}")
    ax.set_ylabel("Weighted contribution to official score")
    fig.savefig(directory / f"{prefix}_attribution.png", dpi=150); plt.close(fig)
    published = []
    for source in sorted(directory.glob("*.png")):
        destination = contained_path(cfg["paths"]["research_reports"]) / "figures" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination); published.append(destination)
    return published


def publish(cfg, root, meta, results, raw_signals, selection, audited, *, confirmation=None):
    prefix = f"alpha_{meta['study_id']}"
    experiments = contained_path(cfg["paths"]["experiment_log"]).parent
    reports = contained_path(cfg["paths"]["research_reports"])
    table = comparison(results, results[0])
    folds, raw_rows, months, quarters, holdings = [], [], [], [], []
    for result in results:
        for run in result["folds"]:
            fold = run["split"]["fold"]
            year = run["split"]["valid_window"][0]//10000
            context = {"candidate": result["id"], "signal": result["signal"]["id"], "controller": result["controller_id"],
                       "fold": fold, "year": year, "exp_id": run["exp_id"]}
            folds.append({**context, **{k: run["metrics"][k] for k in OFFICIAL_METRICS},
                          "reused_run": not run["exp_id"].startswith(meta["study_id"]+"_")})
            monthly, quarterly = periods(cfg, run)
            months.extend({**context, **row} for row in monthly.to_dict("records"))
            quarters.extend({**context, **row} for row in quarterly.to_dict("records"))
            if "holdings_summary" in run:
                holdings.append({**context, **run["holdings_summary"]})
    for result in raw_signals:
        for run in result["folds"]:
            raw_rows.append({"signal": result["signal"]["id"], "components": "+".join(result["signal"]["components"]),
                "weights": json.dumps(result["signal"]["weights"]), "fold": run["split"]["fold"],
                "exp_id": run["exp_id"], **{k: run["metrics"][k] for k in OFFICIAL_METRICS},
                "predictive_score": .4*run["metrics"]["ic_mean"]+.3*run["metrics"]["annual_excess"]})
    confirmation_rows = []
    if confirmation:
        for label, run in confirmation["runs"].items():
            confirmation_rows.append({"layer": label, "exp_id": run["exp_id"], **{k: run["metrics"][k] for k in OFFICIAL_METRICS}})
            monthly, quarterly = periods(cfg, run)
            candidate = "F000" if label == "T030_qstar" else selection["winner"]["id"] if label == "new_complete" else label
            context = {"candidate": candidate, "signal": label, "controller": label, "fold": "confirm2024", "year": 2024, "exp_id": run["exp_id"]}
            months.extend({**context, **row} for row in monthly.to_dict("records"))
            quarters.extend({**context, **row} for row in quarterly.to_dict("records"))
    monthly_table = pd.DataFrame(months)
    for name, frame in [("comparison", table), ("folds", pd.DataFrame(folds)), ("raw_signals", pd.DataFrame(raw_rows)),
                         ("monthly", monthly_table), ("quarterly", pd.DataFrame(quarters)), ("holdings", pd.DataFrame(holdings))]:
        save_csv(experiments / f"{prefix}_{name}.csv", frame)
    for name in ["history_quality", "history_correlations", "history_aliases"]:
        shutil.copyfile(root / f"{name}.csv", experiments / f"{prefix}_{name}.csv")
    if confirmation:
        save_csv(experiments / f"{prefix}_confirmation.csv", pd.DataFrame(confirmation_rows))
    images = figures(cfg, meta, table, monthly_table, selection, confirmation)
    report = reports / f"alpha_research_{meta['study_id']}.md"
    baseline = results[0]
    best = max(results, key=lambda r: r["cv_mean"])
    selected = selection["winner"]
    histories = [r for r in meta["components"] if r["id"] in meta["history_partners"]]
    history_table = pd.DataFrame([{"model": r["id"], "raw_predictive_mean": r["predictive_mean"],
                                  "raw_predictive_worst": r["predictive_worst"], "rounds": r["model_spec"]["rounds"]} for r in histories])
    selected_note = (f"锁定 **{selected['id']}：{' + '.join(selected['signal']['components'])}，权重 {selected['signal']['weights']}，"
                     f"控制器 {selected['controller_id']} / {selected['controller']}**。CV 均分 {selected['cv_mean']:.10f}，"
                     f"最差折 {selected['cv_worst']:.10f}，2023 {selected['scores']['wf2023']:.10f}。") if selected else "没有配置通过预定跨年门槛，保留 S003；本轮不进入新的 2024 确认。"
    top = table.sort_values("cv_mean", ascending=False).head(12)
    lines = [f"# {meta['study_id']}：有限排名融合与冻结控制器研究", "", f"更新时间：{timestamp()}（Asia/Shanghai）。", "",
        "## 协议与来源", "",
        f"实现提交 `{meta['git']['commit']}`，源码摘要 `{meta['source_digest']}`，协议摘要 `{meta['protocol_digest']}`。",
        "仅用 2021–2023 的原始预测筛选历史伙伴并比较完整方案；S005 的 Y4 / Y1 和 S004 的两个控制器均保持冻结。V1 40 特征不变。",
        "历史 50 个 trial 的三折 raw 清单按冻结 SHA 校验，先取 raw PredictiveScore 最高的 10 个并加 T030；同参数/轮数、或三折解析预测完全相同者去重。历史第一伙伴取最高 raw 分，第二个取与 T030 及第一伙伴的平均日排名相关性最低者；相关性 1e-12 内同分按质量和 trial 编号。相关性是研究假设，最终选择使用官方完整分。",
        "", markdown(history_table, ["model", "raw_predictive_mean", "raw_predictive_worst", "rounds"]), "",
        "组成模型为 " + "、".join(c["id"] for c in meta["components"]) + "。新目标固定 T030 的 800 轮；历史伙伴保留其原 trial 的完整参数及实际轮数。",
        f"构造 {len(raw_signals)} 个信号 × {len(meta['controllers'])} 个控制器 = {len(results)} 个完整配置；单模型端点复用真实来源，完整排序/并列组等价时共享评分并公开映射。CV 新训练次数为 0。",
        "融合使用现有 rank_blend 的当日平均百分位。为避免 CSV 解析合并浮点近似并列，融合后用当日平均排名的两倍整数保存；它严格保持原生 rank_blend 的实际顺序和精确并列，原生浮点预测另存无损 Parquet。单模型保留原 pred。输出是排序信号，不解释成收益率。",
        "控制器为 S003 q* band、S004 C018 的按日最小间隔编码和 C042 的 gap(delta=0,rho=0.01)；每折冷启动，处理侧只接收键、预测、涨停标志与历史状态，不接收 y 或缺失 y 掩码。",
        "所有 CSV/评分均为 trade_date、ts_code 升序；官方收益池还删缺 y，换手池仅删涨停，两者分别由官方代码计算。",
        "", "## 完整三折比较", "",
        f"原 S003 完整方案三折均分 {baseline['cv_mean']:.10f}、最差折 {baseline['cv_worst']:.10f}。",
        "选择要求均值提高超过 1e-6，最差折及 2023 不低于参照−1e-6；合格池内按最高均值，同分依次优先最差折、模型更少、编号。",
        selected_note,
        f"不受门槛约束的最高均值是 {best['id']} 的 {best['cv_mean']:.10f}；所有配置均公开。", "",
        markdown(top, ["candidate", "components", "weights", "controller", "cv_mean", "cv_worst", "wf2023", "passes_cv_gate"]),
        "", "完整逐折八指标、raw 指标和排序等价映射分别在 folds / raw_signals / comparison 表；原始 signal 的 PredictiveScore 仅为辅助分析。",
        "", "## 分项归因与弱年表现", ""]
    if selected:
        chosen_row = table[table.candidate == selected["id"]]
        lines += [markdown(chosen_row, ["candidate", "mean_ic", "mean_excess", "mean_turnover", "cv_delta_vs_S003",
            "ic_component_change", "excess_component_change", "stability_component_change"]), "",
            "月度/季度沿用全年已计算的状态，不重新训练或在月初重启。持仓段包含折末右删失，按交易观察计数；实际 Top 从保存后的 pred 解码核对。",
            "", "### 2023 全部月份", "",
            markdown(monthly_table[(monthly_table.candidate.isin(["F000", selected["id"]])) & (monthly_table.year == 2023)],
                     ["candidate", "month", "ic_mean", "annual_excess", "mean_turnover", "final_score"]), ""]
    lines += ["## 2024 一次历史确认", ""]
    if confirmation:
        lines += [f"冻结选择于 `{confirmation['lock_commit']}` 先提交、推送 main，再确认；新增模型训练 {confirmation['new_model_fits']} 次。",
            f"锁定方案 2024 综合分 **{confirmation['final_score']:.10f}**，门槛 {confirmation['threshold']:.10f}；确认门槛结果 **{confirmation['passes_confirmation_gate']}**，后续结论 `{confirmation['recommendation']}`。",
            "2024 不重新选择组成模型、权重、控制器或参数。诊断层按锁定清单预先规定，重合排序共享真实评分。", "",
            markdown(pd.DataFrame(confirmation_rows), ["layer", "ic_mean", "annual_excess", "mean_turnover", "final_score"]), "",
            "### 2024 全部月份（含 1 月）", "",
            markdown(monthly_table[(monthly_table.candidate.isin(["F000", selected["id"]])) & (monthly_table.year == 2024)],
                     ["candidate", "month", "ic_mean", "annual_excess", "mean_turnover", "final_score"]), ""]
    else:
        lines += ["本报告为三折锁定结果。合格候选须先提交、推送冻结清单，才可进行预定的 2024 确认。" if selected else "无合格候选，已按协议保留原方案。", ""]
    lines += ["## 实际核对与交接", "",
        f"所有实际保存评分的官方八指标最大差 {audited['max_official_difference']:.3e}；模型重载最大差 {audited['max_reload_difference']:.3e}。",
        f"原 {audited['initial_records']} 条共享记录保持不变，新增 {audited['new_records']} 条真实记录，共 {audited['total_records']} 条；当前研究新拟合 {audited['new_model_fits']} 次。",
        f"锁定 canonical digest `{selection['selection_digest']}`，预定信号/控制器前缀重放 {len(selection['prefix_replays'])} 折通过。原始数据、官方附件和所有已冻结来源保持核对。",
        "来源、运行清单和完整产物 SHA 在 artifacts 索引；history 表保留初筛、相关性与去重理由。大型模型、native Parquet、预测 CSV 和状态仅保留本地忽略目录。",
        "2021–2024 均为已经使用过的历史验证期，累计筛选存在选择偏差；正式测试得分需要官方隐藏标签。成员 B 独立复核与正式方案/测试初始状态由团队后续接续。本轮不生成比赛 submission，不新增或运行测试套件。",
        "", "## 图表", ""]
    for path in images:
        lines += [f"![{path.stem}](figures/{path.name})", ""]
    write_text(report, "\n".join(lines))
    selection_path = reports / f"alpha_research_{meta['study_id']}_selection.json"
    save_json(selection_path, selection)
    if confirmation:
        save_json(reports / f"alpha_research_{meta['study_id']}_confirmation.json", confirmation)
    audited.update(selection_digest=selection["selection_digest"], prefix_replays=selection["prefix_replays"],
                   historical_partner_ids=meta["history_partners"], unique_complete_schemes=meta["unique_schemes"],
                   evaluated_years=[2021, 2022, 2023]+([2024] if confirmation else []))
    paths = [report, selection_path, *images, *sorted(experiments.glob(f"{prefix}_*.csv"))]
    if confirmation:
        paths.append(reports / f"alpha_research_{meta['study_id']}_confirmation.json")
    audited["published_files"] = {relative(p): {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in paths}
    save_json(root / "audit.json", audited)
    save_json(reports / f"alpha_research_{meta['study_id']}_artifacts.json", audited)

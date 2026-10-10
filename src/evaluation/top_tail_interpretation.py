"""Reporting-only supplement after the immutable S009 numerical run completes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.top_tail_handoff import MODEL_COLORS
from src.evaluation.top_tail_tables import sorted_indices
from src.models.alpha_research import relative, save_csv, write_text
from src.models.optuna_tuning import read_json, save_json
from src.models.target_research import table_markdown
from src.utils.project import ROOT, load_config, sha256, timestamp
from src.utils.research_cache import contained_path


def publish():
    cfg, _ = load_config()
    root = contained_path(cfg["paths"]["research_studies"]) / "S009"
    meta, audit = read_json(root / "study.json"), read_json(root / "audit.json")
    if meta["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Only describe the completed and audited numerical run.")
    for path, expected in meta["source"].items():
        if sha256(contained_path(path)) != expected:
            raise ValueError(f"Executed numerical source changed: {path}")
    for path, entry in audit["published_files"].items():
        if sha256(contained_path(path)) != entry["sha256"]:
            raise ValueError(f"Published input changed: {path}")
    preparation = read_json(root / "context.json")
    rows, daily_rows = [], []
    for entry in audit["runs"]:
        cell = read_json(contained_path(entry["manifest"]))
        info = preparation["contexts"][cell["fold"]]
        if sha256(contained_path(info["path"])) != info["sha256"]:
            raise ValueError("Context changed.")
        context = pd.read_parquet(contained_path(info["path"]))
        path = contained_path(cell["source_prediction"])
        if sha256(path) != meta["frozen_inputs"][relative(path)]:
            raise ValueError("Source prediction changed.")
        values = pd.read_csv(path, usecols=["pred"], dtype={"pred": "float64"}).pred.to_numpy()
        n = context.ts_code.nunique(); shape = (context.trade_date.nunique(), n)
        values = values.reshape(shape)
        flags = context.flag_limit_up.to_numpy().reshape(shape) == 0
        quote = context.quote_valid.to_numpy().reshape(shape)
        labels = np.isfinite(context.y_ret_1d.to_numpy().reshape(shape))
        days = context.trade_date.drop_duplicates().to_numpy()
        local = []
        for t, day in enumerate(days):
            order = sorted_indices(values[t], flags[t]); size = len(order)//10
            boundary = values[t, order[size-1]]
            tied = flags[t] & (values[t] == boundary)
            before = int((flags[t] & (values[t] > boundary)).sum())
            tied_n = int(tied.sum())
            row = {"model": cell["model"], "year": cell["year"], "layer": cell["layer"],
                "trade_date": int(day), "top_n": size, "turn_boundary_tie_n": tied_n,
                "turn_boundary_crossed": before < size < before+tied_n,
                "tie_quote_invalid_n": int((tied & ~quote[t]).sum()),
                "tie_missing_label_n": int((tied & ~labels[t]).sum())}
            local.append(row); daily_rows.append(row)
        f = pd.DataFrame(local)
        directory = contained_path(entry["manifest"]).parent
        ret_ties = pd.read_csv(directory / "ties_daily.csv")
        sens = pd.read_csv(directory / "order_sensitivity.csv")
        rows.append({"model": cell["model"], "year": cell["year"], "layer": cell["layer"], "days": len(f),
            "turn_boundary_crossed_days": int(f.turn_boundary_crossed.sum()),
            "return_boundary_crossed_days": int(ret_ties.boundary_crossed.sum()),
            "turn_boundary_tie_n_mean": f.turn_boundary_tie_n.mean(),
            "tie_missing_label_n_mean": f.tie_missing_label_n.mean(),
            "tie_quote_invalid_n_mean": f.tie_quote_invalid_n.mean(),
            "max_abs_order_score_delta": sens.final_delta.abs().max(),
            "max_abs_order_excess_delta": sens.excess_delta.abs().max(),
            "max_abs_order_turnover_delta": sens.turnover_delta.abs().max(),
            "fallback_turn_top_mean": ret_ties.fallback_in_turn_top.mean(),
            "fallback_return_top_mean": ret_ties.fallback_in_return_top.mean()})
    summary = pd.DataFrame(rows)
    save_csv(root / "tables/turn_boundary_daily.csv", pd.DataFrame(daily_rows))
    experiments = contained_path(cfg["paths"]["experiment_log"]).parent
    summary_path = experiments / "top_tail_S009_turn_boundary_summary.csv"
    save_csv(summary_path, summary)
    boundary = pd.read_csv(experiments / "top_tail_S009_boundary_summary.csv")
    official = pd.read_csv(experiments / "top_tail_S009_official.csv")
    events = boundary[boundary.layer == "raw"].merge(official[["model", "year", "layer", "annual_excess"]],
                                                  on=["model", "year", "layer"], validate="one_to_one")
    events["candidate20_nominal_random_reference"] = .2
    events["candidate30_nominal_random_reference"] = .3
    save_csv(experiments / "top_tail_S009_returns_vs_events.csv", events)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for layer, ax in zip(["raw", "band"], axes):
        for model, color in MODEL_COLORS.items():
            f = summary[(summary.model == model) & (summary.layer == layer)].sort_values("year")
            ax.plot(f.year, f.turn_boundary_crossed_days/f.days*100, marker="o", label=model, color=color)
        ax.set_title(f"Turnover-pool Top10 boundary ties: {layer}")
        ax.set_xticks(range(2021, 2025)); ax.set_ylabel("Days with a straddling exact tie (%)"); ax.grid(alpha=.2)
        ax.set_ylim(0, max(1, (summary.turn_boundary_crossed_days/summary.days*100).max()*1.12))
    axes[0].legend(fontsize=8)
    plot = contained_path(cfg["paths"]["figures"]) / "S009/S009_turn_boundary.png"
    fig.savefig(plot, dpi=150); plt.close(fig)
    published_plot = ROOT / "docs/figures" / plot.name
    published_plot.write_bytes(plot.read_bytes())
    y4 = events[events.model == "Y4"]
    public_report = ROOT / "docs/top_tail_S009.md"
    sensitivity = pd.read_csv(experiments / "top_tail_S009_order_sensitivity.csv")
    worst = sensitivity.loc[sensitivity.final_delta.abs().idxmax()]
    band_delta = sensitivity.loc[sensitivity.layer == "band", "final_delta"].abs().max()
    cohorts = pd.read_csv(experiments / "top_tail_S009_cohorts.csv")
    contributions = cohorts[(cohorts.period_type == "year") & (cohorts.pool == "return") &
                            (cohorts.layer == "band") & cohorts.model.isin(["T030", "Y4"])]
    conclusions = ["## 主要发现与后续决定", "",
        "1. **来源与官方计算已通过。** 38 个主比较、76 个行序敏感性评分与原官方附件核对；"
        "主比较和来源八指标差异均为 0。11 个历史模型重载最大差 8.882e-16，19 次 q* 真实前缀和市场特征前缀通过。"
        "没有新拟合，共享表原 718 条逐条不变。S007/T033 缺原模型，Y1 缺 2024；这些缺口保留。",
        "2. **存在头部改进空间，但不是没有 Alpha。** Y4 raw 在 2021–2024 的 Top20 区域 IC 约 0.016–0.022，"
        "剩余 80% 约 0.099–0.121；限制预测排名范围本身也会降低相关性，不能把全部 IC 改善归因于中部。"
        "5–10% 相对 10–15% 的收益差在三个 CV 年份中两个的 95% 区间跨 0，2024 也跨 0。"
        "年度分层收益整体有序、raw 超额为正，因此结果支持有限头部实验，不证明现有回归无效。",
        "3. **事件召回与收益金额要分开。** Y4 的因果 Top20 候选只召回未来 Top10 收益事件约 14–16%，"
        "Top30 约 21–23%；它仍有正超额。单纯提高极端事件命中未必提高平均组合收益。"
        "S010 保留全市场 B10/B20 对照，R20/R30/L30 还要面对池外事件无法恢复的限制。",
        f"4. **并列问题集中在 raw 换手池。** 收益池边界跨切线并列合计 {int(summary.return_boundary_crossed_days.sum())} 个单元日，"
        f"换手池合计 {int(summary.turn_boundary_crossed_days.sum())} 个单元日（同一交易日可出现于多个模型）。"
        f"最大行序分差为 {worst.final_delta:+.10f}，出现在 {worst.model} / {int(worst.year)} / {worst.layer} / {worst['order']}；"
        f"IC 和超额差为 0，变化来自换手。本次已生成的 q* 分数文件直接改行序最大分差 {band_delta:.1e}。"
        "该项没有重新用置换 raw 运行 band，不能据此断言控制器对任意原始行序都不敏感。",
        "5. **留仓解释受两种池影响。** T030 的四年超额贡献主要来自收益池中的留仓组；"
        "Y4 在 2022/2023 的贡献主要来自不属于昨日换手 Top 的收益池成员，2024 则回到留仓组。"
        "这些补入成员可能由缺 y 过滤产生，不能直接称为控制器买入 Alpha，也不能把缺价占位成员当作可交易仓位。",
        "6. **先进入 S010，暂缓 S011，保留 S003。** CV 的六种市场状态均未在至少两个有足够日数的年份"
        "重复获得满足 IC 条件的正超额优势；2024 单年改善不足以支持门控。按执行前门槛，S010 条件成立、"
        "S011 条件不成立。下一阶段仍按最多 18 次拟合与锁定后一次历史检查执行，正式参照不变。", "",
        "以下为收益池人数加权、按全年日数年化的贡献；三组相加恢复对应官方年化超额。", "",
        table_markdown(contributions, ["model", "year", "cohort", "annual_excess_contribution"]), ""]
    extra = ["## 两种边界的补充判读", "",
        "精确并列不能只看收益池：同一组缺价 raw 0 可能被收益池的缺 y 条件删除，仍保留在换手池。"
        "因此收益边界没有 ties，并不等于所有官方指标对行序都不敏感。以下直接定位换手边界，"
        "并引用已完成的行序评分差；没有重新评分或更换正式排序。", "",
        table_markdown(summary, ["model", "year", "layer", "turn_boundary_crossed_days", "return_boundary_crossed_days",
            "tie_missing_label_n_mean", "max_abs_order_score_delta", "max_abs_order_excess_delta", "max_abs_order_turnover_delta"]), "",
        "收益 Top 与实际换手 Top 的不同还会影响留仓解释：H_ret 中不属于昨日 H_turn 的股票，"
        "可能是官方缺 y 过滤之后的补入者，不能全部叫作控制器新买入。"
        "缺价占位股也不能据此解释为可交易仓位，本报告只复现比赛集合与统计口径。", "",
        "## 事件召回与收益金额的区别", "",
        table_markdown(y4, ["model", "year", "annual_excess", "gap_annualized", "gap_ci_lower", "gap_ci_upper",
            "precision10", "inference_recall20", "inference_recall30"]), "",
        "Top20/30 候选中未来极端 Top10 收益股票的召回，不能直接当作组合盈利能力；"
        "约 0.2 / 0.3 仅是均匀随机候选的名义覆盖参照，实际池、缺标签与 ties 会改变精确随机基线。"
        "Y4 的 raw 超额为正却未捕获多数极端收益事件，说明‘预测收益金额’和‘属于未来高分位的概率’"
        "需要分别检验。该结果不证明二分类优于回归；S010 应先保留全市场 B10/B20 的对照，"
        "同时核对事件命中、收益幅度、波动暴露及完整 q* 后的超额。精排也无法恢复池外股票。", "",
        "本补充只读取已通过输入与表格、派生计数和解释图，不拟合、不重评预测、不改变诊断分流或正式参照。", "",
        f"![换手池边界](figures/{published_plot.name})", ""]
    text = public_report.read_text(encoding="utf-8").split("\n## 两种边界的补充判读", 1)[0].rstrip()
    lead, body = text.split("\n## 官方口径与来源验收", 1)
    lead = lead.split("\n## 主要发现与后续决定", 1)[0].rstrip()
    write_text(public_report, lead+"\n\n"+"\n".join(conclusions)+"\n## 官方口径与来源验收"+body+"\n\n"+"\n".join(extra))
    source_path = ROOT / "src/evaluation/top_tail_interpretation.py"
    audit["descriptive_publication"] = {"time": timestamp(), "source": relative(source_path), "sha256": sha256(source_path),
        "new_model_fits": 0, "new_official_scores": 0, "decision_changed": False}
    audit["visual_inspection_complete"] = False
    audit.pop("visual_inspection", None)
    audit.pop("final_handoff", None)
    published = [public_report, ROOT / "docs/top_tail_S009_decision.json",
                 *sorted((ROOT / "docs/figures").glob("S009_*.png")), *sorted(experiments.glob("top_tail_S009_*.csv"))]
    audit["published_files"] = {relative(p): {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in published}
    audit["descriptive_daily_artifact"] = {"path": relative(root / "tables/turn_boundary_daily.csv"),
                                            "sha256": sha256(root / "tables/turn_boundary_daily.csv")}
    save_json(root / "audit.json", audit); save_json(ROOT / "docs/top_tail_S009_artifacts.json", audit)
    print("S009 interpretive supplement complete; no fits, official scores or decision changes")


if __name__ == "__main__":
    publish()

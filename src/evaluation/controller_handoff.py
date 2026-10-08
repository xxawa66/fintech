"""Add descriptive interpretation and a Pareto figure to completed S004 results.

No predictions, model fitting, scoring, configuration selection or frozen
selection files are changed. The reporting code is indexed separately.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.alpha_research import protocol, read_json, save_json, write_text
from src.utils.project import ROOT, load_config, sha256, timestamp
from src.utils.research_cache import contained_path, digest


def publish_handoff(cfg, study_id):
    root = contained_path(cfg["paths"]["research_studies"]) / study_id
    state, audit = read_json(root / "study.json"), read_json(root / "audit.json")
    selection = read_json(root / "selection.json")
    if state["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Descriptive publication requires completed, audited S004 results.")
    if digest(protocol(cfg)) != state["protocol_digest"]:
        raise ValueError("The numerical research protocol changed.")
    if digest({k: v for k, v in selection.items() if k != "selection_digest"}) != state["selection_digest"]:
        raise ValueError("The frozen controller selection changed.")
    # Extra reporting modules are allowed after completion. Every file that was
    # actually executed by the research must remain byte-identical.
    for path, expected in state["source"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Executed research source changed: {path}")
    for path, entry in audit["published_files"].items():
        if sha256(ROOT / path) != entry["sha256"]:
            raise ValueError(f"Published input changed before descriptive export: {path}")
    prefix = f"alpha_{study_id}"
    experiments = contained_path(cfg["paths"]["experiment_log"]).parent
    table = pd.read_csv(experiments / f"{prefix}_comparison.csv")
    folded = pd.read_csv(experiments / f"{prefix}_folds.csv")
    holdings = pd.read_csv(experiments / f"{prefix}_holdings.csv")
    unique = table[table.alias_of.isna()].copy()
    baseline = table[table.candidate == selection["baseline"]["id"]].iloc[0]
    highest = unique.sort_values("cv_mean", ascending=False).iloc[0]
    band_set_checks, max_portfolio_diff = 0, 0.
    candidate_files = [read_json(path) for path in sorted((root / "candidates").glob("*.json"))]
    old = {r["spec"]["keep_q"]: r for r in candidate_files if r["spec"]["method"] == "band"}
    minimal = {r["spec"]["keep_q"]: r for r in candidate_files if r["spec"]["method"] == "band_minimal"}
    for q, reference in old.items():
        for a, b in zip(reference["folds"], minimal[q]["folds"]):
            for metric in ["annual_excess", "top1_annual_ret", "mean_turnover"]:
                max_portfolio_diff = max(max_portfolio_diff, abs(a["metrics"][metric] - b["metrics"][metric]))
            with np.load(contained_path(cfg["paths"]["metrics"]) / a["exp_id"] / "top_sets.npz", allow_pickle=False) as aa:
                with np.load(contained_path(cfg["paths"]["metrics"]) / b["exp_id"] / "top_sets.npz", allow_pickle=False) as bb:
                    if not np.array_equal(aa["top"], bb["top"]):
                        raise ValueError("Same-set encoding changed a legacy band's actual Top set.")
            band_set_checks += 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = contained_path(cfg["paths"]["figures"]) / study_id
    figure = figures / "pareto_metrics.png"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    points = ax.scatter(unique.mean_ic, unique.mean_excess, c=unique.mean_turnover,
                        cmap="viridis", s=45, alpha=.85)
    pareto = unique[unique.pareto]
    ax.scatter(pareto.mean_ic, pareto.mean_excess, facecolors="none", edgecolors="#c43c39",
               s=110, linewidth=1.3, label="Non-dominated in IC / excess / turnover")
    ax.scatter(baseline.mean_ic, baseline.mean_excess, marker="*", s=170,
               color="black", label="S003 reference")
    for chosen in selection["controllers"]:
        row = table[table.candidate == chosen["id"]].iloc[0]
        ax.annotate(chosen["id"], (row.mean_ic, row.mean_excess),
                    xytext=(7, 12 if chosen["family"] == "band" else -17), textcoords="offset points")
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Mean official Jaccard turnover")
    ax.set(xlabel="Mean daily Rank IC (2021-2023)", ylabel="Mean annual excess (2021-2023)",
           title=f"{len(unique)} distinct controller orderings: three-metric Pareto diagnostics")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(figure, dpi=180); plt.close(fig)
    public = contained_path(cfg["paths"]["research_reports"])
    published_figure = public / "figures" / f"{prefix}_pareto_metrics.png"
    shutil.copyfile(figure, published_figure)
    # Keep the yearly legend above the bars and leave headroom in attribution.
    ids = [baseline.candidate] + [r["id"] for r in selection["controllers"] if r["id"] != baseline.candidate]
    chosen_rows = table.set_index("candidate").loc[ids]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(3)
    for i, (candidate, row) in enumerate(chosen_rows.iterrows()):
        ax.bar(x + (i-(len(chosen_rows)-1)/2)*.23,
               [row.wf2021, row.wf2022, row.wf2023], .23, label=candidate)
    ax.set(xticks=x, xticklabels=["2021", "2022", "2023"], ylabel="Official final score")
    ax.set_title("Frozen research controllers and S003 reference", pad=38)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(figures / "year_scores.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    columns = ["ic_component_change", "excess_component_change", "stability_component_change"]
    for i, column in enumerate(columns):
        ax.bar(np.arange(len(ids)) + (i-1)*.24, chosen_rows[column], .24,
               label=["IC", "Excess", "Stability"][i])
    limits = chosen_rows[columns].to_numpy()
    lo, hi = float(limits.min()), float(limits.max())
    span = max(hi-lo, 1e-6)
    ax.set_ylim(lo-.08*span, hi+.18*span)
    ax.axhline(0, color="black", linewidth=.6)
    ax.set(xticks=np.arange(len(ids)), xticklabels=ids,
           ylabel="Weighted CV score change vs S003", title="Which scoring component changed?")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(figures / "attribution.png", dpi=180); plt.close(fig)
    for name in ["year_scores.png", "attribution.png"]:
        destination = public / "figures" / f"{prefix}_{name}"
        shutil.copyfile(figures / name, destination)
        audit["published_files"][destination.relative_to(ROOT).as_posix()] = {
            "sha256": sha256(destination), "size_bytes": destination.stat().st_size}
    gap = table[table.candidate == "C042"].iloc[0]
    gap2023 = folded[(folded.candidate == "C042") & (folded.fold == "wf2023")].iloc[0]
    base2023 = folded[(folded.candidate == baseline.candidate) & (folded.fold == "wf2023")].iloc[0]
    base_hold = holdings[(holdings.candidate == baseline.candidate) & (holdings.fold == "wf2023")].iloc[0]
    gap_hold = holdings[(holdings.candidate == "C042") & (holdings.fold == "wf2023")].iloc[0]
    bonus_best = table[table.family == "bonus"].sort_values("cv_mean", ascending=False).iloc[0]
    report = public / f"alpha_research_{study_id}.md"
    marker = "## 本轮结论与后续决策"
    content = report.read_text(encoding="utf-8").split(marker)[0].rstrip()
    lines = [content, "", marker, "",
             f"**固定 T030 的本轮控制器网格没有获得实质性提分。** 最高 CV 均分为 {highest.cv_mean:.10f}，",
             f"相对 S003 仅增加 {highest.cv_mean-baseline.cv_mean:.10f}，收益与换手相同，变化全部来自编码后的 IC。",
             "该变化略高于计划的数值替换容差，但不能作为新的 Alpha 提升或隐藏测试改善证据。",
             "本阶段不变更正式推荐，继续保留 S003 完整方案作为参照。", "",
             f"留仓奖励最高均分为 {bonus_best.cv_mean:.10f}（{bonus_best.candidate}），没有超过参照。",
             f"C042 的均分为 {gap.cv_mean:.10f}：2021 / 2022 小幅改善，2023 从 {base2023.final_score:.10f}",
             f"下降到 {gap2023.final_score:.10f}；其 2023 年化超额从 {base2023.annual_excess:.10f}",
             f"下降到 {gap2023.annual_excess:.10f}，换手从 {base2023.mean_turnover:.10f} 升到 {gap2023.mean_turnover:.10f}。",
             "因此 C042 不满足计划的跨年替换条件，只作为第二种结构不同的控制器交给后续融合研究。", "",
             "**旧方案的低换手确实伴随长期保留原始排名已经降低的股票。** 2023 的平均持仓段约",
             f"{base_hold.mean_spell_days:.2f} 个交易观察，持仓股当日原始平均排名分位为 {base_hold.mean_selected_raw_rank:.4f}；",
             f"当日原始 Top 的平均保留比例为 {base_hold.mean_raw_top_retained_fraction:.2%}。",
             f"C042 将平均持仓段缩到 {gap_hold.mean_spell_days:.2f}，原始平均分位升到 {gap_hold.mean_selected_raw_rank:.4f}，",
             "但该年的官方收益反而下降。排序更新更快与实现更多收益，在本数据上没有自动等价。",
             "持仓段在折末有右删失，这些长度用于描述实际留仓，不是无偏的未来持仓期限估计。", "",
             f"**48 个按日编码对照的实际 Top 集合全部相同**；年化超额、Top 绝对收益和换手的最大差为",
             f"{max_portfolio_diff:.3e}。IC 包含涨停股，Top 收益和换手则剔除涨停股，所以同一选股集合下",
             "改变完整 pred 的相对排序，仍能造成小幅 IC 差异。", "",
             "排名差方法中，五种替换上限各自对应的 delta=0 / 0.025 / 0.05 / 0.1 / 0.2",
             "在全部三折的完整排序与并列组上相同；20 个重复配置已映射回首个配置。",
             "这个范围的排名差阈值没有区分最终结果，不能从中声称找到了一个唯一最优 delta。",
             "C042 固定 delta=0 是预定同分时选较早编号的结果。", "",
             "下一步按原计划进入 S005：固定 40 特征与 T030 参数比较训练目标，先看 raw IC 和超额；",
             "再由 S006 比较排名融合与 C018 / C042 / S003 band。正式替换需要完整方案的跨年证据及成员 B 复核。", "",
             "### 三指标非支配图", "",
             f"![IC、超额与换手非支配图](figures/{prefix}_pareto_metrics.png)", "",
             "横轴 IC、纵轴年化超额、颜色为换手；红色边圈按三项均值标记非支配点。",
             "这是三指标描述性诊断，最终候选仍由预定官方综合分规则选取。"]
    write_text(report, "\n".join(lines))
    helper = Path(__file__)
    audit["descriptive_publication"] = {"time": timestamp(), "helper": helper.relative_to(ROOT).as_posix(),
        "helper_sha256": sha256(helper), "same_band_sets_checked": band_set_checks,
        "same_band_portfolio_metric_max_difference": max_portfolio_diff,
        "unique_pareto_points": len(pareto), "new_model_fits": 0, "new_scored_configurations": 0,
        "selection_unchanged": True, "note": "Post-completion interpretation; no search or 2024 evaluation."}
    audit["published_files"][published_figure.relative_to(ROOT).as_posix()] = {
        "sha256": sha256(published_figure), "size_bytes": published_figure.stat().st_size}
    audit["published_files"][report.relative_to(ROOT).as_posix()] = {
        "sha256": sha256(report), "size_bytes": report.stat().st_size}
    audit["reporting_figures"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in figures.glob("*.png")}
    save_json(root / "audit.json", audit)
    save_json(public / f"alpha_research_{study_id}_artifacts.json", audit)
    print(f"descriptive_handoff_passed: sets={band_set_checks}; max_difference={max_portfolio_diff:.3e}; "
          f"pareto_points={len(pareto)}; score_gain={highest.cv_mean-baseline.cv_mean:.10f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study-id", default="S004")
    args = parser.parse_args(argv)
    cfg, _ = load_config(args.config)
    publish_handoff(cfg, args.study_id)


if __name__ == "__main__":
    main()

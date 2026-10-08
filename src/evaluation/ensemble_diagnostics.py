"""Descriptive S006 confirmation attribution; no fitting, scoring or selection."""
from __future__ import annotations

import argparse
import re

import pandas as pd

from src.models.alpha_research import relative, save_csv, write_text
from src.models.ensemble_research import protocol
from src.models.optuna_tuning import read_json, save_json
from src.models.target_research import table_markdown
from src.utils.project import ROOT, load_config, sha256, timestamp
from src.utils.research_cache import contained_path, digest


def publish_diagnostics(cfg, study_id):
    if not re.fullmatch(r"S006(?:_[A-Za-z0-9_-]+)?", study_id):
        raise ValueError("Use a completed S006 study identity.")
    root = contained_path(cfg["paths"]["research_studies"]) / study_id
    meta, audit = read_json(root / "study.json"), read_json(root / "audit.json")
    selection, confirmation = read_json(root / "selection.json"), read_json(root / "confirmation.json")
    if (meta["status"] != "complete" or audit["status"] != "passed"
            or digest(protocol(cfg)) != meta["protocol_digest"]
            or digest({k: v for k, v in selection.items() if k != "selection_digest"}) != meta["selection_digest"]
            or confirmation["selection_digest"] != meta["selection_digest"]):
        raise ValueError("Descriptive diagnostics need the unchanged completed study and selection.")
    # Additional descriptive source is allowed; executed numerical source stays immutable.
    for path, expected in meta["source"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Executed numerical source changed: {path}")
    for path, entry in audit["published_files"].items():
        if sha256(ROOT / path) != entry["sha256"]:
            raise ValueError(f"Published input changed: {path}")
    prefix = f"alpha_{study_id}"
    experiments = contained_path(cfg["paths"]["experiment_log"]).parent
    reports = contained_path(cfg["paths"]["research_reports"])
    scores = pd.read_csv(experiments / f"{prefix}_confirmation.csv")
    base = scores[scores.layer == "T030_qstar"].iloc[0]
    attribution = []
    for row in scores.itertuples(index=False):
        ic = .4*(row.ic_mean-base.ic_mean)
        excess = .3*(row.annual_excess-base.annual_excess)
        stability = -.3*(row.mean_turnover-base.mean_turnover)
        delta = row.final_score-base.final_score
        if abs(ic+excess+stability-delta) > 1e-10:
            raise ValueError("2024 attribution does not add up.")
        attribution.append({"layer": row.layer, "weighted_ic_delta": ic, "weighted_excess_delta": excess,
                            "weighted_stability_delta": stability, "final_delta_vs_S003": delta})
    attribution = pd.DataFrame(attribution)
    save_csv(experiments / f"{prefix}_confirmation_attribution.csv", attribution)
    monthly = pd.read_csv(experiments / f"{prefix}_monthly.csv")
    winner = selection["winner"]["id"]
    base_months = monthly[monthly.candidate == "F000"].set_index(["year", "month"])
    chosen_months = monthly[monthly.candidate == winner].set_index(["year", "month"])
    rows = []
    for key, new in chosen_months.iterrows():
        old = base_months.loc[key]
        rows.append({"year": int(key[0]), "month": int(key[1]), "new_final": new.final_score,
            "S003_final": old.final_score, "final_delta": new.final_score-old.final_score,
            "weighted_ic_delta": .4*(new.ic_mean-old.ic_mean),
            "weighted_excess_delta": .3*(new.annual_excess-old.annual_excess),
            "weighted_stability_delta": -.3*(new.mean_turnover-old.mean_turnover)})
    month_changes = pd.DataFrame(rows)
    save_csv(experiments / f"{prefix}_selected_monthly_changes.csv", month_changes)
    stability_rows = []
    for year, group in month_changes.groupby("year"):
        new_worst, old_worst = group.loc[group.new_final.idxmin()], group.loc[group.S003_final.idxmin()]
        stability_rows.append({"year": int(year), "months": len(group),
            "months_above_S003": int((group.final_delta > 0).sum()),
            "new_worst_month": int(new_worst.month), "new_worst_month_score": new_worst.new_final,
            "S003_worst_month": int(old_worst.month), "S003_worst_month_score": old_worst.S003_final,
            "largest_positive_delta": group.final_delta.max(), "largest_negative_delta": group.final_delta.min()})
    stability = pd.DataFrame(stability_rows)
    save_csv(experiments / f"{prefix}_selected_stability.csv", stability)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected_row = attribution[attribution.layer == "new_complete"].iloc[0]
    values = [selected_row.weighted_ic_delta, selected_row.weighted_excess_delta, selected_row.weighted_stability_delta]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bars = ax.bar(["IC", "Annual excess", "Stability"], values, color=["#3886aa", "#63a273", "#bb874e"])
    ax.axhline(0, color=".25", lw=.8)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.6f}", (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 6 if value >= 0 else -14), textcoords="offset points", ha="center")
    span = max(values)-min(values)
    ax.set_ylim(min(0, min(values))-.25*span, max(0, max(values))+.25*span)
    ax.set_title(f"Locked {winner}, 2024 vs S003; total {sum(values):+.6f}")
    ax.set_ylabel("Weighted contribution to official score")
    image_path = contained_path(cfg["paths"]["figures"]) / study_id / f"{prefix}_confirmation_attribution.png"
    fig.savefig(image_path, dpi=150); plt.close(fig)
    published_image = reports / "figures" / image_path.name
    published_image.write_bytes(image_path.read_bytes())
    january = month_changes[month_changes.month == 202401].iloc[0]
    worst_delta = month_changes[month_changes.year == 2024].sort_values("final_delta").head(3)
    raw = scores.set_index("layer")
    raw_predictive_delta = .4*(raw.loc["new_signal_raw"].ic_mean-raw.loc["T030_raw"].ic_mean) + .3*(raw.loc["new_signal_raw"].annual_excess-raw.loc["T030_raw"].annual_excess)
    verdict = ("锁定候选通过既定确认门槛，交接成员 B 复核，团队后续确定正式方案。" if confirmation["passes_confirmation_gate"] else
               "锁定候选未通过既定确认门槛，本轮结论为保留 S003。预定诊断中更高的其他分数不替代已锁定方案；没有追加模型、权重或控制器搜索。")
    extra = ["## 最终判读与交接决定", "", verdict, "",
        f"三折 CV 均值达到 {selection['winner']['cv_mean']:.10f}，超过 0.4；2024 锁定结果为 {confirmation['final_score']:.10f}，"
        f"相对 S003 变化 {selected_row.final_delta_vs_S003:+.10f}。跨年门槛按事先规则判断，不能仅凭 CV 最高值改用正式方案。",
        f"2024 的 IC 加权变化 {selected_row.weighted_ic_delta:+.10f}，超额项 {selected_row.weighted_excess_delta:+.10f}，"
        f"稳定性项 {selected_row.weighted_stability_delta:+.10f}；IC 的改善被另两项抵消。", "",
        table_markdown(attribution, ["layer", "weighted_ic_delta", "weighted_excess_delta", "weighted_stability_delta", "final_delta_vs_S003"]), "",
        f"同年原始 Y4 信号的 IC 提高，但超额低于 T030，raw PredictiveScore 变化 {raw_predictive_delta:+.10f}。"
        "这说明目标研究在固定预算下的跨年收益优势没有完整保持，不能把原始综合分的微升等同于预测收益项提高。",
        f"预定诊断 Y4 + q* band 得分 {raw.loc['new_signal_qstar'].final_score:.10f}，"
        f"而 T030 + gap 得分 {raw.loc['T030_selected_controller'].final_score:.10f}；控制器在 2024 的取舍与 CV 不同。"
        "这些层用于归因；重新选用其中任何方案需要后续独立冻结协议。", "",
        "### 全年与弱月份", "",
        table_markdown(stability, ["year", "months_above_S003", "new_worst_month", "new_worst_month_score", "S003_worst_month", "S003_worst_month_score"]), "",
        f"2024 年 1 月从 {january.S003_final:.10f} 提高到 {january.new_final:.10f}，"
        f"变化 {january.final_delta:+.10f}；该月改善仍不能抵消全年其他时期的退化。2024 退步最多的三个月如下。", "",
        table_markdown(worst_delta, ["month", "S003_final", "new_final", "final_delta"]), "",
        "月度量是该月日均指标的年化统计，不是年度复合收益；季度、日度及完整月份都保留。",
        "", f"![2024 分项](figures/{published_image.name})", "",
        "该判读读取已通过的实际表格，不拟合、不重评预测、不改变冻结选择。数值源码 / 配置 / 依赖与各产物摘要保留，"
        "本次报告模块单独索引。S004–S006 实际新增训练为 0 + 15 + 1 = 16 次，低于原定 18 次上限。"
        "正式测试预测、提交生成、团队初始状态及成员 B 复核由后续接续。", ""]
    report = reports / f"alpha_research_{study_id}.md"
    text = report.read_text(encoding="utf-8")
    text = text.split("\n## 最终判读与交接决定", 1)[0].rstrip()
    write_text(report, text+"\n\n"+"\n".join(extra))
    audit["descriptive_publication"] = {"time": timestamp(), "reporting_module": relative(ROOT / "src/evaluation/ensemble_diagnostics.py"),
        "sha256": sha256(ROOT / "src/evaluation/ensemble_diagnostics.py"), "new_model_fits": 0,
        "new_scored_configurations": 0, "selection_changed": False}
    paths = [report, reports / f"alpha_research_{study_id}_selection.json", reports / f"alpha_research_{study_id}_confirmation.json",
             *sorted((reports / "figures").glob(f"{prefix}_*.png")), *sorted(experiments.glob(f"{prefix}_*.csv"))]
    audit["published_files"] = {relative(p): {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in paths}
    save_json(root / "audit.json", audit)
    save_json(reports / f"alpha_research_{study_id}_artifacts.json", audit)
    print(f"{study_id} final interpretation published; decision={confirmation['recommendation']}; no new fits/scores")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default="S006")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args(argv)
    cfg, _ = load_config(args.config)
    publish_diagnostics(cfg, args.study_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

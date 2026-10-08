"""Finalize S005 from completed artifacts; no model fitting or new scoring.

Also handles the initial Markdown list/string publication error. The frozen
numerical implementation is verified against Git; only that formatting adapter
may differ. Model, prediction, target and selection identities stay immutable.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import target_research as tr
from src.models.alpha_research import relative, save_csv, write_text
from src.models.optuna_tuning import ensure_record, provenance, read_json, save_json, study_lock
from src.utils.experiments import read_records
from src.utils.project import ROOT, load_config, sha256, timestamp
from src.utils.research_cache import contained_path, digest


def numerical_source_identity(meta, root):
    changed = []
    formatter_path = "src/models/target_research.py"
    for path, expected in meta["source"].items():
        if sha256(ROOT / path) == expected:
            continue
        if path != formatter_path:
            raise ValueError(f"Executed numerical source changed: {path}")
        original = subprocess.check_output(["git", "show", f"{meta['git']['commit']}:{path}"], cwd=ROOT)
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError("Git does not reproduce the executed numerical source bytes.")
        before, after = ast.parse(original.decode("utf-8")), ast.parse((ROOT / path).read_text(encoding="utf-8"))
        adapter = ast.parse('def table_markdown(frame, columns):\n    return "\\n".join(table_lines(frame, columns))').body[0]
        removed = 0
        for node in list(after.body):
            if isinstance(node, ast.FunctionDef) and node.name == "table_markdown":
                if ast.dump(node) != ast.dump(adapter):
                    raise ValueError("Report adapter differs from the allowed newline conversion.")
                after.body.remove(node)
                removed += 1
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    if name.name == "table_markdown" and name.asname == "table_lines":
                        name.asname = None
        if removed != 1 or ast.dump(before) != ast.dump(after):
            raise ValueError("The change is not restricted to the Markdown formatting adapter.")
        frozen_copy = root / "executed_source" / "target_research.py"
        frozen_copy.parent.mkdir(parents=True, exist_ok=True)
        frozen_copy.write_bytes(original)
        changed.append({"path": path, "executed_sha256": expected, "current_sha256": sha256(ROOT / path),
                        "frozen_copy": relative(frozen_copy), "numeric_AST_unchanged": True})
    return changed


def figures_and_findings(cfg, meta, selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    study_id = meta["study_id"]
    prefix = f"alpha_{study_id}"
    experiments = contained_path(cfg["paths"]["experiment_log"]).parent
    directory = contained_path(cfg["paths"]["figures"]) / study_id
    table = pd.read_csv(experiments / f"{prefix}_comparison.csv")
    folds = pd.read_csv(experiments / f"{prefix}_folds.csv")
    monthly = pd.read_csv(experiments / f"{prefix}_monthly.csv")
    importance = pd.read_csv(experiments / f"{prefix}_importance.csv")
    selected = [p["id"] for p in selection["partners"]]
    palette = dict(zip(table.target, plt.get_cmap("tab10").colors[:len(table)]))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(table))
    for i, year in enumerate([2021, 2022, 2023]):
        data = folds[folds.fold == f"wf{year}"].set_index("target").reindex(table.target)
        ax.bar(x+(i-1)*.24, data.predictive_score, .24, label=str(year))
    ax.set_xticks(x, table.target)
    ax.set_ylabel("Raw PredictiveScore (0.4 IC + 0.3 excess)")
    fig.suptitle("S005: fixed T030 parameters, training targets/losses", y=.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .935), ncol=3, frameon=False)
    fig.subplots_adjust(top=.82, bottom=.1, left=.1, right=.98)
    fig.savefig(directory / f"{prefix}_year_scores.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 5.5), constrained_layout=True)
    offsets = {"Y0": (-44, 18), "Y1": (10, 13), "Y2": (12, -23),
               "Y3": (-40, -23), "Y4": (-44, 16), "Y5": (8, -18)}
    for row in table.to_dict("records"):
        ax.scatter(row["mean_ic_mean"], row["mean_annual_excess"], s=110 if row["target"] in selected else 60,
                   color=palette[row["target"]], edgecolors="black" if row["pareto_ic_excess"] else "none")
        ax.annotate(row["target"] + (" *" if row["target"] in selected else ""),
                    (row["mean_ic_mean"], row["mean_annual_excess"]), xytext=offsets[row["target"]],
                    textcoords="offset points", arrowprops={"arrowstyle": "-", "color": ".45", "lw": .7})
    ax.set_xlabel("Mean daily Rank IC (three-fold mean)"); ax.set_ylabel("Annual excess (three-fold mean)")
    ax.set_title("Raw IC/excess; * = frozen S006 partner; border = Pareto point")
    ax.margins(.2)
    fig.savefig(directory / f"{prefix}_pareto.png", dpi=150); plt.close(fig)
    # Normalize within each model/fold before comparing gain across y scales.
    normalized = importance.groupby(["target", "feature"]).gain_share.mean().unstack("target")
    shown = ["Y0"] + selected
    top = normalized[shown].max(axis=1).nlargest(12).index.tolist()[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    y = np.arange(len(top))
    for i, target in enumerate(shown):
        ax.barh(y+(i-1)*.24, normalized.loc[top, target], .24, color=palette[target], label=target)
    ax.set_yticks(y, top); ax.set_xlabel("Mean per-model normalized gain share (three folds)")
    ax.set_title("S005: importance of baseline and frozen partners")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(directory / f"{prefix}_importance.png", dpi=150); plt.close(fig)
    for path in sorted(directory.glob("*.png")):
        destination = contained_path(cfg["paths"]["research_reports"]) / "figures" / path.name
        destination.write_bytes(path.read_bytes())
    baseline = table.set_index("target").loc["Y0"]
    targets = table.set_index("target")
    attribution = []
    for row in table.to_dict("records"):
        delta_ic = .4*(row["mean_ic_mean"]-baseline.mean_ic_mean)
        delta_excess = .3*(row["mean_annual_excess"]-baseline.mean_annual_excess)
        attribution.append({"target": row["target"], "weighted_ic_delta": delta_ic,
            "weighted_excess_delta": delta_excess, "predictive_delta": delta_ic+delta_excess,
            "stability_delta": -.3*(row["mean_mean_turnover"]-baseline.mean_mean_turnover),
            "raw_final_delta": row["raw_final_mean"]-baseline.raw_final_mean})
    attribution = pd.DataFrame(attribution)
    save_csv(experiments / f"{prefix}_attribution.csv", attribution)
    month_rows = []
    for (target, fold), group in monthly.groupby(["target", "fold"]):
        reference = monthly[(monthly.target == "Y0") & (monthly.fold == fold)].set_index("month")
        changes = group.set_index("month").predictive_score - reference.predictive_score
        month_rows.append({"target": target, "fold": fold, "months": len(group),
            "positive_ic_months": int((group.ic_mean > 0).sum()),
            "positive_excess_months": int((group.annual_excess > 0).sum()),
            "months_predictive_above_Y0": int((changes > 0).sum()),
            "worst_predictive_month": int(group.loc[group.predictive_score.idxmin(), "month"]),
            "worst_month_predictive": float(group.predictive_score.min())})
    stability = pd.DataFrame(month_rows)
    save_csv(experiments / f"{prefix}_stability.csv", stability)
    recovery = meta.get("error", meta.get("publication_recovery", {}).get("error"))
    report = contained_path(cfg["paths"]["research_reports"]) / f"alpha_research_{study_id}.md"
    text = report.read_text(encoding="utf-8")
    extra = ["## 补充判读与发布核对", "",
        "冻结伙伴的原始预测变化：" + "；".join(
            f"{target} 均值比 Y0 变化 {targets.loc[target].predictive_mean-baseline.predictive_mean:+.10f}，"
            f"2023 变化 {targets.loc[target].predictive_2023-baseline.predictive_2023:+.10f}" for target in selected) + "。",
        f"Y1 的三折均值 IC 为 {targets.loc['Y1'].mean_ic_mean:.10f}，Y4 为 {targets.loc['Y4'].mean_ic_mean:.10f}；"
        f"年化超额分别为 {targets.loc['Y1'].mean_annual_excess:.10f} / {targets.loc['Y4'].mean_annual_excess:.10f}。"
        f"Y4−Y1 的初筛均分差为 {targets.loc['Y4'].predictive_mean-targets.loc['Y1'].predictive_mean:+.10f}，"
        f"原始官方分差为 {targets.loc['Y4'].raw_final_mean-targets.loc['Y1'].raw_final_mean:+.10f}；"
        "两个排名的区别还包含换手项影响。预定初筛使用 PredictiveScore，冻结顺序为 " + "、".join(selected) + "。",
        "与 T030 的互补性诊断：" + "；".join(
            f"{target} 当日排名相关性 {targets.loc[target].rank_correlation:.6f}，实际可选 Top 重合均值 "
            f"{targets.loc[target].eligible_top_overlap:.2%}" for target in selected) +
        "。该相关性包含缺价兜底行，不等同于有效报价股票的相关性；不同目标尺度还会影响 0 兜底值的排序位置。是否形成融合提分仍须 S006 实际评分。",
        "其他目标相对 Y0 的 PredictiveScore 均值变化：" + "；".join(
            f"{target} {targets.loc[target].predictive_mean-baseline.predictive_mean:+.10f}" for target in ["Y2", "Y3", "Y5"]) +
        "。分项变化见下表，所有结果和模型均保留。",
        "", tr.table_markdown(attribution, ["target", "weighted_ic_delta", "weighted_excess_delta", "predictive_delta", "stability_delta", "raw_final_delta"]),
        "", "### 月度稳定性", "",
        tr.table_markdown(stability[stability.target.isin(shown)], ["target", "fold", "positive_ic_months",
            "positive_excess_months", "months_predictive_above_Y0", "worst_predictive_month", "worst_month_predictive"]),
        "", "这里每年均为 12 个月；月度年化量来自该月均值，不能当作已实现的年度复合收益。2023 仍是三折最弱年份，尚未在完整方案下判断改善是否足够。",
        "", "重要性先在每个模型/折内部把 gain 归一化，再按三折取均值；它描述模型使用的特征，不代表因果贡献。完整 40 列统计在 importance 表。",
        "", f"![重要性](figures/{prefix}_importance.png)", "",
        ("十五次训练均在原实现提交完成。首次报告渲染把表格行列表当成字符串，发布阶段中断；随后只增加换行转换适配和读取已通过产物的发布恢复入口，新增拟合 / 评分 / 选择变更均为 0。" if recovery else
         "已完成研究的图表与判读直接读取原有真实产物；新增拟合 / 评分 / 选择变更均为 0。"),
        "恢复前逐文件核对模型、预测、目标映射、来源和原记录；原数值源码通过 Git 字节摘要与 AST 比较保留，改动仅限表格格式适配。原错误及恢复代码摘要列入交接索引。",
        ""]
    write_text(report, text + "\n" + "\n".join(extra))


def publish_completed(cfg, study_id):
    if not re.fullmatch(r"S005(?:_[A-Za-z0-9_-]+)?", study_id):
        raise ValueError("Reporting identity must be S005 or an S005_* study.")
    root = contained_path(cfg["paths"]["research_studies"]) / study_id
    with study_lock(root):
        meta, selection = read_json(root / "study.json"), read_json(root / "selection.json")
        if (meta.get("completed_fits") != 15 or meta.get("completed_targets") != 5
                or meta["status"] not in {"interrupted", "complete"}
                or digest(tr.protocol(cfg)) != meta["protocol_digest"]
                or digest({k: v for k, v in selection.items() if k != "selection_digest"}) != selection["selection_digest"]):
            raise ValueError("Reporting requires fifteen completed fits and the unchanged frozen selection.")
        if meta["status"] == "interrupted" and "TypeError: sequence item 18" not in meta.get("error", ""):
            raise ValueError("Recovery is limited to the known report formatting interruption.")
        source_changes = numerical_source_identity(meta, root)
        if {p: importlib.metadata.version(p) for p in tr.PACKAGES} != meta["environment"]["packages"]:
            raise ValueError("Dependencies differ from the executed study.")
        if provenance(cfg) != meta["data"]:
            raise ValueError("Original data or official attachments changed.")
        for path, expected in meta["frozen_files"].items():
            if sha256(contained_path(path)) != expected:
                raise ValueError(f"Frozen source changed: {path}")
        candidates = [read_json(root / "candidates" / f"Y{i}.json") for i in range(1, 6)]
        runs = [run for candidate in candidates for run in candidate["folds"]]
        if (len(runs) != 15 or len({run["exp_id"] for run in runs}) != 15
                or any(c["status"] != "COMPLETE" for c in candidates)):
            raise ValueError("Candidate manifests do not contain exactly fifteen valid fits.")
        for run in runs:
            manifest = contained_path(cfg["paths"]["models"]) / run["exp_id"] / "run.json"
            if read_json(manifest) != run or run["status"] != "passed":
                raise ValueError("Model manifest differs from its completed candidate.")
            for path, expected in run["artifact_hashes"].items():
                if sha256(contained_path(path)) != expected:
                    raise ValueError(f"Passed artifact changed: {path}")
            ensure_record(cfg, run, manifest)
        for partner in selection["partners"]:
            candidate = next(c for c in candidates if c["id"] == partner["id"])
            if any(candidate[k] != partner[k] for k in candidate if k != "folds"):
                raise ValueError("Frozen partner metrics or target changed.")
            for run in candidate["folds"]:
                fixed = partner["runs"][run["split"]["fold"]]
                if (fixed["spec"] != run["spec"] or fixed["model_sha256"] != run["artifact_hashes"][run["model"]]
                        or fixed["prediction_sha256"] != run["artifact_hashes"][run["prediction"]]):
                    raise ValueError("Frozen partner artifacts changed.")
        maps = {run["target_mapping"]["mapping"]: run["target_mapping"] for run in runs}
        if len(maps) != 12:
            raise ValueError("Expected twelve unique supervised mappings.")
        for mapped in maps.values():
            if sha256(contained_path(mapped["manifest"])) != mapped["manifest_sha256"]:
                raise ValueError("Target manifest changed.")
            for path, expected in mapped["artifact_hashes"].items():
                if sha256(contained_path(path)) != expected:
                    raise ValueError("Training target artifact changed.")
        records = read_records(contained_path(cfg["paths"]["experiment_log"]))
        if (records[:len(meta["initial_records"])] != meta["initial_records"]
                or len(records) != len(meta["initial_records"]) + 15
                or {r["exp_id"] for r in records[len(meta["initial_records"]):]} != {r["exp_id"] for r in runs}):
            raise ValueError("Old/shared records differ from fifteen completed model fits.")
        source_selection = read_json(contained_path(cfg["alpha_research"]["source_selection"]))
        references = []
        for definition in cfg["target_research"]["folds"]:
            fold = definition["name"]
            run = copy.deepcopy(source_selection["winner"]["folds"][fold]["raw"])
            directory = root / "references" / fold
            matched = pd.read_csv(directory / "agreement_daily.csv")
            run.update(target_id="Y0", target_mapping=None, predictive_score=tr.predictive(run["metrics"]),
                agreement={"rank_correlation": float(matched.rank_correlation.mean()),
                    "eligible_top_overlap": float(matched.eligible_top_overlap.mean()),
                    "eligible_top_jaccard": float(matched.eligible_top_jaccard.mean()),
                    "mean_exact_ties": float(matched.all_market_exact_ties.mean())},
                diagnostics=relative(directory), new_model_fits=0)
            references.append(run)
        baseline = {**selection["baseline"], "folds": references}
        results = [baseline, *candidates]
        recovery = {"time": timestamp(), "new_model_fits": 0, "new_scored_configurations": 0,
                    "selection_changed": False, "source_changes": source_changes,
                    "error": meta.get("error", meta.get("publication_recovery", {}).get("error")),
                    "reporting_code": {p: sha256(ROOT / p) for p in ["src/evaluation/target_handoff.py", "src/models/target_research.py"]}}
        audit = {"status": "passed", "study_id": study_id, "time": timestamp(),
            "implementation_commit": meta["git"]["commit"], "source_digest": meta["source_digest"],
            "protocol_digest": meta["protocol_digest"], "selection_digest": selection["selection_digest"],
            "complete_targets": 5, "failed_fits": 0, "new_model_fits": 15, "reused_reference_folds": 3,
            "evaluated_years": [2021, 2022, 2023], "unique_target_maps": 12,
            "target_prefix_replays": [{"fold": m["identity"]["fold"], "transform": m["identity"]["spec"]["transform"],
                                      **m["prefix_replay"]} for m in maps.values()],
            "initial_records": len(meta["initial_records"]), "new_records": 15, "total_records": len(records),
            "previous_records_unchanged": True, "original_files_unchanged": True,
            "frozen_S003_S004_and_X_cache_unchanged": True,
            "max_official_difference": max(r["official_max_difference"] for r in runs),
            "max_reload_difference": max(r["reload_max_difference"] for r in runs),
            "max_csv_difference": max(r["csv_max_difference"] for r in runs),
            "all_saved_csv_orders_and_ties_match": all(r["csv_order_and_ties_preserved"] for r in runs),
            "data_provenance": meta["data"], "frozen_input_hashes": meta["frozen_files"],
            "target_maps": list(maps.values()), "publication_recovery": recovery,
            "reference_diagnostics": {relative(p): sha256(p) for p in sorted((root / "references").rglob("*.csv"))},
            "runs": [{"exp_id": r["exp_id"], "manifest": relative(contained_path(cfg["paths"]["models"]) / r["exp_id"] / "run.json"),
                      "manifest_sha256": sha256(contained_path(cfg["paths"]["models"]) / r["exp_id"] / "run.json"),
                      "artifact_hashes": r["artifact_hashes"]} for r in runs]}
        tr.publish(cfg, meta, root, results, selection, audit)
        figures_and_findings(cfg, meta, selection)
        reports = contained_path(cfg["paths"]["research_reports"])
        experiments = contained_path(cfg["paths"]["experiment_log"]).parent
        prefix = f"alpha_{study_id}"
        published = [reports / f"alpha_research_{study_id}.md", reports / f"alpha_research_{study_id}_selection.json",
                     *sorted((reports / "figures").glob(f"{prefix}_*.png")), *sorted(experiments.glob(f"{prefix}_*.csv"))]
        audit["published_files"] = {relative(p): {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in published}
        save_json(root / "audit.json", audit)
        save_json(reports / f"alpha_research_{study_id}_artifacts.json", audit)
        meta.update(status="complete", completed_at=meta.get("completed_at", timestamp()),
                    selection_digest=selection["selection_digest"], new_model_fits=15,
                    selected_partners=[p["id"] for p in selection["partners"]], publication_recovery=recovery)
        meta.pop("error", None)
        meta.pop("interrupted_at", None)
        save_json(root / "study.json", meta)
        print(f"{study_id} publication complete; fits=15; official_diff={audit['max_official_difference']}; "
              f"reload_diff={audit['max_reload_difference']}; partners={meta['selected_partners']}; no new fitting/scoring")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default="S005")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args(argv)
    cfg, _ = load_config(args.config)
    publish_completed(cfg, args.study_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

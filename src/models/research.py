"""Day 5–8: eight fixed 2023 experiments, then one locked 2024 confirmation."""
from __future__ import annotations

import argparse
import gc
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.official_eval import OFFICIAL_METRICS
from src.evaluation.research_diagnostics import factor_diagnostics
from src.evaluation.validation import get_folds
from src.features.build_features import feature_names
from src.features.research_features import research_names
from src.models.baseline import RunLog, run_directories, run_one, source_provenance
from src.utils.experiments import check_experiment_id, read_records
from src.utils.project import ROOT, git_state, load_config, project_path, sha256, timestamp, write_json
from src.utils.research_cache import code_hashes, contained_path, digest, prepare_history, protocol_config

PRESET_NAMES = ["base", "trend", "volume", "risk", "all", "without_trend", "without_volume", "without_risk"]


def preset_columns(cfg: dict, preset: dict) -> list[str]:
    columns = feature_names(cfg["features"])
    groups = preset["groups"]
    if len(set(groups)) != len(groups) or set(groups) - {"T", "V", "R"}:
        raise ValueError("Unknown or duplicate research feature group.")
    return columns + [c for g in ["T", "V", "R"] if g in groups for c in cfg["research"]["groups"][g]]


def choose_preset(results: list[dict], tolerance: float) -> str:
    if [r["preset"] for r in results] != PRESET_NAMES or any(r.get("status") != "passed" for r in results):
        raise ValueError("Selection requires all eight successful experiments in protocol order.")
    if any(not np.isfinite(r["metrics"]["final_score"]) for r in results):
        raise ValueError("Selection scores must be finite.")
    maximum = max(r["metrics"]["final_score"] for r in results)
    tied = [r for r in results if maximum - r["metrics"]["final_score"] <= tolerance]
    best = min(tied, key=lambda r: (r["feature_count"], PRESET_NAMES.index(r["preset"])))
    gain = best["metrics"]["final_score"] - results[0]["metrics"]["final_score"]
    return best["preset"] if gain > tolerance else "base"


def promotion(selected: str, screening: dict, base2023: dict, confirmation: dict,
              base2024: dict, tolerance: float) -> str:
    if (selected != "base" and screening["final_score"] > base2023["final_score"] + tolerance
            and confirmation["final_score"] > base2024["final_score"] + tolerance):
        return selected
    return "base"


def verify_lock(selection: dict, expected_sha: str, cfg: dict, source: dict) -> None:
    if digest(selection) != expected_sha:
        raise ValueError("2023 selection lock changed after screening.")
    if selection["protocol_sha256"] != digest(protocol_config(cfg)) or selection["source_sha256"] != digest(source):
        raise ValueError("Code/configuration changed after selection; start a new 2023 study.")
    if selection["selected_preset"] != choose_preset(selection["results"], cfg["research"]["selection_tolerance"]):
        raise ValueError("Locked candidate disagrees with the 2023-only selection rule.")
    for result in selection["results"]:
        path = contained_path(result["manifest_path"])
        if sha256(path) != result["manifest_sha256"]:
            raise ValueError("A frozen 2023 run manifest changed.")


def reference_result(cfg: dict) -> dict:
    records = read_records(project_path(cfg["paths"]["experiment_log"]))
    row = next((r for r in records if r["exp_id"] == cfg["research"]["reference_exp_id"]), None)
    if row is None:
        raise ValueError("The frozen V1 reference is absent from the shared experiment log.")
    model = json.loads(row["params"])
    if model["params"] != cfg["baseline"]["model"]["params"] or model["num_boost_round"] != 200:
        raise ValueError("Research must retain the original V1 parameters and 200 fixed rounds.")
    if cfg["baseline"]["model"]["num_boost_round"] != 200 or cfg["baseline"]["fallback_prediction"] != 0.:
        raise ValueError("Fixed round count and missing-quote fallback must retain the V1 protocol.")
    return {"exp_id": row["exp_id"], "git_commit": row["git_commit"],
            "metrics": {k: float(row[k]) for k in OFFICIAL_METRICS}}


def summarize_run(cfg: dict, preset: dict, info: dict) -> dict:
    manifest = run_directories(cfg, info["exp_id"])["models"] / "run.json"
    return {"preset": preset["name"], "groups": preset["groups"], "exp_id": info["exp_id"],
            "status": info["status"], "feature_count": len(info["feature_columns"]),
            "columns": info["feature_columns"], "metrics": info["metrics"], "split": info["split"],
            "training_samples": info["training_samples"], "prediction": info["prediction"],
            "manifest_path": manifest.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(manifest)}


def study_comparison(results: list[dict]) -> pd.DataFrame:
    base = results[0]["metrics"]
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({"preset": r["preset"], "exp_id": r["exp_id"], "feature_count": r["feature_count"],
                     **m, "delta_score": m["final_score"] - base["final_score"],
                     "ic_contribution": .4 * (m["ic_mean"] - base["ic_mean"]),
                     "excess_contribution": .3 * (m["annual_excess"] - base["annual_excess"]),
                     "turnover_contribution": -.3 * (m["mean_turnover"] - base["mean_turnover"])})
    return pd.DataFrame(rows)


def report_path(cfg: dict, study_id: str) -> Path:
    return contained_path(cfg["paths"]["research_reports"]) / f"feature_research_{study_id}.md"


def write_report(cfg: dict, study_id: str, selection: dict, lock_sha: str, confirmation: dict | None = None) -> None:
    lines = [f"# Day 5–8 特征研究：{study_id}", "", f"更新时间：{timestamp()}", "",
             "## 固定协议", "", "40 个 V1 特征 + 18 个研究特征，模型沿用 V1 的 200 轮、seed=42、8 线程。",
             "训练标签不跨验证边界；缺失日保留；缺失价格固定预测 0。2023 年完成全部 8 组后锁定候选，2024 年最多确认一个新增候选。",
             "", f"研究代码提交：`{selection['git']['commit']}`；源代码指纹：`{selection['source_sha256']}`。",
             f"锁定记录 SHA-256（规范 JSON）：`{lock_sha}`。", "", "## 2023 全年筛选", "",
             "|组合|特征数|Rank IC|年化超额|换手率|综合分|相对 V1|", "|---|---:|---:|---:|---:|---:|---:|"]
    base = selection["results"][0]["metrics"]
    for r in selection["results"]:
        m = r["metrics"]
        lines.append(f"|{r['preset']}|{r['feature_count']}|{m['ic_mean']:.10f}|{m['annual_excess']:.10f}|"
                     f"{m['mean_turnover']:.10f}|{m['final_score']:.10f}|{m['final_score']-base['final_score']:+.10f}|")
    lines += ["", f"按预设规则锁定：**{selection['selected_preset']}**。使用全年官方分，容差 1e-6；近似并列时优先更少特征，再按配置顺序。",
              "月度统计、单因子 IC、Top/Bottom 只做诊断，未用于额外筛选或调参。", "", "## 2024 确认", ""]
    if confirmation is None:
        lines += ["尚未运行。须先把本报告和 2023 实验记录提交到 main，再运行 confirm。"]
    else:
        lines += ["|组合|Rank IC|年化超额|换手率|综合分|", "|---|---:|---:|---:|---:|"]
        for r in confirmation["results"]:
            m = r["metrics"]
            lines.append(f"|{r['preset']}|{m['ic_mean']:.10f}|{m['annual_excess']:.10f}|{m['mean_turnover']:.10f}|{m['final_score']:.10f}|")
        lines += ["", f"V1 的 8 项原始指标复现最大差：{confirmation['reference_max_difference']:.2e}。",
                  f"本阶段推荐：**{confirmation['recommended_preset']}**；只有 2023、2024 同时提高超过 1e-6 才替换 V1。",
                  f"2024 确认代码提交：`{confirmation['git']['commit']}`。"]
    lines += ["", "## 产物和交接", "",
              f"研究清单、锁定记录、对比表及因子诊断：`{cfg['paths']['research_studies']}/{study_id}/`。",
              "各实验模型、预测、逐日/月度/Top-Bottom 表按实验编号保存；共享实验表保留原 20 列。大型产物由 Git 忽略。",
              "成员 A 已执行本报告中有实际结果的阶段。成员 B 的原 V1 复现已完成；本次新增特征和跨年度结论的独立复核仍待成员 B 完成。", ""]
    report_path(cfg, study_id).write_text("\n".join(lines), encoding="utf-8")


def run_study(cfg: dict, config_path: Path, study_id: str, owner: str, phase: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", study_id):
        raise ValueError("Invalid study ID.")
    if [p["name"] for p in cfg["research"]["presets"]] != PRESET_NAMES:
        raise ValueError("The eight screening presets/order must match the agreed protocol.")
    research_names(cfg["research"])
    if cfg["research"]["study_fold"] != "fold1" or cfg["research"]["confirm_fold"] != "fold2":
        raise ValueError("Use fold1 for 2023 selection and fold2 for 2024 confirmation.")
    revision = git_state()
    if revision["branch"] != "main" or revision["dirty"]:
        raise ValueError("Commit the current implementation/results to a clean main before running a phase.")
    source = code_hashes()
    reference = reference_result(cfg)
    root = contained_path(cfg["paths"]["research_studies"]) / study_id
    state_path = root / "study.json"
    if phase == "screen":
        if root.exists() or report_path(cfg, study_id).exists():
            raise FileExistsError("Study/report already exists; retain it and choose a new study ID.")
        ids = [f"{study_id}_screen_smoke"] + [f"{study_id}_screen_{p}" for p in PRESET_NAMES]
        state = {"study_id": study_id, "owner": owner, "status": "screening", "started_at": timestamp(),
                 "screen_git": revision, "protocol_sha256": digest(protocol_config(cfg)),
                 "source_sha256": digest(source), "source_files": source, "reference": reference}
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["status"] != "screened" or (root / "confirmation.json").exists():
            raise ValueError("Confirmation requires a completed, unconfirmed 2023 study.")
        selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
        verify_lock(selection, state["selection_sha256"], cfg, source)
        report = report_path(cfg, study_id)
        if (revision["commit"] == state["screen_git"]["commit"] or not report.exists()
                or state["selection_sha256"] not in report.read_text(encoding="utf-8")):
            raise ValueError("Commit the locked 2023 report and results to main before confirmation.")
        ids = [f"{study_id}_confirm_base"]
        if selection["selected_preset"] != "base":
            ids.append(f"{study_id}_confirm_{selection['selected_preset']}")
    for exp_id in ids:
        check_experiment_id(exp_id, project_path(cfg["paths"]["experiment_log"]),
                            list(run_directories(cfg, exp_id).values()))
    provenance = source_provenance(cfg)
    if phase == "confirm" and provenance != state["provenance"]:
        raise ValueError("Original training data/official attachments changed between phases.")
    root.mkdir(parents=True, exist_ok=True)
    state["provenance"] = provenance
    write_json(state_path, state)
    log = RunLog(root / "preparation.log")
    folds = {f.name: f for f in get_folds(config_path)}
    fold = folds[cfg["research"]["study_fold" if phase == "screen" else "confirm_fold"]]
    def execute(preset, prepared, smoke=False, parity=None):
        if code_hashes() != source:
            raise ValueError("Source code changed during the research phase.")
        name = "smoke" if smoke else preset["name"]
        context = {"study_id": study_id, "phase": phase, "preset": name, "groups": preset["groups"],
                   "feature_version": f"{cfg['research']['pool_version']}_{name}",
                   "source_sha256": digest(source), "protocol_sha256": state["protocol_sha256"]}
        info = run_one(cfg, config_path, fold, f"{study_id}_{phase}_{name}", owner, provenance, revision,
                       smoke, prepared=prepared, columns=preset_columns(cfg, preset),
                       context=context, reference_metrics=parity)
        if code_hashes() != source:
            raise ValueError("Source code changed during a model run.")
        return info
    try:
        if phase == "screen":
            smoke_history = prepare_history(cfg, fold, provenance, log, smoke=True)
            smoke = execute(cfg["research"]["presets"][4], smoke_history, smoke=True)
            state["smoke_exp_id"] = smoke["exp_id"]
            del smoke_history
            gc.collect()
        prepared = prepare_history(cfg, fold, provenance, log)
        write_json(root / f"{phase}_cache.json", prepared.cache)
        results = []
        if phase == "screen":
            presets = cfg["research"]["presets"]
        else:
            names = ["base"] + ([selection["selected_preset"]] if selection["selected_preset"] != "base" else [])
            presets = [next(p for p in cfg["research"]["presets"] if p["name"] == name) for name in names]
        for preset in presets:
            parity = reference["metrics"] if phase == "confirm" and preset["name"] == "base" else None
            info = execute(preset, prepared, parity=parity)
            results.append(summarize_run(cfg, preset, info))
            state[f"{phase}_completed"] = [r["exp_id"] for r in results]
            write_json(state_path, state)
            gc.collect()
        if any(r["split"] != results[0]["split"] or r["training_samples"] != results[0]["training_samples"] for r in results):
            raise AssertionError("Feature variants changed the training/validation cohort.")
        comparison = study_comparison(results)
        comparison.to_csv(root / f"{phase}_comparison.csv", index=False)
        if phase == "screen":
            selected = choose_preset(results, cfg["research"]["selection_tolerance"])
            selection = {"study_id": study_id, "locked_at": timestamp(), "git": revision,
                         "source_sha256": digest(source), "protocol_sha256": digest(protocol_config(cfg)),
                         "selected_preset": selected, "results": results}
            lock_sha = digest(selection)
            write_json(root / "selection.json", selection)
            state.update({"status": "screened", "selection_sha256": lock_sha, "screen_finished_at": timestamp()})
            write_json(state_path, state)
            write_report(cfg, study_id, selection, lock_sha)
            log(f"2023-only selection locked: {selected}; commit its report/results before 2024 confirmation")
            valid = prepared.dataset.loc[prepared.dataset.trade_date.between(fold.valid_start, fold.valid_end)]
            factor_daily, factor_summary = factor_diagnostics(valid, research_names(cfg["research"]))
            factor_daily.to_csv(root / "factor_daily_2023.csv", index=False)
            factor_summary.to_csv(root / "factor_summary_2023.csv", index=False)
        else:
            candidate = results[-1]
            screen_result = next(r for r in selection["results"] if r["preset"] == selection["selected_preset"])
            recommended = promotion(selection["selected_preset"], screen_result["metrics"],
                                    selection["results"][0]["metrics"], candidate["metrics"], results[0]["metrics"],
                                    cfg["research"]["selection_tolerance"])
            confirmation = {"study_id": study_id, "confirmed_at": timestamp(), "git": revision,
                            "selection_sha256": state["selection_sha256"], "results": results,
                            "recommended_preset": recommended,
                            "reference_max_difference": max(abs(results[0]["metrics"][k] - reference["metrics"][k])
                                                            for k in OFFICIAL_METRICS), "member_b_research_review": "pending"}
            write_json(root / "confirmation.json", confirmation)
            state.update({"status": "complete", "finished_at": timestamp(), "recommended_preset": recommended})
            write_json(state_path, state)
            write_report(cfg, study_id, selection, state["selection_sha256"], confirmation)
            log(f"2024 confirmation complete; recommended preset={recommended}")
        print(comparison[["preset", "feature_count", "ic_mean", "annual_excess", "mean_turnover", "final_score"]].to_string(index=False), flush=True)
        return state
    except Exception as error:
        state.update({"status": "failed", "failed_phase": phase, "error": str(error), "traceback": traceback.format_exc()})
        write_json(state_path, state)
        log(f"FAILED: {error}; preserve study artifacts, do not select from incomplete results")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--phase", choices=["screen", "confirm"], required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--owner", default="A")
    args = parser.parse_args(argv)
    cfg, path = load_config(args.config)
    run_study(cfg, path, args.study_id, args.owner, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

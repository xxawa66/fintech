"""Day 9–12: bounded model search, rank blending and a locked cross-year check."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.load_data import KEYS
from src.data.make_dataset import training_arrays
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.model_analysis import analyze_model
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame, load_validation_frame
from src.evaluation.research_diagnostics import monthly_metrics, portfolio_diagnostics
from src.evaluation.turnover import band_scores
from src.evaluation.validation import get_folds, split_train_valid
from src.features.build_features import feature_names
from src.models import ridge_model
from src.models.baseline import RunLog, run_directories, run_one, source_provenance
from src.models.ensemble import align_predictions, prediction_agreement, rank_blend
from src.models.lightgbm_model import load_model as load_lightgbm
from src.utils.experiments import append_record, check_experiment_id, read_records
from src.utils.project import ROOT, git_state, load_config, project_path, sha256, timestamp, write_json
from src.utils.research_cache import code_hashes, contained_path, digest, prepare_history


def protocol(cfg: dict) -> dict:
    return {k: cfg[k] for k in ["paths", "data", "evaluation", "validation", "features",
                              "baseline", "research", "model_research"]}


def validate_protocol(cfg: dict) -> None:
    research = cfg["model_research"]
    if (research["study_fold"] != "fold1" or research["confirm_fold"] != "fold2"
            or len(feature_names(cfg["features"])) != 40
            or cfg["features"]["version"] != "baseline_v1"
            or research["keep_q"] != 0.1 or cfg["evaluation"]["band"]["keep_q"] != 0.1
            or research["initial_state"] != "cold_start"):
        raise ValueError("S002 must retain the agreed folds, V1 features and cold-start band(0.1).")
    if [p["name"] for p in research["lightgbm_presets"]] != [f"L{i}" for i in range(6)]:
        raise ValueError("LightGBM preset order differs from the six agreed configurations.")
    if [p["alpha"] for p in research["ridge_presets"]] != [100., 10000., 1000000.]:
        raise ValueError("Ridge alpha grid differs from the agreed protocol.")
    if research["blend_weights"] != [0., .25, .5, .75, 1.]:
        raise ValueError("Blend grid differs from the five agreed weights.")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def artifact_info(cfg: dict, exp_id: str) -> dict:
    directories = run_directories(cfg, exp_id)
    manifest = directories["models"] / "run.json"
    info = json.loads(manifest.read_text(encoding="utf-8"))
    if info["status"] != "passed":
        raise ValueError(f"Run is not passed: {exp_id}")
    pred = contained_path(info["prediction"]["file"])
    labels = directories["metrics"] / "validation_labels.csv"
    return {"exp_id": exp_id, "manifest": relative(manifest), "manifest_sha256": sha256(manifest),
            "prediction": relative(pred), "prediction_sha256": sha256(pred),
            "labels": relative(labels), "labels_sha256": sha256(labels),
            "metrics": info["metrics"], "model": info["model"],
            "split": info.get("split"), "training_samples": info.get("training_samples"),
            "model_reload_difference": info["prediction"].get("model_reload_max_abs_difference")}


def begin_run(cfg, config_path, fold, exp_id, owner, revision, provenance, model, recipe):
    directories = run_directories(cfg, exp_id)
    check_experiment_id(exp_id, project_path(cfg["paths"]["experiment_log"]), list(directories.values()))
    for path in directories.values():
        path.mkdir(parents=True)
    info = {"exp_id": exp_id, "owner": owner, "status": "running", "started_at": timestamp(),
            "git": revision, "data": provenance, "family": model, "recipe": recipe,
            "model": recipe, "feature_version": "baseline_v1", "feature_columns": feature_names(cfg["features"]),
            "validation_window": [fold.valid_start, fold.valid_end],
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "packages": {n: importlib.metadata.version(n) for n in
                                         ["numpy", "pandas", "lightgbm", "scikit-learn", "scipy"]}},
            "config_path": relative(config_path)}
    (directories["models"] / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(directories["models"] / "features.json", info["feature_columns"])
    write_json(directories["models"] / "run.json", info)
    log = RunLog(directories["metrics"] / "run.log")
    log(f"{exp_id}: {model}; full validation; training window {fold.train_start}-{fold.train_end}")
    return directories, info, log


def finish_prediction(cfg, fold, directories, info, predictions, truth, log, started):
    check_predictions(predictions, truth)
    if len(predictions) != 1_125_300 or predictions.trade_date.nunique() != 242:
        raise ValueError("Full validation must contain 4,650 stocks by 242 days.")
    prediction_path = directories["predictions"] / f"valid_{fold.valid_start // 10000}.csv"
    label_path = directories["metrics"] / "validation_labels.csv"
    predictions.to_csv(prediction_path, index=False)
    truth.to_csv(label_path, index=False)
    persisted = pd.read_csv(prediction_path)
    check_predictions(persisted, truth)
    del persisted
    scored = load_validation_frame(prediction_path, label_path)
    metrics = evaluate_frame(scored)
    differences = compare_official(prediction_path, truth, metrics, cfg["baseline"]["score_tolerance"])
    daily = daily_metrics(scored)
    daily.to_csv(directories["metrics"] / "daily_metrics.csv", index=False)
    monthly_metrics(daily).to_csv(directories["metrics"] / "monthly_metrics.csv", index=False)
    portfolio_diagnostics(scored).to_csv(directories["metrics"] / "portfolio_daily.csv", index=False)
    info.update({"status": "passed", "finished_at": timestamp(), "metrics": metrics,
                 "official_score_differences": differences,
                 "official_score_max_difference": max(differences.values()),
                 "duration_seconds": time.perf_counter() - started,
                 "peak_sampled_process_rss_bytes": log.peak_sampled_rss,
                 "prediction": {"file": relative(prediction_path), "rows": len(predictions), "days": 242,
                                "model_reload_max_abs_difference": info.get("reload_difference")},
                 "checks": {"prediction_coverage": True, "official_score_match": True}})
    if info.get("reload_difference") is not None:
        info["checks"]["model_reload"] = info["reload_difference"] <= 1e-12
    write_json(directories["metrics"] / "metrics.json", metrics)
    write_json(directories["models"] / "run.json", info)
    append_record(project_path(cfg["paths"]["experiment_log"]), {
        "exp_id": info["exp_id"], "date": info["finished_at"], "owner": info["owner"],
        "git_commit": info["git"]["commit"], "config_path": info["config_path"],
        "features": "baseline_v1 (40)", "model": info["family"],
        "params": json.dumps(info["model"], sort_keys=True),
        "train_period": f"{fold.train_start}-{fold.train_end}",
        "valid_period": f"{fold.valid_start}-{fold.valid_end}",
        **{k: metrics[k] for k in OFFICIAL_METRICS},
        "artifact_path": relative(directories["models"] / "run.json"),
        "notes": ("S002; " + info.get("notes", "derived prediction; no additional model training")
                  + "; validation labels used only for scoring; raw/band layers stored separately")})
    log(f"PASS {info['exp_id']}: score={metrics['final_score']:.10f}; official difference={max(differences.values()):.2e}")
    del scored
    return info


def derived_run(cfg, path, fold, exp_id, owner, revision, provenance, predictions,
                truth, family, recipe, extra=None):
    directories, info, log = begin_run(cfg, path, fold, exp_id, owner, revision, provenance, family, recipe)
    if extra:
        info.update(extra)
    started = time.perf_counter()
    try:
        return finish_prediction(cfg, fold, directories, info, predictions, truth, log, started)
    except Exception as error:
        info.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(directories["models"] / "run.json", info)
        raise


def add_band(cfg, path, fold, raw_id, owner, revision, provenance, truth):
    raw = artifact_info(cfg, raw_id)
    scored = load_validation_frame(contained_path(raw["prediction"]), contained_path(raw["labels"]))
    # This transform receives only known X and predictions, never y or a label-validity mask.
    known = scored[KEYS + ["pred", "flag_limit_up"]]
    pred = scored[KEYS].copy()
    pred["pred"] = band_scores(known, cfg["model_research"]["keep_q"]).to_numpy()
    recipe = {"method": "band", "keep_q": cfg["model_research"]["keep_q"],
              "initial_state": "cold_start", "source": raw}
    derived_run(cfg, path, fold, raw_id + "_band", owner, revision, provenance,
                pred, truth, "Band", recipe)
    del scored, pred
    gc.collect()
    return artifact_info(cfg, raw_id + "_band")


def entry(cfg, path, fold, name, spec, raw_id, owner, revision, provenance, truth):
    return {"name": name, "spec": spec, "complexity": len([w for w in spec.get("weights", [1.]) if w > 0]),
            "raw": artifact_info(cfg, raw_id),
            "band": add_band(cfg, path, fold, raw_id, owner, revision, provenance, truth)}


def reference(cfg, path, fold, prepared, study_id, phase, owner, revision, provenance):
    refs = cfg["model_research"]["references"][fold.name]
    source = artifact_info(cfg, refs["raw_exp_id"])
    old_info = json.loads(contained_path(source["manifest"]).read_text(encoding="utf-8"))
    records = {r["exp_id"]: r for r in read_records(project_path(cfg["paths"]["experiment_log"]))}
    if (source["model"]["params"] != cfg["baseline"]["model"]["params"]
            or source["model"]["num_boost_round"] != 200 or old_info["data"] != provenance
            or old_info["feature_columns"] != feature_names(cfg["features"])):
        raise ValueError("Existing V1 reference model/config/data does not match the frozen baseline.")
    _, valid, split = split_train_valid(prepared.dataset, fold)
    pred = pd.read_csv(contained_path(source["prediction"]))
    check_predictions(pred, prepared.truth)
    columns = feature_names(cfg["features"])
    source_model = run_directories(cfg, refs["raw_exp_id"])["models"] / "model.txt"
    model = load_lightgbm(source_model)
    reconstructed = valid[KEYS].copy()
    values = np.full(len(valid), cfg["baseline"]["fallback_prediction"], dtype="float64")
    eligible = valid.quote_valid.to_numpy()
    values[eligible] = model.predict(valid.loc[eligible, columns], num_threads=8)
    reconstructed["pred"] = values
    _, aligned = align_predictions([pred, reconstructed])
    reload_difference = float(np.max(np.abs(aligned[:, 0] - aligned[:, 1])))
    if reload_difference > 1e-12:
        raise AssertionError("Saved baseline model no longer reproduces its prediction file.")
    raw_id = f"{study_id}_{phase}_L0"
    spec = {"family": "lightgbm", "preset": "L0"}
    derived_run(cfg, path, fold, raw_id, owner, revision, provenance, pred, prepared.truth,
                "LightGBMReference", {"reference": source, "model_sha256": sha256(source_model)},
                {"reload_difference": reload_difference, "split": split,
                 "training_samples": source["training_samples"],
                 "notes": "reused frozen V1 model/prediction; no new training"})
    result = entry(cfg, path, fold, "L0", spec, raw_id, owner, revision, provenance, prepared.truth)
    for layer, row_id in [("raw", refs["raw_exp_id"]), ("band", refs["band_record_id"])]:
        differences = {k: abs(result[layer]["metrics"][k] - float(records[row_id][k])) for k in OFFICIAL_METRICS}
        if max(differences.values()) > cfg["baseline"]["score_tolerance"]:
            raise AssertionError(f"Reconstructed {layer} reference disagrees with recorded {row_id}: {differences}")
    del valid, pred, reconstructed, model, aligned, values
    gc.collect()
    return result


def fit_lightgbm(cfg, path, fold, prepared, preset, exp_id, owner, revision, provenance, context):
    specific = copy.deepcopy(cfg)
    specific["baseline"]["model"]["params"].update(preset["overrides"])
    specific["baseline"]["model"]["num_boost_round"] = preset["rounds"]
    return run_one(specific, path, fold, exp_id, owner, provenance, revision, False,
                   prepared=prepared, columns=feature_names(cfg["features"]),
                   context={**context, "feature_version": "baseline_v1_" + preset["name"]})


def fit_ridge(cfg, path, fold, prepared, preset, exp_id, owner, revision, provenance):
    settings = cfg["model_research"]["ridge"]
    params = {k: settings[k] for k in ["solver", "tol", "max_iter", "fit_intercept"]}
    params["alpha"] = preset["alpha"]
    directories, info, log = begin_run(cfg, path, fold, exp_id, owner, revision, provenance,
                                       "Ridge", {"params": params, "preset": preset["name"]})
    started = time.perf_counter()
    try:
        columns = feature_names(cfg["features"])
        train, valid, split = split_train_valid(prepared.dataset[KEYS + columns + ["y_ret_1d", "quote_valid"]], fold)
        x, y, training = training_arrays(train, columns)
        info.update(split=split, training_samples=training, feature_cache=prepared.cache)
        if (split["n_boundary_labels_dropped"] != (4280 if fold.name == "fold1" else 4522)
                or len(valid) != 1_125_300):
            raise AssertionError("Ridge time boundary/cohort differs from the audited fold.")
        del train
        gc.collect()
        fit_start = time.perf_counter()
        bundle = ridge_model.train_model(x, y, params, settings["num_threads"], log)
        info["fit_seconds"] = time.perf_counter() - fit_start
        info["model"] = bundle["info"]
        del x, y
        gc.collect()
        model_path = directories["models"] / "model.joblib"
        ridge_model.save_model(bundle, model_path)
        pd.DataFrame({"feature": columns, "coefficient_standardized": bundle["pipeline"].named_steps["ridge"].coef_,
                      "training_median": bundle["pipeline"].named_steps["imputer"].statistics_,
                      "training_mean": bundle["pipeline"].named_steps["scaler"].mean_,
                      "training_scale": bundle["pipeline"].named_steps["scaler"].scale_}).to_csv(
                          directories["metrics"] / "coefficients_preprocessing.csv", index=False)
        eligible = valid.quote_valid.to_numpy()
        x_valid = valid.loc[eligible, columns]
        predictions = np.full(len(valid), cfg["baseline"]["fallback_prediction"], dtype="float64")
        predictions[eligible] = ridge_model.predict_model(bundle, x_valid, settings["batch_rows"])
        loaded = ridge_model.load_model(model_path)
        reload_predictions = ridge_model.predict_model(loaded, x_valid, settings["batch_rows"])
        difference = float(np.max(np.abs(predictions[eligible] - reload_predictions)))
        if difference > 1e-12:
            raise AssertionError("Saved Ridge preprocessing/model no longer reproduces predictions.")
        frame = valid[KEYS].copy()
        frame["pred"] = predictions
        info.update(reload_difference=difference,
                    fallback_rows=int((~eligible).sum()),
                    notes="full Ridge fit; training-only preprocessing; no early stopping")
        del valid, x_valid, bundle, loaded, reload_predictions, predictions
        gc.collect()
        return finish_prediction(cfg, fold, directories, info, frame, prepared.truth, log, started)
    except Exception as error:
        info.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(directories["models"] / "run.json", info)
        log(f"FAILED {exp_id}; preserve artifacts; do not register an incomplete result")
        raise


def positive(result: dict) -> bool:
    return all(result[layer]["metrics"][key] > 0
               for layer in ["raw", "band"] for key in ["ic_mean", "annual_excess"])


def choose(results: list[dict], tolerance: float, base=None) -> dict:
    eligible = [r for r in results if positive(r)]
    if not eligible:
        if base is not None:
            return base
        # Ridge may still have complementarity even when it is weak by itself.
        eligible = results
    maximum = max(r["band"]["metrics"]["final_score"] for r in eligible)
    tied = [r for r in eligible if maximum - r["band"]["metrics"]["final_score"] <= tolerance]
    best = min(tied, key=lambda r: (r["complexity"], results.index(r)))
    if base and best["band"]["metrics"]["final_score"] <= base["band"]["metrics"]["final_score"] + tolerance:
        return base
    return best


def comparison(results: list[dict]) -> pd.DataFrame:
    base = results[0]
    rows = []
    for r in results:
        row = {"name": r["name"], "family": r["spec"]["family"], "positive_metrics": positive(r)}
        for layer in ["raw", "band"]:
            m, b = r[layer]["metrics"], base[layer]["metrics"]
            row.update({layer + "_" + k: m[k] for k in OFFICIAL_METRICS})
            row.update({layer + "_delta_score": m["final_score"] - b["final_score"],
                        layer + "_ic_contribution": .4 * (m["ic_mean"] - b["ic_mean"]),
                        layer + "_excess_contribution": .3 * (m["annual_excess"] - b["annual_excess"]),
                        layer + "_stability_contribution": -.3 * (m["mean_turnover"] - b["mean_turnover"])})
        rows.append(row)
    return pd.DataFrame(rows)


def report_path(cfg, study_id):
    return contained_path(cfg["paths"]["research_reports"]) / f"model_research_{study_id}.md"


def write_report(cfg, study_id, selection, lock_sha, confirmation=None):
    lines = [f"# Day 9–12 模型研究：{study_id}", "", f"更新时间：{timestamp()}", "",
             "## 固定协议", "", "V1 40 特征；2023 筛选、锁定一个完整候选后做 2024 确认；band=0.1，每折首日冷启动。",
             "Ridge 中位数填充与标准化只在训练样本拟合；模型、融合和 band 均不读取未来信息。",
             f"研究代码提交：`{selection['git']['commit']}`；锁定 SHA-256：`{lock_sha}`。",
             "2024 已被之前的研究使用，本轮是历史跨年确认，不视为完全未使用的留出集。", "",
             "## 2023 全量比较", "",
             "|方案|原始 IC|原始超额|原始换手|原始分|band IC|band 超额|band 换手|band 分|相对 V1+band|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    base = selection["results"][0]
    for r in selection["results"]:
        a, b = r["raw"]["metrics"], r["band"]["metrics"]
        lines.append(f"|{r['name']}|{a['ic_mean']:.6f}|{a['annual_excess']:.6f}|{a['mean_turnover']:.6f}|{a['final_score']:.6f}|"
                     f"{b['ic_mean']:.6f}|{b['annual_excess']:.6f}|{b['mean_turnover']:.6f}|{b['final_score']:.6f}|"
                     f"{b['final_score']-base['band']['metrics']['final_score']:+.6f}|")
    selected = selection["selected"]
    lines += ["", f"按 2023 全年 band 分及预设条件锁定：**{selected['name']}**。",
              "原始和 band 的全年 IC / 超额均为正；近似并列按组件数与预设顺序选择，数值容差 1e-6。",
              "", "候选完整定义：", "", "```json", json.dumps(selected["spec"], ensure_ascii=False, indent=2), "```", "",
              f"新增全量训练：{selection['new_training_count']} 次；融合权重：5 个，均复用模型预测。",
              "", "## 2024 确认", ""]
    if confirmation is None:
        lines += ["尚未执行；须先将本报告、候选锁定清单与 2023 记录提交 main。"]
    else:
        lines += ["|方案|原始 IC|原始超额|原始换手|原始分|band IC|band 超额|band 换手|band 分|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in confirmation["results"]:
            a, b = r["raw"]["metrics"], r["band"]["metrics"]
            lines.append(f"|{r['name']}|{a['ic_mean']:.6f}|{a['annual_excess']:.6f}|{a['mean_turnover']:.6f}|{a['final_score']:.6f}|"
                         f"{b['ic_mean']:.6f}|{b['annual_excess']:.6f}|{b['mean_turnover']:.6f}|{b['final_score']:.6f}|")
        lines += ["", f"本阶段推荐：**{confirmation['recommended']}**。",
                  "替换要求：2023 / 2024 同层 band 综合分都超过 V1+band，原始与 band 的 IC / 超额均为正。",
                  f"确认代码提交：`{confirmation['git']['commit']}`；新增模型训练 {confirmation['new_training_count']} 次。"]
    lines += ["", "## 产物与交接", "",
              f"研究目录：`{cfg['paths']['research_studies']}/{study_id}/`；各模型按实验编号保存，保留原始 / band 两层完整预测与官方 8 指标。",
              f"候选锁定清单：`docs/model_research_{study_id}_selection.json`，包含固定方案、来源与 SHA。",
              "基线来自 A 已有产物，重新加载模型核对预测、恢复本地 band 并核对 B 的 E003 记录；不覆盖 B 原运行目录。",
              "同层分析使用原始对原始、band 对 band；已编码 band 预测的分析关闭再次 band。",
              "大型数据、模型与预测由 Git 忽略；成员 B 对本轮模型与选择结论的独立复核待完成。",
              "最终提交、测试期初始状态决定与正文不超过 8 页的报告属于后续 Day 13–14。", ""]
    report_path(cfg, study_id).write_text("\n".join(lines), encoding="utf-8")


def save_analyses(cfg, root, chosen, base, phase):
    for layer in ["raw", "band"]:
        analyze_model(contained_path(chosen[layer]["prediction"]), contained_path(chosen[layer]["labels"]),
                      root / f"{phase}_analysis_{layer}", name=f"{phase}_{chosen['name']}_{layer}",
                      ref_pred=contained_path(base[layer]["prediction"]),
                      ref_labels=contained_path(base[layer]["labels"]), band_keep_q=0.0)


def verify_lock(cfg, root, selection, state, source):
    if (digest(selection) != state["selection_sha256"] or digest(protocol(cfg)) != selection["protocol_sha256"]
            or digest(source) != selection["source_sha256"]):
        raise ValueError("Locked source/protocol/selection changed; do not use 2024 to repair the search.")
    if choose(selection["results"], cfg["model_research"]["selection_tolerance"], selection["results"][0]) != selection["selected"]:
        raise ValueError("Locked candidate differs from the agreed selection rule.")
    for r in selection["results"]:
        for layer in ["raw", "band"]:
            for key, hash_key in [("manifest", "manifest_sha256"), ("prediction", "prediction_sha256"), ("labels", "labels_sha256")]:
                if sha256(contained_path(r[layer][key])) != r[layer][hash_key]:
                    raise ValueError("A frozen screening artifact was modified.")


def run_study(cfg, path, study_id, owner, phase):
    validate_protocol(cfg)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", study_id):
        raise ValueError("Invalid study identifier.")
    revision = git_state()
    if revision["branch"] != "main" or revision["dirty"]:
        raise ValueError("Commit the implementation or screening results to a clean main before running.")
    source = code_hashes()
    provenance = source_provenance(cfg)
    root = contained_path(cfg["paths"]["research_studies"]) / study_id
    state_path = root / "study.json"
    research = cfg["model_research"]
    if phase == "screen":
        if root.exists() or report_path(cfg, study_id).exists():
            raise FileExistsError("Study/report exists; preserve it and choose a new study ID.")
        root.mkdir(parents=True)
        state = {"study_id": study_id, "owner": owner, "status": "screening", "started_at": timestamp(),
                 "screen_git": revision, "source_sha256": digest(source), "protocol_sha256": digest(protocol(cfg)),
                 "provenance": provenance, "completed": []}
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["status"] != "screened" or (root / "confirmation.json").exists():
            raise ValueError("Confirmation requires an unconfirmed, completed screening phase.")
        selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
        verify_lock(cfg, root, selection, state, source)
        published = contained_path(cfg["paths"]["research_reports"]) / f"model_research_{study_id}_selection.json"
        if (revision["commit"] == state["screen_git"]["commit"]
                or json.loads(published.read_text(encoding="utf-8")) != selection
                or state["selection_sha256"] not in report_path(cfg, study_id).read_text(encoding="utf-8")):
            raise ValueError("Commit the locked 2023 results before confirmation.")
        if provenance != state["provenance"]:
            raise ValueError("Original data/official files changed between phases.")
        state["status"] = "confirming"
    write_json(state_path, state)
    log = RunLog(root / f"{phase}.log")
    fold = next(f for f in get_folds(path) if f.name == research["study_fold" if phase == "screen" else "confirm_fold"])
    new_training_count = 0
    try:
        prepared = prepare_history(cfg, fold, provenance, log)
        write_json(root / f"{phase}_cache.json", prepared.cache)
        base = reference(cfg, path, fold, prepared, study_id, phase, owner, revision, provenance)
        results = [base]
        def completed(result):
            if code_hashes() != source:
                raise ValueError("Code changed during the fixed research phase.")
            results.append(result)
            state["completed"] = [r["name"] for r in results]
            write_json(state_path, state)
            comparison(results).to_csv(root / f"{phase}_comparison.csv", index=False)
            gc.collect()
        def train_component(spec):
            nonlocal new_training_count
            family, name = spec["family"], spec["preset"]
            if name == "L0":
                return base
            exp_id = f"{study_id}_{phase}_{name}"
            if family == "lightgbm":
                preset = next(p for p in research["lightgbm_presets"] if p["name"] == name)
                fit_lightgbm(cfg, path, fold, prepared, preset, exp_id, owner, revision, provenance,
                             {"study_id": study_id, "phase": phase, "preset": name})
            elif family == "ridge":
                preset = next(p for p in research["ridge_presets"] if p["name"] == name)
                fit_ridge(cfg, path, fold, prepared, preset, exp_id, owner, revision, provenance)
            else:
                raise ValueError("Unknown model family in locked recipe.")
            new_training_count += 1
            return entry(cfg, path, fold, name, spec, exp_id, owner, revision, provenance, prepared.truth)
        if phase == "screen":
            for preset in research["lightgbm_presets"][1:]:
                completed(train_component({"family": "lightgbm", "preset": preset["name"]}))
            best_lgb = choose(results, research["selection_tolerance"], base)
            ridge_results = []
            for preset in research["ridge_presets"]:
                result = train_component({"family": "ridge", "preset": preset["name"]})
                completed(result)
                ridge_results.append(result)
            best_ridge = choose(ridge_results, research["selection_tolerance"])
            lgb_frame = pd.read_csv(contained_path(best_lgb["raw"]["prediction"]))
            ridge_frame = pd.read_csv(contained_path(best_ridge["raw"]["prediction"]))
            agreement = prediction_agreement(lgb_frame, ridge_frame)
            agreement.to_csv(root / "model_agreement_daily.csv", index=False)
            state["fusion_components"] = {"lightgbm": best_lgb["name"], "ridge": best_ridge["name"],
                                          "mean_rank_correlation": float(agreement.rank_correlation.mean()),
                                          "mean_top10_overlap": float(agreement.top10_overlap_fraction.mean())}
            for weight in research["blend_weights"]:
                name = f"F{round(weight*100):03d}"
                spec = {"family": "ensemble", "components": [best_lgb["spec"], best_ridge["spec"]],
                        "weights": [weight, 1. - weight], "method": "same_day_percentile_rank"}
                frame = rank_blend([lgb_frame, ridge_frame], spec["weights"])
                raw_id = f"{study_id}_{phase}_{name}"
                derived_run(cfg, path, fold, raw_id, owner, revision, provenance, frame, prepared.truth,
                            "RankEnsemble", {"spec": spec, "sources": [best_lgb["raw"], best_ridge["raw"]]})
                result = entry(cfg, path, fold, name, spec, raw_id, owner, revision, provenance, prepared.truth)
                if weight in (0., 1.):
                    endpoint = best_lgb if weight == 1. else best_ridge
                    for layer in ["raw", "band"]:
                        if max(abs(result[layer]["metrics"][k] - endpoint[layer]["metrics"][k]) for k in OFFICIAL_METRICS) > 1e-10:
                            raise AssertionError("Pure-model fusion endpoint changed official ranking metrics.")
                completed(result)
            selected = choose(results, research["selection_tolerance"], base)
            selection = {"study_id": study_id, "locked_at": timestamp(), "git": revision,
                         "source_sha256": digest(source), "protocol_sha256": digest(protocol(cfg)),
                         "results": results, "selected": selected, "fusion_components": state["fusion_components"],
                         "new_training_count": new_training_count, "protocol": protocol(cfg)}
            lock_sha = digest(selection)
            write_json(root / "selection.json", selection)
            write_json(contained_path(cfg["paths"]["research_reports"]) / f"model_research_{study_id}_selection.json", selection)
            save_analyses(cfg, root, selected, base, phase)
            state.update(status="screened", selection_sha256=lock_sha, screen_finished_at=timestamp(),
                         selected_name=selected["name"], screen_training_count=new_training_count)
            write_report(cfg, study_id, selection, lock_sha)
            log(f"2023 selection locked: {selected['name']}; commit/push results before 2024 confirmation")
        else:
            spec = selection["selected"]["spec"]
            if selection["selected"]["name"] == "L0":
                candidate = base
            elif spec["family"] in {"lightgbm", "ridge"}:
                candidate = train_component(spec)
                completed(candidate)
            else:
                frames = []
                weights = []
                sources = []
                for component, weight in zip(spec["components"], spec["weights"]):
                    if weight <= 0:
                        continue
                    result = train_component(component)
                    if result is not base:
                        completed(result)
                    sources.append(result["raw"])
                    frames.append(pd.read_csv(contained_path(result["raw"]["prediction"])))
                    weights.append(weight)
                frame = rank_blend(frames, weights)
                raw_id = f"{study_id}_{phase}_{selection['selected']['name']}"
                derived_run(cfg, path, fold, raw_id, owner, revision, provenance, frame, prepared.truth,
                            "RankEnsemble", {"spec": spec, "sources": sources})
                candidate = entry(cfg, path, fold, selection["selected"]["name"], spec, raw_id, owner,
                                  revision, provenance, prepared.truth)
                completed(candidate)
            tolerance = research["selection_tolerance"]
            promote = (selection["selected"]["name"] != "L0" and positive(candidate)
                       and selection["selected"]["band"]["metrics"]["final_score"] > selection["results"][0]["band"]["metrics"]["final_score"] + tolerance
                       and candidate["band"]["metrics"]["final_score"] > base["band"]["metrics"]["final_score"] + tolerance)
            confirmation = {"study_id": study_id, "confirmed_at": timestamp(), "git": revision,
                            "selection_sha256": state["selection_sha256"], "results": [base] + ([candidate] if candidate is not base else []),
                            "component_results": results, "new_training_count": new_training_count,
                            "recommended": candidate["name"] if promote else "L0 + band(0.1)",
                            "member_b_independent_review": "pending"}
            write_json(root / "confirmation.json", confirmation)
            save_analyses(cfg, root, candidate, base, phase)
            state.update(status="complete", finished_at=timestamp(), recommended=confirmation["recommended"],
                         confirm_training_count=new_training_count)
            write_report(cfg, study_id, selection, state["selection_sha256"], confirmation)
            log(f"2024 confirmation complete: recommended={confirmation['recommended']}")
        if code_hashes() != source or source_provenance(cfg) != provenance:
            raise AssertionError("Source/original data/official attachments changed during the phase.")
        write_json(state_path, state)
        comparison(results).to_csv(root / f"{phase}_comparison.csv", index=False)
        print(comparison(results)[["name", "raw_ic_mean", "raw_annual_excess", "raw_final_score", "band_final_score", "band_delta_score"]].to_string(index=False), flush=True)
        return state
    except Exception as error:
        state.update(status="failed", failed_phase=phase, error=str(error), traceback=traceback.format_exc())
        write_json(state_path, state)
        log(f"FAILED: {error}; preserve partial results, no selection from an incomplete search")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--phase", choices=["screen", "confirm"], required=True)
    parser.add_argument("--owner", default="A")
    args = parser.parse_args(argv)
    cfg, path = load_config(args.config)
    run_study(cfg, path, args.study_id, args.owner, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

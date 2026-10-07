"""Joint TPE search: causal walk-forward LightGBM and one common band parameter.

The outer validation labels never enter features, training or early stopping.
Only 2021–2023 scores guide search; a committed selection gates 2024 evaluation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import psutil
import yaml
from optuna.trial import TrialState

from src.data.load_data import KEYS
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame, load_validation_frame
from src.evaluation.research_diagnostics import monthly_metrics, portfolio_diagnostics
from src.evaluation.turnover import band_scores
from src.models.baseline import RunLog, source_provenance
from src.models.lightgbm_model import load_model, save_model, train_model
from src.utils.experiments import append_record, read_records
from src.utils.project import ROOT, git_state, load_config, project_path, sha256, timestamp
from src.utils.research_cache import code_hashes, contained_path, digest
from src.utils.tuning_cache import TuningFold, canonical, make_fold, prepare_folds

PACKAGES = ["numpy", "pandas", "scipy", "lightgbm", "pyarrow", "PyYAML", "optuna", "sqlalchemy"]


class InvalidTrial(ValueError):
    """A non-finite model/score is excluded rather than assigned a fake score."""


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n").encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def protocol(cfg: dict) -> dict:
    paths = {k: cfg["paths"][k] for k in ["train", "test_x", "models", "metrics", "predictions",
              "figures", "experiment_log", "research_studies", "research_reports", "optuna_cache"]}
    return {"paths": paths, "data": cfg["data"], "features": cfg["features"],
            "baseline": cfg["baseline"], "optuna": cfg["optuna"],
            "score_formula": "0.4*ic_mean+0.3*annual_excess+0.3*(1-mean_turnover)"}


def provenance(cfg: dict) -> dict:
    value = source_provenance(cfg)
    manifest = read_json(ROOT / "data/manifest.json")
    path = project_path(cfg["paths"]["test_x"])
    original = next(f for f in manifest["files"] if project_path(f["path"]) == path)
    actual = sha256(path)
    if actual != original["sha256"] or path.stat().st_size != original["size_bytes"]:
        raise ValueError("Original test X differs from its manifest.")
    value["test_x"] = {"path": cfg["paths"]["test_x"], "sha256": actual,
                        "size_bytes": path.stat().st_size}
    return value


def assert_identity(cfg: dict, meta: dict, *, check_data=False) -> None:
    if digest(protocol(cfg)) != meta["protocol_digest"] or code_hashes() != meta["source"]:
        raise ValueError("Study code/protocol changed: use a new study ID; preserve existing results.")
    if {p: importlib.metadata.version(p) for p in PACKAGES} != meta["environment"]["packages"]:
        raise ValueError("Study dependency versions changed.")
    if check_data and provenance(cfg) != meta["data"]:
        raise ValueError("Study original data/official attachments changed.")


@contextmanager
def study_lock(root: Path):
    lock = root / "process.lock"
    if lock.exists():
        previous = read_json(lock)
        if psutil.pid_exists(previous["pid"]):
            raise RuntimeError("Study already has an active process; concurrent writers are refused.")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": timestamp()}, stream)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def sample_spec(trial, cfg: dict) -> dict:
    sampled = {}
    for name, space in cfg["optuna"]["search_space"].items():
        if isinstance(space, list):
            sampled[name] = trial.suggest_categorical(name, space)
        elif name in {"num_leaves", "min_data_in_leaf"}:
            sampled[name] = trial.suggest_int(name, **space)
        else:
            sampled[name] = trial.suggest_float(name, **space)
    params = copy.deepcopy(cfg["baseline"]["model"]["params"])
    params.update({k: v for k, v in sampled.items() if k not in {"band_keep_q", "num_boost_round"}})
    if params["max_depth"] > 0:
        params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"])
    params["bagging_freq"] = 1 if params["bagging_fraction"] < 1 else 0
    return {"sampled_params": sampled, "model_params": params,
            "rounds": int(sampled["num_boost_round"]), "keep_q": float(sampled["band_keep_q"]),
            "initial_state": "cold_start"}


def seed_trials(study, cfg: dict) -> None:
    space = cfg["optuna"]["search_space"]
    for name, overrides in [("V1", {}), ("L1", {"num_leaves": 15, "min_data_in_leaf": 500,
                                               "lambda_l2": 5.0})]:
        params = {k: cfg["baseline"]["model"]["params"][k]
                  for k in space if k not in {"num_boost_round", "band_keep_q"}}
        params.update(overrides)
        params.update(num_boost_round=200, band_keep_q=cfg["optuna"]["baseline_keep_q"])
        study.enqueue_trial(params, user_attrs={"role": f"reference_{name}"})
    for q in cfg["optuna"]["warm_keep_qs"]:
        study.enqueue_trial({"band_keep_q": q}, user_attrs={"role": "warm_band"})


def checkpoint(root: Path, study, pending: int | None) -> None:
    sampler = root / "sampler.pkl"
    atomic_bytes(sampler, pickle.dumps(study.sampler, protocol=pickle.HIGHEST_PROTOCOL))
    save_json(root / "checkpoint.json", {"sampler_sha256": sha256(sampler),
              "pending_trial": pending, "finished": [t.number for t in study.get_trials(deepcopy=False)
                                                       if t.state.is_finished()]})


def ensure_record(cfg: dict, info: dict, manifest: Path) -> None:
    record = {"exp_id": info["exp_id"], "date": info["finished_at"], "owner": info["owner"],
              "git_commit": info["git"]["commit"], "config_path": info["config_path"],
              "features": "baseline_v1 (40)", "model": info["family"],
              "params": json.dumps(info["spec"], sort_keys=True),
              "train_period": "-".join(map(str, info["split"]["train_window"])),
              "valid_period": "-".join(map(str, info["split"]["valid_window"])),
              **{k: info["metrics"][k] for k in OFFICIAL_METRICS},
              "artifact_path": relative(manifest), "notes": info["notes"]}
    log_path = project_path(cfg["paths"]["experiment_log"])
    existing = next((r for r in read_records(log_path) if r["exp_id"] == info["exp_id"]), None)
    if existing is not None:
        if existing != {k: str(v) for k, v in record.items()}:
            raise ValueError("Existing experiment record differs from its passed manifest.")
    else:
        append_record(log_path, record)


def passed_run(cfg: dict, exp_id: str, spec: dict) -> dict | None:
    manifest = project_path(cfg["paths"]["models"]) / exp_id / "run.json"
    if not manifest.exists():
        return None
    info = read_json(manifest)
    if info["spec"] != spec:
        raise ValueError("Existing experiment has different parameters.")
    if info["status"] != "passed":
        return None
    for path, expected in info["artifact_hashes"].items():
        if sha256(contained_path(path)) != expected:
            raise ValueError(f"Passed artifact changed: {path}")
    ensure_record(cfg, info, manifest)
    return info


def run_prediction(cfg, config_path, root, meta, data: TuningFold, exp_id, spec,
                   log, *, raw_source=None) -> dict:
    passed = passed_run(cfg, exp_id, spec)
    if passed:
        log(f"reuse passed {exp_id}")
        return passed
    paths = {k: contained_path(cfg["paths"][k]) / exp_id for k in ["models", "metrics", "predictions"]}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    manifest = paths["models"] / "run.json"
    started = time.perf_counter()
    layer = "raw" if raw_source is None else "band"
    info = {"exp_id": exp_id, "status": "running", "started_at": timestamp(),
            "owner": meta["owner"], "git": meta.get("confirm_git", meta["git"]), "data": meta["data"],
            "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
            "config_path": relative(config_path), "spec": spec, "split": data.split,
            "training_samples": data.supervision, "family": "LightGBM" if layer == "raw" else "LightGBM+band",
            "layer": layer, "timings_seconds": {}, "environment": meta["environment"]}
    if raw_source:
        info["raw_source"] = raw_source["exp_id"]
    save_json(manifest, info)
    (paths["models"] / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    prediction_path = paths["predictions"] / f"valid_{data.fold.valid_start // 10000}.csv"
    try:
        if layer == "raw":
            log(f"fit {exp_id}: rounds={spec['rounds']}, leaves={spec['model_params']['num_leaves']}")
            def progress(message):
                done = int(message.split()[2].split("/")[0])
                if done % 100 == 0 or done == spec["rounds"]:
                    log(f"{exp_id}: {message}")
            step = time.perf_counter()
            model = train_model(data.x_train, data.y_train, spec["model_params"], spec["rounds"], progress)
            model_path = paths["models"] / "model.txt"
            save_model(model, model_path)
            info["actual_iterations"] = model.current_iteration()
            pd.DataFrame({"feature": list(data.x_train.columns),
                          "gain": model.feature_importance(importance_type="gain"),
                          "split_count": model.feature_importance(importance_type="split")}).to_csv(
                paths["metrics"] / "feature_importance.csv", index=False)
            info["timings_seconds"]["training"] = time.perf_counter() - step
            values = np.full(len(data.keys), cfg["baseline"]["fallback_prediction"], dtype="float64")
            values[data.eligible] = model.predict(data.x_valid, num_threads=spec["model_params"]["num_threads"])
            restored = load_model(model_path)
            check = restored.predict(data.x_valid, num_threads=spec["model_params"]["num_threads"])
            difference = float(np.max(np.abs(values[data.eligible] - check)))
            if not np.isfinite(difference) or difference > 1e-12:
                raise ValueError("Model reload prediction differs.")
            info.update(reload_max_difference=difference, fallback_rows=int((~data.eligible).sum()))
            pred = canonical(data.keys.assign(pred=values))
            del model, restored, values, check
            gc.collect()
        else:
            scored = load_validation_frame(contained_path(raw_source["prediction"]), data.labels_path)
            pred = scored[KEYS].copy()
            # Never expose labels or label-validity masks to the band transform.
            known = scored[KEYS + ["pred", "flag_limit_up"]]
            pred["pred"] = band_scores(known, spec["keep_q"]).to_numpy()
            pred = canonical(pred)
            del scored, known
        if not np.isfinite(pred.pred.to_numpy()).all():
            raise InvalidTrial("Predictions are not finite.")
        check_predictions(pred, data.truth)
        pred.to_csv(prediction_path, index=False)
        del pred
        scored = load_validation_frame(prediction_path, data.labels_path)
        check_predictions(scored[KEYS + ["pred"]], data.truth)
        metrics = evaluate_frame(scored)
        if not all(np.isfinite(metrics[k]) for k in OFFICIAL_METRICS):
            raise InvalidTrial("Official score contains non-finite metrics.")
        differences = compare_official(prediction_path, data.truth, metrics, cfg["baseline"]["score_tolerance"])
        daily = daily_metrics(scored)
        daily.to_csv(paths["metrics"] / "daily_metrics.csv", index=False)
        monthly_metrics(daily).to_csv(paths["metrics"] / "monthly_metrics.csv", index=False)
        save_json(paths["metrics"] / "metrics.json", metrics)
        info.update(status="passed", finished_at=timestamp(), metrics=metrics,
                    official_differences=differences, official_max_difference=max(differences.values()),
                    duration_seconds=time.perf_counter() - started,
                    prediction=relative(prediction_path), labels=relative(data.labels_path),
                    validation_rows=len(scored), validation_days=int(scored.trade_date.nunique()),
                    peak_sampled_rss_bytes=psutil.Process().memory_info().rss,
                    checks={"keys_complete_unique_finite": True, "official_match": True,
                            "row_order": ["trade_date", "ts_code"], "cold_start": True},
                    notes=f"{meta['study_id']}; {layer}; cold_start; complete fold; "
                          + ("new model fit; no early stopping" if layer == "raw" else "derived band; no additional fit"))
        artifact_paths = [prediction_path, data.labels_path, paths["models"] / "config.yaml",
                          paths["metrics"] / "daily_metrics.csv", paths["metrics"] / "monthly_metrics.csv",
                          paths["metrics"] / "metrics.json"]
        if layer == "raw":
            artifact_paths += [model_path, paths["metrics"] / "feature_importance.csv"]
            info["model"] = relative(model_path)
        info["artifact_hashes"] = {relative(p): sha256(p) for p in artifact_paths}
        save_json(manifest, info)
        ensure_record(cfg, info, manifest)
        log(f"PASS {exp_id}: score={metrics['final_score']:.10f}, official_diff={max(differences.values()):.2e}")
        del scored, daily
        gc.collect()
        return info
    except Exception as error:
        info.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save_json(manifest, info)
        raise


def fold_entry(cfg, path, root, meta, data, prefix, spec, log) -> dict:
    raw = run_prediction(cfg, path, root, meta, data, prefix + "_raw", spec, log)
    band = run_prediction(cfg, path, root, meta, data, prefix + "_band", spec, log, raw_source=raw)
    return {"raw": raw, "band": band}


def aggregate(entries: dict) -> dict:
    values = [v["band"]["metrics"]["final_score"] for v in entries.values()]
    return {"objective": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
            "worst": float(np.min(values)), "scores": {k: v["band"]["metrics"]["final_score"]
                                                       for k, v in entries.items()}}


def candidate_from_trial(trial) -> dict:
    entries = trial.user_attrs["folds"]
    return {"name": f"T{trial.number:03d}", "source_trial": trial.number,
            "spec": trial.user_attrs["spec"], "folds": entries, **aggregate(entries)}


def trial_table(study, root: Path) -> None:
    rows = []
    for t in study.get_trials(deepcopy=False):
        if t.state == TrialState.WAITING:
            continue
        row = {"trial": t.number, "state": t.state.name, "objective": t.value,
               **t.params, "error": t.user_attrs.get("error", "")}
        spec = t.user_attrs.get("spec")
        if spec:
            row["effective_num_leaves"] = spec["model_params"]["num_leaves"]
        entries = t.user_attrs.get("folds", {})
        for name, entry in entries.items():
            for layer in ["raw", "band"]:
                row.update({f"{name}_{layer}_{k}": entry[layer]["metrics"][k] for k in OFFICIAL_METRICS})
        if len(entries) == 3:
            row.update(aggregate(entries))
            row.pop("scores")
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "trials.csv", index=False)


def derived_candidates(cfg, path, root, meta, folds, candidates, best, log):
    refs = [c for c in candidates if c["source_trial"] in {0, 1, best["source_trial"]}]
    seen = {digest({"model": c["spec"]["model_params"], "rounds": c["spec"]["rounds"],
                    "q": c["spec"]["keep_q"]}) for c in candidates}
    new = []
    for source in refs:
        for q in dict.fromkeys([cfg["optuna"]["baseline_keep_q"], best["spec"]["keep_q"]]):
            spec = copy.deepcopy(source["spec"])
            spec["keep_q"] = q
            spec["sampled_params"]["band_keep_q"] = q
            identity = digest({"model": spec["model_params"], "rounds": spec["rounds"], "q": q})
            if identity in seen:
                continue
            name = f"D{len(new):03d}"
            entries = {}
            for data in folds:
                raw = source["folds"][data.fold.name]["raw"]
                band = run_prediction(cfg, path, root, meta, data,
                          f"{meta['study_id']}_{name}_{data.fold.name}_band", spec, log, raw_source=raw)
                entries[data.fold.name] = {"raw": raw, "band": band}
            new.append({"name": name, "source_trial": source["source_trial"], "spec": spec,
                        "folds": entries, "derived": True, **aggregate(entries)})
            seen.add(identity)
    return new


def audit(cfg: dict, meta: dict, root: Path) -> dict:
    assert_identity(cfg, meta, check_data=True)
    records = read_records(project_path(cfg["paths"]["experiment_log"]))
    initial = meta["initial_records"]
    if records[:len(initial)] != initial or len({r["exp_id"] for r in records}) != len(records):
        raise ValueError("Previous experiment records changed or IDs are duplicated.")
    runs = []
    for record in records[len(initial):]:
        if not record["exp_id"].startswith(meta["study_id"] + "_"):
            continue
        manifest = contained_path(record["artifact_path"])
        info = read_json(manifest)
        passed_run(cfg, info["exp_id"], info["spec"])
        runs.append({"exp_id": info["exp_id"], "manifest": relative(manifest),
                     "manifest_sha256": sha256(manifest), "layer": info["layer"],
                     "prediction": info["prediction"], "metrics": info["metrics"],
                     "rows": info["validation_rows"], "days": info["validation_days"],
                     "reload_difference": info.get("reload_max_difference"),
                     "official_difference": info["official_max_difference"],
                     "artifact_hashes": info["artifact_hashes"]})
    result = {"status": "passed", "time": timestamp(), "initial_records": len(initial),
              "study_records": len(runs), "total_records": len(records),
              "new_model_fits": sum(r["layer"] == "raw" for r in runs),
              "max_official_difference": max((r["official_difference"] for r in runs), default=0),
              "max_reload_difference": max((r["reload_difference"] or 0 for r in runs), default=0),
              "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
              "original_files_unchanged": True, "previous_records_unchanged": True, "runs": runs}
    save_json(root / "audit.json", result)
    return result


def report_path(cfg, study_id, suffix=".md") -> Path:
    return contained_path(cfg["paths"]["research_reports"]) / f"optuna_tuning_{study_id}{suffix}"


def write_report(cfg, meta, root, selection, study=None, confirmation=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = contained_path(cfg["paths"]["figures"]) / meta["study_id"]
    figures.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(root / "trials.csv")
    good = table[table.state == "COMPLETE"].sort_values("trial")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(good.trial, good.objective, ".", label="Trial CV mean")
    ax.plot(good.trial, good.objective.cummax(), label="Best so far")
    ax.set(xlabel="Trial", ylabel="Mean official final score (2021–2023)")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "convergence.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(good.band_keep_q, good.objective)
    ax.axvline(.1, color="grey", linestyle="--", label="Previous keep_q=0.1")
    ax.set(xlabel="Band keep_q", ylabel="CV mean score"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "band_score.png", dpi=180); plt.close(fig)
    importance = None
    if study is not None:
        importance = optuna.importance.get_param_importances(study,
                    evaluator=optuna.importance.FanovaImportanceEvaluator(seed=cfg["optuna"]["seed"]))
        save_json(root / "parameter_importance.json", importance)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(importance)[::-1], list(importance.values())[::-1])
        ax.set(xlabel="fANOVA importance (search observations)")
        fig.tight_layout(); fig.savefig(figures / "parameter_importance.png", dpi=180); plt.close(fig)
    elif (root / "parameter_importance.json").exists():
        importance = read_json(root / "parameter_importance.json")
    winner = selection["winner"]
    references = selection["references"]
    lines = [f"# {meta['study_id']}：LightGBM 与 band 联合自动调参", "",
             f"更新时间：{timestamp()}", "",
             f"状态：{'已完成锁定后的 2024 评估' if confirmation else '三折搜索完成，候选已锁定，2024 尚未评估'}。", "",
             "## 固定协议", "",
             "50 个联合 trial；40 特征；2021 / 2022 / 2023 扩展窗口；三折共用一个 keep_q；",
             "目标为三折官方综合分等权均值。模型按固定轮数训练，无验证早停、无 pruning。",
             "keep_q 开放 0–1；前十组有八组集中于 0.05–0.15，另两组检查 0 / 1；",
             "完整参数与 band 共同选优，IC 不设额外筛选门槛。每折独立冷启动，行序为日期、股票代码。", "",
             f"训练实现提交：`{meta['git']['commit']}`；代码摘要：`{meta['source_digest']}`；",
             f"协议摘要：`{meta['protocol_digest']}`；锁定摘要：`{selection['selection_digest']}`。", "",
             f"有效 trial：{len(good)}；失败 trial：{int((table.state == 'FAIL').sum())}；",
             f"派生对照：{len(selection['derived'])}（只评分，无额外训练）。", "",
             "## 候选与同层对照", "",
             "|方案|keep_q|2021|2022|2023|三折均值|标准差|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    displayed = list({c["name"]: c for c in references + selection["derived"]
                      + [selection["tpe_best"], winner]}.values())
    for c in displayed:
        values = [c["scores"][f"wf{year}"] for year in [2021, 2022, 2023]]
        lines.append(f"|{c['name']}|{c['spec']['keep_q']:.10g}|" +
                     "|".join(f"{v:.10f}" for v in values) + f"|{c['objective']:.10f}|{c['std']:.10f}|")
    lines += ["", "最优完整配置：", "", "```json",
              json.dumps(winner["spec"], ensure_ascii=False, indent=2), "```", "",
              "## 分项表现", "", "|年份|原始 IC|band IC|band 年化超额|band 换手|band 综合分|",
              "|---|---:|---:|---:|---:|---:|"]
    for name, entry in winner["folds"].items():
        m, raw = entry["band"]["metrics"], entry["raw"]["metrics"]
        lines.append(f"|{name}|{raw['ic_mean']:.10f}|{m['ic_mean']:.10f}|"
                     f"{m['annual_excess']:.10f}|{m['mean_turnover']:.10f}|{m['final_score']:.10f}|")
    lines += ["", "相对本轮 V1 + band(0.1) 的官方得分归因：", "",
              "|年份|IC 项变化|超额项变化|稳定性项变化|总分变化|",
              "|---|---:|---:|---:|---:|"]
    reference_v1 = next(c for c in references if c["source_trial"] == 0)
    for name, entry in winner["folds"].items():
        m = entry["band"]["metrics"]
        ref = reference_v1["folds"][name]["band"]["metrics"]
        changes = [.4*(m["ic_mean"]-ref["ic_mean"]),
                   .3*(m["annual_excess"]-ref["annual_excess"]),
                   -.3*(m["mean_turnover"]-ref["mean_turnover"])]
        lines.append(f"|{name}|" + "|".join(f"{v:+.10f}" for v in changes)
                     + f"|{sum(changes):+.10f}|")
    if confirmation:
        lines += ["", "## 锁定后的 2024 历史验证", "",
                  f"搜索候选先提交、推送后，以 `{confirmation['lock_commit']}` 启动 2024。", "",
                  "|方案|IC|年化超额|换手|综合分|", "|---|---:|---:|---:|---:|"]
        for label, m in confirmation["comparison"].items():
            lines.append(f"|{label}|{m['ic_mean']:.10f}|{m['annual_excess']:.10f}|"
                         f"{m['mean_turnover']:.10f}|{m['final_score']:.10f}|")
    lines += ["", "## 产物与解释边界", "",
              f"本地产物：`{relative(root)}`；图表：`{relative(figures)}`。",
              "每个通过的原始 / band 实验已按原 20 列格式追加共享表，CV 均值单独记录。",
              "原始数据与三个官方附件不变；完整键、模型重载与落盘官方八指标均在运行内核对。",
              "2023 / 2024 及 band 设计此前已参与研究；2024 并非未触碰的独立留出集。",
              "keep_q 较小时，综合分可能主要受低换手推动；分项归因须与预测能力一起解读。",
              "本轮未新增或运行测试套件，未训练正式全数据提交模型，未生成测试 submission。",
              "50 次搜索不保证全局最优；正式推荐方案的变更需结合本报告及成员 B 的复核。", ""]
    report_path(cfg, meta["study_id"]).write_text("\n".join(lines), encoding="utf-8")


def search(cfg, path, root, meta, log, resume):
    if resume:
        state = read_json(root / "checkpoint.json")
        if sha256(root / "sampler.pkl") != state["sampler_sha256"]:
            raise ValueError("Sampler checkpoint checksum mismatch.")
        sampler = pickle.loads((root / "sampler.pkl").read_bytes())
    else:
        state = {"pending_trial": None}
        sampler = optuna.samplers.TPESampler(seed=cfg["optuna"]["seed"],
                    n_startup_trials=cfg["optuna"]["startup_trials"], multivariate=True)
    study = optuna.create_study(study_name=meta["study_id"], direction="maximize",
                storage="sqlite:///" + (root / "study.db").as_posix(), sampler=sampler,
                pruner=optuna.pruners.NopPruner(), load_if_exists=resume)
    if not resume:
        seed_trials(study, cfg)
        checkpoint(root, study, None)
    folds, cache = prepare_folds(cfg, cfg["optuna"]["folds"], meta["data"], root, log)
    if "search_cache" in meta and meta["search_cache"] != cache:
        raise ValueError("Cached fold inputs/label files changed on resume.")
    meta["search_cache"] = cache
    save_json(root / "study.json", meta)
    active = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
    if active and (len(active) != 1 or active[0].number != state["pending_trial"]):
        raise ValueError("Database and pending-trial checkpoint disagree; preserve them for inspection.")
    while True:
        assert_identity(cfg, meta)
        finished = [t for t in study.get_trials(deepcopy=False) if t.state.is_finished()]
        if len(finished) >= cfg["optuna"]["trial_budget"]:
            break
        if active:
            frozen = active.pop()
            trial = optuna.trial.Trial(study, frozen._trial_id)
            spec = trial.user_attrs["spec"]
        else:
            trial = study.ask()
            spec = sample_spec(trial, cfg)
            trial.set_user_attr("spec", spec)
            trial.set_user_attr("started_at", timestamp())
            checkpoint(root, study, trial.number)
        log(f"TRIAL {trial.number + 1}/{cfg['optuna']['trial_budget']}: keep_q={spec['keep_q']:.10g}, rounds={spec['rounds']}")
        try:
            entries = {}
            for data in folds:
                meta["progress"] = {"trial": trial.number, "fold": data.fold.name, "at": timestamp()}
                save_json(root / "study.json", meta)
                entries[data.fold.name] = fold_entry(cfg, path, root, meta, data,
                            f"{meta['study_id']}_T{trial.number:03d}_{data.fold.name}", spec, log)
                trial.set_user_attr("folds", entries)
            result = aggregate(entries)
            trial.set_user_attr("aggregate", result)
            study.tell(trial, result["objective"])
            log(f"TRIAL_COMPLETE {trial.number}: mean={result['objective']:.10f}, std={result['std']:.10f}")
        except InvalidTrial as error:
            trial.set_user_attr("error", str(error))
            study.tell(trial, state=TrialState.FAIL)
            log(f"TRIAL_FAIL {trial.number}: {error}")
        checkpoint(root, study, None)
        trial_table(study, root)
    candidates = [candidate_from_trial(t) for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    if not candidates or not any(c["source_trial"] == 0 for c in candidates) or not any(c["source_trial"] == 1 for c in candidates):
        raise ValueError("Both V1/L1 controls and at least one valid candidate are required.")
    key = lambda c: (-c["objective"], abs(c["spec"]["keep_q"] - .1), c["source_trial"], c["name"])
    best = min(candidates, key=key)
    derived = derived_candidates(cfg, path, root, meta, folds, candidates, best, log)
    winner = min(candidates + derived, key=key)
    assert_identity(cfg, meta, check_data=True)
    selection = {"schema": 1, "study_id": meta["study_id"], "created_at": timestamp(),
                 "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
                 "data": meta["data"], "winner": winner, "tpe_best": best,
                 "references": [c for c in candidates if c["source_trial"] in {0, 1}],
                 "derived": derived, "trial_summary": [{k: c[k] for k in ["name", "source_trial", "spec", "objective", "std", "worst", "scores"]} for c in candidates],
                 "rule": "max common-band three-fold mean; exact ties: closer to 0.1, earlier trial"}
    selection["selection_digest"] = digest(selection)
    save_json(root / "selection.json", selection)
    save_json(report_path(cfg, meta["study_id"], "_selection.json"), selection)
    meta.update(status="search_complete", selection_digest=selection["selection_digest"],
                winner=winner["name"], search_finished_at=timestamp())
    save_json(root / "study.json", meta)
    audit(cfg, meta, root)
    write_report(cfg, meta, root, selection, study=study)
    log(f"SEARCH_COMPLETE: {winner['name']}, mean={winner['objective']:.10f}, keep_q={winner['spec']['keep_q']:.10g}; commit selection before confirm")


def confirm(cfg, path, root, meta, log):
    selection_path = report_path(cfg, meta["study_id"], "_selection.json")
    selection = read_json(selection_path)
    payload = {k: v for k, v in selection.items() if k != "selection_digest"}
    if digest(payload) != selection["selection_digest"] or selection["selection_digest"] != meta["selection_digest"]:
        raise ValueError("Selection digest mismatch.")
    if git_state()["dirty"]:
        raise ValueError("Commit selection and records before 2024 confirmation.")
    committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"origin/main:{relative(selection_path)}"], encoding="utf-8")
    if json.loads(committed) != selection:
        raise ValueError("The exact selection must be pushed to origin/main first.")
    assert_identity(cfg, meta, check_data=True)
    meta["confirm_git"] = git_state()
    folds, cache = prepare_folds(cfg, [cfg["optuna"]["confirm"]], meta["data"], root, log)
    meta["confirm_cache"] = cache
    save_json(root / "study.json", meta)
    data = folds[0]
    entry = fold_entry(cfg, path, root, meta, data, f"{meta['study_id']}_confirm2024",
                       selection["winner"]["spec"], log)
    # Diagnostic q=0.1 is predeclared and cannot change the locked candidate.
    fixed = copy.deepcopy(selection["winner"]["spec"])
    fixed["keep_q"] = .1
    fixed["sampled_params"]["band_keep_q"] = .1
    comparison = {"selected_raw": entry["raw"]["metrics"], "selected_joint": entry["band"]["metrics"]}
    if fixed != selection["winner"]["spec"]:
        control = run_prediction(cfg, path, root, meta, data, f"{meta['study_id']}_confirm2024_fixed01_band",
                                 fixed, log, raw_source=entry["raw"])
        comparison["selected_model_band01"] = control["metrics"]
    else:
        comparison["selected_model_band01"] = entry["band"]["metrics"]
    references = {"V1_band01": "S002_confirm_L0", "L1_band01": "S002_confirm_L1"}
    frozen_index = read_json(ROOT / "docs/model_research_S002_artifacts.json")["predictions"]
    reference_artifacts = {}
    for label, exp_id in references.items():
        old_manifest = project_path(cfg["paths"]["models"]) / exp_id / "run.json"
        if not old_manifest.exists():
            raise FileNotFoundError(f"Missing frozen reference: {exp_id}; restore documented artifacts.")
        old = read_json(old_manifest)
        old_pred = old.get("prediction", {})
        old_file = old_pred.get("file") if isinstance(old_pred, dict) else old_pred
        frozen = next(r for r in frozen_index if r["exp_id"] == exp_id)
        if (sha256(old_manifest) != frozen["files"]["manifest"]["sha256"]
                or sha256(contained_path(old_file)) != frozen["files"]["prediction"]["sha256"]):
            raise ValueError(f"Frozen reference artifact changed: {exp_id}")
        reference = canonical(pd.read_csv(contained_path(old_file)))
        check_predictions(reference, data.truth)
        scored = reference.merge(data.truth, on=KEYS, validate="one_to_one")
        reference["pred"] = band_scores(scored[KEYS + ["pred", "flag_limit_up"]], .1).to_numpy()
        temporary_pred = root / f"reference_{label}.csv"
        reference.to_csv(temporary_pred, index=False)
        scored = load_validation_frame(temporary_pred, data.labels_path)
        metrics = evaluate_frame(scored)
        diff = compare_official(temporary_pred, data.truth, metrics, cfg["baseline"]["score_tolerance"])
        comparison[label] = metrics
        reference_artifacts[label] = {"source": old_file, "source_sha256": sha256(contained_path(old_file)),
                                      "canonical_band_file": relative(temporary_pred),
                                      "sha256": sha256(temporary_pred), "official_difference": max(diff.values())}
    confirmation = {"status": "passed", "time": timestamp(), "lock_commit": meta["confirm_git"]["commit"],
                    "selection_digest": selection["selection_digest"], "candidate": entry,
                    "comparison": comparison, "references": reference_artifacts}
    save_json(root / "confirmation.json", confirmation)
    meta.update(status="complete", completed_at=timestamp())
    save_json(root / "study.json", meta)
    audited = audit(cfg, meta, root)
    save_json(report_path(cfg, meta["study_id"], "_artifacts.json"), audited)
    write_report(cfg, meta, root, selection, confirmation=confirmation)
    log(f"COMPLETE {meta['study_id']}: 2024={entry['band']['metrics']['final_score']:.10f}; no post-confirmation search")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Joint Optuna/TPE LightGBM+band walk-forward research")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--phase", choices=["search", "confirm"], required=True)
    parser.add_argument("--owner", default="A")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", args.study_id):
        parser.error("Invalid study ID.")
    cfg, path = load_config(args.config)
    if (cfg["optuna"]["trial_budget"] != 50 or cfg["optuna"]["initial_state"] != "cold_start"
            or cfg["features"]["version"] != "baseline_v1"
            or [f["valid"][0] // 10000 for f in cfg["optuna"]["folds"]] != [2021, 2022, 2023]
            or cfg["optuna"]["confirm"]["valid"] != [20240101, 20241231]):
        raise ValueError("Configuration differs from the agreed S003 protocol.")
    root = contained_path(cfg["paths"]["research_studies"]) / args.study_id
    meta_path = root / "study.json"
    if args.phase == "search" and not args.resume:
        if root.exists():
            raise FileExistsError("Study exists; use --resume or a new ID.")
        revision = git_state()
        if revision["branch"] != "main" or revision["dirty"]:
            raise ValueError("Commit the implementation on clean main before starting search.")
        root.mkdir(parents=True)
        source = code_hashes()
        meta = {"schema": 1, "study_id": args.study_id, "owner": args.owner,
                "status": "running", "started_at": timestamp(), "git": revision,
                "data": provenance(cfg), "source": source, "source_digest": digest(source),
                "protocol": protocol(cfg), "protocol_digest": digest(protocol(cfg)),
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "packages": {p: importlib.metadata.version(p) for p in PACKAGES}},
                "initial_records": read_records(project_path(cfg["paths"]["experiment_log"]))}
        save_json(meta_path, meta)
        (root / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    else:
        meta = read_json(meta_path)
        assert_identity(cfg, meta, check_data=True)
    with study_lock(root):
        log = RunLog(root / "run.log")
        try:
            if args.phase == "search":
                search(cfg, path, root, meta, log, args.resume)
            else:
                confirm(cfg, path, root, meta, log)
        except Exception as error:
            meta.update(status="interrupted", error=str(error), traceback=traceback.format_exc())
            save_json(meta_path, meta)
            log(f"STOP: {error}; artifacts preserved")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

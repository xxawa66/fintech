"""One command from original CSV to a full baseline and an honest experiment record.

Run from the repository root with the project's Python environment::

    python -m src.models.baseline --fold fold2 --exp-id E000_baseline --owner A

The automatic 256-stock/10-round preflight is stored separately and is not logged
as a full experiment. Existing experiment directories are never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import yaml

from src.data.clean_data import clean_history
from src.data.load_data import KEYS, X_COLUMNS, load_training_data
from src.data.make_dataset import attach_labels, training_arrays
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame, load_validation_frame
from src.evaluation.validation import get_folds, split_train_valid
from src.features.build_features import build_features
from src.models.lightgbm_model import train_model
from src.utils.experiments import append_record, check_experiment_id
from src.utils.project import ROOT, git_state, load_config, project_path, sha256, timestamp, write_json


def run_directories(cfg: dict, exp_id: str) -> dict[str, Path]:
    paths = {name: (project_path(cfg["paths"][name]) / exp_id).resolve()
             for name in ["models", "metrics", "predictions", "processed"]}
    if any(not path.is_relative_to(ROOT) for path in paths.values()):
        raise ValueError("Experiment artifact directories must stay within the repository.")
    return paths


def source_provenance(cfg: dict) -> dict:
    source = project_path(cfg["paths"]["train"])
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    record = next((item for item in manifest["files"] if project_path(item["path"]) == source), None)
    if record is None:
        raise ValueError("Configured training data is absent from the data manifest.")
    actual = sha256(source)
    if source.stat().st_size != record["size_bytes"] or actual != record["sha256"]:
        raise ValueError("Training file differs from the supplied original data manifest.")
    return {"path": cfg["paths"]["train"], "size_bytes": source.stat().st_size,
            "sha256": actual, "official_attachments": {
                name: sha256(ROOT / name)
                for name in ["evaluate.py", "evaluate.R", "docs/赛题五-更新.pdf"]}}


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.peak_sampled_rss = 0

    def __call__(self, message: str) -> None:
        self.peak_sampled_rss = max(self.peak_sampled_rss, psutil.Process().memory_info().rss)
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def run_one(cfg: dict, config_path: Path, fold, exp_id: str, owner: str,
            provenance: dict, revision: dict, smoke: bool) -> dict:
    directories = run_directories(cfg, exp_id)
    log_path = project_path(cfg["paths"]["experiment_log"])
    check_experiment_id(exp_id, log_path, list(directories.values()))
    for directory in directories.values():
        directory.mkdir(parents=True)
    log = RunLog(directories["metrics"] / "run.log")
    manifest_path = directories["models"] / "run.json"
    started = time.perf_counter()
    info = {"exp_id": exp_id, "owner": owner, "started_at": timestamp(), "status": "running",
            "run_type": "smoke" if smoke else "full", "git": revision, "data": provenance,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "packages": {name: importlib.metadata.version(name) for name in
                                         ["numpy", "pandas", "scipy", "lightgbm", "pyarrow", "PyYAML"]}},
            "artifacts": {name: path.relative_to(ROOT).as_posix() for name, path in directories.items()},
            "timings_seconds": {}}
    write_json(manifest_path, info)
    (directories["models"] / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        step = time.perf_counter()
        log(f"{exp_id}: reading data, {fold.describe()}")
        raw = load_training_data(project_path(cfg["paths"]["train"]))
        if not smoke and len(raw) != 7_900_350:
            raise ValueError("Full baseline expects the complete supplied training dataset.")
        if smoke:
            stocks = sorted(raw["ts_code"].cat.categories.astype(str))[:cfg["baseline"]["smoke_stocks"]]
            raw = raw[raw["ts_code"].isin(stocks)]
        raw = raw[raw["trade_date"].between(fold.train_start, fold.valid_end)]
        raw, cleaning = clean_history(raw)
        truth = raw.loc[raw["trade_date"].between(fold.valid_start, fold.valid_end),
                        KEYS + ["y_ret_1d", "flag_limit_up"]].copy()
        info["cleaning"] = cleaning
        info["timings_seconds"]["read_clean"] = time.perf_counter() - step
        log(f"cleaned {len(raw):,} rows; preserved missing-price calendar rows")

        step = time.perf_counter()
        features, columns = build_features(raw[KEYS + X_COLUMNS], cfg["features"], log)
        features.to_parquet(directories["processed"] / "features.parquet", index=False)
        write_json(directories["models"] / "features.json", columns)
        info["feature_version"] = cfg["features"]["version"]
        info["feature_columns"] = columns
        info["timings_seconds"]["features"] = time.perf_counter() - step
        log(f"saved {len(columns)} historical features")

        step = time.perf_counter()
        dataset = attach_labels(raw, features)
        train, valid, split_info = split_train_valid(dataset, fold)
        del dataset, features, raw
        x_train, y_train, training_info = training_arrays(train, columns)
        del train
        gc.collect()
        info["split"] = split_info
        info["training_samples"] = training_info
        if not smoke and fold.name == "fold2":
            if (len(valid) != 1_125_300 or valid["trade_date"].nunique() != 242
                    or split_info["n_boundary_labels_dropped"] != 4522):
                raise ValueError("Full fold2 coverage/boundary counts differ from audited data.")
        info["timings_seconds"]["split"] = time.perf_counter() - step
        log(f"training {len(y_train):,} labeled rows; validation {len(valid):,}; "
            f"boundary labels dropped {split_info['n_boundary_labels_dropped']}")

        step = time.perf_counter()
        model_cfg = cfg["baseline"]["model"]
        rounds = cfg["baseline"]["smoke_rounds"] if smoke else model_cfg["num_boost_round"]
        model = train_model(x_train, y_train, model_cfg["params"], rounds, log)
        model_path = directories["models"] / "model.txt"
        model.save_model(str(model_path))
        info["model"] = {"params": model_cfg["params"], "num_boost_round": rounds,
                         "actual_iterations": model.current_iteration()}
        del x_train, y_train
        gc.collect()
        info["timings_seconds"]["training"] = time.perf_counter() - step

        step = time.perf_counter()
        eligible = valid["quote_valid"].to_numpy()
        x_valid = valid.loc[eligible, columns]
        predictions = np.full(len(valid), cfg["baseline"]["fallback_prediction"], dtype="float64")
        predictions[eligible] = model.predict(x_valid, num_threads=model_cfg["params"]["num_threads"])
        reloaded = lgb.Booster(model_file=str(model_path))
        loaded_pred = reloaded.predict(x_valid, num_threads=model_cfg["params"]["num_threads"])
        reload_difference = float(np.max(np.abs(predictions[eligible] - loaded_pred)))
        if reload_difference > 1e-12:
            raise AssertionError("Reloaded model predictions differ from fitted model.")
        prediction_frame = valid[KEYS].copy()
        prediction_frame["pred"] = predictions
        check_predictions(prediction_frame, truth)
        prediction_path = directories["predictions"] / f"valid_{fold.valid_start // 10000}.csv"
        prediction_frame.to_csv(prediction_path, index=False)
        label_path = directories["metrics"] / "validation_labels.csv"
        truth.to_csv(label_path, index=False)
        info["prediction"] = {"rows": len(prediction_frame), "days": int(valid["trade_date"].nunique()),
                              "fallback_rows": int((~eligible).sum()),
                              "model_reload_max_abs_difference": reload_difference,
                              "file": prediction_path.relative_to(ROOT).as_posix()}
        del valid, x_valid, predictions, loaded_pred, prediction_frame, model, reloaded
        gc.collect()
        info["timings_seconds"]["prediction_reload"] = time.perf_counter() - step
        log(f"prediction keys complete; fallback rows {info['prediction']['fallback_rows']:,}; reload verified")

        step = time.perf_counter()
        persisted = pd.read_csv(prediction_path)
        check_predictions(persisted, truth)
        del persisted
        scored = load_validation_frame(prediction_path, label_path)
        metrics = evaluate_frame(scored)
        differences = compare_official(prediction_path, truth, metrics, cfg["baseline"]["score_tolerance"])
        daily_metrics(scored).to_csv(directories["metrics"] / "daily_metrics.csv", index=False)
        write_json(directories["metrics"] / "metrics.json", metrics)
        info["metrics"] = metrics
        info["official_score_max_difference"] = max(differences.values())
        info["official_score_differences"] = differences
        info["timings_seconds"]["scoring"] = time.perf_counter() - step
        info["peak_sampled_process_rss_bytes"] = log.peak_sampled_rss
        for name, before in provenance["official_attachments"].items():
            if sha256(ROOT / name) != before:
                raise AssertionError(f"Official attachment changed: {name}")
        if sha256(project_path(cfg["paths"]["train"])) != provenance["sha256"]:
            raise AssertionError("Original training CSV changed during the run.")
        info["duration_seconds"] = time.perf_counter() - started
        info["finished_at"] = timestamp()
        info["checks"] = {"prediction_coverage": True, "model_reload": True,
                          "official_score_match": True, "original_files_unchanged": True}
        info["status"] = "passed"
        write_json(manifest_path, info)
        if not smoke:
            append_record(log_path, {
                "exp_id": exp_id, "date": info["finished_at"], "owner": owner,
                "git_commit": revision["commit"], "config_path": config_path.relative_to(ROOT).as_posix(),
                "features": cfg["features"]["version"] + f" ({len(columns)})", "model": "LightGBM",
                "params": json.dumps(info["model"], sort_keys=True),
                "train_period": f"{fold.train_start}-{fold.train_end}",
                "valid_period": f"{fold.valid_start}-{fold.valid_end}",
                **{name: metrics[name] for name in OFFICIAL_METRICS},
                "artifact_path": manifest_path.relative_to(ROOT).as_posix(),
                "notes": f"full baseline; seconds={info['duration_seconds']:.1f}; "
                         f"fallback={info['prediction']['fallback_rows']}; dirty={revision['dirty']}; "
                         "fixed rounds, validation labels used only for scoring"})
        log(f"PASS final_score={metrics['final_score']:.8f}; official difference={max(differences.values()):.2e}; "
            f"seconds={info['duration_seconds']:.1f}")
        return info
    except Exception as error:
        info.update({"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        write_json(manifest_path, info)
        log(f"FAILED: {error}; no successful experiment record was appended")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full historical LightGBM baseline with official scoring")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--fold", help="Configured fold name, default baseline.default_fold")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--owner", default="A")
    parser.add_argument("--smoke-only", action="store_true", help="Run only the separate 256-stock preflight")
    args = parser.parse_args(argv)
    cfg, config_path = load_config(args.config)
    fold_name = args.fold or cfg["baseline"]["default_fold"]
    fold = next((f for f in get_folds(config_path) if f.name == fold_name), None)
    if fold is None:
        parser.error(f"Unknown fold {fold_name}")
    for exp_id in [args.exp_id, args.exp_id + "__smoke"]:
        check_experiment_id(exp_id, project_path(cfg["paths"]["experiment_log"]),
                            list(run_directories(cfg, exp_id).values()))
    print("Checking original data SHA-256 and recording source version...", flush=True)
    provenance, revision = source_provenance(cfg), git_state()
    run_one(cfg, config_path, fold, args.exp_id + "__smoke", args.owner, provenance, revision, True)
    gc.collect()
    if not args.smoke_only:
        run_one(cfg, config_path, fold, args.exp_id, args.owner, provenance, revision, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

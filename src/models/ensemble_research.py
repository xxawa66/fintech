"""S006: bounded rank blends, frozen controllers, and one locked 2024 check."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.clean_data import clean_history, valid_quote
from src.data.load_data import KEYS, load_training_data
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame, load_validation_frame
from src.evaluation.research_diagnostics import monthly_metrics
from src.evaluation.turnover_controllers import (holdings_diagnostics, official_top, order_fingerprint,
                                                prepare_controller_input, transform)
from src.evaluation.validation import split_train_valid
from src.models.alpha_research import is_canonical, quarter_metrics, relative, save_csv, write_text
from src.models.baseline import RunLog
from src.models.ensemble import rank_blend
from src.models.ensemble_sources import FrozenFiles, component_frames, daily_correlations, prepare_inputs
from src.models.lightgbm_model import load_model
from src.models.optuna_tuning import ensure_record, provenance, read_json, run_prediction, save_json, study_lock
from src.models.target_research import target_mapping, target_spec
from src.utils.experiments import read_records
from src.utils.project import ROOT, git_state, load_config, sha256, timestamp
from src.utils.research_cache import code_hashes, contained_path, digest
from src.utils.target_cache import TargetFold
from src.utils.tuning_cache import canonical, prepare_folds

PACKAGES = ["numpy", "pandas", "scipy", "lightgbm", "pyarrow", "PyYAML", "matplotlib", "optuna", "psutil"]


def protocol(cfg):
    s = cfg["ensemble_research"]
    if (s["phase"] != "S006" or s["signal_budget"] != 18 or s["configuration_budget"] != 54
            or s["confirm_fit_budget"] != 3 or s["history_top_k"] != 10
            or s["base_weights"] != [.25, .5, .75] or s["history_partner_count"] != 2
            or s["row_order"] != ["trade_date", "ts_code"] or s["initial_state"] != "cold_start"
            or s["folds"] != cfg["optuna"]["folds"] or s["confirm"] != cfg["optuna"]["confirm"]
            or cfg["features"]["expected_count"] != 40
            or s["blend_serialization"] != "twice_daily_average_rank_integer"):
        raise ValueError("S006 protocol differs from the approved finite matrix.")
    return {"schema": 1, "ensemble_research": s, "features": cfg["features"],
            "source_selection": cfg["alpha_research"]["source_selection"],
            "source_artifacts": cfg["alpha_research"]["source_artifacts"],
            "paths": {k: cfg["paths"][k] for k in ["train", "test_x", "models", "metrics", "predictions",
                "figures", "experiment_log", "research_studies", "research_reports", "optuna_cache"]},
            "score_tolerance": cfg["baseline"]["score_tolerance"], "fallback": cfg["baseline"]["fallback_prediction"],
            "selection": "CV mean improvement >1e-6; worst/2023 no worse; tol ties worst/fewer models/ID; one confirmation"}


def assert_identity(cfg, meta, *, originals=False):
    if (digest(protocol(cfg)) != meta["protocol_digest"] or code_hashes() != meta["source"]
            or {p: importlib.metadata.version(p) for p in PACKAGES} != meta["environment"]["packages"]):
        raise ValueError("Source/protocol/packages changed; preserve study and use a new identity.")
    if originals:
        if provenance(cfg) != meta["data"]:
            raise ValueError("Competition data or official files changed.")
        for path, expected in meta["frozen_files"].items():
            if sha256(contained_path(path)) != expected:
                raise ValueError(f"Frozen input changed: {path}")


def signal_matrix(cfg, prepared):
    ids = [c["id"] for c in prepared["components"]]
    result = [{"components": [name], "weights": [1.], "method": "single_original_pred"} for name in ids]
    for partner in ids[1:]:
        result.extend({"components": ["T030", partner], "weights": [w, 1-w], "method": "same_day_percentile_rank"}
                      for w in cfg["ensemble_research"]["base_weights"])
    target = next((c["id"] for c in prepared["components"] if c["kind"] == "target"), None)
    history = prepared["history_partners"]
    if target and history and len({"T030", target, history[0]}) == 3:
        result.append({"components": ["T030", target, history[0]], "weights": [1/3]*3,
                       "method": "same_day_percentile_rank"})
    if len(result) > 18 or len({digest(r) for r in result}) != len(result):
        raise ValueError("Signal matrix exceeds its predeclared budget or has duplicate endpoints.")
    return [{"id": f"U{i:03d}", **s} for i, s in enumerate(result)]


def blend_signal(signal, frames):
    selected = [frames[name] for name in signal["components"]]
    if len(selected) == 1:
        return selected[0].copy(), "original_pred", None
    native = rank_blend(selected, signal["weights"])
    # A monotone integer representation preserves the *actual* native floating
    # order and exact tie groups, including near ties. CSV parsing cannot merge
    # distinct groups separated by at least one integer.
    twice = native.pred.groupby(native.trade_date).rank(method="average").to_numpy()*2
    if not np.array_equal(twice, twice.astype("int32")):
        raise ValueError("Average ranks did not produce exact doubled integers.")
    encoded = native[KEYS].copy()
    encoded["pred"] = twice.astype("int32")
    shape = (native.trade_date.nunique(), 4650)
    if order_fingerprint(native.pred.to_numpy().reshape(shape)) != order_fingerprint(twice.reshape(shape)):
        raise ValueError("Serialization encoding changed native rank_blend order/ties.")
    return encoded, "twice_daily_average_rank_integer", native


def load_derived(cfg, exp_id, spec):
    manifest = contained_path(cfg["paths"]["models"]) / exp_id / "run.json"
    if not manifest.exists():
        return None
    info = read_json(manifest)
    if info["status"] != "passed" or info["spec"] != spec:
        raise ValueError("Interrupted/failed or different derived run is preserved, not overwritten.")
    for path, expected in info["artifact_hashes"].items():
        if sha256(contained_path(path)) != expected:
            raise ValueError("Passed derived artifact changed.")
    ensure_record(cfg, info, manifest)
    return info


def score_derived(cfg, config_path, meta, exp_id, definition, data, truth, signal, controller, result,
                  components, log, *, native=None, signal_encoding="original_pred"):
    spec = {"signal": signal, "controller": controller, "initial_state": "cold_start",
            "row_order": ["trade_date", "ts_code"], "signal_encoding": signal_encoding,
            "output_encoding": result.encoding,
            "components": [{"id": name, "model_params": components[name]["spec"]["model_params"],
                "rounds": components[name]["spec"]["rounds"], "source_exp_id": components[name]["exp_id"],
                "prediction_sha256": components[name]["artifact_hashes"][components[name]["prediction"]]}
                for name in signal["components"]]}
    passed = load_derived(cfg, exp_id, spec)
    if passed:
        return passed
    paths = {k: contained_path(cfg["paths"][k]) / exp_id for k in ["models", "metrics", "predictions"]}
    for path in paths.values():
        if path.exists():
            raise ValueError("Incomplete derived output is preserved; no automatic overwrite.")
        path.mkdir(parents=True)
    manifest = paths["models"] / "run.json"
    original = next(iter(components.values()))
    started = time.perf_counter()
    info = {"exp_id": exp_id, "status": "running", "started_at": timestamp(), "owner": meta["owner"],
            "git": meta.get("confirm_git", meta["git"]), "source_digest": meta["source_digest"],
            "protocol_digest": meta["protocol_digest"], "config_path": relative(config_path), "spec": spec,
            "family": "rank_blend+"+controller["method"], "split": original["split"],
            "training_samples": {name: components[name]["training_samples"] for name in signal["components"]},
            "new_model_fits": 0, "signal": signal["id"], "controller": controller, "labels": original["labels"],
            "component_models": {name: {"model": components[name]["model"],
                "sha256": components[name]["artifact_hashes"][components[name]["model"]],
                "fallback_rows": components[name]["fallback_rows"]} for name in signal["components"]}}
    save_json(manifest, info)
    write_text(paths["models"] / "config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    try:
        prediction = paths["predictions"] / f"valid_{definition['valid'][0]//10000}.csv"
        pred = data.known[KEYS].copy()
        pred["pred"] = result.scores.ravel()
        check_predictions(pred, truth)
        save_csv(prediction, pred)
        scored = load_validation_frame(prediction, contained_path(info["labels"]))
        check_predictions(scored[KEYS+["pred"]], truth)
        if not is_canonical(scored):
            raise ValueError("Saved scoring rows are not canonical.")
        saved = scored.pred.to_numpy().reshape(data.raw.shape)
        if order_fingerprint(saved) != order_fingerprint(result.scores):
            raise ValueError("Saved CSV changed full order or tie groups.")
        metrics = evaluate_frame(scored)
        differences = compare_official(prediction, truth, metrics, cfg["baseline"]["score_tolerance"])
        daily = daily_metrics(scored)
        diagnostics, stocks, lengths, summary, top = holdings_diagnostics(data, saved, result.operations, result.intended_top)
        for name, frame in [("daily_metrics", daily), ("monthly_metrics", monthly_metrics(daily)),
            ("quarterly_metrics", quarter_metrics(daily)), ("holdings_daily", diagnostics),
            ("holdings_stocks", stocks), ("holding_length_histogram", lengths)]:
            save_csv(paths["metrics"] / f"{name}.csv", frame)
        save_json(paths["metrics"] / "metrics.json", metrics)
        save_json(paths["metrics"] / "holdings_summary.json", summary)
        np.savez_compressed(paths["metrics"] / "top_sets.npz", dates=data.dates, stocks=data.stocks, top=top)
        if native is not None:
            native.to_parquet(paths["predictions"] / "native_rank_blend.parquet", index=False)
        info.update(status="passed", finished_at=timestamp(), metrics=metrics, official_differences=differences,
            official_max_difference=max(differences.values()), prediction=relative(prediction),
            validation_rows=len(scored), validation_days=len(daily), duration_seconds=time.perf_counter()-started,
            order_fingerprint=order_fingerprint(saved), holdings_summary=summary,
            notes=f"{meta['study_id']}; {signal['id']}; {controller['method']}; no new fit; complete fold; "
                  "cold_start; canonical rows; native order/ties preserved; saved-file official check")
        files = [p for path in paths.values() for p in path.glob("*") if p.is_file() and p != manifest]
        info["artifact_hashes"] = {relative(p): sha256(p) for p in files}
        save_json(manifest, info)
        ensure_record(cfg, info, manifest)
        log(f"PASS {exp_id}: final={metrics['final_score']:.10f}; IC={metrics['ic_mean']:.6f}; "
            f"excess={metrics['annual_excess']:.6f}; turn={metrics['mean_turnover']:.6f}; official_diff=0; seconds={info['duration_seconds']:.1f}")
        return info
    except Exception:
        info.update(status="failed", failed_at=timestamp(), error=traceback.format_exc())
        save_json(manifest, info)
        raise


def summarize(candidate, runs, fingerprints, alias_of=None):
    values = np.array([r["metrics"]["final_score"] for r in runs])
    return {**candidate, "status": "COMPLETE", "folds": runs, "alias_of": alias_of,
            "order_fingerprints": fingerprints, "cv_mean": float(values.mean()),
            "cv_std": float(values.std()), "cv_worst": float(values.min()),
            "mean_ic": float(np.mean([r["metrics"]["ic_mean"] for r in runs])),
            "mean_excess": float(np.mean([r["metrics"]["annual_excess"] for r in runs])),
            "mean_turnover": float(np.mean([r["metrics"]["mean_turnover"] for r in runs])),
            "scores": {r["split"]["fold"]: r["metrics"]["final_score"] for r in runs}}


def select_candidate(results, baseline, tol):
    for r in results:
        r["passes_cv_gate"] = (r["cv_mean"] > baseline["cv_mean"]+tol and r["cv_worst"] >= baseline["cv_worst"]-tol
                               and r["scores"]["wf2023"] >= baseline["scores"]["wf2023"]-tol)
    pool = [r for r in results if r["passes_cv_gate"]]
    if not pool:
        return None
    highest = max(r["cv_mean"] for r in pool)
    tied = [r for r in pool if highest-r["cv_mean"] <= tol]
    return min(tied, key=lambda r: (-r["cv_worst"], len(r["signal"]["components"]), r["id"]))


def prefix_replay(signal, controller, frames, truth, saved_run):
    stop = int(np.sort(truth.trade_date.unique())[truth.trade_date.nunique()//2-1])
    truncated = {name: frame[frame.trade_date <= stop].reset_index(drop=True) for name, frame in frames.items()}
    pred, _, _ = blend_signal(signal, truncated)
    known = pred.merge(truth[KEYS+["flag_limit_up"]], on=KEYS, validate="one_to_one")
    data = prepare_controller_input(known)
    result = transform(data, controller)
    saved = pd.read_csv(contained_path(saved_run["prediction"]))
    actual = saved[saved.trade_date <= stop].pred.to_numpy().reshape(data.raw.shape)
    if (order_fingerprint(result.scores) != order_fingerprint(actual)
            or not np.array_equal(official_top(data, actual), result.intended_top)):
        raise ValueError("Locked complete scheme failed actual-data prefix replay.")
    return {"fold": saved_run["split"]["fold"], "end": stop, "rows": len(pred), "passed": True}


def audit(cfg, root, meta, results, raw_signals, *, confirmation=None):
    assert_identity(cfg, meta, originals=True)
    used = {run["exp_id"]: run for r in results+raw_signals for run in r["folds"]}
    if confirmation:
        used.update({run["exp_id"]: run for run in confirmation["runs"].values()})
    new_runs = {k: v for k, v in used.items() if k.startswith(meta["study_id"]+"_")}
    for run in used.values():
        for name, expected in run["artifact_hashes"].items():
            if sha256(contained_path(name)) != expected:
                raise ValueError(f"Used reference/output artifact changed: {name}")
    for run in new_runs.values():
        path = contained_path(cfg["paths"]["models"]) / run["exp_id"] / "run.json"
        if read_json(path) != run or run["status"] != "passed":
            raise ValueError("Passed experiment manifest changed.")
        for name, expected in run["artifact_hashes"].items():
            if sha256(contained_path(name)) != expected:
                raise ValueError(f"Passed output changed: {name}")
        ensure_record(cfg, run, path)
    records = read_records(contained_path(cfg["paths"]["experiment_log"]))
    if (records[:len(meta["initial_records"])] != meta["initial_records"]
            or len(records) != len(meta["initial_records"])+len(new_runs)
            or {r["exp_id"] for r in records[len(meta["initial_records"]):]} != set(new_runs)):
        raise ValueError("Shared records do not match the actual passed runs.")
    return {"status": "passed", "time": timestamp(), "study_id": meta["study_id"],
        "implementation_commit": meta["git"]["commit"], "source_digest": meta["source_digest"],
        "protocol_digest": meta["protocol_digest"], "data_provenance": meta["data"],
        "frozen_input_hashes": meta["frozen_files"], "original_files_unchanged": True, "previous_records_unchanged": True,
        "initial_records": len(meta["initial_records"]), "new_records": len(new_runs), "total_records": len(records),
        "new_model_fits": sum(r.get("new_model_fits", 0) for r in new_runs.values()),
        "configured_complete_schemes": len(results), "raw_signals": len(raw_signals),
        "max_official_difference": max((r.get("official_max_difference", 0) for r in new_runs.values()), default=0),
        "max_reload_difference": max((r.get("reload_max_difference", 0) for r in new_runs.values()), default=0),
        "runs": [{"exp_id": r["exp_id"], "manifest": relative(contained_path(cfg["paths"]["models"]) / r["exp_id"] / "run.json"),
                  "manifest_sha256": sha256(contained_path(cfg["paths"]["models"]) / r["exp_id"] / "run.json"),
                  "artifact_hashes": r["artifact_hashes"]} for r in new_runs.values()]}


def screen(cfg, config_path, root, meta, frames, truths, log):
    from src.evaluation.ensemble_handoff import publish
    signals, controllers = signal_matrix(cfg, meta), meta["controllers"]
    if len(controllers) > 3 or len(signals)*len(controllers) > cfg["ensemble_research"]["configuration_budget"]:
        raise ValueError("The complete scheme matrix exceeds the approved budget.")
    components = {c["id"]: c for c in meta["components"]}
    results, raws, registry = [], [], {}
    for signal in signals:
        assert_identity(cfg, meta)
        log(f"signal {signal['id']} ({len(raws)+1}/{len(signals)}): {signal['components']} weights={signal['weights']}")
        inputs, raw_runs, native_outputs, encodings = {}, [], {}, {}
        for definition in cfg["ensemble_research"]["folds"]:
            fold = definition["name"]
            source_runs = {name: components[name]["folds"][fold] for name in signal["components"]}
            pred, encoding, native = blend_signal(signal, {name: frames[name][fold] for name in signal["components"]})
            known = pred.merge(truths[fold][KEYS+["flag_limit_up"]], on=KEYS, validate="one_to_one")
            data = prepare_controller_input(known)
            inputs[fold], native_outputs[fold], encodings[fold] = data, native, encoding
            if len(signal["components"]) == 1:
                raw_run = next(iter(source_runs.values()))
            else:
                raw_run = score_derived(cfg, config_path, meta, f"{meta['study_id']}_{signal['id']}_{fold}_raw", definition,
                    data, truths[fold], signal, {"method": "raw"}, transform(data, {"method": "raw"}), source_runs,
                    log, native=native, signal_encoding=encoding)
            raw_runs.append(raw_run)
        raws.append(summarize({"id": signal["id"], "signal": signal, "controller": {"method": "raw"}}, raw_runs,
                              [order_fingerprint(inputs[f["name"]].raw) for f in cfg["ensemble_research"]["folds"]]))
        for ctrl in controllers:
            candidate = {"id": f"F{len(results):03d}", "signal": signal, "controller_id": ctrl["id"], "controller": ctrl["spec"]}
            outputs = [transform(inputs[f["name"]], ctrl["spec"]) for f in cfg["ensemble_research"]["folds"]]
            fingerprints = [order_fingerprint(r.scores) for r in outputs]
            equivalent = registry.get(tuple(fingerprints))
            if equivalent:
                completed = summarize(candidate, equivalent["folds"], fingerprints, equivalent["id"])
            else:
                runs = []
                for definition, result in zip(cfg["ensemble_research"]["folds"], outputs):
                    fold = definition["name"]
                    if signal["components"] == ["T030"]:
                        run = ctrl["cv_runs"][fold]
                        saved = pd.read_csv(contained_path(run["prediction"])).pred.to_numpy().reshape(inputs[fold].raw.shape)
                        if (order_fingerprint(saved) != order_fingerprint(result.scores)
                                or not np.array_equal(official_top(inputs[fold], saved), result.intended_top)):
                            raise ValueError("T030 controller does not reproduce frozen S004.")
                    else:
                        sources = {name: components[name]["folds"][fold] for name in signal["components"]}
                        run = score_derived(cfg, config_path, meta, f"{meta['study_id']}_{candidate['id']}_{fold}", definition,
                            inputs[fold], truths[fold], signal, ctrl["spec"], result, sources, log,
                            signal_encoding=encodings[fold])
                    runs.append(run)
                completed = summarize(candidate, runs, fingerprints)
                registry[tuple(fingerprints)] = completed
            save_json(root / "candidates" / f"{candidate['id']}.json", completed)
            results.append(completed)
            meta.update(completed_candidates=len(results), last_candidate=candidate["id"], updated_at=timestamp())
            save_json(root / "study.json", meta)
            log(f"COMPLETE {candidate['id']}: mean={completed['cv_mean']:.10f}; worst={completed['cv_worst']:.10f}")
        del inputs, native_outputs, outputs
        gc.collect()
    baseline = results[0]
    if abs(baseline["cv_mean"]-meta["baseline"]["cv_mean"]) > 1e-10:
        raise ValueError("S003 full-scheme baseline no longer reproduces.")
    winner = select_candidate(results, baseline, cfg["ensemble_research"]["selection_tolerance"])
    for completed in results:
        save_json(root / "candidates" / f"{completed['id']}.json", completed)
    chosen = None
    replays = []
    if winner:
        chosen = {k: v for k, v in winner.items() if k != "folds"}
        chosen["components"] = [components[name] for name in winner["signal"]["components"]]
        chosen["cv_runs"] = {r["split"]["fold"]: r for r in winner["folds"]}
        for definition, run in zip(cfg["ensemble_research"]["folds"], winner["folds"]):
            fold = definition["name"]
            replays.append(prefix_replay(winner["signal"], winner["controller"],
                {name: frames[name][fold] for name in winner["signal"]["components"]}, truths[fold], run))
    selection = {"schema": 1, "study_id": meta["study_id"], "created_at": timestamp(), "git": meta["git"],
        "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
        "source_selection_digest": meta["source_selection_digest"], "controller_selection_digest": meta["controller_selection_digest"],
        "target_selection_digest": meta["target_selection_digest"], "winner": chosen,
        "baseline": {k: v for k, v in baseline.items() if k != "folds"},
        "highest_mean_unrestricted": {k: v for k, v in max(results, key=lambda r: r["cv_mean"]).items() if k != "folds"},
        "history_partners": meta["history_partners"], "prefix_replays": replays,
        "row_order": ["trade_date", "ts_code"], "initial_state": "cold_start", "blend_serialization": "twice_daily_average_rank_integer",
        "confirm_definition": cfg["ensemble_research"]["confirm"], "confirm_fit_budget": 3,
        "confirm_layers": ["T030_raw", "T030_qstar", "component_raw", "new_signal_raw", "new_signal_qstar", "T030_selected_controller", "new_complete"],
        "confirm_threshold": cfg["ensemble_research"]["confirm_threshold"], "rule": meta["protocol"]["selection"],
        "screen_new_model_fits": 0, "formal_adoption": False, "evaluated_years": [2021, 2022, 2023]}
    selection["selection_digest"] = digest(selection)
    save_json(root / "selection.json", selection)
    save_json(root / "raw_signals.json", raws)
    audited = audit(cfg, root, meta, results, raws)
    meta.update(status="screen_selected" if winner else "complete_no_candidate", screen_finished_at=timestamp(),
                selection_digest=selection["selection_digest"], raw_signal_count=len(raws), unique_schemes=len(registry))
    save_json(root / "study.json", meta)
    publish(cfg, root, meta, results, raws, selection, audited)
    log(f"S006 screen complete: signals={len(raws)}; schemes={len(results)}; unique={len(registry)}; "
        f"winner={winner['id'] if winner else None}; best={max(r['cv_mean'] for r in results):.10f}; no new fits")


def confirmation_data(cfg, root, meta, reference, log):
    definitions = [cfg["ensemble_research"]["confirm"]]
    data_folds, cache = prepare_folds(cfg, definitions, meta["data"], root, log)
    data = data_folds[0]
    source_study = read_json(contained_path(cfg["target_research"]["source_study"]))
    original_cache = source_study["confirm_cache"]
    if any(cache[k] != original_cache[k] for k in ["identity", "path", "parquet_sha256", "cleaning"]):
        raise ValueError("Confirmation X differs from verified S003 cache.")
    raw = load_training_data(contained_path(cfg["paths"]["train"]))
    raw = raw[raw.trade_date.between(data.fold.train_start, data.fold.valid_end)]
    raw, _ = clean_history(raw)
    train, valid, split = split_train_valid(raw, data.fold)
    mask = np.isfinite(train.y_ret_1d.to_numpy()) & valid_quote(train).to_numpy()
    selected = train.loc[mask]
    if (not selected.index.equals(data.x_train.index) or not np.array_equal(selected.y_ret_1d.to_numpy(), data.y_train)
            or split != reference["split"] or data.supervision != reference["training_samples"]
            or selected.trade_date.max() >= split["last_train_day"]
            or sha256(data.labels_path) != reference["artifact_hashes"][reference["labels"]]):
        raise ValueError("Confirmation supervision/boundary/labels differ from frozen T030.")
    training = selected[KEYS+["y_ret_1d"]].reset_index(drop=True)
    training["ts_code"] = training.ts_code.astype(str)
    key_digest = hashlib.sha256(pd.util.hash_pandas_object(training[KEYS], index=False).to_numpy(dtype="<u8").tobytes()).hexdigest()
    del raw, train, valid, selected
    gc.collect()
    meta["confirm_cache"] = cache
    save_json(root / "study.json", meta)
    return TargetFold(data, training, key_digest)


def fit_confirm(cfg, config_path, root, meta, fold, component, log):
    spec = {"model_params": component["model_spec"]["model_params"], "rounds": component["model_spec"]["rounds"],
            "component_id": component["id"], "target_transform": target_spec(component["target"]),
            "training_keys_digest": fold.key_digest, "source_cv_model": component["id"]}
    y, mapping = target_mapping(root, fold, target_spec(component["target"]), log)
    manifest_path = root / "targets" / fold.data.fold.name / digest(mapping["identity"]) / "target.json"
    mapping.update(manifest=relative(manifest_path), manifest_sha256=sha256(manifest_path))
    spec.update(target_mapping=mapping["mapping"], target_mapping_sha256=mapping["artifact_hashes"][mapping["mapping"]])
    exp_id = f"{meta['study_id']}_confirm2024_{component['id']}_raw"
    manifest = contained_path(cfg["paths"]["models"]) / exp_id / "run.json"
    if manifest.exists() and read_json(manifest)["status"] != "passed":
        raise ValueError("Interrupted/failed confirmation fit is preserved; no automatic extra fit.")
    if not manifest.exists() and any((contained_path(cfg["paths"][k])/exp_id).exists() for k in ["models", "metrics", "predictions"]):
        raise ValueError("Incomplete confirmation directories are preserved.")
    info = run_prediction(cfg, config_path, root, meta, replace(fold.data, y_train=y), exp_id, spec, log)
    restored = load_model(contained_path(info["model"]))
    native = np.full(len(fold.data.keys), cfg["baseline"]["fallback_prediction"], dtype="float64")
    native[fold.data.eligible] = restored.predict(fold.data.x_valid, num_threads=8)
    native = canonical(fold.data.keys.assign(pred=native)).pred.to_numpy()
    actual = pd.read_csv(contained_path(info["prediction"])).pred.to_numpy()
    shape = (fold.data.truth.trade_date.nunique(), 4650)
    if np.max(abs(native-actual)) > 1e-12 or order_fingerprint(native.reshape(shape)) != order_fingerprint(actual.reshape(shape)):
        raise ValueError("Saved confirmation model order/ties changed.")
    info.update(new_model_fits=1, target_mapping=mapping, csv_order_and_ties_preserved=True,
                csv_max_difference=float(np.max(abs(native-actual))))
    save_json(manifest, info)
    ensure_record(cfg, info, manifest)
    del native, actual, restored, y
    gc.collect()
    return info


def cached_confirm(cfg, root, meta, fold, component, checks, log):
    """Reuse a known S002 native model only after exact parameter/source checks.

    T030 has its own S003 frozen reference. Other models are never assumed to
    have a 2024 artifact merely because they have a CV artifact.
    """
    if component["kind"] != "historical":
        return None
    index_path = contained_path("docs/model_research_S002_artifacts.json")
    index = read_json(index_path)
    for entry in index["predictions"]:
        if not entry["exp_id"].startswith("S002_confirm_L"):
            continue
        old = read_json(contained_path(entry["files"]["manifest"]["path"]))
        block = old["model"].get("reference", {}).get("model", old["model"])
        if (block.get("params") != component["model_spec"]["model_params"]
                or block.get("num_boost_round") != component["model_spec"]["rounds"]):
            continue
        checks.check(index_path)
        for file in entry["files"].values():
            checks.check(file["path"], file["sha256"])
        if "reference" in old["model"]:
            model_reference = old["model"]["reference"]
            checks.check(model_reference["manifest"], model_reference["manifest_sha256"])
            model_path = contained_path(cfg["paths"]["models"]) / model_reference["exp_id"] / "model.txt"
            checks.check(model_path, old["model"]["model_sha256"])
        else:
            fitted = next(r for r in index["full_fits"] if r["exp_id"] == old["exp_id"])
            model_path = contained_path(fitted["model_path"])
            checks.check(model_path, fitted["sha256"])
        if (old["status"] != "passed" or old["training_samples"] != fold.data.supervision
                or old["split"]["train_window"] != fold.data.split["train_window"]
                or old["split"]["valid_window"] != fold.data.split["valid_window"]):
            raise ValueError("Matching cached model has inconsistent supervision or windows.")
        prediction = contained_path(entry["files"]["prediction"]["path"])
        pred = pd.read_csv(prediction, dtype={"ts_code": "str", "trade_date": "int64"})
        check_predictions(pred, fold.data.truth)
        if not is_canonical(pred):
            log(f"cached {old['exp_id']} is not canonical; it is not a verified S006 component")
            continue
        restored = load_model(model_path)
        native = np.full(len(fold.data.keys), cfg["baseline"]["fallback_prediction"], dtype="float64")
        native[fold.data.eligible] = restored.predict(fold.data.x_valid, num_threads=8)
        native = canonical(fold.data.keys.assign(pred=native)).pred.to_numpy()
        saved = pred.pred.to_numpy()
        shape = (fold.data.truth.trade_date.nunique(), 4650)
        if (np.max(abs(native-saved)) > 1e-12
                or order_fingerprint(native.reshape(shape)) != order_fingerprint(saved.reshape(shape))):
            log(f"cached {old['exp_id']} does not preserve native prediction order; no reuse")
            continue
        scored = load_validation_frame(prediction, fold.data.labels_path)
        metrics = evaluate_frame(scored)
        if any(abs(metrics[k]-old["metrics"][k]) > 1e-10 for k in OFFICIAL_METRICS):
            raise ValueError("Cached component does not reproduce its frozen official metrics.")
        differences = compare_official(prediction, fold.data.truth, metrics, cfg["baseline"]["score_tolerance"])
        directory = root / "references" / old["exp_id"]
        save_csv(directory / "daily_metrics.csv", daily_metrics(scored))
        info = {"exp_id": old["exp_id"], "status": "passed", "spec": component["model_spec"],
            "split": fold.data.split, "training_samples": fold.data.supervision, "prediction": relative(prediction),
            "model": relative(model_path), "labels": relative(fold.data.labels_path), "metrics": metrics,
            "validation_rows": len(pred), "validation_days": shape[0], "diagnostics": relative(directory),
            "new_model_fits": 0, "fallback_rows": int((~fold.data.eligible).sum()),
            "reload_max_difference": float(np.max(abs(native-saved))), "actual_iterations": block["actual_iterations"],
            "official_max_difference": max(differences.values()),
            "source_manifest": entry["files"]["manifest"], "artifact_hashes": {
                relative(prediction): sha256(prediction), relative(model_path): sha256(model_path),
                relative(fold.data.labels_path): sha256(fold.data.labels_path),
                relative(directory / "daily_metrics.csv"): sha256(directory / "daily_metrics.csv")}}
        log(f"reuse verified cached 2024 model {old['exp_id']} for {component['id']}; no new fit")
        return info
    return None


def confirm(cfg, config_path, root, meta, log):
    from src.evaluation.ensemble_handoff import publish
    selection_path = contained_path(cfg["paths"]["research_reports"]) / f"alpha_research_{meta['study_id']}_selection.json"
    selection = read_json(selection_path)
    if (digest({k: v for k, v in selection.items() if k != "selection_digest"}) != meta["selection_digest"]
            or selection["winner"] is None or git_state()["dirty"]):
        raise ValueError("Confirmation needs the exact selected candidate committed on clean main.")
    origin = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    if git_state()["commit"] != origin:
        raise ValueError("Confirmation must start from the pushed main commit.")
    committed = subprocess.check_output(["git", "show", f"origin/main:{relative(selection_path)}"], cwd=ROOT, encoding="utf-8")
    if read_json(root / "selection.json") != selection or __import__("json").loads(committed) != selection:
        raise ValueError("The exact selection must already be pushed to origin/main.")
    assert_identity(cfg, meta, originals=True)
    meta["confirm_git"] = git_state()
    source_index = read_json(contained_path(cfg["alpha_research"]["source_artifacts"]))
    file_checks = FrozenFiles(meta["frozen_files"])
    references = {}
    for layer, exp_id in [("T030_raw", "S003_confirm2024_raw"), ("T030_qstar", "S003_confirm2024_band")]:
        saved = next(r for r in source_index["runs"] if r["exp_id"] == exp_id)
        file_checks.check(saved["manifest"], saved["manifest_sha256"])
        run = read_json(contained_path(saved["manifest"]))
        if run["status"] != "passed":
            raise ValueError("Frozen confirmation reference did not pass.")
        for path, expected in run["artifact_hashes"].items():
            file_checks.check(path, expected)
        references[layer] = run
    meta["frozen_files"] = file_checks.files
    save_json(root / "study.json", meta)
    selected = selection["winner"]
    missing = [c for c in selected["components"] if c["id"] != "T030"]
    if len(missing) > 3:
        raise ValueError("Locked scheme exceeds the three-fit confirmation budget.")
    source_runs = {"T030": references["T030_raw"]}
    if missing:
        fold = confirmation_data(cfg, root, meta, references["T030_raw"], log)
        for component in missing:
            assert_identity(cfg, meta)
            reused = cached_confirm(cfg, root, meta, fold, component, file_checks, log)
            source_runs[component["id"]] = reused or fit_confirm(cfg, config_path, root, meta, fold, component, log)
        meta["frozen_files"] = file_checks.files
        save_json(root / "study.json", meta)
        truth = fold.data.truth.astype({"ts_code": "str", "trade_date": "int64"})
    else:
        truth = pd.read_csv(contained_path(references["T030_raw"]["labels"]), dtype={"ts_code": "str", "trade_date": "int64"})
    frames = {name: pd.read_csv(contained_path(run["prediction"]), dtype={"ts_code": "str", "trade_date": "int64"}) for name, run in source_runs.items()}
    for frame in frames.values():
        check_predictions(frame, truth)
        if not is_canonical(frame):
            raise ValueError("Confirmation component rows are not canonical.")
    definition = cfg["ensemble_research"]["confirm"]
    signal = selected["signal"]
    pred, encoding, native = blend_signal(signal, frames)
    data = prepare_controller_input(pred.merge(truth[KEYS+["flag_limit_up"]], on=KEYS, validate="one_to_one"))
    outputs = {**references, **{f"component_raw_{name}": run for name, run in source_runs.items() if name != "T030"}}
    registry = {}
    if len(signal["components"]) == 1:
        outputs["new_signal_raw"] = source_runs[signal["components"][0]]
    else:
        outputs["new_signal_raw"] = score_derived(cfg, config_path, meta, f"{meta['study_id']}_confirm2024_signal_raw", definition,
            data, truth, signal, {"method": "raw"}, transform(data, {"method": "raw"}), source_runs, log,
            native=native, signal_encoding=encoding)
    cases = [("new_signal_qstar", signal, meta["controllers"][0]["spec"], data, encoding),
             ("new_complete", signal, selected["controller"], data, encoding)]
    base_signal = {"id": "U000", "components": ["T030"], "weights": [1.], "method": "single_original_pred"}
    base_known = frames["T030"].merge(truth[KEYS+["flag_limit_up"]], on=KEYS, validate="one_to_one")
    base_data = prepare_controller_input(base_known)
    cases.append(("T030_selected_controller", base_signal, selected["controller"], base_data, "original_pred"))
    for label, case_signal, controller, case_data, case_encoding in cases:
        if case_signal == base_signal and controller == meta["controllers"][0]["spec"]:
            outputs[label] = references["T030_qstar"]
            continue
        transformed = transform(case_data, controller)
        signature = order_fingerprint(transformed.scores)
        if signature in registry:
            outputs[label] = registry[signature]
            continue
        outputs[label] = score_derived(cfg, config_path, meta, f"{meta['study_id']}_confirm2024_{label}", definition,
            case_data, truth, case_signal, controller, transformed, source_runs, log, signal_encoding=case_encoding)
        registry[signature] = outputs[label]
    final = outputs["new_complete"]["metrics"]["final_score"]
    confirmation = {"status": "passed", "time": timestamp(), "lock_commit": meta["confirm_git"]["commit"],
        "selection_digest": selection["selection_digest"], "runs": outputs,
        "new_model_fits": sum(r.get("new_model_fits", 0) for r in source_runs.values()),
        "final_score": final, "threshold": selection["confirm_threshold"],
        "passes_confirmation_gate": final >= selection["confirm_threshold"]-cfg["ensemble_research"]["selection_tolerance"],
        "recommendation": "candidate_for_B_review" if final >= selection["confirm_threshold"]-cfg["ensemble_research"]["selection_tolerance"] else "retain_S003",
        "no_post_confirmation_search": True, "formal_adoption": False}
    confirmation["prefix_replay"] = prefix_replay(signal, selected["controller"], frames, truth, outputs["new_complete"])
    save_json(root / "confirmation.json", confirmation)
    results = [read_json(p) for p in sorted((root / "candidates").glob("*.json"))]
    raws = read_json(root / "raw_signals.json")
    audited = audit(cfg, root, meta, results, raws, confirmation=confirmation)
    if audited["new_model_fits"] > 3:
        raise ValueError("Actual fit count exceeded the fixed confirmation budget.")
    meta.update(status="complete", completed_at=timestamp(), new_model_fits=audited["new_model_fits"],
                confirmation_score=final, recommendation=confirmation["recommendation"])
    save_json(root / "study.json", meta)
    publish(cfg, root, meta, results, raws, selection, audited, confirmation=confirmation)
    log(f"S006 complete: 2024={final:.10f}; new_fits={audited['new_model_fits']}; {confirmation['recommendation']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study-id", default="S006")
    parser.add_argument("--phase", choices=["screen", "confirm"], default="screen")
    parser.add_argument("--owner", default="A")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"S006(?:_[A-Za-z0-9_-]+)?", args.study_id):
        raise ValueError("Use S006 or an independent S006_* identity.")
    cfg, config_path = load_config(args.config)
    fixed_protocol = protocol(cfg)
    root = contained_path(cfg["paths"]["research_studies"]) / args.study_id
    revision = git_state()
    if revision["branch"] != "main":
        raise ValueError("The user requires work on main.")
    if args.phase == "screen" and not args.resume:
        origin = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
        if revision["dirty"] or revision["commit"] != origin or root.exists():
            raise ValueError("Commit/push fixed protocol on clean main and use a new identity.")
        if shutil.disk_usage(ROOT).free < 15*2**30:
            raise ValueError("At least 15 GiB is needed for full forecasts and diagnostics.")
        root.mkdir(parents=True)
    elif not (root / "study.json").exists():
        raise ValueError("No study metadata exists to continue.")
    log = RunLog(root / "run.log")
    with study_lock(root):
        try:
            if args.phase == "screen":
                prepared, frames, truths = prepare_inputs(cfg, root, log)
                if args.resume:
                    meta = read_json(root / "study.json")
                    if meta["status"] in {"screen_selected", "complete", "complete_no_candidate"}:
                        raise ValueError("Completed screen cannot be rerun or reselected.")
                    assert_identity(cfg, meta)
                    if any(prepared[k] != meta[k] for k in prepared):
                        raise ValueError("Frozen input/partner identity changed during resume.")
                else:
                    meta = {"schema": 1, "study_id": args.study_id, "owner": args.owner, "status": "screen_running",
                        "started_at": timestamp(), "git": revision, "source": code_hashes(),
                        "protocol": fixed_protocol, "protocol_digest": digest(fixed_protocol),
                        "initial_records": read_records(contained_path(cfg["paths"]["experiment_log"])),
                        "environment": {"python": sys.version, "platform": platform.platform(),
                            "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}, **prepared}
                    meta["source_digest"] = digest(meta["source"])
                    save_json(root / "study.json", meta)
                    write_text(root / "config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
                screen(cfg, config_path, root, meta, frames, truths, log)
            else:
                meta = read_json(root / "study.json")
                if meta["status"] != "screen_selected":
                    raise ValueError("Confirmation requires an unconfirmed eligible selection.")
                confirm(cfg, config_path, root, meta, log)
        except Exception:
            if (root / "study.json").exists():
                saved = read_json(root / "study.json")
                if saved["status"] not in {"complete", "complete_no_candidate"}:
                    saved.update(status="interrupted", error=traceback.format_exc(), interrupted_at=timestamp())
                    save_json(root / "study.json", saved)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

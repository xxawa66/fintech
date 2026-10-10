"""S009 complete saved-prediction diagnostics; no model fitting or optimisation."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import re
import shutil
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from src.data.load_data import KEYS
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame
from src.evaluation.paired_block_bootstrap import (block_indices, comparisons, interval, paired_draws, scalar_interval)
from src.evaluation.research_diagnostics import monthly_metrics
from src.evaluation.top_tail_sources import FrozenInputs, prepare_context, prepare_sources
from src.evaluation.top_tail_tables import period_tables, top_matrix, trace_tables
from src.evaluation.turnover import band_scores
from src.evaluation.turnover_controllers import order_fingerprint
from src.features.build_features import feature_names
from src.models.alpha_research import is_canonical, relative, save_csv, write_text
from src.models.baseline import RunLog
from src.models.lightgbm_model import load_model
from src.models.optuna_tuning import provenance, read_json, save_json, study_lock
from src.utils.experiments import read_records
from src.utils.project import ROOT, git_state, load_config, sha256, timestamp
from src.utils.research_cache import code_hashes, contained_path, digest

PACKAGES = ["numpy", "pandas", "scipy", "lightgbm", "pyarrow", "PyYAML", "matplotlib"]


def protocol(cfg):
    s = cfg["top_tail_diagnostics"]
    if (s["phase"] != "S009" or s["model_fit_budget"] != 0 or s["comparison_units"] != 38
            or s["keep_q"] != .0022778298112255263 or s["models"] != ["T030", "Y1", "Y4", "Market", "T033"]
            or s["bin_edges"] != [0., .05, .10, .15, .20, .30, .50, .70, 1.]
            or s["bootstrap"] != {"block_length": 20, "repetitions": 1000, "seed": 42}
            or s["row_orders"] != ["canonical", "reverse_code", "permuted_seed42"]):
        raise ValueError("S009 protocol differs from the accepted finite diagnostic plan.")
    return {"schema": 1, "diagnostics": s, "features": cfg["features"],
            "folds": [*cfg["optuna"]["folds"], cfg["optuna"]["confirm"]],
            "paths": {k: cfg["paths"][k] for k in ["train", "test_x", "research_studies", "research_reports",
                "research_handoff_inputs", "experiment_log", "figures"]}}


def read_prediction(path, context, *, precision=None):
    pred = pd.read_csv(contained_path(path), dtype={"ts_code": "str", "trade_date": "int32", "pred": "float64"},
                       float_precision=precision)
    check_predictions(pred, context[KEYS])
    if not is_canonical(pred) or not pred[KEYS].equals(context[KEYS]):
        raise ValueError("Source rows are not canonical and completely aligned.")
    return pred


def native_prediction(case, context, cfg, preparation, root, log):
    run = case["raw"]["run"]
    if "model" not in run:
        return None, {"status": "unavailable_model_not_saved"}
    year = case["year"]
    columns = feature_names(cfg["features"])
    directory = contained_path(preparation["feature_cache"]["path"])
    features = pd.read_parquet(directory / "features.parquet", columns=KEYS+columns,
                               filters=[("trade_date", ">=", year*10000+101), ("trade_date", "<=", year*10000+1231)])
    features["ts_code"] = features.ts_code.astype(str)
    features["trade_date"] = features.trade_date.astype("int32")
    features = features.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    if not features[KEYS].equals(context[KEYS]):
        raise ValueError("Native model X rows differ from scored keys.")
    if features[columns].dtypes.ne("float32").any():
        raise ValueError("Native inference X must keep the frozen float32 dtype.")
    model = load_model(contained_path(run["model"]))
    values = np.zeros(len(context), dtype="float64")
    good = context.quote_valid.to_numpy()
    values[good] = model.predict(features.loc[good, columns], num_threads=8)
    if not np.isfinite(values).all():
        raise ValueError("Reloaded native prediction is not finite.")
    path = root / "native" / case["model"] / f"{case['fold']}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    log(f"reload native {case['model']} {case['fold']}; no fitting")
    del model, features
    gc.collect()
    return values, {"status": "available", "model": run["model"], "prediction": relative(path), "sha256": sha256(path)}


def order_sensitivity(pred, scored, source_path, context, directory, baseline):
    n = context.ts_code.nunique()
    shape = (len(pred)//n, n)
    records, hashes = [], {}
    original_lines = contained_path(source_path).read_bytes().splitlines(keepends=True)
    if len(original_lines) != len(pred)+1:
        raise ValueError("Prediction line records differ from parsed rows.")
    offsets = np.arange(shape[0])[:, None]*n
    orders = {"reverse_code": (offsets+np.arange(n)[::-1]).ravel(),
              "permuted_seed42": (offsets+np.asarray([np.random.default_rng(42+int(d)).permutation(n)
                    for d in context.trade_date.drop_duplicates()])).ravel()}
    for name, indices in orders.items():
        path = directory / f"sensitivity_{name}.csv"
        # Move the exact source bytes; reserialization could otherwise change ties.
        with path.open("wb") as file:
            file.write(original_lines[0])
            file.writelines(original_lines[int(i)+1] for i in indices)
        reordered = scored.iloc[indices].reset_index(drop=True)
        metrics = evaluate_frame(reordered)
        differences = compare_official(path, reordered[KEYS+["y_ret_1d", "flag_limit_up"]], metrics, 1e-10)
        records.append({"order": name, **metrics, "final_delta": metrics["final_score"]-baseline["final_score"],
                        "ic_delta": metrics["ic_mean"]-baseline["ic_mean"],
                        "excess_delta": metrics["annual_excess"]-baseline["annual_excess"],
                        "turnover_delta": metrics["mean_turnover"]-baseline["mean_turnover"],
                        "official_max_difference": max(differences.values()), "prediction": relative(path)})
        hashes[relative(path)] = sha256(path)
    return pd.DataFrame(records), hashes


def analyze_pair(cfg, root, case, context, preparation, log):
    settings = cfg["top_tail_diagnostics"]
    pred = read_prediction(case["raw"]["prediction"], context)
    band = read_prediction(case["band"]["prediction"], context)
    n = context.ts_code.nunique()
    shape = (context.trade_date.nunique(), n)
    raw = pred.pred.to_numpy().reshape(shape)
    eligible = context.flag_limit_up.to_numpy().reshape(shape) == 0
    replay = band_scores(pred.assign(flag_limit_up=context.flag_limit_up), settings["keep_q"]).to_numpy().reshape(shape)
    saved_band = band.pred.to_numpy().reshape(shape)
    replay_max = float(np.max(abs(replay-saved_band)))
    replay_order_match = order_fingerprint(replay) == order_fingerprint(saved_band)
    replay_top_match = np.array_equal(top_matrix(replay, eligible), top_matrix(saved_band, eligible))
    if replay_max > 1e-12 or not replay_top_match:
        raise ValueError("Saved raw to q* replay changes values or actual Top sets.")
    stop = shape[0]//2
    prefix_input = pred.iloc[:stop*n].assign(flag_limit_up=context.flag_limit_up.iloc[:stop*n].to_numpy())
    prefix = band_scores(prefix_input, settings["keep_q"]).to_numpy().reshape((stop, n))
    if not np.array_equal(prefix, replay[:stop]):
        raise ValueError("Actual q* prefix replay changed the earlier scores.")
    native, native_info = native_prediction(case, context, cfg, preparation, root, log)
    if native is not None:
        native_difference = float(np.max(abs(native-pred.pred.to_numpy())))
        native_info["max_value_difference"] = native_difference
        native_info["order_and_ties_match"] = order_fingerprint(native.reshape(shape)) == order_fingerprint(raw)
        if native_difference > 1e-12:
            raise ValueError("Reloaded native model differs from saved raw.")
    records = []
    for layer, source, frame in [("raw", case["raw"], pred), ("band", case["band"], band)]:
        directory = root / "cells" / case["model"] / case["fold"] / layer
        directory.mkdir(parents=True, exist_ok=True)
        scored = frame.merge(context[KEYS+["y_ret_1d", "flag_limit_up"]], on=KEYS, validate="one_to_one")
        metrics = evaluate_frame(scored)
        official = compare_official(contained_path(source["prediction"]), context, metrics, 1e-10)
        source_difference = max(abs(metrics[k]-source["metrics"][k]) for k in OFFICIAL_METRICS)
        if source_difference > 1e-10:
            raise ValueError("Saved-file official metrics differ from frozen summary.")
        daily = daily_metrics(scored)
        monthly = monthly_metrics(daily)
        save_csv(directory / "daily_metrics.csv", daily)
        save_csv(directory / "monthly_metrics.csv", monthly)
        values = frame.pred.to_numpy().reshape(shape)
        tables, masks, spells = trace_tables(context, raw, values, daily, settings)
        for key, table in tables.items():
            save_csv(directory / f"{key}.csv", table)
        np.savez_compressed(directory / "actual_top_sets.npz", **masks)
        identifiers = {"model": case["model"], "fold": case["fold"], "year": case["year"], "layer": layer}
        for key, table in period_tables(tables, identifiers).items():
            save_csv(directory / f"{key}_periods.csv", table)
        rounded = read_prediction(source["prediction"], context, precision="round_trip").pred.to_numpy().reshape(shape)
        parser = {"max_value_difference": float(np.max(abs(rounded-values))),
                  "order_and_ties_match": order_fingerprint(rounded) == order_fingerprint(values),
                  "actual_turn_top_match": np.array_equal(top_matrix(rounded, eligible), masks["turn_top"])}
        sensitivity, additional_hashes = order_sensitivity(frame, scored, source["prediction"], context, directory, metrics)
        save_csv(directory / "order_sensitivity.csv", sensitivity)
        boundary = tables["boundary_daily"]
        boundary_ci = scalar_interval(boundary.gap_5_10_minus_10_15,
            **{**settings["bootstrap"], "seed": settings["bootstrap"]["seed"]+case["year"]})
        info = {**identifiers, "status": "passed", "source_prediction": source["prediction"],
            "source_manifest": source["manifest"], "rows": len(frame), "days": shape[0], "metrics": metrics,
            "official_differences": official, "official_max_difference": max(official.values()),
            "source_max_difference": source_difference, "parser_comparison": parser,
            "native": native_info if layer == "raw" else {"status": "not_applicable_encoded_scores"},
            "qstar_replay": {"max_value_difference": replay_max, "order_and_ties_match": replay_order_match,
                             "actual_turn_top_match": replay_top_match, "prefix_passed": True, "prefix_days": stop},
            "spells": spells, "boundary_ci": {k: (float(v) if np.isfinite(v) else None) for k, v in boundary_ci.items()},
            "new_model_fits": 0, "new_shared_records": 0, "finished_at": timestamp()}
        paths = sorted(p for p in directory.iterdir() if p.is_file())
        info["artifact_hashes"] = {relative(p): sha256(p) for p in paths}
        info["artifact_hashes"].update(additional_hashes)
        if layer == "raw" and native_info["status"] == "available":
            info["artifact_hashes"][native_info["prediction"]] = native_info["sha256"]
        save_json(directory / "cell.json", info)
        records.append(info)
        log(f"PASS {case['model']} {case['fold']} {layer}: score={metrics['final_score']:.10f}, "
            f"official Δ={info['official_max_difference']:.2e}; no new fit/record")
    return records


def collect(root, cells, context_paths, settings):
    datasets = {name: [] for name in ["official", "monthly", "bins", "cumulative", "cohorts", "subgroups",
                                        "boundary", "ties", "holdings", "order_sensitivity", "daily", "spells"]}
    for cell in cells:
        ids = {k: cell[k] for k in ["model", "fold", "year", "layer"]}
        directory = root / "cells" / cell["model"] / cell["fold"] / cell["layer"]
        datasets["official"].append(pd.DataFrame([{**ids, **cell["metrics"],
            "rows": cell["rows"], "days": cell["days"], "official_max_difference": cell["official_max_difference"],
            "source_max_difference": cell["source_max_difference"], "native_status": cell["native"]["status"],
            "native_max_difference": cell["native"].get("max_value_difference"),
            "native_order_match": cell["native"].get("order_and_ties_match"),
            "parser_order_match": cell["parser_comparison"]["order_and_ties_match"],
            "band_replay_order_match": cell["qstar_replay"]["order_and_ties_match"]}]))
        for name, filename in [("monthly", "monthly_metrics.csv"), ("bins", "bins_periods.csv"),
            ("cumulative", "cumulative_periods.csv"), ("cohorts", "cohorts_periods.csv"), ("subgroups", "subgroups_periods.csv"),
            ("boundary", "boundary_daily.csv"), ("ties", "ties_daily.csv"), ("holdings", "holdings_daily.csv"),
            ("order_sensitivity", "order_sensitivity.csv"), ("daily", "daily_metrics.csv")]:
            frame = pd.read_csv(directory / filename)
            for key, value in ids.items():
                if key not in frame:
                    frame[key] = value
            if name == "daily":
                context = pd.read_parquet(contained_path(context_paths[cell['fold']]["path"]),
                                          columns=["trade_date", "state_vol", "state_breadth", "state_trend"])
                frame = frame.merge(context.drop_duplicates(), on="trade_date", validate="one_to_one")
            datasets[name].append(frame)
        datasets["spells"].append(pd.DataFrame([{**ids, **cell["spells"]}]))
    result = {name: pd.concat(frames, ignore_index=True) for name, frames in datasets.items()}
    result["bootstrap"] = comparisons(result["daily"], settings["bootstrap"])
    return result


def state_comparison(daily, settings):
    rows = []
    for (model, layer, year), frame in daily.groupby(["model", "layer", "year"], sort=True):
        if model == "T030":
            continue
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        base = daily[(daily.model == "T030") & (daily.layer == layer) & (daily.year == year)].sort_values("trade_date").reset_index(drop=True)
        ix = block_indices(len(frame), settings["repetitions"], settings["block_length"], settings["seed"]+int(year))
        for family in ["vol", "breadth", "trend"]:
            for state in sorted(frame[f"state_{family}"].unique()):
                mask = frame[f"state_{family}"].to_numpy() == state
                a, b = frame[mask], base[mask]
                draws = paired_draws(frame, base, ix, mask)
                ci = interval(draws["annual_excess"])
                rows.append({"model": model, "layer": layer, "year": year, "family": family, "state": state,
                    "days": int(mask.sum()), "annual_excess": a.top_excess.mean()*252,
                    "reference_excess": b.top_excess.mean()*252,
                    "delta_excess": (a.top_excess.mean()-b.top_excess.mean())*252,
                    "delta_ic": a.ic.mean()-b.ic.mean(), "delta_turnover": a.turnover.mean()-b.turnover.mean(),
                    "delta_score": .4*(a.ic.mean()-b.ic.mean())+75.6*(a.top_excess.mean()-b.top_excess.mean())-.3*(a.turnover.mean()-b.turnover.mean()),
                    "excess_ci_lower": ci["lower"], "excess_ci_upper": ci["upper"], "valid_draws": ci["valid_draws"]})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default="S009")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"S009(?:_[A-Za-z0-9_-]+)?", args.study_id):
        raise ValueError("Use an S009 study identity.")
    cfg, config_path = load_config(args.config)
    specification = protocol(cfg)
    root = contained_path(cfg["paths"]["research_studies"]) / args.study_id
    revision, source = git_state(), code_hashes()
    if revision["branch"] != "main" or revision["dirty"]:
        raise ValueError("S009 starts from committed clean main.")
    if revision["commit"] != subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip():
        raise ValueError("S009 implementation must be pushed before execution.")
    meta_path = root / "study.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        if meta["status"] == "complete" or not args.resume:
            raise ValueError("Preserve existing study; only unchanged incomplete studies can resume.")
        if meta["source"] != source or meta["protocol_digest"] != digest(specification):
            raise ValueError("Source/protocol changed; preserve evidence and use a new identity.")
    else:
        if root.exists():
            raise ValueError("Incomplete preexisting study directory must be preserved.")
        root.mkdir(parents=True)
        meta = {"study_id": args.study_id, "status": "preparing", "started_at": timestamp(), "git": revision,
            "source": source, "source_digest": digest(source), "protocol": specification,
            "protocol_digest": digest(specification), "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
            "initial_records": read_records(contained_path(cfg["paths"]["experiment_log"])), "new_model_fits": 0,
            "new_shared_records": 0, "completed_pairs": []}
        save_json(meta_path, meta)
        write_text(root / "config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    if shutil.disk_usage(root).free < 5*1024**3:
        raise ValueError("S009 needs space for context and byte-preserving sensitivity files.")
    with study_lock(root):
        log = RunLog(root / "run.log")
        try:
            frozen = FrozenInputs(meta.get("frozen_inputs"))
            meta["data"] = provenance(cfg)
            cases = prepare_sources(cfg, frozen)
            preparation_path = root / "context.json"
            if preparation_path.exists():
                preparation = read_json(preparation_path)
                for entry in preparation["contexts"].values():
                    if sha256(contained_path(entry["path"])) != entry["sha256"]:
                        raise ValueError("Prepared causal context changed.")
            else:
                preparation = prepare_context(cfg, root, frozen, meta["data"], log)
                save_json(preparation_path, preparation)
            meta.update(status="running", frozen_inputs=frozen.hashes)
            save_json(meta_path, meta)
            cells = []
            for case in cases:
                pair_id = f"{case['model']}_{case['fold']}"
                if pair_id in meta["completed_pairs"]:
                    cells.extend(read_json(root / "cells" / case["model"] / case["fold"] / layer / "cell.json") for layer in ["raw", "band"])
                    continue
                log(f"BEGIN {len(meta['completed_pairs'])+1}/19 {pair_id}")
                context = pd.read_parquet(contained_path(preparation["contexts"][case['fold']]["path"]))
                cells.extend(analyze_pair(cfg, root, case, context, preparation, log))
                meta["completed_pairs"].append(pair_id)
                save_json(meta_path, meta)
                del context
                gc.collect()
            if len(cells) != 38:
                raise ValueError("S009 did not complete the declared comparison coverage.")
            tables = collect(root, cells, preparation["contexts"], cfg["top_tail_diagnostics"])
            tables["states"] = state_comparison(tables["daily"], cfg["top_tail_diagnostics"]["bootstrap"])
            from src.evaluation.top_tail_handoff import publish
            publish(cfg, root, meta, cells, tables, preparation, log)
            meta.update(status="complete", completed_at=timestamp(), completed_cells=38, peak_rss=log.peak_sampled_rss)
            save_json(meta_path, meta)
            log("S009 complete: 38 real saved-file cases; zero new fits/shared records")
        except BaseException as error:
            meta.update(status="failed", failed_at=timestamp(), error=repr(error))
            save_json(meta_path, meta)
            write_text(root / "failure_traceback.txt", traceback.format_exc())
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

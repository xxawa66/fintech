"""Frozen research inputs and unsupervised history-partner selection for S006."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.baseline_checks import check_predictions
from src.evaluation.official_eval import OFFICIAL_METRICS
from src.evaluation.turnover_controllers import prepare_controller_input
from src.models.alpha_research import is_canonical, relative, save_csv, verify_sources
from src.models.optuna_tuning import read_json
from src.models.target_research import predictive
from src.utils.project import sha256
from src.utils.research_cache import contained_path, digest


class FrozenFiles:
    def __init__(self, files=None):
        self.files = dict(files or {})
        self.checked = set()

    def check(self, path, expected=None):
        path = contained_path(path)
        name = relative(path)
        if name in self.checked:
            if expected is not None and self.files[name] != expected:
                raise ValueError(f"Conflicting frozen checksums: {name}")
            return self.files[name]
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen input changed: {name}")
        self.files[name] = actual
        self.checked.add(name)
        return actual


def frozen_selection(path, index_path, files):
    files.check(path); files.check(index_path)
    selection, index = read_json(contained_path(path)), read_json(contained_path(index_path))
    if (digest({k: v for k, v in selection.items() if k != "selection_digest"}) != selection["selection_digest"]
            or index["selection_digest"] != selection["selection_digest"] or index["status"] != "passed"):
        raise ValueError("Selection and frozen audit digest disagree.")
    return selection, index


def frozen_run(cfg, exp_id, index, files, *, full=True):
    saved = next((r for r in index["runs"] if r["exp_id"] == exp_id), None)
    if saved is None:
        raise ValueError(f"Missing audited source run: {exp_id}")
    files.check(saved["manifest"], saved["manifest_sha256"])
    run = read_json(contained_path(saved["manifest"]))
    if run["status"] != "passed" or run["exp_id"] != exp_id:
        raise ValueError("Source run did not pass.")
    if "metrics" in saved and any(abs(run["metrics"][k]-saved["metrics"][k]) > 1e-10 for k in OFFICIAL_METRICS):
        raise ValueError("Frozen source metrics changed.")
    if full:
        for path, expected in run["artifact_hashes"].items():
            files.check(path, expected)
    return run


def component(name, kind, runs, target=None):
    first = next(iter(runs.values()))
    specs = [{"model_params": r["spec"]["model_params"], "rounds": r["spec"]["rounds"]} for r in runs.values()]
    if any(s != specs[0] for s in specs) or any(not np.isfinite(r["metrics"][k]) for r in runs.values() for k in OFFICIAL_METRICS):
        raise ValueError("A component changed parameters between folds or has invalid metrics.")
    scores = [predictive(r["metrics"]) for r in runs.values()]
    return {"id": name, "kind": kind, "target": target or {"transform": "raw", "objective": "regression", "metric": "l2"},
            "model_spec": specs[0], "folds": runs, "predictive_mean": float(np.mean(scores)),
            "predictive_worst": float(min(scores)), "predictive_std": float(np.std(scores)),
            "mean_ic": float(np.mean([r["metrics"]["ic_mean"] for r in runs.values()])),
            "mean_excess": float(np.mean([r["metrics"]["annual_excess"] for r in runs.values()]))}


def component_frames(item, truths):
    frames = {}
    for fold, run in item["folds"].items():
        pred = pd.read_csv(contained_path(run["prediction"]), dtype={"ts_code": "str", "trade_date": "int64"})
        check_predictions(pred, truths[fold])
        if not is_canonical(pred):
            raise ValueError("Frozen prediction row order is not canonical.")
        if (run["split"]["fold"] != fold or not pred.trade_date.between(*run["split"]["valid_window"]).all()
                or len(pred) != run["validation_rows"]):
            raise ValueError("Component prediction does not cover its exact fold.")
        frames[fold] = pred
    return frames


def value_signature(frames):
    return tuple(hashlib.sha256(f.pred.to_numpy(dtype="<f8").tobytes()).hexdigest() for f in frames.values())


def daily_correlations(left, right):
    rows = []
    for fold in left:
        a, b = left[fold], right[fold]
        if not a[KEYS].equals(b[KEYS]):
            raise ValueError("History correlations require identical canonical keys.")
        days = np.sort(a.trade_date.unique())
        aa = a.pred.groupby(a.trade_date).rank(method="average", pct=True).to_numpy().reshape(len(days), 4650)
        bb = b.pred.groupby(b.trade_date).rank(method="average", pct=True).to_numpy().reshape(len(days), 4650)
        aa -= aa.mean(axis=1, keepdims=True); bb -= bb.mean(axis=1, keepdims=True)
        corr = (aa*bb).sum(axis=1)/np.sqrt((aa*aa).sum(axis=1)*(bb*bb).sum(axis=1))
        if not np.isfinite(corr).all():
            raise ValueError("Non-finite unsupervised rank correlation.")
        rows.extend({"fold": fold, "trade_date": int(day), "rank_correlation": float(value)} for day, value in zip(days, corr))
    return pd.DataFrame(rows)


def prepare_inputs(cfg, root: Path, log):
    s003, checked, base_sources = verify_sources(cfg)
    files = FrozenFiles(checked["frozen_files"])
    index003 = read_json(contained_path(cfg["alpha_research"]["source_artifacts"]))
    section = cfg["ensemble_research"]
    s004, index004 = frozen_selection(section["controller_selection"], section["controller_artifacts"], files)
    s005, index005 = frozen_selection(section["target_selection"], section["target_artifacts"], files)
    if (s004["source_selection_digest"] != s003["selection_digest"]
            or s005["source_selection_digest"] != s003["selection_digest"]
            or s005["controller_selection_digest"] != s004["selection_digest"]):
        raise ValueError("S003/S004/S005 handoff sources are inconsistent.")
    truths = {}
    for definition, source in zip(section["folds"], base_sources):
        fold = definition["name"]
        truth = pd.read_csv(contained_path(source["raw"]["labels"]), dtype={"ts_code": "str", "trade_date": "int64"})
        if (not is_canonical(truth) or len(truth) != 4650*truth.trade_date.nunique()
                or truth.ts_code.nunique() != 4650 or not truth.trade_date.between(*definition["valid"]).all()):
            raise ValueError("Frozen complete fold labels changed.")
        truths[fold] = truth
    base = component("T030", "historical", {s["definition"]["name"]: s["raw"] for s in base_sources})
    fixed_targets = []
    for selected in s005["partners"]:
        runs = {fold: frozen_run(cfg, r["exp_id"], index005, files) for fold, r in selected["runs"].items()}
        for fold, run in runs.items():
            saved = selected["runs"][fold]
            if (run["spec"] != saved["spec"] or run["artifact_hashes"][run["prediction"]] != saved["prediction_sha256"]
                    or run["artifact_hashes"][run["model"]] != saved["model_sha256"]):
                raise ValueError("Selected target model/forecast differs from S005.")
        fixed_targets.append(component(selected["id"], "target", runs, selected["target"]))
    controllers = [{"id": "Q0", "source": "S003 q*", "spec": {"method": "band", "keep_q": s003["winner"]["spec"]["keep_q"]},
                    "cv_runs": {f["name"]: frozen_run(cfg, f"S004_{s004['baseline']['id']}_{f['name']}", index004, files)
                                for f in section["folds"]}}]
    for source in base_sources:
        actual = controllers[0]["cv_runs"][source["definition"]["name"]]
        if any(abs(actual["metrics"][k]-source["band"]["metrics"][k]) > 1e-10 for k in OFFICIAL_METRICS):
            raise ValueError("S004 baseline does not reproduce frozen S003.")
    for chosen in s004["controllers"]:
        controllers.append({"id": f"Q{len(controllers)}", "source": chosen["id"], "spec": chosen["spec"],
                            "cv_runs": {fold: frozen_run(cfg, r["exp_id"], index004, files) for fold, r in chosen["runs"].items()}})
    history = []
    for trial in range(50):
        runs = {f["name"]: frozen_run(cfg, f"S003_T{trial:03d}_{f['name']}_raw", index003, files, full=False)
                for f in section["folds"]}
        history.append(component(f"T{trial:03d}", "historical", runs))
    history.sort(key=lambda r: (-r["predictive_mean"], int(r["id"][1:])))
    pool = history[:section["history_top_k"]]
    if not any(r["id"] == "T030" for r in pool):
        pool.append(next(r for r in history if r["id"] == "T030"))
    representatives, frames, signatures, params_seen, value_seen, aliases = [], {}, {}, {}, {}, {}
    # T030 is always retained, including when an identical parameter endpoint exists.
    pool = [next(r for r in pool if r["id"] == "T030")] + [r for r in pool if r["id"] != "T030"]
    for item in pool:
        parameter_id = digest(item["model_spec"])
        if parameter_id in params_seen:
            aliases[item["id"]] = {"representative": params_seen[parameter_id], "reason": "same model parameters and rounds"}
            continue
        for run in item["folds"].values():
            for path, expected in run["artifact_hashes"].items():
                files.check(path, expected)
        actual = component_frames(item, truths)
        value_id = value_signature(actual)
        if value_id in value_seen:
            aliases[item["id"]] = {"representative": value_seen[value_id], "reason": "identical parsed forecasts in all folds"}
            continue
        representatives.append(item); frames[item["id"]] = actual; signatures[item["id"]] = value_id
        params_seen[parameter_id] = item["id"]; value_seen[value_id] = item["id"]
    others = [r for r in representatives if r["id"] != "T030"]
    others.sort(key=lambda r: (-r["predictive_mean"], int(r["id"][1:])))
    partners, correlations = [], []
    if others:
        first = others[0]
        partners.append(first)
        rows = []
        for candidate in others[1:]:
            pair_means = []
            for reference in ["T030", first["id"]]:
                daily = daily_correlations(frames[candidate["id"]], frames[reference])
                pair_means.append(float(daily.groupby("fold").rank_correlation.mean().mean()))
                correlations.extend({"candidate": candidate["id"], "reference": reference, **r} for r in daily.to_dict("records"))
            rows.append({"id": candidate["id"], "mean_correlation_to_two": float(np.mean(pair_means)),
                         "predictive_mean": candidate["predictive_mean"], "trial": int(candidate["id"][1:])})
        if rows:
            lowest = min(r["mean_correlation_to_two"] for r in rows)
            tied = [r for r in rows if r["mean_correlation_to_two"]-lowest <= 1e-12]
            second = min(tied, key=lambda r: (-r["predictive_mean"], r["trial"]))
            partners.append(next(r for r in others if r["id"] == second["id"]))
    components = [base, *fixed_targets, *partners]
    for item in fixed_targets:
        frames[item["id"]] = component_frames(item, truths)
    frames = {item["id"]: frames[item["id"]] for item in components}
    selected_ids, pool_ids = {p["id"] for p in partners}, {p["id"] for p in pool}
    quality = pd.DataFrame([{k: r[k] for k in ["id", "predictive_mean", "predictive_worst", "predictive_std", "mean_ic", "mean_excess"]}
        | {"trial": int(r["id"][1:]), "preselected": r["id"] in pool_ids, "selected": r["id"] in selected_ids,
           "alias_of": aliases.get(r["id"], {}).get("representative"), "params_digest": digest(r["model_spec"]),
           "rounds": r["model_spec"]["rounds"]} for r in history])
    save_csv(root / "history_quality.csv", quality)
    save_csv(root / "history_correlations.csv", pd.DataFrame(correlations))
    save_csv(root / "history_aliases.csv", pd.DataFrame([{"id": k, **v} for k, v in aliases.items()],
                                                       columns=["id", "representative", "reason"]))
    log(f"history frozen: top10+T030={len(pool)}; unique={len(representatives)}; partners={[p['id'] for p in partners]}; components={[c['id'] for c in components]}")
    prepared = {"data": checked["data"], "frozen_files": files.files,
                "source_selection_digest": s003["selection_digest"], "controller_selection_digest": s004["selection_digest"],
                "target_selection_digest": s005["selection_digest"], "components": components, "controllers": controllers,
                "history_partners": [p["id"] for p in partners], "history_preselection": [p["id"] for p in pool],
                "history_aliases": aliases, "baseline": s004["baseline"]}
    return prepared, frames, truths

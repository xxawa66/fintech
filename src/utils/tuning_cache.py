"""A verified 40-feature X cache and causal, expanding-window tuning folds."""
from __future__ import annotations

import gc
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import clean_history
from src.data.load_data import KEYS, X_COLUMNS, load_training_data
from src.data.make_dataset import attach_labels, training_arrays
from src.evaluation.validation import Fold, split_train_valid
from src.features.build_features import build_features, feature_names
from src.utils.project import ROOT, project_path, sha256, write_json
from src.utils.research_cache import code_hashes, contained_path, digest


@dataclass
class TuningFold:
    fold: Fold
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_valid: pd.DataFrame
    keys: pd.DataFrame
    eligible: np.ndarray
    truth: pd.DataFrame
    labels_path: Path
    split: dict
    supervision: dict


def make_fold(item: dict) -> Fold:
    return Fold(item["name"], *item["train"], *item["valid"])


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["ts_code"] = frame.ts_code.astype(str)
    frame["trade_date"] = frame.trade_date.astype("int32")
    return frame.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)


def prepare_folds(cfg: dict, definitions: list[dict], provenance: dict,
                  study_root: Path, log) -> tuple[list[TuningFold], dict]:
    """Calculate X over continuous history once; attach labels only afterwards.

    All folds live in memory as selected arrays; large raw/history frames are
    released before fitting. Each fit still constructs its own LightGBM Dataset.
    """
    folds = [make_fold(item) for item in definitions]
    start, end = min(f.train_start for f in folds), max(f.valid_end for f in folds)
    raw = load_training_data(project_path(cfg["paths"]["train"]))
    if len(raw) != 7_900_350:
        raise ValueError("Tuning requires the complete original training CSV.")
    raw = raw[raw.trade_date.between(start, end)]
    raw, cleaning = clean_history(raw)
    columns = feature_names(cfg["features"])
    if len(columns) != 40:
        raise ValueError("S003 is restricted to the frozen 40-feature V1.")
    relevant = {p: h for p, h in code_hashes().items()
                if p.startswith("src/features/") or p in {
                    "src/data/clean_data.py", "src/data/load_data.py", "src/utils/tuning_cache.py"}}
    identity = {"schema": 1, "raw_sha256": provenance["sha256"],
                "history": [start, end], "rows": len(raw), "features": cfg["features"],
                "code": relevant, "packages": {n: importlib.metadata.version(n)
                                               for n in ["numpy", "pandas", "pyarrow"]}}
    directory = contained_path(cfg["paths"]["optuna_cache"]) / digest(identity)
    parquet, metadata = directory / "features.parquet", directory / "cache.json"
    if directory.exists() and not metadata.exists():
        raise ValueError("Incomplete feature cache: preserve it and inspect before reuse.")
    if not directory.exists():
        directory.mkdir(parents=True)
        log(f"building 40-feature X cache: {len(raw):,} historical rows through {end}")
        features, actual = build_features(raw[KEYS + X_COLUMNS], cfg["features"], log)
        if actual != columns:
            raise ValueError("Feature column order changed.")
        features.to_parquet(parquet, index=False)
        write_json(metadata, {"status": "passed", "identity": identity,
                              "columns": columns, "parquet_sha256": sha256(parquet)})
        del features
        gc.collect()
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    if saved["identity"] != identity or saved["parquet_sha256"] != sha256(parquet):
        raise ValueError("Feature cache source/checksum mismatch.")
    features = pd.read_parquet(parquet)
    features["ts_code"] = pd.Categorical(features.ts_code, categories=raw.ts_code.cat.categories)
    features["trade_date"] = features.trade_date.astype(raw.trade_date.dtype)
    if list(features.columns) != KEYS + columns or not features[KEYS].equals(raw[KEYS]):
        raise ValueError("Feature cache keys, row order or columns changed.")
    if any(features[c].dtype != np.float32 for c in columns):
        raise ValueError("Feature dtypes must remain float32.")
    if np.isinf(features[columns].to_numpy()).any():
        raise ValueError("Feature cache contains infinity.")
    dataset = attach_labels(raw, features)
    del raw, features
    gc.collect()
    result = []
    for fold in folds:
        train, valid, split = split_train_valid(dataset, fold)
        x_train, y_train, supervision = training_arrays(train, columns)
        eligible = valid.quote_valid.to_numpy()
        truth = canonical(valid[KEYS + ["y_ret_1d", "flag_limit_up"]])
        truth["flag_limit_up"] = truth.flag_limit_up.astype("int8")
        labels = study_root / "labels" / f"{fold.name}.csv"
        labels.parent.mkdir(parents=True, exist_ok=True)
        if labels.exists():
            existing = pd.read_csv(labels)
            # Compare the CSV representation: default CSV parsing can round y.
            if (not canonical(existing)[KEYS].equals(truth[KEYS])
                    or len(existing) != len(truth)):
                raise ValueError("Existing fold label keys changed.")
        else:
            truth.to_csv(labels, index=False)
        if (truth.ts_code.nunique() != 4650 or truth.duplicated(KEYS).any()
                or len(truth) != 4650 * truth.trade_date.nunique()
                or split["n_boundary_rows"] != 4650
                or split["n_train_labels_after"] + split["n_boundary_labels_dropped"]
                != split["n_train_labels_before"]):
            raise ValueError("Fold panel coverage or boundary-label accounting failed.")
        result.append(TuningFold(fold, x_train, y_train, valid.loc[eligible, columns],
                                 valid[KEYS].copy(), eligible, truth, labels, split, supervision))
        log(f"{fold.name}: train={len(y_train):,}, valid={len(truth):,}, "
            f"boundary_labels={split['n_boundary_labels_dropped']}")
        del train, valid
        gc.collect()
    del dataset
    gc.collect()
    cache = {"identity": identity, "path": directory.relative_to(ROOT).as_posix(),
             "parquet_sha256": saved["parquet_sha256"], "cleaning": cleaning,
             "folds": {f.fold.name: {"split": f.split, "supervision": f.supervision,
                                     "labels": f.labels_path.relative_to(ROOT).as_posix(),
                                     "labels_sha256": sha256(f.labels_path)} for f in result}}
    log("prepared causal tuning folds; released raw/history frames")
    return result, cache

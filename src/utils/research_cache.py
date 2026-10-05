"""Verified X-only cache shared by experiments with the same historical cutoff."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import clean_history
from src.data.load_data import KEYS, X_COLUMNS, load_training_data
from src.data.make_dataset import attach_labels
from src.features.build_features import build_features, feature_names
from src.features.research_features import research_names
from src.models.baseline import PreparedHistory
from src.utils.project import ROOT, project_path, sha256, timestamp, write_json


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def code_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}


def protocol_config(cfg: dict) -> dict:
    """Progress/documentation commits do not change the numerical protocol."""
    return {k: cfg[k] for k in ["paths", "data", "evaluation", "validation", "features", "baseline", "research"]}


def contained_path(value: str | Path) -> Path:
    path = project_path(value).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("Research artifacts must stay within a subdirectory of the repository.")
    return path


def cache_identity(raw: pd.DataFrame, cfg: dict, fold, provenance: dict) -> dict:
    feature_code = {p: h for p, h in code_hashes().items()
                    if p.startswith("src/features/") or p in {
                        "src/data/clean_data.py", "src/data/load_data.py", "src/utils/research_cache.py"}}
    stocks = sorted(raw.ts_code.astype(str).unique().tolist())
    return {"schema": 1, "training_sha256": provenance["sha256"],
            "history_window": [int(fold.train_start), int(fold.valid_end)],
            "actual_window": [int(raw.trade_date.min()), int(raw.trade_date.max())],
            "rows": len(raw), "stock_count": len(stocks), "stocks_sha256": digest(stocks),
            "features": cfg["features"], "research": {k: cfg["research"][k] for k in
                                                     ["pool_version", "windows", "groups"]},
            "feature_code": feature_code,
            "packages": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "pyarrow"]}}


def validate_cache(directory: Path, identity: dict, raw: pd.DataFrame,
                   columns: list[str]) -> tuple[pd.DataFrame, dict]:
    metadata = json.loads((directory / "cache.json").read_text(encoding="utf-8"))
    parquet = directory / "features.parquet"
    if (metadata.get("status") != "passed" or metadata.get("identity") != identity
            or sha256(parquet) != metadata.get("parquet_sha256")):
        raise ValueError("Feature cache identity or checksum mismatch; preserve it for inspection.")
    features = pd.read_parquet(parquet)
    if list(features.columns) != KEYS + columns or len(features) != len(raw):
        raise ValueError("Feature cache columns/row count disagree with the protocol.")
    if isinstance(raw.ts_code.dtype, pd.CategoricalDtype):
        features["ts_code"] = pd.Categorical(features.ts_code, categories=raw.ts_code.cat.categories)
    features["trade_date"] = features.trade_date.astype(raw.trade_date.dtype)
    if not features[KEYS].equals(raw[KEYS]):
        raise ValueError("Feature cache keys/order disagree with the original historical rows.")
    if any(features[c].dtype != np.float32 for c in columns):
        raise ValueError("Feature cache has changed numerical dtypes.")
    if np.isinf(features[columns].to_numpy()).any():
        raise ValueError("Feature cache contains infinite values.")
    return features, metadata


def cached_features(raw: pd.DataFrame, cfg: dict, fold, provenance: dict, log) -> tuple[pd.DataFrame, dict]:
    identity = cache_identity(raw, cfg, fold, provenance)
    fingerprint = digest(identity)
    root = contained_path(cfg["paths"]["research_cache"])
    root.mkdir(parents=True, exist_ok=True)
    directory = root / fingerprint
    columns = feature_names(cfg["features"]) + research_names(cfg["research"])
    hit = directory.exists()
    if not hit:
        log(f"building X-only cache {fingerprint[:12]} ({len(raw):,} rows)")
        features, actual_columns = build_features(raw[KEYS + X_COLUMNS], cfg["features"], log,
                                                  research=cfg["research"])
        if actual_columns != columns:
            raise AssertionError("Feature cache column contract changed.")
        with tempfile.TemporaryDirectory(prefix="building_", dir=root) as temporary:
            staging = Path(temporary) / "complete"
            staging.mkdir()
            parquet = staging / "features.parquet"
            features.to_parquet(parquet, index=False)
            write_json(staging / "cache.json", {"status": "passed", "created_at": timestamp(),
                       "identity": identity, "columns": columns, "parquet_sha256": sha256(parquet)})
            # The final directory appears only after both files have been completed.
            staging.rename(directory)
        del features
        gc.collect()
    features, metadata = validate_cache(directory, identity, raw, columns)
    record = {"fingerprint": fingerprint, "hit": hit, "identity": identity,
              "path": directory.relative_to(ROOT).as_posix(),
              "parquet_sha256": metadata["parquet_sha256"], "columns": columns}
    log(f"verified feature cache {fingerprint[:12]}; hit={hit}")
    return features, record


def prepare_history(cfg: dict, fold, provenance: dict, log, smoke: bool = False) -> PreparedHistory:
    started = time.perf_counter()
    raw = load_training_data(project_path(cfg["paths"]["train"]))
    if len(raw) != 7_900_350:
        raise ValueError("Research expects the full supplied training CSV before selecting its history.")
    if smoke:
        stocks = sorted(raw.ts_code.cat.categories.astype(str))[:cfg["baseline"]["smoke_stocks"]]
        raw = raw[raw.ts_code.isin(stocks)]
    raw = raw[raw.trade_date.between(fold.train_start, fold.valid_end)]
    raw, cleaning = clean_history(raw)
    truth = raw.loc[raw.trade_date.between(fold.valid_start, fold.valid_end),
                    KEYS + ["y_ret_1d", "flag_limit_up"]].copy()
    features, cache = cached_features(raw, cfg, fold, provenance, log)
    dataset = attach_labels(raw, features)
    cache["preparation_seconds"] = time.perf_counter() - started
    del raw, features
    gc.collect()
    return PreparedHistory(dataset, truth, cleaning, cache, fold.name)

"""Reuse S003's unchanged X cache and recover actual supervised training keys."""
from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import clean_history, valid_quote
from src.data.load_data import KEYS, load_training_data
from src.evaluation.validation import split_train_valid
from src.models.optuna_tuning import read_json
from src.utils.project import project_path, sha256
from src.utils.research_cache import contained_path
from src.utils.tuning_cache import TuningFold, prepare_folds


@dataclass
class TargetFold:
    data: TuningFold
    training: pd.DataFrame
    key_digest: str


def prepare_target_folds(cfg, root: Path, meta: dict, sources, log):
    folds, cache = prepare_folds(cfg, cfg["target_research"]["folds"], meta["data"], root, log)
    frozen_cache = read_json(contained_path(cfg["target_research"]["source_study"]))["search_cache"]
    if any(cache[k] != frozen_cache[k] for k in ["identity", "path", "parquet_sha256", "cleaning"]):
        raise ValueError("S005 must reuse the exact verified S003 X cache.")
    # Recover date/stock keys through the same sorted history and mask. Matching
    # the original X index, raw labels and split proves the alignment directly.
    raw = load_training_data(project_path(cfg["paths"]["train"]))
    raw = raw[raw.trade_date.between(min(f.fold.train_start for f in folds),
                                   max(f.fold.valid_end for f in folds))]
    raw, _ = clean_history(raw)
    result = []
    for data, source in zip(folds, sources):
        train, valid, split = split_train_valid(raw, data.fold)
        mask = np.isfinite(train.y_ret_1d.to_numpy()) & valid_quote(train).to_numpy()
        selected = train.loc[mask]
        if (not selected.index.equals(data.x_train.index)
                or not np.array_equal(selected.y_ret_1d.to_numpy(), data.y_train)
                or split != data.split or split != source["raw"]["split"]
                or data.supervision != source["raw"]["training_samples"]
                or sha256(data.labels_path) != source["raw"]["artifact_hashes"][source["raw"]["labels"]]
                or selected.trade_date.max() >= split["last_train_day"]):
            raise ValueError("Recovered supervised keys/labels or boundary purge differ from T030.")
        training = selected[KEYS + ["y_ret_1d"]].reset_index(drop=True)
        training["ts_code"] = training.ts_code.astype(str)
        hashed = pd.util.hash_pandas_object(training[KEYS], index=False).to_numpy(dtype="<u8")
        key_digest = hashlib.sha256(hashed.tobytes()).hexdigest()
        result.append(TargetFold(data, training, key_digest))
        log(f"aligned {data.fold.name}: {len(training):,} supervised keys, "
            f"purged {split['n_boundary_labels_dropped']:,} labels; target dates end {training.trade_date.max()}")
        del train, valid, selected
        gc.collect()
    del raw
    gc.collect()
    return result, cache

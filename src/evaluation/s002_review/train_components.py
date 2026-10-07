"""成员 B 独立复核 S002：在本机重建 L1 / R2 两折预测。

- 不写入共享实验表 experiments/experiment_log.csv
- 不覆盖 A 的产物目录（outputs/models|metrics|predictions 下 A 的实验编号）
- 复现口径：与 configs/project.yaml 的 model_research 段一致（V1 40 特征、L1/R2 参数、固定轮数）

用法::

    python src/evaluation/s002_review/train_components.py --fold fold1
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.make_dataset import training_arrays          # noqa: E402
from src.data.load_data import KEYS                        # noqa: E402
from src.evaluation.validation import get_folds, split_train_valid  # noqa: E402
from src.features.build_features import feature_names      # noqa: E402
from src.models import lightgbm_model, ridge_model         # noqa: E402
from src.models.baseline import RunLog, source_provenance  # noqa: E402
from src.utils.project import load_config, project_path, sha256, write_json  # noqa: E402
from src.utils.research_cache import prepare_history       # noqa: E402

OUT = ROOT / "tmp" / "review_s002_b" / "artifacts"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--fold", required=True, choices=["fold1", "fold2"])
    args = parser.parse_args(argv)

    cfg, config_path = load_config(args.config)
    research = cfg["model_research"]
    fold = next(f for f in get_folds(config_path) if f.name == args.fold)
    target = OUT / args.fold
    target.mkdir(parents=True, exist_ok=True)
    log = RunLog(target / "run.log")

    provenance = source_provenance(cfg)
    log(f"原始训练数据 SHA 核对通过：{provenance['sha256'][:16]}… "
        f"({provenance['size_bytes']:,} 字节)")
    started = time.perf_counter()

    prepared = prepare_history(cfg, fold, provenance, log)
    columns = feature_names(cfg["features"])
    assert len(columns) == 40, columns
    write_json(target / "cache.json", prepared.cache)

    frame = prepared.dataset[KEYS + columns + ["y_ret_1d", "quote_valid"]]
    train, valid, split = split_train_valid(frame, fold)
    del frame
    gc.collect()
    x, y, stats = training_arrays(train, columns)
    del train
    gc.collect()
    log(f"{fold.name}: 训练 {len(y):,} 行；验证 {len(valid):,} 行；"
        f"屏蔽跨界标签 {split['n_boundary_labels_dropped']}")

    provenance_record = {
        "fold": fold.name, "train_window": [fold.train_start, fold.train_end],
        "valid_window": [fold.valid_start, fold.valid_end],
        "split": split, "training_samples": stats,
        "data_sha256": provenance["sha256"],
        "official_attachments": provenance["official_attachments"],
        "feature_columns": columns, "config_sha256": sha256(config_path),
    }
    write_json(target / "provenance.json", provenance_record)

    eligible = valid["quote_valid"].to_numpy()
    keys = valid[KEYS].copy()

    # ---- L1: V1 参数 + num_leaves=15, min_data_in_leaf=500, lambda_l2=5.0 ----
    l1_preset = next(p for p in research["lightgbm_presets"] if p["name"] == "L1")
    params = dict(cfg["baseline"]["model"]["params"])
    params.update(l1_preset["overrides"])
    step = time.perf_counter()
    model = lightgbm_model.train_model(x, y, params, l1_preset["rounds"], log)
    lgb_values = np.full(len(valid), cfg["baseline"]["fallback_prediction"], dtype="float64")
    lgb_values[eligible] = model.predict(valid.loc[eligible, columns], num_threads=8)
    lgb_seconds = time.perf_counter() - step
    lgb_frame = keys.copy()
    lgb_frame["pred"] = lgb_values
    lgb_frame.to_csv(target / "l1_valid.csv", index=False)
    log(f"L1 完成：{l1_preset['rounds']} 轮，{lgb_seconds:.1f}s，"
        f"实际迭代 {model.current_iteration()}，兜底行 {int((~eligible).sum()):,}")

    # ---- R2: Ridge alpha=10000, lsqr ----
    settings = research["ridge"]
    ridge_params = {k: settings[k] for k in ["solver", "tol", "max_iter", "fit_intercept"]}
    ridge_params["alpha"] = next(p["alpha"] for p in research["ridge_presets"] if p["name"] == "R2")
    step = time.perf_counter()
    bundle = ridge_model.train_model(x, y, ridge_params, settings["num_threads"], log)
    ridge_values = np.full(len(valid), cfg["baseline"]["fallback_prediction"], dtype="float64")
    ridge_values[eligible] = ridge_model.predict_model(
        bundle, valid.loc[eligible, columns], settings["batch_rows"])
    ridge_seconds = time.perf_counter() - step
    ridge_frame = keys.copy()
    ridge_frame["pred"] = ridge_values
    ridge_frame.to_csv(target / "r2_valid.csv", index=False)
    log(f"R2 完成：alpha={ridge_params['alpha']:.0f}，{ridge_seconds:.1f}s，"
        f"迭代 {bundle['info']['n_iter']}，警告 {len(bundle['info']['warnings'])}")

    truth = prepared.truth.copy()
    truth.to_csv(target / "validation_labels.csv", index=False)

    provenance_record.update({
        "l1_seconds": lgb_seconds, "ridge_seconds": ridge_seconds,
        "l1_iterations": model.current_iteration(), "ridge_iterations": bundle["info"]["n_iter"],
        "ridge_warnings": bundle["info"]["warnings"],
        "fallback_rows": int((~eligible).sum()),
        "total_seconds": time.perf_counter() - started,
    })
    write_json(target / "provenance.json", provenance_record)
    log(f"{fold.name} 组件训练完成，用时 {provenance_record['total_seconds']:.1f}s，产物 {target}")
    print(json.dumps(provenance_record, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

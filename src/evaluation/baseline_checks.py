"""Pipeline-level coverage checks and comparison against the unmodified official file."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.official_eval import OFFICIAL_METRICS, _load_official_evaluator
from src.utils.project import ROOT


def check_predictions(predictions: pd.DataFrame, truth: pd.DataFrame) -> None:
    if list(predictions.columns) != KEYS + ["pred"]:
        raise ValueError("Predictions must contain exactly ts_code, trade_date, pred.")
    if not np.isfinite(predictions["pred"].to_numpy()).all():
        raise ValueError("Predictions contain missing/infinite values.")
    for frame in [predictions, truth]:
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError("Missing or duplicate prediction/validation keys.")
    def normalized_keys(frame):
        return frame[KEYS].astype({"ts_code": "str", "trade_date": "int64"}).sort_values(
            KEYS, kind="stable").reset_index(drop=True)
    if not normalized_keys(predictions).equals(normalized_keys(truth)):
        raise ValueError("Prediction keys differ from complete validation keys.")


def compare_official(pred_path: Path, truth: pd.DataFrame, metrics: dict, tolerance: float) -> dict:
    if not all(np.isfinite(metrics[name]) for name in OFFICIAL_METRICS):
        raise ValueError("Official summary metrics must all be finite.")
    with tempfile.TemporaryDirectory(prefix="baseline-official-") as temporary:
        folder = Path(temporary)
        truth[KEYS + ["y_ret_1d"]].to_csv(folder / "测试集_Y.csv", index=False)
        truth[KEYS + ["flag_limit_up"]].to_csv(folder / "测试集_X.csv", index=False)
        official = _load_official_evaluator(ROOT).evaluate(str(pred_path), str(folder))
    differences = {name: float(abs(metrics[name] - official[name])) for name in OFFICIAL_METRICS}
    if not all(np.isfinite(value) and value <= tolerance for value in differences.values()):
        raise AssertionError(f"Scoring differs from official evaluator: {differences}")
    return differences

"""Align complete prediction keys and blend same-day percentile ranks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.baseline_checks import check_predictions


def align_predictions(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray]:
    if not frames:
        raise ValueError("At least one prediction frame is required.")
    first = frames[0].copy()
    check_predictions(first, first[KEYS])
    first["ts_code"] = first.ts_code.astype(str)
    first["trade_date"] = first.trade_date.astype("int64")
    keys = first.sort_values(["trade_date", "ts_code"], kind="stable")[KEYS].reset_index(drop=True)
    values = []
    for frame in frames:
        check_predictions(frame, keys)
        normalized = frame.astype({"ts_code": "str", "trade_date": "int64"}).sort_values(
            ["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
        if not normalized[KEYS].equals(keys):
            raise ValueError("Prediction key alignment failed.")
        values.append(normalized.pred.to_numpy(dtype="float64"))
    return keys, np.column_stack(values)


def rank_blend(frames: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    weights_array = np.asarray(weights, dtype="float64")
    if (len(weights) != len(frames) or not np.isfinite(weights_array).all()
            or (weights_array < 0).any() or abs(weights_array.sum() - 1.) > 1e-12):
        raise ValueError("Weights must be finite, nonnegative and sum to one.")
    keys, values = align_predictions(frames)
    ranks = pd.DataFrame(values).groupby(keys.trade_date.to_numpy()).rank(method="average", pct=True)
    result = keys.copy()
    result["pred"] = ranks.to_numpy() @ weights_array
    check_predictions(result, keys)
    return result


def prediction_agreement(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Unsupervised complementarity diagnostics; labels are never used here."""
    keys, values = align_predictions([left, right])
    ranked = pd.DataFrame(values).groupby(keys.trade_date.to_numpy()).rank(method="average", pct=True)
    rows = []
    for day, positions in keys.groupby("trade_date").indices.items():
        a, b = ranked.iloc[positions, 0], ranked.iloc[positions, 1]
        n = max(len(positions) // 10, 1)
        aa = set(np.argsort(-values[positions, 0])[:n])
        bb = set(np.argsort(-values[positions, 1])[:n])
        rows.append({"trade_date": int(day), "rank_correlation": float(a.corr(b)),
                     "top10_overlap_fraction": len(aa & bb) / n,
                     "top10_jaccard": len(aa & bb) / len(aa | bb)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError("Retain the existing prediction; choose a new output file.")
    result = rank_blend([pd.read_csv(p) for p in args.pred], args.weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

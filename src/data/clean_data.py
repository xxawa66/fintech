"""Keep calendar rows, validate keys, and make missing-value handling explicit."""
import numpy as np
import pandas as pd

from src.data.load_data import KEYS, PRICE_COLUMNS, VALUE_COLUMNS, X_COLUMNS


def valid_quote(frame: pd.DataFrame) -> pd.Series:
    prices = frame[PRICE_COLUMNS]
    return prices.notna().all(axis=1) & (prices > 0).all(axis=1)


def clean_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = set(KEYS + X_COLUMNS + ["y_ret_1d"])
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if frame.empty or frame[KEYS].isna().any().any():
        raise ValueError("Empty history or missing stock/date keys.")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate stock/date keys; no rows were silently removed.")
    pd.to_datetime(frame["trade_date"].unique().astype(str), format="%Y%m%d", errors="raise")
    result = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
    stats = {"rows": len(result), "stocks": int(result["ts_code"].nunique()),
             "days": int(result["trade_date"].nunique()), "nonfinite_x_to_nan": {}}
    for name in VALUE_COLUMNS:
        bad = np.isinf(result[name].to_numpy())
        stats["nonfinite_x_to_nan"][name] = int(bad.sum())
        if bad.any():
            result.loc[bad, name] = np.nan
    for name in ["flag_limit_up", "flag_limit_down"]:
        if not result[name].isin([0, 1]).all():
            raise ValueError(f"Invalid limit flag: {name}")
    prices = result[PRICE_COLUMNS]
    invalid = ((prices <= 0).any(axis=1)
               | (result["high"] < result["low"])
               | (result["high"] < prices[["open", "close"]].max(axis=1))
               | (result["low"] > prices[["open", "close"]].min(axis=1))
               | (result["vol"] < 0) | (result["amount"] < 0))
    if invalid.any():
        raise ValueError(f"Unexpected invalid quote/volume rows: {int(invalid.sum())}")
    stats.update({"missing_label_rows": int(result["y_ret_1d"].isna().sum()),
                  "nonfinite_label_rows": int(np.isinf(result["y_ret_1d"].to_numpy()).sum()),
                  "invalid_quote_rows": int((~valid_quote(result)).sum()),
                  "zero_volume_rows": int(result["vol"].eq(0).sum())})
    return result, stats

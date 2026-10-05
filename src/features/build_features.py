"""Build the complete fixed feature set without access to any target column."""
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.clean_data import valid_quote
from src.data.load_data import KEYS, X_COLUMNS
from src.features.cross_section_features import percentile_rank
from src.features.price_features import price_features
from src.features.technical_features import technical_features
from src.features.volume_features import volume_features

CANDLES = ["intraday_ret", "high_low_range", "high_close", "close_low", "open_gap", "close_position"]
STATES = ["flag_limit_up", "flag_limit_down", "zero_volume"]


def feature_names(settings: dict) -> list[str]:
    names = [f"ret_{w}" for w in settings["return_windows"]] + CANDLES
    for prefix, field in [("ma_bias", "ma_windows"), ("volatility", "volatility_windows"),
                          ("vol_ratio", "volume_windows"), ("amount_ratio", "amount_windows"),
                          ("vol_change", "volume_change_windows"), ("price_position", "position_windows")]:
        names += [f"{prefix}_{w}" for w in settings[field]]
    names += STATES + [f"rank_{source}" for source in settings["rank_sources"]]
    if len(names) != settings["expected_count"] or len(set(names)) != len(names):
        raise ValueError("Unexpected feature count or duplicate names.")
    return names


def build_features(x: pd.DataFrame, settings: dict,
                   progress: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if set(x.columns) != set(KEYS + X_COLUMNS):
        raise ValueError("Feature input must contain exactly keys and raw X, with no labels.")
    if x.duplicated(KEYS).any():
        raise ValueError("Feature input keys are not unique.")
    frame = x.sort_values(KEYS, kind="stable").reset_index(drop=True)
    names = feature_names(settings)
    positions = {name: i for i, name in enumerate(names)}
    matrix = np.full((len(frame), len(names)), np.nan, dtype="float32")
    groups = frame.groupby("ts_code", sort=False, observed=True)
    total = frame["ts_code"].nunique()
    for done, (_, stock) in enumerate(groups, 1):
        values = price_features(stock, settings)
        values.update(volume_features(stock, settings))
        values.update(technical_features(stock, settings, values["ret_1"]))
        rows = stock.index.to_numpy()
        for name, series in values.items():
            matrix[rows, positions[name]] = series.to_numpy(dtype="float32")
        if progress and (done % 500 == 0 or done == total):
            progress(f"feature stocks {done}/{total}")
    for name in STATES[:2]:
        matrix[:, positions[name]] = frame[name].to_numpy(dtype="float32")
    matrix[:, positions["zero_volume"]] = frame["vol"].eq(0).to_numpy(dtype="float32")
    eligible = valid_quote(frame)
    for source in settings["rank_sources"]:
        rank = percentile_rank(pd.Series(matrix[:, positions[source]], index=frame.index),
                               frame["trade_date"], eligible)
        matrix[:, positions[f"rank_{source}"]] = rank.to_numpy(dtype="float32")
    matrix[np.isinf(matrix)] = np.nan
    result = pd.concat([frame[KEYS], pd.DataFrame(matrix, columns=names, copy=False)], axis=1)
    return result, names

"""Join labels after X-only feature calculation; filter supervision after splitting."""
import numpy as np
import pandas as pd

from src.data.clean_data import valid_quote
from src.data.load_data import KEYS


def attach_labels(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if not raw[KEYS].equals(features[KEYS]):
        raise ValueError("Feature keys/order changed during feature calculation.")
    return pd.concat([features, raw[["y_ret_1d"]], valid_quote(raw).rename("quote_valid")], axis=1)


def training_arrays(train: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, np.ndarray, dict]:
    mask = np.isfinite(train["y_ret_1d"].to_numpy()) & train["quote_valid"].to_numpy()
    stats = {"candidate_rows": len(train), "used_rows": int(mask.sum()),
             "dropped_rows": int((~mask).sum())}
    if not mask.any():
        raise ValueError("No valid supervised training samples.")
    return train.loc[mask, columns], train.loc[mask, "y_ret_1d"].to_numpy(dtype="float64"), stats

"""Read original CSV without losing stock codes or label precision."""
from pathlib import Path

import pandas as pd

KEYS = ["ts_code", "trade_date"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
VALUE_COLUMNS = PRICE_COLUMNS + ["vol", "amount"]
X_COLUMNS = VALUE_COLUMNS + ["flag_limit_up", "flag_limit_down"]


def load_training_data(path: str | Path) -> pd.DataFrame:
    dtypes = {"ts_code": "category", "trade_date": "int32",
              **{name: "float64" for name in VALUE_COLUMNS + ["y_ret_1d"]},
              "flag_limit_up": "int8", "flag_limit_down": "int8"}
    return pd.read_csv(path, usecols=KEYS + X_COLUMNS + ["y_ret_1d"], dtype=dtypes)

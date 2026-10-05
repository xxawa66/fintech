"""Trailing windows include today and require all observations in the window."""
import pandas as pd

from src.features.price_features import safe_ratio


def technical_features(stock: pd.DataFrame, settings: dict, ret_1: pd.Series) -> dict[str, pd.Series]:
    close = stock["close"]
    values = {f"ma_bias_{w}": safe_ratio(close, close.rolling(w, min_periods=w).mean()) - 1
              for w in settings["ma_windows"]}
    values.update({f"volatility_{w}": ret_1.rolling(w, min_periods=w).std(ddof=1)
                   for w in settings["volatility_windows"]})
    for w in settings["position_windows"]:
        minimum = close.rolling(w, min_periods=w).min()
        maximum = close.rolling(w, min_periods=w).max()
        values[f"price_position_{w}"] = safe_ratio(close - minimum, maximum - minimum)
    return values

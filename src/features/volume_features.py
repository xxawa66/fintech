"""Relative volume/amount features; real zero volume is retained."""
import pandas as pd

from src.features.price_features import safe_ratio


def volume_features(stock: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    vol, amount = stock["vol"], stock["amount"]
    values = {f"vol_ratio_{w}": safe_ratio(vol, vol.rolling(w, min_periods=w).mean())
              for w in settings["volume_windows"]}
    values.update({f"amount_ratio_{w}": safe_ratio(amount, amount.rolling(w, min_periods=w).mean())
                   for w in settings["amount_windows"]})
    values.update({f"vol_change_{w}": safe_ratio(vol, vol.shift(w)) - 1
                   for w in settings["volume_change_windows"]})
    return values

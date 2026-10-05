"""Price formulas for one stock in ascending calendar-row order."""
import pandas as pd


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def price_features(stock: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    close, open_, high, low = (stock[name] for name in ["close", "open", "high", "low"])
    values = {f"ret_{w}": safe_ratio(close, close.shift(w)) - 1
              for w in settings["return_windows"]}
    values.update({"intraday_ret": safe_ratio(close, open_) - 1,
                   "high_low_range": safe_ratio(high, low) - 1,
                   "high_close": safe_ratio(high, close) - 1,
                   "close_low": safe_ratio(close, low) - 1,
                   "open_gap": safe_ratio(open_, close.shift(1)) - 1,
                   "close_position": safe_ratio(close - low, high - low)})
    return values

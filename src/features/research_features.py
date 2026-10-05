"""18 fixed causal research features. Targets are never accessible here."""
import numpy as np
import pandas as pd

from src.features.price_features import safe_ratio


def research_names(settings: dict) -> list[str]:
    names = [name for group in ["T", "V", "R"] for name in settings["groups"][group]]
    if len(names) != 18 or len(set(names)) != 18:
        raise ValueError("Research requires three disjoint groups of six features.")
    if any(len(settings["groups"][g]) != 6 for g in ["T", "V", "R"]):
        raise ValueError("Each research group must contain six features.")
    return names


def stock_research_features(stock: pd.DataFrame, base: dict, settings: dict) -> dict[str, pd.Series]:
    close, vol = stock["close"], stock["vol"]
    ret = base["ret_1"]
    windows = settings["windows"]
    w = windows["path"]
    delta = close.diff()
    efficiency = safe_ratio(close - close.shift(w), delta.abs().rolling(w, min_periods=w).sum())
    rolling = ret.rolling(w, min_periods=w)
    stdev = rolling.std(ddof=1)
    values = {
        f"trend_efficiency_{w}": efficiency,
        f"up_ratio_{w}": ret.gt(0).astype("float64").where(ret.notna()).rolling(w, min_periods=w).mean(),
        f"downside_rms_{w}": np.sqrt(ret.clip(upper=0).pow(2).rolling(w, min_periods=w).mean()),
        f"return_skew_{w}": rolling.skew().where(stdev > 0),
        f"return_kurt_{w}": rolling.kurt().where(stdev > 0),
    }
    w = windows["correlation"]
    logvol = np.log1p(vol)
    def correlation(left, right):
        lr, rr = left.rolling(w, min_periods=w), right.rolling(w, min_periods=w)
        corr = lr.corr(right)
        return corr.where((lr.std(ddof=1) > 0) & (rr.std(ddof=1) > 0) & np.isfinite(corr)).clip(-1, 1)
    values[f"corr_ret_logvol_{w}"] = correlation(ret, logvol)
    values[f"corr_price_vol_{w}"] = correlation(close, vol)
    for w in windows["signed_volume"]:
        signed = np.sign(ret) * vol
        values[f"signed_vol_ratio_{w}"] = safe_ratio(
            signed.rolling(w, min_periods=w).sum(), vol.rolling(w, min_periods=w).sum())
    w = windows["volume_cv"]
    values[f"volume_cv_{w}"] = safe_ratio(vol.rolling(w, min_periods=w).std(ddof=1),
                                          vol.rolling(w, min_periods=w).mean())
    for short, long in windows["volatility_pairs"]:
        values[f"volatility_ratio_{short}_{long}"] = safe_ratio(
            base[f"volatility_{short}"], base[f"volatility_{long}"])
    w = windows["risk_return"]
    values[f"risk_adj_ret_{w}"] = safe_ratio(base[f"ret_{w}"], base[f"volatility_{w}"] * np.sqrt(w))
    previous = close.shift(1)
    parts = pd.concat([stock["high"] - stock["low"],
                       (stock["high"] - previous).abs(), (stock["low"] - previous).abs()], axis=1)
    # A missing previous close must propagate rather than being skipped by max().
    tr = safe_ratio(parts.max(axis=1, skipna=False), previous)
    values["true_range_relative"] = tr
    short, long = windows["atr"]
    values[f"atr_ratio_{short}_{long}"] = safe_ratio(
        tr.rolling(short, min_periods=short).mean(), tr.rolling(long, min_periods=long).mean())
    expected = {name for name in research_names(settings) if not name.startswith("rank_")}
    if set(values) != expected:
        raise ValueError("Research windows and configured column names disagree.")
    return values

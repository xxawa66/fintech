"""Rank only current X; labels must never select the ranking universe."""
import pandas as pd


def percentile_rank(values: pd.Series, dates: pd.Series, eligible: pd.Series) -> pd.Series:
    return values.where(eligible).groupby(dates, sort=False, observed=True).rank(
        pct=True, method="average", na_option="keep")

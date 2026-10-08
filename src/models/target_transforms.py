"""Training-only daily targets, with explicit keys and reproducible statistics."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.data.load_data import KEYS


def array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def daily_target(training: pd.DataFrame, spec: dict, valid_start: int):
    """Return targets in the exact input order; never accept validation labels.

    Input rows have already passed the boundary purge and quote/label mask.
    No statistics from another date, validation period or predictor enter here.
    """
    if list(training.columns) != KEYS + ["y_ret_1d"]:
        raise ValueError("Targets require the actual supervised keys and raw y only.")
    if (training.empty or training[KEYS].isna().any().any()
            or training.duplicated(KEYS).any()
            or not (training.trade_date < valid_start).all()):
        raise ValueError("Missing/duplicate keys or validation dates in target construction.")
    raw = training.y_ret_1d.to_numpy(dtype="float64")
    if not np.isfinite(raw).all():
        raise ValueError("Missing labels cannot become ranked or imputed targets.")
    dates = training.trade_date.to_numpy()
    series = pd.Series(raw)
    groups = series.groupby(dates, sort=True)
    daily = groups.agg(n="size", raw_min="min", raw_max="max", raw_mean="mean")
    daily["raw_std"] = groups.std(ddof=0)
    method = spec["transform"]
    if method == "raw":
        values = raw.copy()
    elif method == "daily_rank":
        if spec != {"transform": "daily_rank", "ties": "average"}:
            raise ValueError("Daily rank ties must use average percentile ranks.")
        values = groups.rank(method="average", pct=True).to_numpy(dtype="float64")
        means = pd.Series(values).groupby(dates).mean().to_numpy()
        expected = (daily.n.to_numpy() + 1) / (2 * daily.n.to_numpy())
        if (not np.allclose(means, expected, atol=1e-12, rtol=0)
                or not ((values > 0) & (values <= 1)).all()):
            raise ValueError("Daily rank target invariant failed.")
    elif method in {"daily_winsor", "daily_winsor_zscore"}:
        if spec.get("quantiles") != [0.01, 0.99] or spec.get("interpolation") != "linear":
            raise ValueError("Winsorization is fixed at daily linear 1%/99% quantiles.")
        daily["q01"] = groups.quantile(.01, interpolation="linear")
        daily["q99"] = groups.quantile(.99, interpolation="linear")
        lo = pd.Series(dates).map(daily.q01).to_numpy()
        hi = pd.Series(dates).map(daily.q99).to_numpy()
        clipped = np.clip(raw, lo, hi)
        daily["clipped_rows"] = pd.Series(clipped != raw).groupby(dates).sum()
        clip_groups = pd.Series(clipped).groupby(dates, sort=True)
        daily["winsor_mean"] = clip_groups.mean()
        daily["winsor_std"] = clip_groups.std(ddof=0)
        if method == "daily_winsor":
            values = clipped
        else:
            if spec.get("ddof") != 0 or spec.get("zero_std") != 0.:
                raise ValueError("Daily z-score requires ddof=0 and zero-std output 0.")
            center = pd.Series(dates).map(daily.winsor_mean).to_numpy()
            scale = pd.Series(dates).map(daily.winsor_std).to_numpy()
            values = np.divide(clipped - center, scale, out=np.zeros_like(raw), where=scale > 0)
            transformed = pd.Series(values).groupby(dates, sort=True)
            nonzero = daily.winsor_std.to_numpy() > 0
            if (not np.allclose(transformed.mean().to_numpy(), 0, atol=1e-10, rtol=0)
                    or not np.allclose(transformed.std(ddof=0).to_numpy()[nonzero], 1,
                                       atol=1e-10, rtol=0)
                    or np.any(values[scale == 0] != 0)):
                raise ValueError("Daily standardized target invariant failed.")
    else:
        raise ValueError(f"Unknown approved target transform: {method}")
    if not np.isfinite(values).all():
        raise ValueError("Target transform produced non-finite values.")
    transformed = pd.Series(values).groupby(dates, sort=True)
    daily["target_min"] = transformed.min()
    daily["target_max"] = transformed.max()
    daily["target_mean"] = transformed.mean()
    daily["target_std"] = transformed.std(ddof=0)
    daily.index.name = "trade_date"
    daily = daily.reset_index()
    summary = {"rows": len(raw), "days": len(daily), "stocks": int(training.ts_code.nunique()),
               "start": int(dates.min()), "end": int(dates.max()), "spec": spec,
               "raw_digest": array_digest(raw), "target_digest": array_digest(values),
               "raw_mean": float(raw.mean()), "raw_std": float(raw.std(ddof=0)),
               "target_mean": float(values.mean()), "target_std": float(values.std(ddof=0)),
               "target_min": float(values.min()), "target_max": float(values.max()),
               "target_q01": float(np.quantile(values, .01)),
               "target_q99": float(np.quantile(values, .99)),
               "zero_std_days": int((daily.target_std == 0).sum()),
               "clipped_rows": int(daily.get("clipped_rows", pd.Series([0])).sum()),
               "all_finite": True, "training_only": True, "input_order_preserved": True}
    return values, daily, summary

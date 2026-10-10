"""Paired date-block uncertainty on already realised continuous-path statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_indices(n, repetitions=1000, block_length=20, seed=42):
    if n < 1 or repetitions < 1 or block_length < 1:
        raise ValueError("Block sampling needs positive sizes.")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n/block_length))))
    return ((starts[:, :, None]+np.arange(block_length)) % n).reshape(repetitions, -1)[:, :n]


def mean_draws(values, indices, mask=None):
    values = np.asarray(values, dtype="float64")
    good = np.isfinite(values)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    sampled = np.where(good, values, 0.)[indices]
    counts = good[indices].sum(axis=1)
    return np.divide(sampled.sum(axis=1), counts,
                     out=np.full(len(indices), np.nan), where=counts > 0)


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"lower": np.nan, "upper": np.nan, "positive_fraction": np.nan, "valid_draws": 0}
    return {"lower": float(np.quantile(values, .025)), "upper": float(np.quantile(values, .975)),
            "positive_fraction": float((values > 0).mean()), "valid_draws": len(values)}


def paired_draws(left, right, indices, mask=None):
    if not np.array_equal(left.trade_date.to_numpy(), right.trade_date.to_numpy()):
        raise ValueError("Paired bootstrap dates differ.")
    delta = {k: mean_draws(left[k], indices, mask)-mean_draws(right[k], indices, mask)
             for k in ["ic", "top_excess", "turnover"]}
    return {"annual_excess": 252*delta["top_excess"],
            "final_score": .4*delta["ic"]+.3*252*delta["top_excess"]-.3*delta["turnover"]}


def scalar_interval(values, *, block_length=20, repetitions=1000, seed=42, mask=None):
    values = np.asarray(values, dtype="float64")
    ix = block_indices(len(values), repetitions, block_length, seed)
    good = np.isfinite(values)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    point = float(values[good].mean()) if good.any() else np.nan
    return {"mean": point, "days": int(good.sum()), **interval(mean_draws(values, ix, mask))}


def comparisons(daily, settings):
    rows, pools = [], {}
    for (model, layer, year), frame in daily.groupby(["model", "layer", "year"], sort=True):
        if model == "T030":
            continue
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        base = daily[(daily.model == "T030") & (daily.layer == layer) & (daily.year == year)]
        base = base.sort_values("trade_date").reset_index(drop=True)
        ix = block_indices(len(frame), settings["repetitions"], settings["block_length"], settings["seed"]+int(year))
        draws = paired_draws(frame, base, ix)
        pools.setdefault((model, layer), {})[int(year)] = draws
        for metric, values in draws.items():
            column, scale = ("top_excess", 252) if metric == "annual_excess" else (None, None)
            if column:
                point = scale*(frame[column].mean()-base[column].mean())
            else:
                point = (.4*(frame.ic.mean()-base.ic.mean())+75.6*(frame.top_excess.mean()-base.top_excess.mean())
                         -.3*(frame.turnover.mean()-base.turnover.mean()))
            rows.append({"model": model, "layer": layer, "period": str(year), "metric": metric,
                         "delta": point, **interval(values)})
    for (model, layer), years in pools.items():
        periods = {"CV2021_2023": [2021, 2022, 2023]}
        if len(years) == 4:
            periods["Historical2021_2024"] = [2021, 2022, 2023, 2024]
        for period, requested in periods.items():
            if not all(y in years for y in requested):
                continue
            for metric in ["final_score", "annual_excess"]:
                subset = [r for r in rows if r["model"] == model and r["layer"] == layer
                          and r["period"] in [str(y) for y in requested] and r["metric"] == metric]
                samples = np.mean([years[y][metric] for y in requested], axis=0)
                rows.append({"model": model, "layer": layer, "period": period, "metric": metric,
                             "delta": np.mean([r["delta"] for r in subset]), **interval(samples)})
    return pd.DataFrame(rows)

"""Causal S004 controllers; no target column or target-validity mask is accepted.

The legacy band remains in turnover.py. The minimal encoding uses its *actual*
eligible Top set, including its historical tie policy, so it isolates encoding.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.official_eval import TOP_MIN_VALID
from src.evaluation.turnover import band_scores

KNOWN_COLUMNS = ["ts_code", "trade_date", "pred", "flag_limit_up"]


@dataclass
class ControllerInput:
    known: pd.DataFrame
    dates: np.ndarray
    stocks: np.ndarray
    raw: np.ndarray
    eligible: np.ndarray
    rank: np.ndarray
    raw_top: np.ndarray
    legacy_cache: dict = field(default_factory=dict)


@dataclass
class ControllerResult:
    scores: np.ndarray
    intended_top: np.ndarray
    operations: pd.DataFrame
    encoding: str


def official_top(data: ControllerInput, scores: np.ndarray) -> np.ndarray:
    """Exactly reproduce pandas' official descending sort on the canonical rows."""
    if scores.shape != data.raw.shape or not np.isfinite(scores).all():
        raise ValueError("Controller scores have changed shape or are not finite.")
    top = np.zeros_like(data.eligible)
    for t, ok in enumerate(data.eligible):
        indices = np.flatnonzero(ok)
        if len(indices) < TOP_MIN_VALID:
            continue
        order = pd.Series(scores[t, indices], index=indices).sort_values(ascending=False).index
        top[t, order[:max(len(indices) // 10, 1)]] = True
    return top


def prepare_controller_input(known: pd.DataFrame) -> ControllerInput:
    if set(known) != set(KNOWN_COLUMNS):
        raise ValueError("Controller input must contain only keys, pred and flag_limit_up.")
    frame = known[KNOWN_COLUMNS].astype({"ts_code": "str", "trade_date": "int64"}).sort_values(
        ["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    if frame.empty or frame[["ts_code", "trade_date"]].isna().any().any():
        raise ValueError("Missing controller keys.")
    if frame.duplicated(["ts_code", "trade_date"]).any() or not np.isfinite(frame.pred).all():
        raise ValueError("Duplicate keys or non-finite input predictions.")
    if not frame.flag_limit_up.isin([0, 1]).all():
        raise ValueError("Limit-up flags must be known binary X values.")
    wide = frame.pivot(index="trade_date", columns="ts_code", values="pred").sort_index()
    if wide.isna().any().any() or len(frame) != wide.size:
        raise ValueError("This study requires the complete balanced stock-date panel.")
    flags = frame.flag_limit_up.to_numpy().reshape(wide.shape)
    data = ControllerInput(frame, wide.index.to_numpy(), wide.columns.to_numpy(dtype=str),
                           wide.to_numpy(dtype="float64"), flags == 0,
                           wide.rank(axis=1, method="average", pct=True).to_numpy(),
                           np.zeros(wide.shape, dtype=bool))
    data.raw_top = official_top(data, data.raw)
    return data


def _ordered(indices: np.ndarray, row: np.ndarray) -> np.ndarray:
    # Column indices are sorted ts_code; smaller code wins an exact tie.
    return indices[np.lexsort((indices, -row[indices]))]


def _minimal_encoding(data: ControllerInput, top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = data.rank.copy()
    shifts = np.zeros(len(data.dates))
    for t in range(len(data.dates)):
        selected = np.flatnonzero(top[t])
        dropped = np.flatnonzero(data.eligible[t] & ~top[t])
        if not len(selected) or not len(dropped):
            continue
        gap = float(data.rank[t, dropped].max() - data.rank[t, selected].min())
        shifts[t] = max(0., gap + 1e-9)
        scores[t, dropped] -= shifts[t]
    if not np.array_equal(official_top(data, scores), top):
        raise ValueError("Encoded scores do not reproduce the intended eligible Top sets.")
    return scores, shifts


def _legacy(data: ControllerInput, q: float) -> tuple[np.ndarray, np.ndarray]:
    if q not in data.legacy_cache:
        values = band_scores(data.known, q).to_numpy(dtype="float64").reshape(data.raw.shape)
        data.legacy_cache[q] = (values, official_top(data, values))
    return data.legacy_cache[q]


def transform(data: ControllerInput, spec: dict) -> ControllerResult:
    method = spec["method"]
    operations = pd.DataFrame({"trade_date": data.dates})
    if method == "raw" or (method == "bonus" and spec["beta"] == 0):
        return ControllerResult(data.raw.copy(), data.raw_top.copy(), operations, "original_pred")
    if method in {"band", "band_minimal"}:
        q = float(spec["keep_q"])
        if not 0 <= q <= 1:
            raise ValueError("keep_q must lie in [0, 1].")
        legacy, top = _legacy(data, q)
        if method == "band":
            return ControllerResult(legacy.copy(), top.copy(), operations, "legacy_band")
        scores, shifts = _minimal_encoding(data, top)
        operations["encoding_shift"] = shifts
        operations["legacy_top_equal"] = True
        return ControllerResult(scores, top.copy(), operations, "daily_minimal_shift")
    if method == "bonus":
        beta = float(spec["beta"])
        if not 0 < beta <= 1:
            raise ValueError("Positive bonus beta must lie in (0, 1].")
        scores = data.rank.copy()
        top = np.zeros_like(data.eligible)
        previous = np.zeros(len(data.stocks), dtype=bool)
        for t, ok in enumerate(data.eligible):
            adjusted = data.rank[t].copy()
            adjusted[previous & ok] += beta
            # A strict monotone ordinal representation preserves all unequal u
            # relationships; exact u ties are broken by ts_code over all stocks.
            order = _ordered(np.arange(len(data.stocks)), adjusted)
            scores[t, order] = np.arange(len(order), 0, -1) / (len(order) + 1.)
            eligible_order = order[ok[order]]
            if len(eligible_order) < TOP_MIN_VALID:
                previous[:] = False
                continue
            top[t, eligible_order[:max(len(eligible_order) // 10, 1)]] = True
            previous = top[t].copy()
        if not np.array_equal(official_top(data, scores), top):
            raise ValueError("Bonus output and intended Top sets disagree.")
        return ControllerResult(scores, top, operations, "strict_adjusted_order_ts_code_ties")
    if method != "gap":
        raise ValueError(f"Unknown controller method: {method}")
    delta, rho = float(spec["delta"]), float(spec["rho"])
    if not 0 <= delta <= 1 or not 0 < rho <= 1:
        raise ValueError("Gap delta/rho are outside their allowed ranges.")
    top = np.zeros_like(data.eligible)
    previous = np.zeros(len(data.stocks), dtype=bool)
    rows = []
    for t, ok in enumerate(data.eligible):
        indices = np.flatnonzero(ok)
        forced = int((previous & ~ok).sum())
        row = {"trade_date": data.dates[t], "active_swap_count": 0,
               "forced_ineligible_count": forced, "forced_resize_count": 0,
               "capacity_fill_count": 0, "active_swap_cap": 0}
        if len(indices) < TOP_MIN_VALID:
            previous[:] = False
            rows.append(row)
            continue
        size = max(len(indices) // 10, 1)
        order = _ordered(indices, data.rank[t])
        survivors = _ordered(np.flatnonzero(previous & ok), data.rank[t])
        row["forced_resize_count"] = max(0, len(survivors) - size)
        chosen = set(survivors[:size].tolist())
        for i in order:
            if len(chosen) >= size:
                break
            if int(i) not in chosen:
                chosen.add(int(i))
                row["capacity_fill_count"] += 1
        old = [int(i) for i in survivors[:size]]
        cap = int(np.floor(rho * size))
        row["active_swap_cap"] = cap
        newcomers = [int(i) for i in order if int(i) not in chosen and not previous[i]]
        for incoming in newcomers:
            if not old or row["active_swap_count"] >= cap:
                break
            outgoing = old[-1]
            if data.rank[t, incoming] - data.rank[t, outgoing] <= delta:
                break
            chosen.remove(outgoing)
            chosen.add(incoming)
            old.pop()
            row["active_swap_count"] += 1
        top[t, list(chosen)] = True
        previous = top[t].copy()
        rows.append(row)
    scores, shifts = _minimal_encoding(data, top)
    operations = pd.DataFrame(rows)
    operations["encoding_shift"] = shifts
    return ControllerResult(scores, top, operations, "daily_minimal_shift")


def order_fingerprint(scores: np.ndarray) -> str:
    """Same ranks and tie groups on canonical rows imply the same scoring order."""
    twice = (pd.DataFrame(scores).rank(axis=1, method="average").to_numpy() * 2).astype("<u4")
    return hashlib.sha256(twice.tobytes(order="C")).hexdigest()


def holdings_diagnostics(data: ControllerInput, scores: np.ndarray, operations: pd.DataFrame,
                         intended: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
    """Describe the Top sets decoded from actual saved predictions, without y."""
    top = official_top(data, scores)
    if not np.array_equal(top, intended):
        raise ValueError("Saved predictions changed the controller Top sets.")
    n = len(data.stocks)
    streak = np.zeros(n, dtype=int)
    longest = np.zeros(n, dtype=int)
    starts = np.zeros(n, dtype=int)
    rank_sum = np.zeros(n)
    closed_lengths = []
    previous = np.zeros(n, dtype=bool)
    rows = []
    for t, current in enumerate(top):
        ended = previous & ~current
        closed_lengths.extend(streak[ended].tolist())
        starts[current & ~previous] += 1
        streak = np.where(current, streak + 1, 0)
        longest = np.maximum(longest, streak)
        rank_sum += np.where(current, data.rank[t], 0.)
        unavailable = int((previous & ~data.eligible[t]).sum())
        removed = int((previous & ~current).sum())
        selected_ranks = data.rank[t, current]
        raw_size = int(data.raw_top[t].sum())
        overlap = int((current & data.raw_top[t]).sum())
        rows.append({"trade_date": data.dates[t], "n_eligible": int(data.eligible[t].sum()),
                     "n_top": int(current.sum()), "entered": int((current & ~previous).sum()),
                     "exited": removed, "forced_ineligible_exits": unavailable,
                     "eligible_exits": removed - unavailable,
                     "raw_top_retained_fraction": overlap / raw_size if raw_size else np.nan,
                     "selected_raw_rank_mean": selected_ranks.mean() if len(selected_ranks) else np.nan,
                     "selected_raw_rank_median": np.median(selected_ranks) if len(selected_ranks) else np.nan,
                     "selected_raw_rank_min": selected_ranks.min() if len(selected_ranks) else np.nan,
                     "selected_below_top20_fraction": (selected_ranks < .8).mean() if len(selected_ranks) else np.nan,
                     "mean_current_streak": streak[current].mean() if current.any() else np.nan,
                     "max_current_streak": int(streak.max()),
                     "eligible_score_ties": int(pd.Series(scores[t, data.eligible[t]]).duplicated().sum())})
        previous = current.copy()
    censored = streak[previous].tolist()
    days = top.sum(axis=0)
    stock = pd.DataFrame({"ts_code": data.stocks, "days_selected": days, "spell_count": starts,
                          "max_consecutive_days": longest,
                          "mean_raw_rank_when_selected": np.divide(rank_sum, days,
                              out=np.full(n, np.nan), where=days > 0)})
    lengths = np.asarray(closed_lengths + censored, dtype=int)
    counts = np.bincount(lengths, minlength=len(data.dates) + 1)
    censored_counts = np.bincount(censored, minlength=len(data.dates) + 1)
    histogram = pd.DataFrame({"length_trading_days": np.arange(1, len(counts)),
                              "spell_count": counts[1:], "right_censored_count": censored_counts[1:]})
    daily = pd.DataFrame(rows).merge(operations, on="trade_date", how="left", validate="one_to_one")
    summary = {"stocks_ever_selected": int((days > 0).sum()), "spell_count": len(lengths),
               "right_censored_spells": len(censored),
               "mean_spell_days": float(lengths.mean()) if len(lengths) else 0.,
               "median_spell_days": float(np.median(lengths)) if len(lengths) else 0.,
               "p95_spell_days": float(np.quantile(lengths, .95)) if len(lengths) else 0.,
               "max_spell_days": int(longest.max()),
               "mean_selected_raw_rank": float(daily.selected_raw_rank_mean.mean()),
               "mean_raw_top_retained_fraction": float(daily.raw_top_retained_fraction.mean()),
               "mean_selected_below_top20_fraction": float(daily.selected_below_top20_fraction.mean()),
               "eligible_exits": int(daily.eligible_exits.sum()),
               "forced_ineligible_exits": int(daily.forced_ineligible_exits.sum()),
               "top_set_sha256": hashlib.sha256(top.tobytes()).hexdigest(),
               "saved_top_matches_intended": True,
               "spell_note": "Fold-end spells are right censored; all lengths use trading observations."}
    if "active_swap_count" in daily:
        if (daily.active_swap_count > daily.active_swap_cap).any():
            raise ValueError("An active replacement exceeded the predeclared daily cap.")
        summary["active_swaps"] = int(daily.active_swap_count.sum())
        summary["forced_resize_exits"] = int(daily.forced_resize_count.sum())
        summary["capacity_fills"] = int(daily.capacity_fill_count.sum())
    return daily, stock, histogram, summary, top

"""Descriptive S009 tables from fixed complete predictions and realised labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def sorted_indices(values, eligible):
    indices = np.flatnonzero(eligible)
    return pd.Series(values[indices], index=indices).sort_values(ascending=False).index.to_numpy()


def top_mask(values, eligible):
    result = np.zeros(len(values), dtype=bool)
    ix = sorted_indices(values, eligible)
    if len(ix) >= 100:
        result[ix[:max(len(ix)//10, 1)]] = True
    return result


def top_matrix(values, eligible):
    return np.asarray([top_mask(v, ok) for v, ok in zip(values, eligible)])


def mean_or_nan(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def cohort_row(day, pool, group, selected, y, market, prev, raw_rank, valid, total):
    n = int(selected.sum())
    known = selected & np.isfinite(y)
    n_labels = int(known.sum())
    mean = mean_or_nan(y[known])
    return {"trade_date": day, "pool": pool, "cohort": group, "n": n, "n_labels": n_labels,
            "label_coverage": n_labels/n if n else np.nan, "mean_ret": mean,
            "excess": mean-market, "weight": n/total if total else np.nan,
            "ret_contribution": (n/total*mean if n else 0.) if pool == "return" else np.nan,
            "excess_contribution": (n/total*(mean-market) if n else 0.) if pool == "return" else np.nan,
            "mean_raw_rank": mean_or_nan(raw_rank[selected]),
            "invalid_quote_n": int((selected & ~valid).sum()), "first_day": prev is None}


def trace_tables(context, raw, scores, daily, settings):
    dates = context.trade_date.drop_duplicates().to_numpy()
    stocks = context[context.trade_date == dates[0]].ts_code.to_numpy(dtype=str)
    shape = (len(dates), len(stocks))
    if len(stocks) != 4650 or len(context) != np.prod(shape):
        raise ValueError("Diagnostics require the complete 4650-stock daily panel.")
    y = context.y_ret_1d.to_numpy().reshape(shape)
    eligible = context.flag_limit_up.to_numpy().reshape(shape) == 0
    quote = context.quote_valid.to_numpy().reshape(shape)
    raw_rank = np.asarray([rankdata(v, method="average")/len(v) for v in raw])
    ret_top, turn_top = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    rows = {k: [] for k in ["bins_daily", "cumulative_daily", "boundary_daily", "ties_daily",
                           "cohorts_daily", "holdings_daily", "subgroups_daily", "bin_medians"]}
    pooled = {i: {} for i in range(len(settings["bin_edges"])-1)}
    previous = None
    streak = np.zeros(len(stocks), dtype=int)
    closed = []
    for t, day in enumerate(dates):
        label_ok = np.isfinite(y[t])
        return_ok = eligible[t] & label_ok
        order = sorted_indices(scores[t], return_ok)
        turn_order = sorted_indices(scores[t], eligible[t])
        if len(order) < 100 or len(turn_order) < 100:
            raise ValueError("Unexpected skipped day in frozen complete source.")
        k, kt = max(len(order)//10, 1), max(len(turn_order)//10, 1)
        ret_top[t, order[:k]], turn_top[t, turn_order[:kt]] = True, True
        market = float(y[t, order].mean())
        true_rank = rankdata(y[t, order], method="average")/len(order)
        events10 = np.zeros(len(stocks), dtype=bool)
        events20 = events10.copy()
        events10[order] = true_rank > .9
        events20[order] = true_rank > .8
        target_allowed = eligible[t] & quote[t] & label_ok
        true_allowed = np.zeros(len(stocks), dtype=bool)
        ix_allowed = np.flatnonzero(target_allowed)
        if len(ix_allowed):
            true_allowed[ix_allowed] = rankdata(y[t, ix_allowed], method="average")/len(ix_allowed) > .9
        inference_order = np.flatnonzero(eligible[t] & quote[t])
        # Fixed causal tie policy for the prospective S010 pool, not official scoring.
        inference_order = inference_order[np.lexsort((inference_order, -raw[t, inference_order]))]
        recall = {}
        for p in [.2, .3]:
            chosen = inference_order[:max(int(len(inference_order)*p), 1)]
            key = f"{int(p*100)}"
            recall[f"inference_recall{key}"] = true_allowed[chosen].sum()/true_allowed.sum() if true_allowed.any() else np.nan
            recall[f"inference_candidate{key}_n"] = len(chosen)
            recall[f"inference_candidate{key}_label_coverage"] = label_ok[chosen].mean() if len(chosen) else np.nan
        bin_returns = []
        for b, (lower, upper) in enumerate(zip(settings["bin_edges"], settings["bin_edges"][1:])):
            start, end = int(len(order)*lower), int(len(order)*upper)
            selected = order[start:end]
            values = y[t, selected]
            value = mean_or_nan(values)
            bin_returns.append(value)
            rows["bins_daily"].append({"trade_date": int(day), "bin": b,
                "lower": lower, "upper": upper, "n": len(values), "mean_ret": value,
                "median_ret": float(np.median(values)) if len(values) else np.nan,
                "excess": value-market, "positive_return_fraction": (values > 0).mean() if len(values) else np.nan,
                "start_boundary_tie": bool(start > 0 and scores[t, order[start-1]] == scores[t, order[start]]),
                "end_boundary_tie": bool(end < len(order) and scores[t, order[end-1]] == scores[t, order[end]])})
            pooled[b].setdefault(int(day)//100, []).append(values)
        for p in [.05, .10, .20, .30]:
            chosen = order[:max(int(len(order)*p), 1)]
            rows["cumulative_daily"].append({"trade_date": int(day), "top_fraction": p, "n": len(chosen),
                "mean_ret": float(y[t, chosen].mean()), "excess": float(y[t, chosen].mean()-market),
                "precision_true10": float(events10[chosen].mean()),
                "recall_true10": events10[chosen].sum()/events10.sum() if events10.any() else np.nan,
                "precision_true20": float(events20[chosen].mean()),
                "recall_true20": events20[chosen].sum()/events20.sum() if events20.any() else np.nan})
        near = order[:max(len(order)//5, 1)]
        rest = order[max(len(order)//5, 1):]
        def region_ic(ix):
            if len(ix) < 30 or np.unique(scores[t, ix]).size < 2:
                return np.nan
            return float(spearmanr(scores[t, ix], y[t, ix]).statistic)
        rows["boundary_daily"].append({"trade_date": int(day), "gap_5_10_minus_10_15": bin_returns[1]-bin_returns[2],
            "top20_ic": region_ic(near), "remaining80_ic": region_ic(rest),
            "precision10": float(events10[order[:k]].mean()),
            "recall20_return_pool": events10[order[:len(order)//5]].sum()/events10.sum() if events10.any() else np.nan,
            "true10_n": int(events10.sum()), "true20_n": int(events20.sum()), **recall,
            "bins_monotone": bool(np.all(np.diff(bin_returns) <= 0))})
        unique, counts = np.unique(scores[t], return_counts=True)
        boundary = scores[t, order[k-1]]
        tied = return_ok & (scores[t] == boundary)
        before = int((return_ok & (scores[t] > boundary)).sum())
        after = before+int(tied.sum())
        zeros = scores[t] == 0
        fallback = ~quote[t] & (raw[t] == 0)
        rows["ties_daily"].append({"trade_date": int(day), "unique_scores": len(unique),
            "duplicate_count": len(stocks)-len(unique), "tied_stock_count": int(counts[counts > 1].sum()),
            "largest_tie": int(counts.max()), "largest_tie_value": float(unique[np.argmax(counts)]),
            "boundary_tie_n": int(tied.sum()), "boundary_tie_start": before,
            "boundary_tie_end": after, "boundary_crossed": before < k < after,
            "return_top_n": k, "turn_top_n": kt, "zero_score_n": int(zeros.sum()),
            "zero_rank_mean": mean_or_nan(rankdata(scores[t], method="average")[zeros]/len(stocks)),
            "raw_valid_quote_zero_n": int((quote[t] & (raw[t] == 0)).sum()),
            "fallback_zero_n": int(fallback.sum()), "fallback_in_turn_top": int((fallback & turn_top[t]).sum()),
            "fallback_in_return_top": int((fallback & ret_top[t]).sum()),
            "return_top_quote_invalid_n": int((ret_top[t] & ~quote[t]).sum()),
            "return_turn_overlap": int((ret_top[t] & turn_top[t]).sum()),
            "return_turn_jaccard": (ret_top[t] & turn_top[t]).sum()/(ret_top[t] | turn_top[t]).sum()})
        current = turn_top[t]
        if previous is not None:
            closed.extend(streak[previous & ~current].tolist())
        streak = np.where(current, streak+1, 0)
        old = np.zeros(len(stocks), dtype=bool) if previous is None else previous
        cohorts = [("initial", current)] if previous is None else [("retained", current & old), ("entered", current & ~old)]
        for group, selected in cohorts:
            rows["cohorts_daily"].append(cohort_row(int(day), "turn", group, selected, y[t], market,
                                                   previous, raw_rank[t], quote[t], kt))
        cohorts = [("initial", ret_top[t])] if previous is None else [("retained", ret_top[t] & old), ("entered", ret_top[t] & ~old)]
        pieces = []
        for group, selected in cohorts:
            row = cohort_row(int(day), "return", group, selected, y[t], market, previous, raw_rank[t], quote[t], k)
            rows["cohorts_daily"].append(row); pieces.append(row)
        recovered = sum(v["ret_contribution"] for v in pieces)
        realised = float(y[t, order[:k]].mean())
        if abs(recovered-realised) > 1e-12 or abs(sum(v["excess_contribution"] for v in pieces)-(realised-market)) > 1e-12:
            raise ValueError("Return-cohort contributions do not reproduce official Top.")
        if abs(realised-daily.iloc[t].top1_ret) > 1e-12:
            raise ValueError("Actual Top set differs from official daily metrics.")
        exited = old & ~current
        rank_eligible = np.full(len(stocks), np.nan)
        ix_elig = np.flatnonzero(eligible[t])
        rank_eligible[ix_elig] = rankdata(raw[t, ix_elig], method="average")/len(ix_elig)
        rows["holdings_daily"].append({"trade_date": int(day), "selected_n": kt,
            "entered_n": int((current & ~old).sum()), "retained_n": int((current & old).sum()),
            "forced_ineligible_exits": int((old & ~eligible[t]).sum()),
            "eligible_exits": int((exited & eligible[t]).sum()),
            "below_q_exits": int((exited & eligible[t] & (rank_eligible < settings['keep_q'])).sum()),
            "resize_exits_minimum": max(int((old & eligible[t]).sum())-kt, 0),
            "mean_streak": mean_or_nan(streak[current]), "max_streak": int(streak.max()),
            "mean_raw_rank": mean_or_nan(raw_rank[t, current]),
            "raw_top_overlap": int((current & top_mask(raw[t], eligible[t])).sum())/kt,
            "return_recovery_difference": abs(recovered-realised), "first_day": previous is None})
        for dimension in ["liquidity_group", "stock_vol_group"]:
            groups = context[dimension].to_numpy().reshape(shape)[t]
            for group in range(5):
                selected = ret_top[t] & (groups == group)
                pool = return_ok & (groups == group)
                if not pool.any():
                    continue
                mean = mean_or_nan(y[t, selected])
                rows["subgroups_daily"].append({"trade_date": int(day), "dimension": dimension,
                    "group": group, "n_pool": int(pool.sum()), "n_top": int(selected.sum()),
                    "mean_ret": mean, "global_market_excess": mean-market,
                    "subgroup_market_ret": mean_or_nan(y[t, pool]),
                    "within_subgroup_excess": mean-mean_or_nan(y[t, pool])})
        previous = current.copy()
    for b, months in pooled.items():
        year = int(dates[0])//10000
        sets = {str(m): parts for m, parts in months.items()}
        sets[str(year)] = [v for parts in months.values() for v in parts]
        for quarter in range(1, 5):
            sets[f"{year}Q{quarter}"] = [v for month, parts in months.items()
                                        if (month % 100-1)//3+1 == quarter for v in parts]
        for period, parts in sets.items():
            values = np.concatenate(parts) if parts else np.array([])
            rows["bin_medians"].append({"period": period, "bin": b, "pooled_median_ret": float(np.median(values)) if len(values) else np.nan})
    lengths = np.asarray(closed+streak[previous].tolist(), dtype=int)
    rows["spell_histogram"] = pd.DataFrame({"length": np.arange(1, len(dates)+1),
        "spells": np.bincount(lengths, minlength=len(dates)+1)[1:],
        "right_censored": np.bincount(streak[previous], minlength=len(dates)+1)[1:]})
    rows = {name: pd.DataFrame(values) for name, values in rows.items()}
    return rows, {"return_top": ret_top, "turn_top": turn_top,
                  "dates": dates, "stocks": stocks}, {"mean_spell_days": float(lengths.mean()),
                  "median_spell_days": float(np.median(lengths)), "p95_spell_days": float(np.quantile(lengths, .95)),
                  "max_spell_days": int(lengths.max()), "spells": len(lengths),
                  "right_censored": int(previous.sum())}


def period_tables(tables, identifiers):
    result = {}
    group_cols = {"bins": ["bin", "lower", "upper"], "cumulative": ["top_fraction"],
        "cohorts": ["pool", "cohort"], "subgroups": ["dimension", "group"]}
    for kind, groups in group_cols.items():
        frame = tables[kind+"_daily"].copy()
        frame["month"] = frame.trade_date//100
        frame["quarter"] = frame.trade_date//10000*10+((frame.trade_date//100 % 100-1)//3+1)
        parts = []
        for period, period_columns in [("year", []), ("month", ["month"]), ("quarter", ["quarter"])]:
            for key, group in frame.groupby(groups+period_columns, dropna=False, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                record = {**identifiers, **dict(zip(groups+period_columns, key)), "period_type": period,
                          "days": int(group.trade_date.nunique())}
                for col in group.select_dtypes(include=["number", "bool"]).columns:
                    if col not in groups+period_columns+["trade_date"]:
                        record[col+"_mean"] = float(group[col].mean())
                value_col = "excess" if "excess" in group else "global_market_excess"
                record["annual_excess"] = 252*group[value_col].mean()
                record["positive_excess_days"] = float((group.loc[group[value_col].notna(), value_col] > 0).mean())
                if kind == "cohorts":
                    calendar = frame
                    for col in period_columns:
                        calendar = calendar[calendar[col] == record[col]]
                    record["period_days"] = calendar.trade_date.nunique()
                    record["annual_ret_contribution"] = 252*group.ret_contribution.sum()/record["period_days"] if record["pool"] == "return" else np.nan
                    record["annual_excess_contribution"] = 252*group.excess_contribution.sum()/record["period_days"] if record["pool"] == "return" else np.nan
                parts.append(record)
        result[kind] = pd.DataFrame(parts)
    medians = tables["bin_medians"].copy()
    bin_table = result["bins"]
    def period_name(row):
        if row.period_type == "year":
            return str(identifiers["year"])
        if row.period_type == "month":
            return str(int(row.month))
        return f"{identifiers['year']}Q{int(row.quarter)%10}"
    bin_table["period"] = bin_table.apply(period_name, axis=1)
    result["bins"] = bin_table.merge(medians, on=["period", "bin"], how="left", validate="one_to_one")
    return result

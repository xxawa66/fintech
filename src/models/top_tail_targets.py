"""S010 targets and complete causal signals, with exact tie-preserving encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.models.target_transforms import array_digest

ARMS = [
    {"id": "B10", "pool": 1., "objective": "binary", "event": .90},
    {"id": "B20", "pool": 1., "objective": "binary", "event": .80},
    {"id": "R20", "pool": .20, "objective": "binary", "event": .90},
    {"id": "R30", "pool": .30, "objective": "binary", "event": .90},
    {"id": "L30", "pool": .30, "objective": "lambdarank", "cuts": [.60, .80, .90, .95]},
]


def panel_shape(frame):
    n = frame.ts_code.nunique()
    days = frame.trade_date.drop_duplicates().to_numpy()
    if (len(frame) != n*len(days) or frame.duplicated(KEYS).any()
            or not np.all(np.diff(frame.trade_date.to_numpy()) >= 0)):
        raise ValueError("A complete canonical panel is required.")
    stocks = frame.ts_code.iloc[:n].to_numpy(dtype=str)
    if not np.array_equal(frame.ts_code.to_numpy(dtype=str), np.tile(stocks, len(days))):
        raise ValueError("Canonical stock order changed across dates.")
    return len(days), n


def causal_candidates(known, fraction):
    """Only X eligibility and past-trained predictions enter the candidate pool."""
    if set(known) != set(KEYS + ['quote_valid', 'flag_limit_up', 'oof_raw']):
        raise ValueError("Candidate selection must not receive labels or label masks.")
    shape = panel_shape(known)
    first = known.oof_raw.to_numpy().reshape(shape)
    eligible = (known.quote_valid & known.flag_limit_up.eq(0)).to_numpy().reshape(shape)
    selected = np.zeros(shape, dtype=bool)
    for t in range(shape[0]):
        positions = np.flatnonzero(eligible[t])
        # Canonical code order is the declared secondary key at exact ties.
        order = positions[np.argsort(-first[t, positions], kind='stable')]
        selected[t, order[:int(np.floor(len(order)*fraction))]] = True
    return selected.ravel()


def training_targets(frame, valid_start):
    """Define labels over the full supervised pool before any candidate filtering."""
    if frame.trade_date.max() >= valid_start:
        raise ValueError("Validation dates entered the supervised target map.")
    allowed = frame.quote_valid & frame.flag_limit_up.eq(0) & np.isfinite(frame.y_ret_1d)
    p = pd.Series(np.nan, index=frame.index, dtype='float64')
    p.loc[allowed] = frame.loc[allowed, 'y_ret_1d'].groupby(frame.loc[allowed, 'trade_date']).rank(
        method='average', pct=True)
    result = frame[KEYS + ['y_ret_1d']].copy()
    result['allowed'] = allowed.to_numpy()
    result['p_y'] = p.to_numpy()
    result['B10'] = np.where(allowed, (p > .90).astype(int), -1).astype('int8')
    result['B20'] = np.where(allowed, (p > .80).astype(int), -1).astype('int8')
    result['L30'] = np.where(allowed, np.sum(p.to_numpy()[:, None] > [.60, .80, .90, .95], axis=1), -1).astype('int8')
    known = frame[KEYS + ['quote_valid', 'flag_limit_up', 'oof_raw']]
    result['candidate20'] = causal_candidates(known, .20)
    result['candidate30'] = causal_candidates(known, .30)
    stats = result.groupby('trade_date', sort=True).agg(rows=('allowed', 'size'),
        supervised=('allowed', 'sum'), candidate20=('candidate20', 'sum'), candidate30=('candidate30', 'sum'))
    selected = result.loc[allowed]
    g = selected.groupby('trade_date')
    stats['B10_positive'] = g.B10.sum(); stats['B20_positive'] = g.B20.sum()
    stats['B10_single_class'] = g.B10.nunique().eq(1); stats['B20_single_class'] = g.B20.nunique().eq(1)
    for label in range(5):
        stats[f'ordinal_{label}'] = selected.L30.eq(label).groupby(selected.trade_date).sum()
    # Equal raw returns on a date must have identical labels.
    if selected.groupby(['trade_date', 'y_ret_1d'])[['B10','B20','L30']].nunique().to_numpy().max() != 1:
        raise ValueError("Equal-return target ties were split.")
    summary = {'supervised_rows': int(allowed.sum()), 'missing_or_disallowed_rows': int((~allowed).sum()),
               'raw_digest': array_digest(selected.y_ret_1d.to_numpy()),
               'percentile_digest': array_digest(selected.p_y.to_numpy()),
               'B10_positive_fraction': float(selected.B10.mean()), 'B20_positive_fraction': float(selected.B20.mean()),
               'target_ties_preserved': True, 'labels_defined_before_candidate_selection': True}
    return result, stats.reset_index(), summary


def complete_signal(known, secondary, arm):
    """Encode the complete native ordering into exact integer average ranks.

    Binary: secondary scores, including original zero quote fallbacks.
    Rerank: (candidate membership, secondary score, first-stage raw score).
    The shared zero-fallback group's rank is subtracted so its raw code stays 0.
    """
    shape = panel_shape(known)
    secondary = np.asarray(secondary, dtype='float64').reshape(shape)
    first = known.oof_raw.to_numpy().reshape(shape)
    quote = known.quote_valid.to_numpy().reshape(shape)
    if not np.isfinite(secondary).all() or not np.isfinite(first).all():
        raise ValueError("Native predictors must be finite.")
    candidate = (causal_candidates(known, arm['pool']).reshape(shape)
                 if arm['pool'] < 1 else quote)
    result = np.empty(shape, dtype='float64')
    for t in range(shape[0]):
        if arm['pool'] == 1:
            values = secondary[t]
            order = np.argsort(values, kind='stable')
            change = np.r_[True, values[order][1:] != values[order][:-1]]
        else:
            group = candidate[t].astype('int8')
            sec = np.where(candidate[t], secondary[t], 0.)
            order = np.lexsort((np.arange(shape[1]), first[t], sec, group))
            change = np.r_[True, (group[order][1:] != group[order][:-1]) |
                (sec[order][1:] != sec[order][:-1]) | (first[t,order][1:] != first[t,order][:-1])]
        starts = np.flatnonzero(change); ends = np.r_[starts[1:], shape[1]]
        ranks = np.empty(shape[1], dtype='float64')
        for start, end in zip(starts, ends):
            ranks[order[start:end]] = start+1+end
        fallback = np.flatnonzero(~quote[t])
        if len(fallback):
            if np.unique(ranks[fallback]).size != 1:
                raise ValueError("Quote fallbacks no longer share their original zero tie.")
            ranks -= ranks[fallback[0]]
        # Integral codes retain every native tuple tie and all strict inequalities.
        if not np.array_equal(ranks, np.round(ranks)) or np.any(ranks[order][1:] < ranks[order][:-1]):
            raise ValueError("Native signal coding is not monotone.")
        result[t] = ranks
    return result.ravel(), candidate.ravel()

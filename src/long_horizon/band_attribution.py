"""探针：把 band 剔除归因从「22 个交易日」扩到四折全窗口。

统计在选池方案下，band 每天的剔除到底由什么触发：
  涨停（flag_limit_up 或次日不可交易） / 排名分位跌穿 keep_q / 其他
并给出持仓名单的漂移率（首日名单在期末还剩多少）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.turnover import band_scores
from src.long_horizon.rerank_eval import _fill, pool_rerank, read_valid_panel
from src.long_horizon.run import get_optuna_folds
from src.utils.project import load_config, project_path

cfg, _ = load_config("configs/project.yaml")
TRAIN = project_path(cfg["paths"]["train"])
FOLD_NAMES = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KEEP = 0.0022778298112255263

PAIRS = [("H05_F2T2", "H20_F2T2", 0.20), ("H05_F2T2", "H20_F2T2", 1.00)]


def attr_fold(fn, fold, pa_name, pb_name, q):
    panel = read_valid_panel(TRAIN, fold.valid_start, fold.valid_end)
    pf = pd.read_parquet(f"outputs/long_horizon/LH003/{fn}/raw_predictions.parquet")
    pf["ts_code"] = pf["ts_code"].astype(str)
    pf["trade_date"] = pf["trade_date"].astype("int64")
    cols = sorted({pa_name, pb_name})
    m = panel.merge(pf[KEYS + cols], on=KEYS, how="left", validate="one_to_one")
    del panel, pf
    dates = m["trade_date"].to_numpy()
    flag = m["flag_limit_up"].to_numpy()
    pa = _fill(m[pa_name].to_numpy("float64"))
    pb = _fill(m[pb_name].to_numpy("float64"))
    vec = _fill(pool_rerank(pa, pb, flag, dates, q))

    d = m[KEYS + ["y_ret_1d", "flag_limit_up"]].copy()
    d["pred"] = vec
    d["pred_band"] = _fill(band_scores(d, KEEP).to_numpy("float64"))
    d["pa"] = pa
    d["pb"] = pb
    d["in_pool"] = np.nan
    # 池内外标记
    for dt, g in d.groupby("trade_date"):
        e = g[g.flag_limit_up == 0]
        if len(e) < 100:
            continue
        n_pool = max(int(round(len(e) * min(q, 1.0))), 1)
        pool = set(e.nlargest(n_pool, "pa").ts_code)
        d.loc[g.index, "in_pool"] = g.ts_code.isin(pool).astype(float)

    names = d.ts_code.unique()
    code_idx = {c: i for i, c in enumerate(names)}
    day_list = sorted(d.trade_date.unique())

    def top_set(dt, col):
        g = d[(d.trade_date == dt) & (d.flag_limit_up == 0)]
        if len(g) < 100:
            return set()
        return set(g.nlargest(max(len(g) // 10, 1), col).ts_code)

    prev = None
    prev_raw = None
    recs = []
    held_first = None
    for dt in day_list:
        cur = top_set(dt, "pred_band")
        raw = top_set(dt, "pred")
        if held_first is None and cur:
            held_first = cur
        if prev is not None and cur:
            dropped = prev - cur
            g = d[(d.trade_date == dt) & d.ts_code.isin(dropped)]
            recs.append({
                "date": dt, "n_hold": len(prev), "n_drop": len(dropped),
                "n_new": len(cur - prev),
                "drop_limit": int((g.flag_limit_up == 1).sum()),
                "drop_notrade": int(g.y_ret_1d.isna().sum()),
                "drop_in_pool": int((g.in_pool == 1).sum()),
                "drop_pct_lt_keep": int((g.pred < KEEP).sum()),
                "hold_pct_lt_keep": int((d[(d.trade_date == dt) &
                                           d.ts_code.isin(prev & cur)].pred < KEEP).sum()),
                "raw_turnover": 1 - len(raw & prev_raw) / max(len(raw | prev_raw), 1)
                if prev_raw else np.nan,
            })
        prev, prev_raw = cur, raw

    r = pd.DataFrame(recs)
    last = top_set(day_list[-1], "pred_band")
    surv = len(held_first & last) / max(len(held_first), 1) if held_first else np.nan
    return {
        "fold": fn, "direction": f"{pa_name}>{pb_name}", "q": q,
        "days": len(day_list),
        "drop_total": int(r.n_drop.sum()), "drop_per_day": float(r.n_drop.mean()),
        "drop_limit": int(r.drop_limit.sum()), "drop_notrade": int(r.drop_notrade.sum()),
        "drop_pct_lt_keep": int(r.drop_pct_lt_keep.sum()),
        "drop_in_pool": int(r.drop_in_pool.sum()),
        "hold_pct_lt_keep": int(r.hold_pct_lt_keep.sum()),
        "raw_turnover_mean": float(r.raw_turnover.mean()),
        "band_turnover_mean": float(
            r.n_drop.sum() / (r.n_drop.sum() + (r.n_hold - r.n_drop).sum())),
        "first_last_overlap": float(surv),
    }, r


def main() -> None:
    folds_all = {f.name: f for f in get_optuna_folds(cfg)}
    out = []
    for pa_name, pb_name, q in PAIRS:
        for fn in FOLD_NAMES:
            row, _ = attr_fold(fn, folds_all[fn], pa_name, pb_name, q)
            out.append(row)
            print(row, flush=True)
    df = pd.DataFrame(out)
    df.to_csv("outputs/long_horizon/_probe_band_attr.csv", index=False)
    pd.set_option("display.width", 300)
    print("\n===== 四折 band 剔除归因（全窗口，非 22 天）=====")
    print(df.round(4).to_string(index=False))
    print("\n===== 按方案汇总 =====")
    g = df.groupby(["direction", "q"]).agg(
        折数=("fold", "size"), 交易日=("days", "sum"),
        日均剔除=("drop_per_day", "mean"), 剔除合计=("drop_total", "sum"),
        因涨停=("drop_limit", "sum"), 因停牌无价=("drop_notrade", "sum"),
        因分位跌破keep=("drop_pct_lt_keep", "sum"), 剔除中在池内=("drop_in_pool", "sum"),
        持仓中分位跌破keep=("hold_pct_lt_keep", "sum"),
        裸口径换手=("raw_turnover_mean", "mean"),
        首末名单留存=("first_last_overlap", "mean"))
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()

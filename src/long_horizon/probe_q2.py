"""探针 2：把 band 每天的保留/补足中间量打出来，看持仓到底冻不冻。

用法（仓库根目录）::

    python src/long_horizon/probe_q2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.official_eval import TOP_MIN_VALID
from src.evaluation.turnover import TOP_FRACTION, _wide_rank
from src.evaluation.validation import get_folds
from src.long_horizon.rerank_eval import pool_rerank, read_valid_panel
from src.utils.project import load_config, project_path

cache = Path("outputs/long_horizon/_probe_panel.parquet")
if cache.exists():
    panel = pd.read_parquet(cache)
    print("panel 来自缓存")
else:
    cfg, _ = load_config("configs/project.yaml")
    csv_path = project_path(cfg["paths"]["train"])
    fold = [f for f in get_folds() if f.name == "fold1"][0]
    panel = read_valid_panel(csv_path, fold.valid_start, fold.valid_end).reset_index(drop=True)
    pred = pd.read_parquet("outputs/long_horizon/LH003/fold1/raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel["trade_date"] = panel["trade_date"].astype("int64")
    panel = panel.merge(pred, on=KEYS, how="left", validate="one_to_one")
    panel.to_parquet(cache, index=False)
    print("panel 已缓存")

a = panel["H01"].to_numpy("float64")
b = panel["H10_F2T2"].to_numpy("float64")
flag = panel["flag_limit_up"].to_numpy()
dates = panel["trade_date"].to_numpy()
codes = panel["ts_code"].to_numpy()
keep_q = 0.0022778298112255263


def trace(pred_vec, tag):
    scored = panel[KEYS + ["y_ret_1d", "flag_limit_up"]].assign(pred=pred_vec)
    re_ = _wide_rank(scored, True)
    vals = re_.to_numpy("float64")
    d = re_.index.to_numpy()
    cols = np.asarray(re_.columns)
    prev, rows = [], []
    hold = []
    for t in range(len(d)):
        row = vals[t]
        ok = ~np.isnan(row)
        n = int(ok.sum())
        if n < TOP_MIN_VALID:
            prev = []
            hold.append(set())
            continue
        n_top = max(n // TOP_FRACTION, 1)
        keep = [i for i in prev if ok[i] and row[i] >= keep_q]
        k0 = len(keep)
        if len(keep) > n_top:
            keep = list(np.argsort(-row[keep])[:n_top])
        if len(keep) < n_top:
            chosen = set(keep)
            for idx in np.argsort(-np.where(ok, row, -np.inf)):
                if len(keep) >= n_top:
                    break
                if idx not in chosen:
                    keep.append(int(idx))
                    chosen.add(int(idx))
        rows.append((d[t], n, n_top, len(prev), k0, len(keep)))
        prev = keep
        hold.append(set(cols[keep]))
    print(f"--- {tag} 前 6 天：日期 / 候选数 / n_top / 昨日持仓 / 保留数 / 持仓数")
    for r in rows[:6]:
        print("   ", r)
    j_self = np.mean([len(hold[0] & h) / len(hold[0] | h) for h in hold if h])
    j_prev = np.mean([len(hold[i] & hold[i - 1]) / len(hold[i] | hold[i - 1])
                      for i in range(1, len(hold)) if hold[i] and hold[i - 1]])
    print(f"    与首日持仓平均重合 {j_self:.4f}；日间持仓自重合 {j_prev:.6f}")
    return hold


h_base = trace(a, "baseline H01")
for q in (0.02, 0.10, 0.20):
    h = trace(pool_rerank(a, b, flag, dates, q), f"H10_F2T2 q={q}")
    jj = np.mean([len(x & y) / len(x | y) for x, y in zip(h_base, h) if x or y])
    print(f"    → 与 baseline 持仓的逐日平均重合 {jj:.4f}")
    for i in (1, 2, 3, 5, 10, 30, 60, 120, 240):
        if i < len(h) and h_base[i] and h[i]:
            x, y = h_base[i], h[i]
            print(f"      day{i:>3}: 重合 {len(x & y) / len(x | y):.4f}"
                  f"（双方各 {len(x)}/{len(y)} 只）")
    print()

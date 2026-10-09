"""探针：q（池比例）到底改变了什么。

比较 baseline(H01) 与 H01>H10_F2T2 在各 q 下的
(1) 未加 band 的 Top 1/10 集合、(2) 加 band 后的持仓集合，
看 q 到多小才开始真正换股。

用法（仓库根目录，需先有 ``outputs/long_horizon/LH003/fold1/raw_predictions.parquet``）::

    python src/long_horizon/probe_q.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.official_eval import TOP_MIN_VALID
from src.evaluation.turnover import TOP_FRACTION, band_scores
from src.evaluation.validation import get_folds
from src.long_horizon.rerank_eval import _fill, pool_rerank, read_valid_panel
from src.utils.project import load_config, project_path

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
print(f"panel {len(panel):,} 行 / {panel['trade_date'].nunique()} 天；"
      f"H01 缺失 {int(panel['H01'].isna().sum()):,}")

a = panel["H01"].to_numpy("float64")
b = panel["H10_F2T2"].to_numpy("float64")
flag = panel["flag_limit_up"].to_numpy()
dates = panel["trade_date"].to_numpy()
codes = panel["ts_code"].to_numpy()
keep_q = 0.0022778298112255263

order = np.argsort(dates, kind="stable")
uniq, starts = np.unique(dates[order], return_index=True)
starts = np.append(starts, len(order))


def day_blocks():
    for k in range(len(uniq)):
        idx = order[starts[k]:starts[k + 1]]
        ok = flag[idx] == 0
        n = int(ok.sum())
        if n < TOP_MIN_VALID:
            yield idx, ok, 0
            continue
        yield idx, ok, max(n // TOP_FRACTION, 1)


def top_sets(score, banded):
    """返回每天 top 集合。banded=True 时分数已编码，直接用；否则按 pred 排名。"""
    out = []
    for idx, ok, n_top in day_blocks():
        if n_top == 0:
            out.append(set())
            continue
        v = score[idx]
        key = np.where(ok, v, -np.inf)
        sel = np.argsort(-key, kind="stable")[:n_top]
        out.append(set(codes[idx][sel]))
    return out


def jac(x, y):
    return np.mean([len(p & q) / len(p | q) for p, q in zip(x, y) if p or q])


ref_raw = top_sets(a, False)
ref_band = top_sets(band_scores(
    panel[KEYS + ["y_ret_1d", "flag_limit_up"]].assign(pred=a), keep_q).to_numpy("float64"),
    True)

print()
print(f"{'q':>5} | {'raw Top1/10':>12} | {'band 持仓':>10} | {'band IC':>8} | {'band 超额':>9} | {'换手':>7}")
for q in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]:
    out = pool_rerank(a, b, flag, dates, q)
    scored = panel[KEYS + ["y_ret_1d", "flag_limit_up"]].assign(pred=out)
    banded = band_scores(scored, keep_q).to_numpy("float64")
    j_raw = jac(ref_raw, top_sets(out, False))
    j_band = jac(ref_band, top_sets(banded, True))
    from src.evaluation.official_eval import evaluate_frame
    m = evaluate_frame(scored.assign(pred=_fill(banded)))
    print(f"{q:>5.2f} | {j_raw:>12.4f} | {j_band:>10.4f} | {m['ic_mean']:>8.4f} | "
          f"{m['annual_excess']:>9.4f} | {m['mean_turnover']:>7.4f}")

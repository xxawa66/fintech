"""探针：各长周期模型 Top 1/10 名单在「滚动持有 h 天」口径下的顶部超额。

回答的问题：用长周期模型选出的 Top 1/10，在随后 5/20 天里的平均表现，
是否真的强于 H01（1 日模型）——即 band 冻结持仓（≈首日选股长期持有）时，
超额项是否应该由长周期名单来挣。

三个口径：
  A  每日重选、持有 1 天  ← 官方口径，band 下被冻结成「首日持有」
  B  每日重选、但名单持有 h 天（评估选股质量本身）
  C  首日选一次、持有整折（band 的真实行为）

全部口径都用「超额 = top组收益 − 当日候选全集收益」年化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.long_horizon.rerank_eval import read_valid_panel
from src.utils.project import load_config, project_path

cfg, _ = load_config("configs/project.yaml")
TRAIN = project_path(cfg["paths"]["train"])

FOLDS = {
    "wf2021": (20210101, 20211231),
    "wf2022": (20220101, 20221231),
    "wf2023": (20230101, 20231231),
    "confirm2024": (20240101, 20241231),
}
MODELS = ["H01", "H05_F1T2", "H05_F2T2", "H10_F2T2", "H20_F2T2", "H30_F2T2"]
HLIST = [1, 5, 10, 20]
ANN = 252


def fold_matrices(fold: str, ys: int, ye: int):
    panel = read_valid_panel(TRAIN, ys, ye)
    pred = pd.read_parquet(f"outputs/long_horizon/LH003/{fold}/raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    models = [m for m in MODELS if m in pred.columns]
    m = panel.merge(pred[KEYS + models], on=KEYS, how="inner", validate="one_to_one")
    m = m.sort_values(KEYS, kind="stable")

    wide_y = m.pivot(index="trade_date", columns="ts_code", values="y_ret_1d")
    wide_flag = m.pivot(index="trade_date", columns="ts_code", values="flag_limit_up")
    elig = wide_flag.values == 0
    logr = np.log1p(np.nan_to_num(wide_y.values.astype(float), nan=0.0))

    fwd = {}
    for h in HLIST:
        acc = np.zeros_like(logr)
        for k in range(1, h + 1):
            acc += np.roll(logr, -k, axis=0)
        acc[len(acc) - h:] = np.nan
        fwd[h] = acc

    scores = {
        mm: m.pivot(index="trade_date", columns="ts_code", values=mm).values.astype(float)
        for mm in models
    }
    return wide_y.index.values, elig, wide_y.values.astype(float), fwd, scores, models


def rolling(fwd_h, S, elig):
    """每日重选 Top1/10，返回 (top 组收益均值, 超额) —— 均为 h 日累计口径。"""
    tops, exs = [], []
    for i in range(fwd_h.shape[0]):
        rf, rs = fwd_h[i], S[i]
        mask = elig[i] & np.isfinite(rf) & np.isfinite(rs)
        if mask.sum() < 100:
            continue
        idx = np.flatnonzero(mask)
        s, f = rs[idx], rf[idx]
        k = max(len(idx) // 10, 1)
        sel = np.argpartition(-s, k - 1)[:k]
        tops.append(f[sel].mean())
        exs.append(f[sel].mean() - f.mean())
    return float(np.mean(tops)), float(np.mean(exs))


def main() -> None:
    all_rows = []
    for fold, (ys, ye) in FOLDS.items():
        dates, elig, y1, fwd, scores, models = fold_matrices(fold, ys, ye)
        print(f"[{fold}] {len(dates)} 个交易日 / {elig.shape[1]} 只股票", flush=True)
        for mm in models:
            S = scores[mm]
            row = {"fold": fold, "model": mm}
            for h in HLIST:
                t, e = rolling(fwd[h], S, elig)
                row[f"top_h{h}"] = t * ANN / h
                row[f"ex_h{h}"] = e * ANN / h
            # 口径 C：首日选一次，整折持有（每日等权再平衡）
            rs = S[0]
            mask = elig[0] & np.isfinite(rs)
            idx = np.flatnonzero(mask)
            k = max(len(idx) // 10, 1)
            held = idx[np.argpartition(-rs[idx], k - 1)[:k]]
            daily = np.nan_to_num(y1[:, held], nan=0.0).mean(axis=1)
            mk = np.nan_to_num(y1, nan=0.0).mean(axis=1)
            row["top_C"] = float(daily.mean()) * ANN
            row["ex_C"] = float((daily - mk).mean()) * ANN
            # 首日名单与 H01 首日名单的重合率
            all_rows.append(row)
            print("   ", {k2: (round(v, 4) if isinstance(v, float) else v)
                          for k2, v in row.items() if k2 not in ("fold", "model")}, flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv("outputs/long_horizon/_probe_hold_h.csv", index=False)

    print("\n================ 四折均值 ================")
    g = df.groupby("model")[["top_h1", "ex_h1", "top_h5", "ex_h5", "top_h10", "ex_h10",
                             "top_h20", "ex_h20", "top_C", "ex_C"]].mean()
    base = g.loc["H01"]
    show = pd.DataFrame({
        "top_h1": g.top_h1, "ex_h1": g.ex_h1,
        "top_h5": g.top_h5, "ex_h5": g.ex_h5, "ex_h5-H01": g.ex_h5 - base.ex_h5,
        "top_h20": g.top_h20, "ex_h20": g.ex_h20, "ex_h20-H01": g.ex_h20 - base.ex_h20,
        "ex_C": g.ex_C, "ex_C-H01": g.ex_C - base.ex_C,
    })
    pd.set_option("display.width", 260)
    print(show.round(4).to_string())


if __name__ == "__main__":
    main()

"""LH001 诊断：首日冻结组超额的时间衰减（按月与前缀窗口分解）。

回答的问题：2024 折上所有模型的首日冻结组全年超额都≈0，究竟是“选股不行”
还是“年初的 alpha 快速衰减”？把每个模型冻结组的逐月超额（年化）与
前缀窗口（前 21 / 63 / 126 / 252 个交易日）年化超额算出来即可区分。

输入为本实验的既有产物（不重训）：
- ``outputs/long_horizon/LH001/{fold}/raw_predictions.parquet``（键 + 各模型 raw 预测）
- ``experiments/LH001_day1_sets.csv``（每折每模型的首日 Top 1/10 成员）
标签与涨停标志从原始训练集按键合并（官方口径过滤：剔涨停、剔 y 缺失）。

用法（仓库根目录）::

    python -m src.long_horizon.diagnostics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.utils.project import ROOT, load_config, project_path

PREFIX_WINDOWS = [21, 63, 126, 252]


def load_valid_labels(valid_path: str, fold_start: int, fold_end: int) -> pd.DataFrame:
    labels = pd.read_csv(valid_path,
                         usecols=KEYS + ["y_ret_1d", "flag_limit_up"],
                         dtype={"ts_code": "string", "trade_date": "int32"})
    return labels[labels["trade_date"].between(fold_start, fold_end)]


def frozen_excess_series(valid: pd.DataFrame, members: set[str]) -> pd.DataFrame:
    """冻结组的逐日超额（官方过滤口径），带 trade_date 列。"""
    member = valid["ts_code"].isin(members).to_numpy()
    y = valid["y_ret_1d"].to_numpy(dtype="float64")
    flag = valid["flag_limit_up"].to_numpy()
    dates = valid["trade_date"].to_numpy()
    rows = []
    for day in np.unique(dates):
        day_rows = dates == day
        market = day_rows & np.isfinite(y) & (flag == 0)
        held = market & member
        if market.sum() >= 100 and held.sum() > 0:
            rows.append({"trade_date": int(day),
                         "excess": float(y[held].mean() - y[market].mean())})
    return pd.DataFrame(rows)


def main() -> int:
    cfg, _ = load_config("configs/project.yaml")
    settings = cfg["long_horizon"]
    study = settings["study"]
    out_root = ROOT / "outputs" / "long_horizon" / study
    day1 = pd.read_csv(ROOT / "experiments" / f"{study}_day1_sets.csv",
                       dtype={"ts_code": "string"})
    folds = {"fold1": (20230101, 20231231), "fold2": (20240101, 20241231)}

    monthly_frames: list[pd.DataFrame] = []
    prefix_frames: list[pd.DataFrame] = []
    for fold, (start, end) in folds.items():
        preds = pd.read_parquet(out_root / fold / "raw_predictions.parquet")
        valid = load_valid_labels(project_path(cfg["paths"]["train"]), start, end)
        valid = preds[KEYS].merge(valid, on=KEYS, how="inner", validate="one_to_one")
        for model in [c for c in preds.columns if c not in KEYS]:
            members = set(day1[(day1["fold"] == fold) & (day1["model"] == model)]["ts_code"])
            series = frozen_excess_series(valid, members)
            series["month"] = (series["trade_date"] // 10000 * 100
                               + series["trade_date"] // 100 % 100)
            monthly = series.groupby("month")["excess"].agg(["mean", "count"])
            monthly = monthly.assign(
                model=model, fold=fold,
                annual_excess=(monthly["mean"] * 252).round(4)).reset_index()
            monthly_frames.append(monthly[["fold", "model", "month", "count", "annual_excess"]])
            prefix_rows = []
            ordered = series.sort_values("trade_date").reset_index(drop=True)
            for window in PREFIX_WINDOWS:
                head = ordered.head(window)
                prefix_rows.append({
                    "fold": fold, "model": model, "window_days": window,
                    "n_days": int(len(head)),
                    "annual_excess": round(float(head["excess"].mean() * 252), 4)})
            prefix_frames.extend(prefix_rows)

    monthly = pd.concat(monthly_frames, ignore_index=True)
    prefix = pd.DataFrame(prefix_frames)
    monthly.to_csv(out_root / "frozen_monthly_excess.csv", index=False)
    prefix.to_csv(out_root / "frozen_prefix_excess.csv", index=False)

    pd.set_option("display.width", 200)
    print("===== 首日冻结组年化超额：前缀窗口分解 =====")
    print(prefix.pivot(index=["fold", "model"], columns="window_days",
                       values="annual_excess").to_string())
    print("\n===== 首日冻结组年化超额：逐月 =====")
    print(monthly.pivot(index=["fold", "model"], columns="month",
                        values="annual_excess").to_string())
    print(f"\n已写入 {out_root / 'frozen_monthly_excess.csv'} 与 "
          f"{out_root / 'frozen_prefix_excess.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

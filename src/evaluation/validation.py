"""时间验证切分：固定时间窗口，训练标签不跨验证边界。

切分定义在 ``configs/project.yaml`` 的 ``validation`` 段：

- Fold 1：训练 2018-01-02 ~ 2022-12-31，验证 2023 全年
- Fold 2：训练 2018-01-02 ~ 2023-12-31，验证 2024 全年（模型选择时权重更高，
  正式测试从 2025 开始）
- final：全量训练 2018-01-02 ~ 2024-12-31，正式预测 2025-01-02 ~ 2026-06-08

边界规则（与 AGENTS.md 一致）：

1. **训练标签不跨验证边界。** ``y_ret_1d(t) = close(t+1)/close(t) - 1``，训练窗口
   最后一个交易日的标签由验证期首个交易日收盘价计算，属于跨界信息——切分时将这些
   标签置 NaN（不动原始数据），训练侧按"删除无标签样本"的基线约定剔除。
   对停牌等已缺失标签的行该操作是空操作，无双重处理问题。
2. **验证集保留期内全部官方标签**，包括验证期末日（如 20241231）由测试期价格
   计算的 4,525 个标签——官方评分本身就使用实现收益，保留才与正式口径一致。
3. **特征历史边界。** 特征必须在 ``[数据起点, 验证终点]`` 的连续历史上一次性计算，
   再按本模块切分；禁止训练/验证各自独立计算（否则验证期开头的 rolling/lag 特征
   会缺历史）。rolling/lag 只向过去取值即可，不构成未来信息。最终预测时同理：
   在训练 X + 测试 X 拼接后的全历史（2018–2026）上计算特征。

用法（在仓库根目录运行）::

    from src.evaluation.validation import get_folds, split_train_valid, split_final
    folds = get_folds()
    for fold in folds:
        train, valid, info = split_train_valid(df, fold)

实数据核对::

    python -m src.evaluation.validation --check
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATE_COL = "trade_date"
LABEL_COL = "y_ret_1d"


@dataclass(frozen=True)
class Fold:
    """单个时间验证折，窗口均为闭区间（YYYYMMDD 整数）。"""

    name: str
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int

    def __post_init__(self) -> None:
        if not (self.train_start <= self.train_end < self.valid_start <= self.valid_end):
            raise ValueError(f"折 {self.name} 窗口非法: {self}")

    @property
    def feature_history_start(self) -> int:
        """特征计算的历史起点：从数据起点（= 训练窗口起点）开始。"""
        return self.train_start

    def describe(self) -> str:
        return (f"{self.name}: train {self.train_start}–{self.train_end} "
                f"valid {self.valid_start}–{self.valid_end}")


def _load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "project.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_folds(config_path: str | Path | None = None) -> list[Fold]:
    """从 configs/project.yaml 读取验证折定义。"""
    cfg = _load_config(config_path)
    folds = []
    for item in cfg["validation"]["folds"]:
        (ts, te), (vs, ve) = item["train"], item["valid"]
        folds.append(Fold(item["name"], ts, te, vs, ve))
    if not folds:
        raise ValueError("configs/project.yaml 中未定义 validation.folds")
    return folds


def get_final_split(config_path: str | Path | None = None) -> dict:
    """最终训练/预测窗口：全量训练 2018–2024，正式预测 2025–2026。"""
    cfg = _load_config(config_path)
    return cfg["validation"]["final"]


def _boundary_crossing_mask(train_dates: pd.Series, all_dates: np.ndarray, train_end: int) -> pd.Series:
    """训练日 t 的标签跨界 = 全局下一交易日 > train_end（即标签用到验证期价格）。

    全局交易日历取自完整数据（训练+验证）的 ``trade_date`` 去重升序；平衡面板下
    每日全体股票在场，日历与股票无关。只有训练窗口最后一个交易日满足该条件。
    """
    pos = np.searchsorted(all_dates, train_dates.to_numpy(), side="right")
    has_next = pos < len(all_dates)
    next_date = np.where(has_next, all_dates[np.minimum(pos, len(all_dates) - 1)], -1)
    return pd.Series(has_next & (next_date > train_end), index=train_dates.index)


def split_train_valid(
    df: pd.DataFrame,
    fold: Fold,
    date_col: str = DATE_COL,
    label_col: str = LABEL_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """按折切分数据；训练集跨界标签置 NaN（原始数据不改，返回副本）。

    参数 ``df`` 应为**完整历史**（含训练+验证期间，最好已是特征计算后的数据），
    至少含 ``date_col`` 与 ``label_col``。返回 ``(train, valid, info)``；
    ``info`` 记录窗口、跨界处理与标签覆盖情况，供实验记录引用。
    """
    if date_col not in df or label_col not in df:
        raise ValueError(f"df 缺少列 {date_col} / {label_col}")
    all_dates = np.sort(df[date_col].unique())

    d = df[date_col]
    train = df[(d >= fold.train_start) & (d <= fold.train_end)].copy()
    valid = df[(d >= fold.valid_start) & (d <= fold.valid_end)].copy()

    n_train_labels_before = int(train[label_col].notna().sum())
    crossing = _boundary_crossing_mask(train[date_col], all_dates, fold.train_end)
    n_crossing_rows = int(crossing.sum())
    n_crossing_labels = int(train.loc[crossing, label_col].notna().sum())
    train.loc[crossing, label_col] = np.nan

    last_train_day = int(train[date_col].max()) if len(train) else -1
    next_day_pos = int(np.searchsorted(all_dates, last_train_day, side="right"))
    next_trading_day = int(all_dates[next_day_pos]) if next_day_pos < len(all_dates) else -1

    info = {
        "fold": fold.name,
        "train_window": [fold.train_start, fold.train_end],
        "valid_window": [fold.valid_start, fold.valid_end],
        "n_train_rows": len(train),
        "n_valid_rows": len(valid),
        "last_train_day": last_train_day,
        "next_trading_day": next_trading_day,
        "n_boundary_rows": n_crossing_rows,
        "n_boundary_labels_dropped": n_crossing_labels,
        "n_train_labels_before": n_train_labels_before,
        "n_train_labels_after": int(train[label_col].notna().sum()),
        "n_valid_labels": int(valid[label_col].notna().sum()),
        "valid_label_coverage": float(valid[label_col].notna().mean()) if len(valid) else 0.0,
    }
    return train, valid, info


def split_final(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
    date_col: str = DATE_COL,
    label_col: str = LABEL_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """最终提交用切分：全量 2018–2024 训练（含期末日官方标签，官方已给出且无验证
    边界可言），2025–2026 为预测目标。"""
    final = get_final_split(config_path)
    (ts, te), (xs, xe) = final["train_all"], final["test"]
    d = df[date_col]
    train = df[(d >= ts) & (d <= te)].copy()
    test = df[(d >= xs) & (d <= xe)].copy()
    info = {
        "train_window": [ts, te],
        "test_window": [xs, xe],
        "n_train_rows": len(train),
        "n_train_labels": int(train[label_col].notna().sum()) if label_col in train else None,
        "n_test_rows": len(test),
        "note": "最终训练保留 20241231 官方标签（由 20250102 收盘价计算，官方提供）",
    }
    return train, test, info


def run_check(data_path: str | Path | None = None,
              config_path: str | Path | None = None,
              test_path: str | Path | None = None) -> list[dict]:
    """在真实数据上核对切分：逐折打印行数、边界处理与标签覆盖。

    训练集文件核对折内切分；测试集文件核对最终预测窗口的行数（测试集无标签，
    只核键覆盖）。两份文件的路径取自 ``configs/project.yaml`` 的 ``paths``。
    """
    cfg = _load_config(config_path)
    root = Path(__file__).resolve().parents[2]
    if data_path is None:
        data_path = root / cfg["paths"]["train"]
    df = pd.read_csv(data_path, usecols=["ts_code", DATE_COL, LABEL_COL])
    dates = np.sort(df[DATE_COL].unique())
    print(f"训练数据: {data_path}（{len(df):,} 行，{dates[0]}–{dates[-1]}）\n")

    infos = []
    for fold in get_folds(config_path):
        train, valid, info = split_train_valid(df, fold)
        infos.append(info)
        print(fold.describe())
        print(f"  训练行数 {info['n_train_rows']:,}，标签 {info['n_train_labels_before']:,} → "
              f"{info['n_train_labels_after']:,}（跨界剔除 {info['n_boundary_labels_dropped']:,}）")
        print(f"  训练期最后交易日 {info['last_train_day']}，下一交易日 {info['next_trading_day']}"
              f"（> {fold.train_end}，其标签跨界，已置 NaN）")
        print(f"  验证行数 {info['n_valid_rows']:,}，标签 {info['n_valid_labels']:,}"
              f"（覆盖 {info['valid_label_coverage']:.2%}）")
        # 结构核对：窗口不重叠、日期落在各自窗口内、跨界标签守恒、切分时序正确
        assert set(train[DATE_COL].unique()).isdisjoint(set(valid[DATE_COL].unique())), "训练/验证日期重叠"
        assert train[DATE_COL].between(fold.train_start, fold.train_end).all(), "训练集含窗口外日期"
        assert valid[DATE_COL].between(fold.valid_start, fold.valid_end).all(), "验证集含窗口外日期"
        assert info["n_train_labels_after"] + info["n_boundary_labels_dropped"] == info["n_train_labels_before"]
        assert fold.train_end < fold.valid_start
        print()

    final = get_final_split(config_path)
    (ts, te), (xs, xe) = final["train_all"], final["test"]
    _, _, info = split_final(df, config_path)
    if test_path is None:
        test_path = root / cfg["paths"]["test_x"]
    if Path(test_path).exists():
        test = pd.read_csv(test_path, usecols=["ts_code", DATE_COL])
        test_dates = np.sort(test[DATE_COL].unique())
        n_test = len(test)
        n_test_in_window = int(((test[DATE_COL] >= xs) & (test[DATE_COL] <= xe)).sum())
        assert n_test == n_test_in_window, "测试集含有预测窗口之外的行"
        print(f"测试数据: {test_path}（{n_test:,} 行，{test_dates[0]}–{test_dates[-1]}）")
    else:
        n_test = None
        print(f"测试数据: {test_path} 不存在，跳过预测窗口核对")
    print(f"final: 训练 {info['train_window']} {info['n_train_rows']:,} 行"
          f"（标签 {info['n_train_labels']:,}，含 20241231 官方标签），"
          f"预测 {xs}–{xe}" + (f" {n_test:,} 行" if n_test is not None else ""))
    print("\ncheck PASS：切分窗口、边界处理、标签覆盖与键覆盖均符合约定。")
    return infos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="时间验证切分（固定窗口，标签不跨边界）")
    parser.add_argument("--check", action="store_true", help="在真实训练集上核对切分")
    parser.add_argument("--data", help="训练数据 CSV（默认取 configs/project.yaml 的 paths.train）")
    parser.add_argument("--config", help="配置文件路径（默认 configs/project.yaml）")
    args = parser.parse_args(argv)

    if not args.check:
        parser.error("当前仅支持 --check")
    run_check(args.data, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

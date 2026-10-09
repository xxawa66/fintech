"""LH001 长周期选股实验：5/10/20/30 日前瞻模型的 raw pred 评估（不叠 band）。

背景：S003 锁定的 T030（keep_q≈0.0023）在留仓带下首日进入 Top 1/10 的股票
几乎全年不变，初始集合的质量直接决定超额项。本实验训练 horizon∈{5,10,20,30}
的长周期模型，以 **raw pred 口径**回答：长周期模型选出的首日 Top 组是否比
1 日模型更好。不叠加任何换手/band 机制，不以官方综合分为优化目标。

协议：
- 特征：冻结的 40 个 V1 特征，全部只使用 t 日及以前的信息；
- 监督：``y_ret_{h}d = close(t+h)/close(t) - 1``，仅作训练目标；训练窗口末尾
  h 个交易日的标签使用验证期价格，按边界规则剔除（``src/long_horizon/labels.py``）；
- 模型：与 T030 完全相同的 LightGBM 参数与轮数，从
  ``docs/optuna_tuning_S003_selection.json`` 的 winner.spec 直读，不重调参；
- 评估（每折每模型，**排名口径为主**）：
  1. **top10% 排名重合度**（主指标）：每日把预测与实际各自的 h 日收益排序，
     取前 1/10 组成两个集合，重合率 = |交集| / 组大小；并报 Jaccard 与两组
     实现收益均值（仅作旁证，不作判据）；
  2. 自身 horizon 的逐日 Rank IC（>=30 有效样本）；
  3. raw 官方 8 指标与首日 Top 组的实现超额（旁证，不作判据）。
- 模型选择：看**两折**（2023 / 2024）的排名重合度，不取只对单折好的 h。

特征集 / 训练目标变体（``configs/project.yaml: long_horizon.variants``）：

- ``features``：``v1`` = 原 40 个特征；``long`` = V1 + 长周期增量特征
  （``src/long_horizon/features.py``，120/250 日窗口，共 59 个）；
- ``target``：``raw`` = 原始 h 日收益；``rank`` = **当日截面百分位排名**
  （``cross_sectional_rank``），与排名判据对齐、屏蔽市场共同涨跌。
- 变体命名 ``H{h}_{tag}``，如 ``H10_F2T2``；参照模型 ``H01`` = 1 日 + V1 + raw。
- 正确性校验：h=1 重算标签与官方 y_ret_1d 逐位一致；H01（1 日基线）raw 指标
  应复现 S003 的 wf2023 / confirm2024 raw 结果（相同环境与监督路径）。

生产接入路径（仅当实验证明选股更优后才考虑，本实验不跑）：
首日 pred 替换为长周期预测、其余交易日 pred 不变，由留仓带机制
（``src/evaluation/turnover.band_scores``）维持首日集合。

折来源（``--fold-source``）：

- ``validation``（默认）：``configs/project.yaml validation.folds`` 的两折
  （fold1 = 2023、fold2 = 2024）；
- ``optuna``：``configs/project.yaml optuna.folds`` 的 walk-forward 折
  （wf2021 / wf2022 / wf2023）加 ``optuna.confirm``（confirm2024）。
  四折时间窗（2021–2024）都落在 S003 调参的连续历史内，可复用同一份特征缓存。

用法（仓库根目录）::

    python -m src.long_horizon.run                 # 跑 fold1 + fold2
    python -m src.long_horizon.run --folds fold2   # 只跑 2024 折
    python -m src.long_horizon.run --study LH006 --fold-source optuna \\
        --folds wf2021,wf2022,wf2023,confirm2024 --variants F2T2 --horizons 20
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.clean_data import clean_history
from src.data.load_data import KEYS, X_COLUMNS, load_training_data
from src.data.make_dataset import attach_labels, training_arrays
from src.evaluation.official_eval import (ANNUALIZATION_DAYS, IC_MIN_VALID,
                                           OFFICIAL_METRICS, TOP_MIN_VALID,
                                           evaluate_frame)
from src.evaluation.validation import Fold, get_folds, split_train_valid
from src.features.build_features import build_features
from src.long_horizon.features import build_long_features
from src.long_horizon.labels import (WINDOW_KINDS, forward_labels,
                                     forward_window_labels, label_name,
                                     training_label_mask,
                                     verify_against_official, window_label_name)
from src.long_horizon.selection_quality import top_group_quality
from src.models.lightgbm_model import load_model, save_model, train_model
from src.utils.project import ROOT, load_config, project_path, write_json

_T0 = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - _T0:7.1f}s] {message}", flush=True)


def rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return float("nan")


def json_safe(obj):
    """递归把非有限浮点（NaN/inf）转成 None，配合 write_json 的 allow_nan=False。"""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def get_optuna_folds(cfg: dict) -> list[Fold]:
    """S003 的 walk-forward 折（wf2021 / wf2022 / wf2023）+ 2024 确认折。

    读 ``configs/project.yaml`` 的 ``optuna`` 段——与 S003 调参用的是同一份定义，
    只是把验证窗口从「两折」扩到「四折（2021–2024）」。
    """
    items = list(cfg["optuna"]["folds"])
    confirm = cfg["optuna"].get("confirm")
    if confirm:
        items.append(confirm)
    return [Fold(i["name"], *i["train"], *i["valid"]) for i in items]


def expected_raw_metrics(selection: dict, fold: Fold) -> dict | None:
    """从 S003 锁定产物读取同折 raw 基线指标（wf2021–wf2023 / confirm2024）。"""
    year = fold.valid_start // 10000
    entries = selection["winner"]["folds"]
    if f"wf{year}" in entries:
        return entries[f"wf{year}"]["raw"]["metrics"]
    path = ROOT / "docs/optuna_tuning_S003_confirmation.json"
    if year == 2024 and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["candidate"]["raw"]["metrics"]
    return None


def fit_predict(name: str, x_train: pd.DataFrame, y_train: np.ndarray,
                valid: pd.DataFrame, eligible: np.ndarray, columns: list[str],
                params: dict, rounds: int, fallback: float,
                model_dir: Path, log_fn=log) -> tuple[np.ndarray, dict]:
    """训练一个 LightGBM 并在验证期出 raw 预测（不合格报价行填 fallback）。

    若同名模型文件已存在（同一次研究协议下先前运行保存），直接加载复用，
    不重复训练；``info["reused"]`` 记录该事实。
    """
    model_path = model_dir / f"{name}.txt"
    reused = model_path.exists()
    if reused:
        booster = load_model(model_path)
        log_fn(f"{name}: 复用已保存模型 {model_path.name}")
    else:
        started = time.perf_counter()
        booster = train_model(x_train, y_train, params, rounds)
        save_model(booster, model_path)
    pred = np.full(len(valid), fallback, dtype="float64")
    pred[eligible] = booster.predict(valid.loc[eligible, columns],
                                     num_threads=params["num_threads"])
    if not np.isfinite(pred).all():
        raise ValueError(f"{name} 预测含非有限值")
    importance = pd.DataFrame({
        "feature": list(x_train.columns),
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    info = {"n_train_samples": int(len(y_train)),
            "reused": reused,
            "top_features": importance.head(10).to_dict(orient="records")}
    if not reused:
        info["training_seconds"] = round(time.perf_counter() - started, 1)
        log_fn(f"{name}: 训练 {info['training_seconds']}s，样本 {info['n_train_samples']:,}；RSS {rss_gb():.2f} GB")
    del booster
    gc.collect()
    return pred, info


def day1_top_set(valid: pd.DataFrame, pred: np.ndarray, eligible: np.ndarray) -> list[str]:
    """首日 Top 1/10 的成员（官方换手候选口径：报价有效且非涨停）。"""
    dates = valid["trade_date"].to_numpy()
    first = dates.min()
    mask = (dates == first) & eligible & (valid["flag_limit_up"].to_numpy() == 0)
    codes = valid["ts_code"].astype(str).to_numpy()[mask]
    order = np.argsort(-pred[mask], kind="stable")
    n_top = max(int(mask.sum()) // 10, 1)
    return [str(c) for c in codes[order[:n_top]]]


def frozen_set_annual_excess(valid: pd.DataFrame, members: set[str]) -> dict:
    """首日选出的固定组合持有整个验证期的实现年化超额（等权，官方过滤口径）。

    逐日超额 = 成员内 y_ret_1d 有效行的等权均值 − 全市场 y_ret_1d 有效行均值
    （与官方 Top 超额同口径：剔涨停、剔 y 缺失），日均超额 × 252 年化。
    成员当日全部无有效收益的交易日跳过并计数。
    """
    codes = valid["ts_code"].astype(str).to_numpy()
    y = valid["y_ret_1d"].to_numpy(dtype="float64")
    flag = valid["flag_limit_up"].to_numpy()
    dates = valid["trade_date"].to_numpy()
    member = np.isin(codes, np.array(sorted(members), dtype=object))
    excess: list[float] = []
    skipped = 0
    for day in np.unique(dates):
        day_rows = dates == day
        market = day_rows & np.isfinite(y) & (flag == 0)
        if market.sum() < TOP_MIN_VALID:
            skipped += 1
            continue
        held = market & member
        if held.sum() == 0:
            skipped += 1
            continue
        excess.append(float(y[held].mean() - y[market].mean()))
    annual = float(np.mean(excess) * ANNUALIZATION_DAYS) if excess else float("nan")
    return {"frozen_annual_excess": annual, "frozen_days": len(excess),
            "frozen_days_skipped": skipped}


def top_decile_rank_agreement(valid: pd.DataFrame, pred: np.ndarray, label_col: str,
                              eligible: np.ndarray) -> dict:
    """逐日对比「预测 top 1/10 组」与「实际 top 1/10 组」的**排名重合度**。

    排名口径（不是收益率数值口径）：把预测与实际各自的 h 日收益排序，
    取前 1/10 组成两个集合，看两个集合的重合比例。
    - 投资域与官方换手候选一致：报价有效、非涨停，且**实际 h 日收益可算**
      （验证期末尾 h 个交易日无 close(t+h)，自动排除，天数记入 ``days``）。
    - 组大小 k = floor(N/10)，N 为该日投资域股票数。
    - 重合率 = |预测组 ∩ 实际组| / k；Jaccard = |∩| / |∪|。
    """
    y = valid[label_col].to_numpy(dtype="float64")
    flag = valid["flag_limit_up"].to_numpy()
    dates = valid["trade_date"].to_numpy()
    ok = eligible & (flag == 0) & np.isfinite(y)
    overlaps: list[float] = []
    jaccards: list[float] = []
    pred_mean: list[float] = []
    oracle_mean: list[float] = []
    sizes: list[int] = []
    for day in np.unique(dates):
        rows = ok & (dates == day)
        n = int(rows.sum())
        if n < TOP_MIN_VALID:
            continue
        k = max(n // 10, 1)
        idx = np.flatnonzero(rows)
        a = y[idx]
        pred_top = np.argsort(-pred[idx], kind="stable")[:k]
        actual_top = np.argsort(-a, kind="stable")[:k]
        inter = int(np.intersect1d(pred_top, actual_top).size)
        overlaps.append(inter / k)
        jaccards.append(inter / (2 * k - inter))
        pred_mean.append(float(a[pred_top].mean()))
        oracle_mean.append(float(a[actual_top].mean()))
        sizes.append(k)
    ov = np.asarray(overlaps, dtype="float64")
    return {"top10_overlap_mean": float(ov.mean()) if len(ov) else float("nan"),
            "top10_overlap_std": float(ov.std(ddof=1)) if len(ov) > 1 else 0.0,
            "top10_jaccard_mean": float(np.mean(jaccards)) if jaccards else float("nan"),
            "top10_overlap_days": len(ov),
            "pred_top10_realized_h": float(np.mean(pred_mean)) if pred_mean else float("nan"),
            "oracle_top10_realized_h": float(np.mean(oracle_mean)) if oracle_mean else float("nan"),
            "top10_group_size": int(np.mean(sizes)) if sizes else 0}


def daily_ic_stats(valid: pd.DataFrame, pred: np.ndarray, label_col: str,
                   eligible: np.ndarray) -> dict:
    """验证期内 pred 与指定 horizon 标签的逐日 Rank IC 统计（官方门槛 30 样本）。"""
    y = valid[label_col].to_numpy(dtype="float64")
    ok = eligible & np.isfinite(y)
    dates = valid["trade_date"].to_numpy()
    ics: list[float] = []
    for day in np.unique(dates):
        rows = ok & (dates == day)
        if rows.sum() >= IC_MIN_VALID:
            ics.append(float(spearmanr(pred[rows], y[rows])[0]))
    values = np.asarray(ics, dtype="float64")
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"horizon_ic_mean": float(values.mean()) if len(values) else float("nan"),
            "horizon_ic_std": std,
            "horizon_ic_positive_ratio": float((values > 0).mean()) if len(values) else float("nan"),
            "horizon_ic_days": len(values)}


def cross_sectional_rank(dates: np.ndarray, values: np.ndarray) -> np.ndarray:
    """把训练标签换成**当日截面百分位排名**（(0,1]），与排名口径的判据对齐。

    只是对监督目标做单调变换：每个交易日内部按该日样本排名，屏蔽掉市场共同涨跌，
    让模型专注学「同日谁更强」。不引入任何未来信息——排名用的仍是同一批训练标签。
    """
    frame = pd.DataFrame({"d": dates, "v": values})
    return frame.groupby("d", sort=False, observed=True)["v"].rank(
        pct=True, method="average").to_numpy(dtype="float64")


def run_fold(cfg: dict, horizons: list[int], spec: dict, selection: dict,
             dataset: pd.DataFrame, columns_by_set: dict[str, list[str]],
             variants: list[dict], fold: Fold,
             out_root: Path) -> tuple[list[dict], list[dict], dict]:
    params, rounds = spec["model_params"], int(spec["rounds"])
    fallback = float(cfg["baseline"]["fallback_prediction"])
    fold_dir = out_root / fold.name
    model_dir = fold_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    dates_sorted = np.sort(dataset["trade_date"].unique())
    train, valid, split = split_train_valid(dataset, fold)
    eligible = valid["quote_valid"].to_numpy()
    log(f"{fold.name}: 训练 {split['n_train_rows']:,} 行 / 验证 {split['n_valid_rows']:,} 行"
        f"（边界剔除 1d 标签 {split['n_boundary_labels_dropped']:,}）")

    preds: dict[str, np.ndarray] = {}
    infos: dict[str, dict] = {}
    meta: dict[str, tuple[int, str, str, str]] = {}

    # --- 参照 H01：1 日 + V1 特征 + raw 目标（= S003/T030 的监督路径） ---
    ref_columns = columns_by_set["v1"]
    x_train, y_train, array_info = training_arrays(train, ref_columns)
    preds["H01"], infos["H01"] = fit_predict("H01", x_train, y_train, valid,
                                             eligible, ref_columns, params, rounds,
                                             fallback, model_dir)
    infos["H01"]["supervision"] = array_info
    meta["H01"] = (1, "F1T1", "raw", "v1")
    del x_train, y_train
    gc.collect()

    # --- 变体：特征集 × 训练目标 × horizon（参数与轮数不变） ---
    for variant in variants:
        tag, feat_set, target = variant["tag"], variant["features"], variant["target"]
        columns = columns_by_set[feat_set]
        for h in horizons:
            name = f"H{h:02d}_{tag}"
            # 监督目标：raw/rank 用端点标签 y_ret_{h}d；窗口类目标用窗口聚合标签
            sup_col = (label_name(h) if target in ("raw", "rank")
                       else window_label_name(target, h))
            mask = training_label_mask(train, sup_col, dates_sorted, h, fold.train_end)
            if not mask.any():
                raise ValueError(f"{fold.name} {name}: 无有效监督样本")
            x_train = train.loc[mask, columns]
            y_raw = train.loc[mask, sup_col].to_numpy(dtype="float64")
            if target == "rank":
                y_train = cross_sectional_rank(train.loc[mask, "trade_date"].to_numpy(), y_raw)
            else:
                y_train = y_raw
            preds[name], infos[name] = fit_predict(name, x_train, y_train, valid,
                                                   eligible, columns, params, rounds,
                                                   fallback, model_dir)
            infos[name]["supervision"] = {
                "candidate_rows": int(len(train)), "used_rows": int(mask.sum()),
                "dropped_rows": int((~mask).sum()), "supervision_label": sup_col,
                "feature_set": feat_set, "n_features": len(columns), "target": target}
            meta[name] = (h, tag, target, feat_set)
            del x_train, y_train
            gc.collect()

    # --- 评估：排名重合度（主）+ horizon IC + 官方 raw 指标（旁证） ---
    base_set = day1_top_set(valid, preds["H01"], eligible)
    expected = expected_raw_metrics(selection, fold)
    rows: list[dict] = []
    day1_rows: list[dict] = []
    for name, pred in preds.items():
        h, tag, target, feat_set = meta[name]
        scored = valid[KEYS + ["y_ret_1d", "flag_limit_up"]].assign(pred=pred)
        metrics = evaluate_frame(scored)
        members = day1_top_set(valid, pred, eligible)
        frozen = frozen_set_annual_excess(valid, set(members))
        label_col = "y_ret_1d" if h == 1 else label_name(h)
        agree = top_decile_rank_agreement(valid, pred, label_col, eligible)
        # 宽容度 + 下行风险（排名口径：不计较是否严丝合缝落在 top10%，
        # 但在意「选进去就掉队」）。与 agree 同日口径，随机基线见模块 docstring。
        quality = top_group_quality(valid[label_col].to_numpy(dtype="float64"), pred,
                                    valid["trade_date"].to_numpy(), eligible,
                                    valid["flag_limit_up"].to_numpy())
        # 次要基准：实际组改由「窗口内处于截面 top 10% 的天数占比」定义
        # （持久性基准：真正该被选中的是整段窗口都待在顶部的股票）
        stay: dict[str, float | int] = {}
        stay_col = window_label_name("topfrac", h)
        if stay_col in valid.columns:
            sq = top_group_quality(valid[stay_col].to_numpy(dtype="float64"), pred,
                                   valid["trade_date"].to_numpy(), eligible,
                                   valid["flag_limit_up"].to_numpy())
            stay = {f"stay_{k}": v for k, v in sq.items()}
        if h == 1:
            # 官方 ic_mean 即自身 horizon（1 日）IC，同一口径不再重算
            horizon_ic = {"horizon_ic_mean": metrics["ic_mean"],
                          "horizon_ic_std": float("nan"),
                          "horizon_ic_positive_ratio": metrics["ic_positive_ratio"],
                          "horizon_ic_days": metrics["n_days_ic"]}
        else:
            horizon_ic = daily_ic_stats(valid, pred, label_name(h), eligible)
        row = {"fold": fold.name, "model": name, "variant": tag,
               "feature_set": feat_set, "target": target, "horizon": h,
               "n_features": len(columns_by_set[feat_set]),
               "n_train_samples": infos[name]["n_train_samples"],
               **{k: metrics[k] for k in OFFICIAL_METRICS},
               "day1_n": len(members),
               "day1_overlap_H01": len(set(members) & set(base_set)),
               **frozen, **horizon_ic, **agree, **quality, **stay}
        if name == "H01" and expected is not None:
            row["expected_s003_raw_final"] = expected["final_score"]
            row["raw_final_diff_vs_s003"] = abs(metrics["final_score"] - expected["final_score"])
        rows.append(row)
        day1_rows.extend({"fold": fold.name, "model": name, "rank": i + 1,
                          "ts_code": code} for i, code in enumerate(members))
        log(f"  {name}: 排名重合率 {agree['top10_overlap_mean']:.4f} "
            f"(Jaccard {agree['top10_jaccard_mean']:.4f}, {agree['top10_overlap_days']} 天) | "
            f"宽容 hit15 {quality['hit_15']:.4f} hit20 {quality['hit_20']:.4f} "
            f"(基线 0.150/0.200) | 平均实际百分位 {quality['mean_actual_pct']:.4f} "
            f"(基线 0.500) | 掉队率 后50% {quality['downside_50']:.4f} "
            f"后10% {quality['downside_10']:.4f} | "
            f"horizon IC {horizon_ic['horizon_ic_mean']:+.6f} | "
            f"预测组实现收益 {agree['pred_top10_realized_h']:+.4f} vs "
            f"实际组 {agree['oracle_top10_realized_h']:+.4f} | raw final {metrics['final_score']:.6f}")

    fold_json = {
        "fold": fold.name, "split": split, "keep_q_spec": spec["keep_q"],
        "rounds": rounds, "model_params": params, "models": infos,
        "label_verify_passed": True,
        "rows": rows,
    }
    write_json(fold_dir / "fold_report.json", json_safe(fold_json))
    pred_frame = valid[KEYS].copy()
    for name, pred in preds.items():
        pred_frame[name] = pred
    pred_frame.to_parquet(fold_dir / "raw_predictions.parquet", index=False)
    log(f"{fold.name}: 产物写入 {fold_dir}")
    return rows, day1_rows, fold_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="长周期选股实验（raw pred，排名重合度口径，不叠 band）")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--folds", default="fold1,fold2",
                        help="要跑的折，逗号分隔（configs/project.yaml validation.folds 的名字）")
    parser.add_argument("--study", default=None,
                        help="覆盖 configs 的 long_horizon.study（决定产物目录与汇总文件名）")
    parser.add_argument("--variants", default=None,
                        help="只跑 configs long_horizon.variants 里这些 tag，逗号分隔（默认全跑）")
    parser.add_argument("--horizons", default=None,
                        help="覆盖 configs long_horizon.horizons，逗号分隔（如 20）")
    parser.add_argument("--fold-source", default="validation",
                        choices=["validation", "optuna"],
                        help="validation=两折(fold1/fold2)；optuna=四折(wf2021–wf2023+confirm2024)")
    args = parser.parse_args(argv)

    cfg, _ = load_config(args.config)
    settings = cfg["long_horizon"]
    horizons = [int(h) for h in settings["horizons"]]
    if args.horizons:
        horizons = [int(h) for h in str(args.horizons).split(",") if h.strip()]
        if not horizons:
            raise ValueError("--horizons 为空")
    variants = list(settings["variants"])
    if args.variants:
        wanted_tags = [t.strip() for t in args.variants.split(",") if t.strip()]
        variants = [v for v in variants if v["tag"] in wanted_tags]
        if not variants:
            raise ValueError(f"未匹配到变体 {wanted_tags}")
    study = args.study or settings["study"]
    selection = json.loads((ROOT / settings["spec_source"]).read_text(encoding="utf-8"))
    spec = selection["winner"]["spec"]
    log(f"模型配置直读 {settings['spec_source']}：rounds={spec['rounds']}，"
        f"leaves={spec['model_params']['num_leaves']}，horizons={horizons}（不重调参）")
    log(f"变体：{[(v['tag'], v['features'], v['target']) for v in variants]}")

    log("读取训练集并清洗 ...")
    raw = load_training_data(project_path(cfg["paths"]["train"]))
    raw, cleaning = clean_history(raw)
    log(f"  {len(raw):,} 行 / {raw['trade_date'].nunique()} 个交易日；RSS {rss_gb():.2f} GB")

    log("计算 40 个 V1 特征（全历史一次算完，只向后看）...")
    features, columns_v1 = build_features(raw[KEYS + X_COLUMNS], cfg["features"], log)
    dataset = attach_labels(raw, features)
    del features
    gc.collect()

    columns_by_set: dict[str, list[str]] = {"v1": list(columns_v1)}
    if any(v["features"] == "long" for v in variants):
        log("计算长周期专属增量特征（120/250 日窗口，只向后看）...")
        lh_frame, columns_lh = build_long_features(raw[KEYS + X_COLUMNS],
                                                   settings["long_features"], log)
        if not lh_frame[KEYS].equals(dataset[KEYS]):
            raise ValueError("长周期特征的键/顺序与 V1 特征不一致。")
        dataset = pd.concat([dataset, lh_frame[columns_lh]], axis=1)
        columns_by_set["long"] = list(columns_v1) + list(columns_lh)
        del lh_frame
        gc.collect()
        log(f"  长周期特征 {len(columns_lh)} 个；long 特征集共 {len(columns_by_set['long'])} 个")

    log("计算多周期前瞻标签并校验 h=1 与官方 y_ret_1d 逐位一致 ...")
    labels = forward_labels(raw, [1] + horizons)
    verify = verify_against_official(labels[label_name(1)].to_numpy(),
                                     raw["y_ret_1d"], raw["trade_date"].to_numpy())
    log(f"  校验通过：{verify}")
    for h in horizons:
        dataset[label_name(h)] = labels[label_name(h)].to_numpy()
    del labels
    gc.collect()

    log("计算窗口一致性标签（窗口内逐日截面排名聚合，只作监督目标）...")
    # 只算需要的种类（每列约 1.89 MB/万行），topfrac 恒需（持久性基准评估要用）
    kinds = sorted({v["target"] for v in variants if v["target"] in WINDOW_KINDS}
                   | {"topfrac"})
    win_labels = forward_window_labels(raw, horizons, kinds=tuple(kinds))
    for h in horizons:
        for kind in kinds:
            dataset[window_label_name(kind, h)] = win_labels[
                window_label_name(kind, h)].to_numpy()
    for h in horizons:
        col = window_label_name("topfrac", h)
        arr = win_labels[col].to_numpy()
        log(f"  h={h}: 目标 {kinds}；topfrac 覆盖 {np.isfinite(arr).mean():.2%}，"
            f"均值 {np.nanmean(arr):.4f}")
    del raw, win_labels
    gc.collect()

    wanted = [name.strip() for name in args.folds.split(",") if name.strip()]
    fold_pool = get_optuna_folds(cfg) if args.fold_source == "optuna" else get_folds()
    folds = [f for f in fold_pool if f.name in wanted]
    if not folds:
        raise ValueError(f"未找到折 {wanted}（fold_source={args.fold_source}）")
    for f in folds:
        log(f"折 {f.describe()}")
    out_root = project_path("outputs/long_horizon") / study
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_day1: list[dict] = []
    fold_jsons: dict[str, dict] = {}
    for fold in folds:
        rows, day1_rows, fold_json = run_fold(cfg, horizons, spec, selection,
                                              dataset, columns_by_set, variants,
                                              fold, out_root)
        all_rows.extend(rows)
        all_day1.extend(day1_rows)
        fold_jsons[fold.name] = {"split": fold_json["split"], "rows": rows}

    fold_tag = "_".join(f.name for f in folds)
    summary = pd.DataFrame(all_rows)
    summary_path = ROOT / "experiments" / f"{study}_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    # 带折标签的副本：分折并行运行时互不覆盖，便于事后合并
    summary.to_csv(ROOT / "experiments" / f"{study}_summary_{fold_tag}.csv", index=False)
    day1_frame = pd.DataFrame(all_day1)
    day1_frame.to_csv(ROOT / "experiments" / f"{study}_day1_sets.csv", index=False)
    day1_frame.to_csv(ROOT / "experiments" / f"{study}_day1_sets_{fold_tag}.csv", index=False)
    report = json_safe({
        "study": study, "horizons": horizons, "variants": variants,
        "folds": [f.name for f in folds],
        "feature_sets": {k: len(v) for k, v in columns_by_set.items()},
        "spec_source": settings["spec_source"], "spec": spec,
        "cleaning": cleaning, "label_verify": verify,
        "fold_details": fold_jsons, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    write_json(out_root / "experiment_report.json", report)
    write_json(out_root / f"experiment_report_{fold_tag}.json", report)

    pd.set_option("display.width", 260)
    print(f"\n===== {study} 汇总（raw pred，选股质量口径）=====")
    show = summary[["fold", "model", "target", "n_features", "top10_overlap_mean",
                    "hit_15", "hit_20", "mean_actual_pct", "downside_50",
                    "pred_excess_h", "oracle_excess_h", "horizon_ic_mean",
                    "annual_excess", "final_score"]]
    print(show.round(4).to_string(index=False))

    def _pivot(metric: str, label: str, ascending: bool) -> None:
        if metric not in summary.columns:
            return
        piv = summary.pivot_table(index="model", columns="fold", values=metric)
        cols = list(piv.columns)
        piv["两折均值"] = piv[cols].mean(axis=1)
        piv["两折最差"] = piv[cols].min(axis=1)
        print(f"\n----- {label} -----")
        print(piv.round(4).sort_values("两折均值", ascending=ascending).to_string())

    print("\n随机基线：hit_15=0.150 hit_20=0.200 | mean_actual_pct=0.500 | "
          "downside_50=0.500 | 严格 hit_10=0.100")
    _pivot("hit_20", "宽容命中率 hit_20（预测组落在实际 top20% 的比例）", False)
    _pivot("mean_actual_pct", "预测组平均实际百分位（越接近 1 越好）", False)
    _pivot("downside_50", "掉队率 downside_50（越低越好）", True)
    if "stay_hit_20" in summary.columns:
        _pivot("stay_hit_20", "持久性基准 stay_hit_20（实际组=窗口内 top10% 天数占比最高者）", False)
    log(f"汇总已写入 {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

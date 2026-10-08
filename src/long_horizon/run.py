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
- 评估（每折每模型，全部官方口径）：
  1. raw 官方 8 指标（``evaluate_frame``，无 band）；
  2. 首日 Top 1/10 组（换手候选口径：当日报价有效且非涨停）**持有全验证期**
     的实现年化超额；
  3. 自身 horizon 的逐日 Rank IC（>=30 有效样本）。
- 正确性校验：h=1 重算标签与官方 y_ret_1d 逐位一致；H01（1 日基线）raw 指标
  应复现 S003 的 wf2023 / confirm2024 raw 结果（相同环境与监督路径）。

生产接入路径（仅当实验证明选股更优后才考虑，本实验不跑）：
首日 pred 替换为长周期预测、其余交易日 pred 不变，由留仓带机制
（``src/evaluation/turnover.band_scores``）维持首日集合。

用法（仓库根目录）::

    python -m src.long_horizon.run                 # 跑 fold1 + fold2
    python -m src.long_horizon.run --folds fold2   # 只跑 2024 折
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
from src.long_horizon.labels import (forward_labels, label_name,
                                     training_label_mask,
                                     verify_against_official)
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


def expected_raw_metrics(selection: dict, fold: Fold) -> dict | None:
    """从 S003 锁定产物读取同折 raw 基线指标（wf2023 / confirm2024）。"""
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


def run_fold(cfg: dict, horizons: list[int], spec: dict, selection: dict,
             dataset: pd.DataFrame, columns: list[str], fold: Fold,
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

    # --- 1 日基线 H01：与 S003 相同的监督路径（应复现其 raw 指标） ---
    x_train, y_train, array_info = training_arrays(train, columns)
    preds["H01"], infos["H01"] = fit_predict("H01", x_train, y_train, valid,
                                             eligible, columns, params, rounds,
                                             fallback, model_dir)
    infos["H01"]["supervision"] = array_info
    del x_train, y_train
    gc.collect()

    # --- 长周期模型：同参数，目标换成 y_ret_{h}d（末尾 h 日跨界标签剔除） ---
    for h in horizons:
        name = f"H{h:02d}"
        mask = training_label_mask(train, label_name(h), dates_sorted, h, fold.train_end)
        if not mask.any():
            raise ValueError(f"{fold.name} {name}: 无有效监督样本")
        x_train = train.loc[mask, columns]
        y_train = train.loc[mask, label_name(h)].to_numpy(dtype="float64")
        preds[name], infos[name] = fit_predict(name, x_train, y_train, valid,
                                               eligible, columns, params, rounds,
                                               fallback, model_dir)
        infos[name]["supervision"] = {
            "candidate_rows": int(len(train)), "used_rows": int(mask.sum()),
            "dropped_rows": int((~mask).sum())}
        del x_train, y_train
        gc.collect()

    # --- 评估：raw 官方指标 + 首日 Top 组实现超额 + 自身 horizon IC ---
    base_set = day1_top_set(valid, preds["H01"], eligible)
    expected = expected_raw_metrics(selection, fold)
    rows: list[dict] = []
    day1_rows: list[dict] = []
    for name, pred in preds.items():
        h = int(name[1:])
        scored = valid[KEYS + ["y_ret_1d", "flag_limit_up"]].assign(pred=pred)
        metrics = evaluate_frame(scored)
        members = day1_top_set(valid, pred, eligible)
        frozen = frozen_set_annual_excess(valid, set(members))
        if h == 1:
            # 官方 ic_mean 即自身 horizon（1 日）IC，同一口径不再重算
            horizon_ic = {"horizon_ic_mean": metrics["ic_mean"],
                          "horizon_ic_std": float("nan"),
                          "horizon_ic_positive_ratio": metrics["ic_positive_ratio"],
                          "horizon_ic_days": metrics["n_days_ic"]}
        else:
            horizon_ic = daily_ic_stats(valid, pred, label_name(h), eligible)
        row = {"fold": fold.name, "model": name, "horizon": h,
               "n_train_samples": infos[name]["n_train_samples"],
               **{k: metrics[k] for k in OFFICIAL_METRICS},
               "day1_n": len(members),
               "day1_overlap_H01": len(set(members) & set(base_set)),
               **frozen, **horizon_ic}
        if name == "H01" and expected is not None:
            row["expected_s003_raw_final"] = expected["final_score"]
            row["raw_final_diff_vs_s003"] = abs(metrics["final_score"] - expected["final_score"])
        rows.append(row)
        day1_rows.extend({"fold": fold.name, "model": name, "rank": i + 1,
                          "ts_code": code} for i, code in enumerate(members))
        log(f"  {name}: raw final {metrics['final_score']:.6f} | IC {metrics['ic_mean']:+.6f} | "
            f"raw 超额 {metrics['annual_excess']:+.4f} | 冻结组年化超额 "
            f"{frozen['frozen_annual_excess']:+.4f} | 自身 horizon IC "
            f"{horizon_ic['horizon_ic_mean']:+.6f} | 与 H01 首日重合 "
            f"{row['day1_overlap_H01']}/{len(members)}")

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
    parser = argparse.ArgumentParser(description="LH001 长周期选股实验（raw pred，不叠 band）")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--folds", default="fold1,fold2",
                        help="要跑的折，逗号分隔（configs/project.yaml validation.folds 的名字）")
    args = parser.parse_args(argv)

    cfg, _ = load_config(args.config)
    settings = cfg["long_horizon"]
    horizons = [int(h) for h in settings["horizons"]]
    selection = json.loads((ROOT / settings["spec_source"]).read_text(encoding="utf-8"))
    spec = selection["winner"]["spec"]
    log(f"模型配置直读 {settings['spec_source']}：rounds={spec['rounds']}，"
        f"leaves={spec['model_params']['num_leaves']}，horizons={horizons}（不重调参）")

    log("读取训练集并清洗 ...")
    raw = load_training_data(project_path(cfg["paths"]["train"]))
    raw, cleaning = clean_history(raw)
    log(f"  {len(raw):,} 行 / {raw['trade_date'].nunique()} 个交易日；RSS {rss_gb():.2f} GB")

    log("计算 40 个 V1 特征（全历史一次算完，只向后看）...")
    features, columns = build_features(raw[KEYS + X_COLUMNS], cfg["features"], log)
    dataset = attach_labels(raw, features)
    del features
    gc.collect()

    log("计算多周期前瞻标签并校验 h=1 与官方 y_ret_1d 逐位一致 ...")
    labels = forward_labels(raw, [1] + horizons)
    verify = verify_against_official(labels[label_name(1)].to_numpy(),
                                     raw["y_ret_1d"], raw["trade_date"].to_numpy())
    log(f"  校验通过：{verify}")
    for h in horizons:
        dataset[label_name(h)] = labels[label_name(h)].to_numpy()
    del raw, labels
    gc.collect()

    wanted = [name.strip() for name in args.folds.split(",") if name.strip()]
    folds = [f for f in get_folds() if f.name in wanted]
    if not folds:
        raise ValueError(f"未找到折 {wanted}（configs/project.yaml validation.folds）")
    out_root = project_path("outputs/long_horizon") / settings["study"]
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_day1: list[dict] = []
    fold_jsons: dict[str, dict] = {}
    for fold in folds:
        rows, day1_rows, fold_json = run_fold(cfg, horizons, spec, selection,
                                              dataset, columns, fold, out_root)
        all_rows.extend(rows)
        all_day1.extend(day1_rows)
        fold_jsons[fold.name] = {"split": fold_json["split"], "rows": rows}

    summary = pd.DataFrame(all_rows)
    summary_path = ROOT / "experiments" / f"{settings['study']}_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    day1_frame = pd.DataFrame(all_day1)
    day1_frame.to_csv(ROOT / "experiments" / f"{settings['study']}_day1_sets.csv", index=False)
    write_json(out_root / "experiment_report.json", json_safe({
        "study": settings["study"], "horizons": horizons,
        "spec_source": settings["spec_source"], "spec": spec,
        "cleaning": cleaning, "label_verify": verify,
        "folds": fold_jsons, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))

    pd.set_option("display.width", 240)
    print("\n===== LH001 汇总（raw pred，官方口径）=====")
    show = summary[["fold", "model", "horizon", "ic_mean", "annual_excess",
                    "mean_turnover", "final_score", "frozen_annual_excess",
                    "horizon_ic_mean", "day1_overlap_H01"]]
    print(show.to_string(index=False))
    log(f"汇总已写入 {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

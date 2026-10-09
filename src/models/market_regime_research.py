"""S007 市场状态特征研究：固定 T030 配置，仅替换特征集，四折因果验证。

计划来源：《下一步》第十六节与第 5 步（S007 Market Regime Features）：
加入市场宽度、横截面波动、市场成交状态、涨跌停比例等市场状态特征，
只使用赛题允许字段；重点检查 2023（三折最弱年）与 2024 年 1 月是否改善。

协议要点：

- **固定模型**：LightGBM 参数、轮数、种子全部取自 S003 锁定的 T030 winner spec，
  不重新搜索；**不新增任何换手控制器层**，换手层沿用既有 ``band_scores`` 留仓带，
  keep_q 取 T030 的 q*（0.0022778298...）为主口径，另存 keep_q=0.1 作参考。
- **两个特征臂**：``baseline`` = 原 40 特征 V1（用于在本管线内复现 T030 冻结结果，
  逐折断言一致）；``market`` = 40 + 14 个市场状态特征（11 基础 + 3 交互）。
- **特征因果**：市场特征在 [数据起点, 20241231] 的连续历史上一次性计算后再切分；
  40 特征直接复用 S003 的已校验 parquet 缓存（身份与校验和逐项核对）。
- **切分**：wf2021 / wf2022 / wf2023 + confirm2024，冷启动，训练标签不跨边界
  （复用 ``split_train_valid``，与 S003 完全同一实现）。
- **成功标准**（《下一步》第二十节）：2023 折 0.35710 → 0.38+ 且 2024 折保持 0.38+。
- **不触碰测试期**（2025–2026 标签与折结构已不干净，本轮不使用）。

用法（仓库根目录）::

    python -m src.models.market_regime_research

产物：outputs/metrics/research_studies/S007/（逐折 manifest/metrics/prediction）、
experiments/market_regime_S007_*.csv、docs/market_regime_S007.md。
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import clean_history
from src.data.load_data import KEYS, load_training_data
from src.data.make_dataset import attach_labels, training_arrays
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame
from src.evaluation.research_diagnostics import monthly_metrics
from src.evaluation.turnover import band_scores
from src.evaluation.validation import Fold, split_train_valid
from src.features.build_features import feature_names
from src.features.market_features import (MARKET_ALL, MARKET_BASE, MARKET_INTERACTION,
                                          MARKET_WINDOW, market_daily_table,
                                          market_interactions)
from src.models.baseline import RunLog, source_provenance
from src.models.lightgbm_model import train_model
from src.models.optuna_tuning import save_json
from src.utils.experiments import append_record, read_records
from src.utils.project import ROOT, git_state, load_config, project_path, sha256, timestamp
from src.utils.research_cache import code_hashes, contained_path, digest

STUDY_ID = "S007"
FOLD_KEYS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
ARMS = {"baseline": 40, "market": 40 + len(MARKET_ALL)}
BASELINE_TOLERANCE = 1.0e-9


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_references() -> tuple[dict, dict]:
    """S003 锁定的 T030 三折与 2024 确认折官方指标（用于复现断言与对照）。"""
    selection = read_json(ROOT / "docs/optuna_tuning_S003_selection.json")
    confirmation = read_json(ROOT / "docs/optuna_tuning_S003_confirmation.json")
    winner = selection["winner"]
    folds = {name: {layer: entry[layer]["metrics"] for layer in ["raw", "band"]}
             for name, entry in winner["folds"].items()}
    folds["confirm2024"] = {"raw": confirmation["comparison"]["selected_raw"],
                            "band": confirmation["comparison"]["selected_joint"]}
    return winner, folds


def cache_identity(cfg: dict, provenance: dict, start: int, end: int, rows: int) -> dict:
    # market_features.py 不参与 40 特征 V1 的计算，排除后才能与 S003 缓存 identity 逐位对齐。
    relevant = {p: h for p, h in code_hashes().items()
                if (p.startswith("src/features/") and p != "src/features/market_features.py")
                or p in {"src/data/clean_data.py", "src/data/load_data.py",
                         "src/utils/tuning_cache.py"}}
    return {"schema": 1, "raw_sha256": provenance["sha256"], "history": [start, end],
            "rows": rows, "features": cfg["features"], "code": relevant,
            "packages": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "pyarrow"]}}


def load_v1_features(cfg: dict, provenance: dict, raw: pd.DataFrame, log) -> pd.DataFrame:
    """复用并校验 S003 的 40 特征 X 缓存（身份、校验和、列序、键对齐逐项核对）。"""
    columns = feature_names(cfg["features"])
    if len(columns) != 40:
        raise ValueError("S007 expects the frozen 40-feature V1 as its base arm.")
    directory = (contained_path(cfg["paths"]["optuna_cache"])
                 / digest(cache_identity(cfg, provenance, int(raw.trade_date.min()),
                                         int(raw.trade_date.max()), len(raw))))
    metadata = read_json(directory / "cache.json")
    parquet = directory / "features.parquet"
    if metadata.get("status") != "passed" or sha256(parquet) != metadata.get("parquet_sha256"):
        raise ValueError("V1 feature cache checksum mismatch.")
    if metadata["identity"] != cache_identity(cfg, provenance, int(raw.trade_date.min()),
                                              int(raw.trade_date.max()), len(raw)):
        raise ValueError("V1 feature cache identity mismatch; inspect before reuse.")
    if metadata["columns"] != columns:
        raise ValueError("V1 feature cache columns changed.")
    features = pd.read_parquet(parquet)
    features["ts_code"] = pd.Categorical(features.ts_code, categories=raw.ts_code.cat.categories)
    features["trade_date"] = features.trade_date.astype(raw.trade_date.dtype)
    if list(features.columns) != KEYS + columns or not features[KEYS].equals(raw[KEYS]):
        raise ValueError("V1 feature cache keys/row order changed.")
    if any(features[c].dtype != np.float32 for c in columns):
        raise ValueError("V1 feature cache dtypes changed.")
    if np.isinf(features[columns].to_numpy()).any():
        raise ValueError("V1 feature cache contains infinity.")
    log(f"verified S003 V1 cache: {directory.name[:12]} ({len(features):,} rows, 40 features)")
    return features


def build_dataset(cfg: dict, provenance: dict, log) -> tuple[pd.DataFrame, dict]:
    """40 特征 + 标签 + 市场状态特征（广播合并，保持行序），返回合并后数据帧与审计信息。"""
    raw = load_training_data(project_path(cfg["paths"]["train"]))
    if len(raw) != 7_900_350:
        raise ValueError("S007 requires the complete original training CSV.")
    raw = raw[raw.trade_date.between(20180102, 20241231)]
    raw, cleaning = clean_history(raw)
    features = load_v1_features(cfg, provenance, raw, log)
    log(f"computing market daily table (window={MARKET_WINDOW})")
    market = market_daily_table(features, raw)
    dataset = attach_labels(raw, features)
    # attach_labels 只带 y_ret_1d/quote_valid；评分与留仓带还需要 flag_limit_up（行序与 raw 一致）。
    dataset["flag_limit_up"] = raw["flag_limit_up"].to_numpy().astype("int8")
    del raw, features
    gc.collect()
    keys_before = dataset[KEYS].copy()
    dataset = dataset.merge(market, on="trade_date", how="left", validate="many_to_one")
    if not dataset[KEYS].equals(keys_before):
        raise ValueError("Market merge changed panel row order.")
    dataset[MARKET_INTERACTION] = market_interactions(dataset)
    for name in MARKET_ALL:
        if dataset[name].dtype != np.float32:
            raise ValueError(f"Market feature {name} dtype changed.")
    audit = {"rows": len(dataset), "days": int(dataset.trade_date.nunique()),
             "market_window": MARKET_WINDOW, "cleaning": cleaning,
             "market_base": MARKET_BASE, "market_interaction": MARKET_INTERACTION}
    log(f"dataset ready: {len(dataset):,} rows, 40 V1 + {len(MARKET_ALL)} market features")
    return dataset, audit


def fold_definitions(cfg: dict) -> list[Fold]:
    items = list(cfg["optuna"]["folds"]) + [cfg["optuna"]["confirm"]]
    return [Fold(item["name"], *item["train"], *item["valid"]) for item in items]


def evaluate_arm_fold(cfg: dict, dataset: pd.DataFrame, columns: list[str], fold: Fold,
                      spec: dict, out_dir: Path, log) -> dict:
    """单臂单折：切分 → 训练 → 预测 → raw/band(q*)/band(0.1) 官方口径评分。"""
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        saved = read_json(metrics_path)
        if saved.get("spec_digest") == digest(spec) and saved.get("columns") == columns:
            log(f"reuse {out_dir.name}: band q*={saved['band_qstar']['final_score']:.10f}")
            return saved
        raise ValueError(f"Existing S007 metrics disagree with current spec: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, valid, split = split_train_valid(dataset, fold)
    x_train, y_train, supervision = training_arrays(train, columns)
    eligible = valid.quote_valid.to_numpy()
    truth = valid[KEYS + ["y_ret_1d", "flag_limit_up"]].copy()
    truth["flag_limit_up"] = truth.flag_limit_up.astype("int8")
    truth = truth.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    del train
    gc.collect()
    log(f"fit {out_dir.name}: rounds={spec['rounds']}, features={len(columns)}")
    model = train_model(x_train, y_train, spec["model_params"], spec["rounds"],
                        lambda m: log(f"{out_dir.name}: {m}"))
    values = np.full(len(valid), 0.0, dtype="float64")
    values[eligible] = model.predict(valid.loc[eligible, columns],
                                     num_threads=spec["model_params"]["num_threads"])
    importance = pd.DataFrame({"feature": columns,
                               "gain": model.feature_importance(importance_type="gain"),
                               "split": model.feature_importance(importance_type="split")})
    del model, x_train, y_train
    gc.collect()
    if not np.isfinite(values[eligible]).all():
        raise ValueError("Model predictions are not finite.")
    pred = valid[KEYS].assign(pred=values).sort_values(
        ["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    del values
    check_predictions(pred, truth)
    raw_path = out_dir / "prediction_raw.csv"
    pred.to_csv(raw_path, index=False)
    scored = pred.merge(truth, on=KEYS, validate="one_to_one")
    raw_metrics = evaluate_frame(scored)
    raw_differences = compare_official(raw_path, truth, raw_metrics,
                                       cfg["baseline"]["score_tolerance"])
    layers = {"raw": raw_metrics}
    paths = {"raw": raw_path}
    differences = {"raw": raw_differences}
    for label, keep_q in [("band_qstar", spec["keep_q"]), ("band_01", 0.1)]:
        banded = scored[KEYS + ["pred", "flag_limit_up"]].copy()
        banded["pred"] = band_scores(banded, keep_q).to_numpy()
        band_pred = banded[KEYS + ["pred"]]
        band_path = out_dir / f"prediction_{label}.csv"
        band_pred.to_csv(band_path, index=False)
        check_predictions(band_pred, truth)
        band_scored = band_pred.merge(truth, on=KEYS, validate="one_to_one")
        metrics = evaluate_frame(band_scored)
        differences[label] = compare_official(band_path, truth, metrics,
                                              cfg["baseline"]["score_tolerance"])
        layers[label] = metrics
        paths[label] = band_path
        del banded, band_scored
    daily = daily_metrics(scored)
    monthly = monthly_metrics(daily)
    daily.to_csv(out_dir / "daily_metrics.csv", index=False)
    monthly.to_csv(out_dir / "monthly_metrics.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    result = {"study_id": STUDY_ID, "arm": out_dir.parent.name, "fold": fold.name,
              "fold_window": [fold.train_start, fold.train_end, fold.valid_start, fold.valid_end],
              "split": split, "supervision": supervision, "spec_digest": digest(spec),
              "columns": columns, "metrics": layers,
              "official_differences": differences,
              "prediction": {k: relative(v) for k, v in paths.items()},
              "duration_seconds": time.perf_counter() - started,
              "finished_at": timestamp()}
    save_json(metrics_path, result)
    log(f"PASS {out_dir.name}: raw={layers['raw']['final_score']:.10f}, "
        f"band q*={layers['band_qstar']['final_score']:.10f}, "
        f"band 0.1={layers['band_01']['final_score']:.10f}")
    del train, valid, scored, pred, truth
    gc.collect()
    return result


def verify_baseline(results: dict, frozen: dict) -> None:
    """baseline 臂必须逐折复现 S003 冻结的 T030 raw / band(q*) 指标。"""
    for fold_name, entry in results.items():
        for layer in ["raw", "band_qstar"]:
            reference = frozen[fold_name]["raw" if layer == "raw" else "band"]
            for metric in OFFICIAL_METRICS:
                difference = abs(entry["metrics"][layer][metric] - reference[metric])
                if difference > BASELINE_TOLERANCE:
                    raise AssertionError(
                        f"baseline arm {fold_name} {layer} {metric} differs from frozen T030 "
                        f"by {difference:.3e}")


def attribution_table(results: dict) -> pd.DataFrame:
    """market 臂相对 baseline 臂（同 band q*）的官方分项归因。"""
    rows = []
    for fold_name in FOLD_KEYS:
        base = results[("baseline", fold_name)]["metrics"]["band_qstar"]
        market = results[("market", fold_name)]["metrics"]["band_qstar"]
        changes = {"ic_term": 0.4 * (market["ic_mean"] - base["ic_mean"]),
                   "excess_term": 0.3 * (market["annual_excess"] - base["annual_excess"]),
                   "stability_term": -0.3 * (market["mean_turnover"] - base["mean_turnover"])}
        rows.append({"fold": fold_name,
                     "baseline_final": base["final_score"], "market_final": market["final_score"],
                     "delta_final": market["final_score"] - base["final_score"],
                     "delta_ic": market["ic_mean"] - base["ic_mean"],
                     "delta_annual_excess": market["annual_excess"] - base["annual_excess"],
                     "delta_turnover": market["mean_turnover"] - base["mean_turnover"],
                     **changes, "sum_terms": sum(changes.values())})
    return pd.DataFrame(rows)


def fold_metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for (arm, fold_name), entry in results.items():
        for layer, metrics in entry["metrics"].items():
            rows.append({"arm": arm, "fold": fold_name, "layer": layer,
                         **{k: metrics[k] for k in OFFICIAL_METRICS}})
    return pd.DataFrame(rows)


def monthly_table(results: dict) -> pd.DataFrame:
    frames = []
    for (arm, fold_name), entry in results.items():
        monthly = pd.read_csv(project_path(entry["prediction"]["band_qstar"]).parent
                              / "monthly_metrics.csv")
        frames.append(monthly.assign(arm=arm, fold=fold_name))
    return pd.concat(frames, ignore_index=True)


def importance_table(results: dict) -> pd.DataFrame:
    frames = []
    for (arm, fold_name), entry in results.items():
        if arm != "market":
            continue
        importance = pd.read_csv(project_path(entry["prediction"]["band_qstar"]).parent
                                 / "feature_importance.csv")
        market_rows = importance[importance.feature.isin(MARKET_ALL)].copy()
        frames.append(market_rows.assign(fold=fold_name))
    return pd.concat(frames, ignore_index=True)


def append_log_records(cfg: dict, results: dict) -> None:
    log_path = project_path(cfg["paths"]["experiment_log"])
    existing = {r["exp_id"] for r in read_records(log_path)}
    revision = git_state()
    for (arm, fold_name), entry in results.items():
        exp_id = f"{STUDY_ID}_{arm}_{fold_name}_band"
        if exp_id in existing:
            continue
        metrics = entry["metrics"]["band_qstar"]
        keep_q = "q*_0.0022778298112255263"
        record = {"exp_id": exp_id, "date": entry["finished_at"], "owner": "A",
                  "git_commit": revision["commit"], "config_path": "configs/project.yaml",
                  "features": "baseline_v1 (40)" if arm == "baseline" else "baseline_v1+market (54)",
                  "model": "LightGBM T030 spec + band keep_q=" + keep_q,
                  "params": json.dumps({"arm": arm, "spec_digest": entry["spec_digest"]},
                                       sort_keys=True),
                  "train_period": f"{entry['fold_window'][0]}-{entry['fold_window'][1]}",
                  "valid_period": f"{entry['fold_window'][2]}-{entry['fold_window'][3]}",
                  **{k: metrics[k] for k in OFFICIAL_METRICS},
                  "artifact_path": entry["prediction"]["band_qstar"],
                  "notes": f"S007 market regime study; {arm} arm; band q* layer"}
        append_record(log_path, record)


def write_report(cfg: dict, results: dict, frozen: dict, winner_spec: dict,
                 audit: dict, attribution: pd.DataFrame, folds_table: pd.DataFrame) -> None:
    arms_mean = {arm: float(np.mean([results[(arm, f)]["metrics"]["band_qstar"]["final_score"]
                                     for f in FOLD_KEYS[:3]])) for arm in ARMS}
    lines = ["# S007：市场状态特征研究（固定 T030 配置）", "",
             f"更新时间：{timestamp()}", "",
             "## 协议", "",
             "固定 S003 锁定的 T030 LightGBM 配置（800 轮、seed 42，不重搜），仅替换特征集：",
             "baseline = 原 40 特征 V1；market = 40 + 14 个市场状态特征",
             "（11 个日度横截面聚合与 20 日滚动体制统计 + 3 个动量/反转交互）。",
             "市场特征只用赛题允许字段、只使用 t 日及以前信息，在连续历史上一次算完再切分。",
             "换手层沿用既有留仓带（不新增控制器层）：主口径 keep_q = T030 q*，另存 0.1 参考。",
             "四折：wf2021 / wf2022 / wf2023 + confirm2024，冷启动，训练标签不跨验证边界。",
             "baseline 臂逐折复现 S003 冻结 T030 结果（断言 1e-9 内一致）后，与 market 臂同层对比。", "",
             "成功标准（《下一步》第二十节）：2023 折 0.35710 → 0.38+，且 2024 折保持 0.38+。", "",
             "## 特征清单", "",
             "基础（11）：mkt_ret_mean、mkt_ret_median、mkt_ret_std（横截面离散度）、mkt_adv_ratio、",
             "mkt_limit_up_ratio、mkt_limit_down_ratio、mkt_amount_chg（全市场成交额对数变化）、",
             "mkt_ret_mean_20、mkt_ret_vol_20、mkt_breadth_20、mkt_mom_20（20 日市场复利动量）。", "",
             "交互（3）：mkt_x_mom_vol20 = ret_20 × mkt_ret_vol_20；",
             "mkt_x_mom_breadth = ret_20 × mkt_breadth_20；mkt_x_rev_mktmom = ret_1 × mkt_mom_20。", "",
             "## 结果（band keep_q=q*，官方口径）", "",
             "|臂|2021|2022|2023|三折均值|2024|", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        values = [results[(arm, f)]["metrics"]["band_qstar"]["final_score"] for f in FOLD_KEYS]
        lines.append(f"|{arm}|" + "|".join(f"{v:.10f}" for v in values)
                     + f"|{arms_mean[arm]:.10f}|{values[3]:.10f}|")
    lines += ["", "## 分项（band q*，逐折）", "",
              "|折|臂|IC|年化超额|换手|综合分|", "|---|---|---:|---:|---:|---:|"]
    for fold_name in FOLD_KEYS:
        for arm in ARMS:
            m = results[(arm, fold_name)]["metrics"]["band_qstar"]
            lines.append(f"|{fold_name}|{arm}|{m['ic_mean']:.10f}|{m['annual_excess']:.10f}|"
                         f"{m['mean_turnover']:.10f}|{m['final_score']:.10f}|")
    lines += ["", "## market 相对 baseline 的归因（band q*）", "",
              "|折|总分变化|IC 项|超额项|稳定项|ΔIC|Δ超额|Δ换手|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in attribution.iterrows():
        lines.append(f"|{row['fold']}|{row['delta_final']:+.10f}|{row['ic_term']:+.10f}|"
                     f"{row['excess_term']:+.10f}|{row['stability_term']:+.10f}|"
                     f"{row['delta_ic']:+.10f}|{row['delta_annual_excess']:+.10f}|"
                     f"{row['delta_turnover']:+.10f}|")
    lines += ["", "## 市场状态特征的重要性（market 臂，gain）", "",
              "|折|特征|gain|split|", "|---|---|---:|---:|"]
    importance = importance_table(results)
    for fold_name in FOLD_KEYS:
        subset = importance[importance.fold == fold_name].sort_values("gain", ascending=False)
        for _, row in subset.iterrows():
            lines.append(f"|{fold_name}|{row['feature']}|{row['gain']:.1f}|{row['split']}|")
    lines += ["", "## 产物与解释边界", "",
              "研究产物：outputs/metrics/research_studies/S007/；汇总表："
              "experiments/market_regime_S007_{folds,comparison,monthly,importance}.csv。",
              "40 特征直接复用 S003 已校验 parquet 缓存；市场特征由本仓库"
              " src/features/market_features.py 计算。",
              "本轮结论只覆盖 2021–2024 历史验证；测试期（2025–2026）标签已不干净，未参与本轮。",
              "固定单一超参配置，未与特征集做联合搜索；若 market 臂有效，是否需要按特征数"
              "重调 feature_fraction 属后续 S008 的问题。", ""]
    report = contained_path(cfg["paths"]["research_reports"]) / f"market_regime_{STUDY_ID}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    summary_path = ROOT / "experiments" / f"market_regime_{STUDY_ID}_summary.json"
    save_json(summary_path, {
        "study_id": STUDY_ID, "created_at": timestamp(), "protocol": audit,
        "winner_spec": winner_spec, "arms_mean_2021_2023": arms_mean,
        "folds": {f"{arm}/{fold}": results[(arm, fold)]["metrics"]
                  for arm in ARMS for fold in FOLD_KEYS},
        "attribution": attribution.to_dict(orient="records")})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S007 market regime feature study")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--arms", default="baseline,market",
                        help="comma separated: baseline,market")
    args = parser.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if any(a not in ARMS for a in arms):
        parser.error(f"arms must be among {sorted(ARMS)}")
    cfg, _ = load_config(args.config)
    revision = git_state()
    if revision["dirty"]:
        raise ValueError("S007 must run from a clean working tree.")
    winner, frozen = frozen_references()
    spec = winner["spec"]
    provenance = source_provenance(cfg)
    root = contained_path(cfg["paths"]["research_studies"]) / STUDY_ID
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "process.lock"
    if lock.exists():
        raise RuntimeError("S007 study lock exists; inspect before rerunning.")
    lock.write_text(json.dumps({"pid": __import__("os").getpid(), "started_at": timestamp()}),
                    encoding="utf-8")
    try:
        log = RunLog(root / "run.log")
        dataset, audit = build_dataset(cfg, provenance, log)
        columns = {"baseline": feature_names(cfg["features"]),
                   "market": feature_names(cfg["features"]) + MARKET_ALL}
        results = {}
        for fold in fold_definitions(cfg):
            for arm in arms:
                out_dir = root / arm / fold.name
                results[(arm, fold.name)] = evaluate_arm_fold(
                    cfg, dataset, columns[arm], fold, spec, out_dir, log)
            gc.collect()
        if "baseline" in arms:
            verify_baseline({k[1]: v for k, v in results.items() if k[0] == "baseline"}, frozen)
            log("baseline arm reproduces frozen T030 within 1e-9 on all folds")
        if arms == ["baseline", "market"]:
            attribution = attribution_table(results)
            folds_table = fold_metrics_table(results)
            attribution.to_csv(ROOT / "experiments" / f"market_regime_{STUDY_ID}_comparison.csv",
                               index=False)
            folds_table.to_csv(ROOT / "experiments" / f"market_regime_{STUDY_ID}_folds.csv",
                               index=False)
            monthly_table(results).to_csv(
                ROOT / "experiments" / f"market_regime_{STUDY_ID}_monthly.csv", index=False)
            importance_table(results).to_csv(
                ROOT / "experiments" / f"market_regime_{STUDY_ID}_importance.csv", index=False)
            append_log_records(cfg, results)
            write_report(cfg, results, frozen, spec, audit, attribution, folds_table)
            log("S007 complete: report and experiment tables written")
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

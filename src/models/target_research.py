"""S005: five fixed training targets/losses, three folds, fifteen fits."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.load_data import KEYS
from src.evaluation.baseline_checks import check_predictions
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame, load_validation_frame
from src.evaluation.research_diagnostics import monthly_metrics
from src.evaluation.turnover_controllers import official_top, order_fingerprint, prepare_controller_input
from src.models.alpha_research import (is_canonical, quarter_metrics, relative, save_csv,
                                       table_markdown as table_lines, verify_sources, write_text)
from src.models.baseline import RunLog
from src.models.lightgbm_model import load_model
from src.models.optuna_tuning import (ensure_record, provenance, read_json, run_prediction,
                                      save_json, study_lock)
from src.models.target_transforms import array_digest, daily_target
from src.utils.experiments import read_records
from src.utils.project import ROOT, git_state, load_config, sha256, timestamp
from src.utils.research_cache import code_hashes, contained_path, digest
from src.utils.target_cache import TargetFold, prepare_target_folds
from src.utils.tuning_cache import canonical

PACKAGES = ["numpy", "pandas", "scipy", "lightgbm", "pyarrow", "PyYAML", "matplotlib", "optuna", "psutil"]
APPROVED_TARGETS = [
    {"id": "Y1", "transform": "daily_rank", "ties": "average", "objective": "regression", "metric": "l2"},
    {"id": "Y2", "transform": "daily_winsor", "quantiles": [.01, .99], "interpolation": "linear",
     "objective": "regression", "metric": "l2"},
    {"id": "Y3", "transform": "daily_winsor_zscore", "quantiles": [.01, .99], "interpolation": "linear",
     "ddof": 0, "zero_std": 0., "objective": "regression", "metric": "l2"},
    {"id": "Y4", "transform": "daily_winsor_zscore", "quantiles": [.01, .99], "interpolation": "linear",
     "ddof": 0, "zero_std": 0., "objective": "huber", "metric": "huber", "alpha": .9},
    {"id": "Y5", "transform": "raw", "objective": "regression_l1", "metric": "l1"},
]


def table_markdown(frame, columns):
    return "\n".join(table_lines(frame, columns))


def protocol(cfg):
    section = cfg["target_research"]
    if (section["phase"] != "S005" or section["model_fit_budget"] != 15 or section["rounds"] != 800
            or section["targets"] != APPROVED_TARGETS or section["partner_count"] != 2
            or section["selection_objective"] != "mean_raw_predictive_score"
            or section["folds"] != cfg["optuna"]["folds"]
            or section["folds"] != cfg["alpha_research"]["folds"]
            or section["row_order"] != ["trade_date", "ts_code"]
            or cfg["features"]["expected_count"] != 40):
        raise ValueError("S005 protocol differs from the approved fixed 15-fit plan.")
    return {"schema": 1, "target_research": section, "features": cfg["features"],
            "source_selection": cfg["alpha_research"]["source_selection"],
            "source_artifacts": cfg["alpha_research"]["source_artifacts"],
            "paths": {k: cfg["paths"][k] for k in ["train", "test_x", "models", "metrics", "predictions",
                "figures", "experiment_log", "research_studies", "research_reports", "optuna_cache"]},
            "score_tolerance": cfg["baseline"]["score_tolerance"],
            "fallback_prediction": cfg["baseline"]["fallback_prediction"],
            "selection": "mean(0.4*raw_ic_mean+0.3*raw_annual_excess); tol then worst fold then ID"}


def verify_inputs(cfg):
    selection, checked, sources = verify_sources(cfg)
    for key in ["controller_selection", "controller_artifacts", "source_study"]:
        path = contained_path(cfg["target_research"][key])
        checked["frozen_files"][relative(path)] = sha256(path)
    controllers = read_json(contained_path(cfg["target_research"]["controller_selection"]))
    controller_index = read_json(contained_path(cfg["target_research"]["controller_artifacts"]))
    if (digest({k: v for k, v in controllers.items() if k != "selection_digest"}) != controllers["selection_digest"]
            or controller_index["selection_digest"] != controllers["selection_digest"]
            or controller_index["status"] != "passed"):
        raise ValueError("S004 frozen controller handoff differs from its completed audit.")
    # Only the three research folds enter training or scoring. The source study
    # manifest is provenance; its historical 2024 metrics do not guide S005.
    source_study = read_json(contained_path(cfg["target_research"]["source_study"]))
    if (source_study["status"] != "complete" or source_study["selection_digest"] != selection["selection_digest"]
            or source_study["data"] != checked["data"]):
        raise ValueError("The verified S003 study/cache source changed.")
    cache = source_study["search_cache"]
    for path, expected in cache["identity"]["code"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Original feature/cache implementation changed: {path}")
    for name in ["features.parquet", "cache.json"]:
        path = contained_path(cache["path"]) / name
        checked["frozen_files"][relative(path)] = sha256(path)
    if checked["frozen_files"][cache["path"] + "/features.parquet"] != cache["parquet_sha256"]:
        raise ValueError("S003 feature cache bytes changed.")
    checked["controller_selection_digest"] = controllers["selection_digest"]
    return selection, checked, sources


def assert_identity(cfg, meta, *, original_files=False):
    if (digest(protocol(cfg)) != meta["protocol_digest"] or code_hashes() != meta["source"]
            or {p: importlib.metadata.version(p) for p in PACKAGES} != meta["environment"]["packages"]):
        raise ValueError("Code/protocol/packages changed; preserve S005 and use a new identity.")
    if original_files:
        if provenance(cfg) != meta["data"]:
            raise ValueError("Original competition data/attachments changed.")
        for path, expected in meta["frozen_files"].items():
            if sha256(contained_path(path)) != expected:
                raise ValueError(f"Frozen input changed: {path}")


def target_spec(target):
    return {k: v for k, v in target.items() if k not in {"id", "objective", "metric", "alpha"}}


def target_mapping(root, fold: TargetFold, spec, log):
    identity = {"schema": 1, "fold": fold.data.fold.name, "supervised_keys": fold.key_digest,
                "raw_digest": array_digest(fold.data.y_train), "spec": spec}
    directory = root / "targets" / fold.data.fold.name / digest(identity)
    manifest = directory / "target.json"
    if directory.exists():
        if not manifest.exists():
            raise ValueError("Incomplete target mapping is preserved; use a new study ID.")
        info = read_json(manifest)
        if info["identity"] != identity or info["status"] != "passed":
            raise ValueError("Existing target map identity differs.")
        for path, expected in info["artifact_hashes"].items():
            if sha256(contained_path(path)) != expected:
                raise ValueError("Saved target map changed.")
        mapped = pd.read_parquet(contained_path(info["mapping"]))
        values = mapped.transformed_y.to_numpy()
        if (not mapped[KEYS + ["y_ret_1d"]].equals(fold.training)
                or array_digest(values) != info["summary"]["target_digest"]):
            raise ValueError("Saved target keys or raw/transformed labels changed.")
        return values, info
    directory.mkdir(parents=True)
    values, daily, summary = daily_target(fold.training, spec, fold.data.fold.valid_start)
    mapped = fold.training.copy()
    mapped["training_position"] = np.arange(len(mapped), dtype="int64")
    mapped["transformed_y"] = values
    mapping = directory / "mapping.parquet"
    mapped.to_parquet(mapping, index=False)
    recovered = pd.read_parquet(mapping)
    if (not recovered.equals(mapped) or not np.array_equal(recovered.transformed_y.to_numpy(), values)):
        raise ValueError("Saved supervised target mapping is not lossless.")
    dates = np.sort(fold.training.trade_date.unique())
    stop = int(dates[len(dates) // 2 - 1])
    mask = fold.training.trade_date.to_numpy() <= stop
    replay, replay_daily, _ = daily_target(fold.training.loc[mask].reset_index(drop=True), spec,
                                          fold.data.fold.valid_start)
    if (not np.array_equal(replay, values[mask])
            or not replay_daily.equals(daily[daily.trade_date <= stop].reset_index(drop=True))):
        raise ValueError("Daily targets failed actual training-prefix causality replay.")
    stats_path, summary_path = directory / "daily_statistics.csv", directory / "summary.json"
    save_csv(stats_path, daily)
    save_json(summary_path, summary)
    info = {"status": "passed", "identity": identity, "mapping": relative(mapping), "summary": summary,
            "key_digest": fold.key_digest, "daily_statistics": relative(stats_path),
            "prefix_replay": {"end": stop, "rows": int(mask.sum()), "passed": True},
            "artifact_hashes": {relative(p): sha256(p) for p in [mapping, stats_path, summary_path]}}
    save_json(manifest, info)
    info["manifest"] = relative(manifest)
    info["manifest_sha256"] = sha256(manifest)
    # Do not put the manifest's own hash inside its bytes.
    log(f"target {fold.data.fold.name} {spec['transform']}: mapping/prefix passed; "
        f"range=({summary['target_min']:.4f},{summary['target_max']:.4f})")
    del mapped, recovered
    gc.collect()
    return values, info


def predictive(metrics):
    return .4 * metrics["ic_mean"] + .3 * metrics["annual_excess"]


def agreement(scored, reference):
    """Unsupervised agreement: full-market ranks, eligible Top without y masks."""
    if not scored[KEYS].equals(reference[KEYS]):
        raise ValueError("Agreement diagnostics require canonical aligned keys.")
    data = prepare_controller_input(scored[KEYS + ["pred", "flag_limit_up"]])
    ref_ranks = reference.pred.groupby(reference.trade_date).rank(method="average", pct=True).to_numpy().reshape(data.raw.shape)
    current_top = official_top(data, data.raw)
    base_top = official_top(data, reference.pred.to_numpy().reshape(data.raw.shape))
    rows = []
    for t, date in enumerate(data.dates):
        a, b = set(np.flatnonzero(current_top[t]).tolist()), set(np.flatnonzero(base_top[t]).tolist())
        rc = float(np.corrcoef(data.rank[t], ref_ranks[t])[0, 1])
        rows.append({"trade_date": int(date), "rank_correlation": rc,
                     "eligible_top_overlap": len(a & b) / len(a) if a else 0.,
                     "eligible_top_jaccard": len(a & b) / len(a | b) if a | b else 0.,
                     "all_market_exact_ties": int(len(data.stocks) - np.unique(data.raw[t]).size)})
    result = pd.DataFrame(rows)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Prediction agreement contains a non-finite diagnostic.")
    return result


def add_diagnostics(cfg, root, fold: TargetFold, source, info, target_id, target_info=None):
    if target_id == "Y0":
        directory = root / "references" / fold.data.fold.name
    else:
        directory = contained_path(cfg["paths"]["metrics"]) / info["exp_id"]
    directory.mkdir(parents=True, exist_ok=True)
    scored = load_validation_frame(contained_path(info["prediction"]), fold.data.labels_path)
    reference = load_validation_frame(contained_path(source["raw"]["prediction"]), fold.data.labels_path)
    check_predictions(scored[KEYS + ["pred"]], fold.data.truth)
    if not is_canonical(scored) or not scored[KEYS].equals(reference[KEYS]):
        raise ValueError("Raw prediction scoring keys/order differ from T030.")
    if target_id == "Y0":
        metrics = evaluate_frame(scored)
        if max(abs(metrics[k] - info["metrics"][k]) for k in OFFICIAL_METRICS) > 1e-10:
            raise ValueError("Y0 raw metrics no longer reproduce.")
    else:
        restored = load_model(contained_path(info["model"]))
        native = np.full(len(fold.data.keys), cfg["baseline"]["fallback_prediction"], dtype="float64")
        native[fold.data.eligible] = restored.predict(fold.data.x_valid, num_threads=8)
        native = canonical(fold.data.keys.assign(pred=native)).pred.to_numpy()
        saved = scored.pred.to_numpy()
        shape = (scored.trade_date.nunique(), 4650)
        if (np.max(np.abs(native - saved)) > 1e-12
                or order_fingerprint(native.reshape(shape)) != order_fingerprint(saved.reshape(shape))):
            raise ValueError("Saved CSV changed model prediction order or exact ties.")
        info["csv_order_and_ties_preserved"] = True
        info["csv_max_difference"] = float(np.max(np.abs(native - saved)))
        del native, saved, restored
    daily = daily_metrics(scored)
    monthly, quarterly = monthly_metrics(daily), quarter_metrics(daily)
    for frame in [monthly, quarterly]:
        frame["predictive_score"] = .4 * frame.ic_mean + .3 * frame.annual_excess
    matches = agreement(scored, reference)
    diagnostic_paths = []
    for name, frame in [("daily_metrics", daily), ("monthly_metrics", monthly),
                         ("quarterly_metrics", quarterly), ("agreement_daily", matches)]:
        path = directory / f"{name}.csv"
        save_csv(path, frame)
        diagnostic_paths.append(path)
    info = copy.deepcopy(info)
    info.update(target_id=target_id, target_mapping=target_info,
                predictive_score=predictive(info["metrics"]),
                agreement={"rank_correlation": float(matches.rank_correlation.mean()),
                           "eligible_top_overlap": float(matches.eligible_top_overlap.mean()),
                           "eligible_top_jaccard": float(matches.eligible_top_jaccard.mean()),
                           "mean_exact_ties": float(matches.all_market_exact_ties.mean())},
                diagnostics=relative(directory), new_model_fits=0 if target_id == "Y0" else 1,
                signal_semantics="sorting signal; no inverse transform in prediction",
                features="baseline_v1 (40)")
    info["artifact_hashes"].update({relative(p): sha256(p) for p in diagnostic_paths})
    if target_id != "Y0":
        manifest = contained_path(cfg["paths"]["models"]) / info["exp_id"] / "run.json"
        save_json(manifest, info)
        ensure_record(cfg, info, manifest)
    del scored, reference, daily, matches
    gc.collect()
    return info


def run_target(cfg, config_path, root, meta, fold: TargetFold, source, target, log):
    exp_id = f"{meta['study_id']}_{target['id']}_{fold.data.fold.name}_raw"
    target_y, mapped = target_mapping(root, fold, target_spec(target), log)
    # Identical transform identities share one immutable map (Y3 == Y4).
    target_manifest = root / "targets" / fold.data.fold.name / digest(mapped["identity"]) / "target.json"
    mapped.update(manifest=relative(target_manifest), manifest_sha256=sha256(target_manifest))
    params = copy.deepcopy(meta["base_spec"]["model_params"])
    params.update({k: target[k] for k in ["objective", "metric"]})
    if "alpha" in target:
        params["alpha"] = target["alpha"]
    unchanged = {k: v for k, v in params.items() if k not in {"objective", "metric", "alpha"}}
    frozen = {k: v for k, v in meta["base_spec"]["model_params"].items() if k not in {"objective", "metric", "alpha"}}
    if unchanged != frozen:
        raise ValueError("Structural/regularization parameters differ from frozen T030.")
    spec = {"model_params": params, "rounds": 800, "target_id": target["id"],
            "target_transform": target_spec(target), "target_mapping": mapped["mapping"],
            "target_mapping_sha256": mapped["artifact_hashes"][mapped["mapping"]],
            "training_keys_digest": fold.key_digest, "source_model": "S003 T030", "layer": "raw"}
    manifest = contained_path(cfg["paths"]["models"]) / exp_id / "run.json"
    if manifest.exists():
        previous = read_json(manifest)
        if previous["status"] != "passed":
            raise ValueError("Interrupted/failed model fit is preserved and cannot consume an extra retry.")
    elif any((contained_path(cfg["paths"][k]) / exp_id).exists() for k in ["models", "metrics", "predictions"]):
        raise ValueError("Incomplete model artifacts are preserved; no automatic retry.")
    fitted = replace(fold.data, y_train=target_y)
    info = run_prediction(cfg, config_path, root, meta, fitted, exp_id, spec, log)
    info = add_diagnostics(cfg, root, fold, source, info, target["id"], mapped)
    if info["actual_iterations"] != 800:
        log(f"NOTE {exp_id}: requested=800, actual={info['actual_iterations']}; no validation early stopping")
    del fitted, target_y
    gc.collect()
    return info


def summarize(target, runs):
    scores = [r["predictive_score"] for r in runs]
    return {"id": target["id"], "target": target, "folds": runs, "status": "COMPLETE",
            "predictive_mean": float(np.mean(scores)), "predictive_std": float(np.std(scores)),
            "predictive_worst": float(min(scores)), "predictive_2023": runs[-1]["predictive_score"],
            "raw_final_mean": float(np.mean([r["metrics"]["final_score"] for r in runs])),
            "mean_metrics": {k: float(np.mean([r["metrics"][k] for r in runs])) for k in OFFICIAL_METRICS},
            "agreement": {k: float(np.mean([r["agreement"][k] for r in runs])) for k in runs[0]["agreement"]}}


def rank_candidates(results, tol):
    pool, ranked = list(results), []
    while pool:
        highest = max(r["predictive_mean"] for r in pool)
        tied = [r for r in pool if highest - r["predictive_mean"] <= tol]
        chosen = sorted(tied, key=lambda r: (-r["predictive_worst"], r["id"]))[0]
        ranked.append(chosen)
        pool.remove(chosen)
    return ranked


def publish(cfg, meta, root, results, selection, audit):
    experiments = contained_path(cfg["paths"]["experiment_log"]).parent
    report_dir = contained_path(cfg["paths"]["research_reports"])
    figures = contained_path(cfg["paths"]["figures"]) / meta["study_id"]
    figures.mkdir(parents=True, exist_ok=True)
    prefix = f"alpha_{meta['study_id']}"
    selected = {r["id"] for r in selection["partners"]}
    comparison, folds, months, quarters, agreement_rows, statistics, importance = [], [], [], [], [], [], []
    for r in results:
        comparison.append({"target": r["id"], "transform": r["target"]["transform"],
            "objective": r["target"]["objective"], "selected_for_S006": r["id"] in selected,
            "predictive_mean": r["predictive_mean"], "predictive_std": r["predictive_std"],
            "predictive_worst": r["predictive_worst"], "predictive_2023": r["predictive_2023"],
            "raw_final_mean": r["raw_final_mean"], **{"mean_"+k: v for k, v in r["mean_metrics"].items()},
            **r["agreement"]})
        for run in r["folds"]:
            fold = run["split"]["fold"]
            context = {"target": r["id"], "fold": fold, "exp_id": run["exp_id"]}
            folds.append({**context, **run["metrics"], "predictive_score": run["predictive_score"],
                          **run["agreement"], "new_model_fits": run["new_model_fits"],
                          "actual_iterations": run["actual_iterations"], "prediction": run["prediction"]})
            directory = contained_path(run["diagnostics"])
            for rows, name in [(months, "monthly_metrics"), (quarters, "quarterly_metrics"),
                               (agreement_rows, "agreement_daily")]:
                rows.extend({**context, **row} for row in pd.read_csv(directory / f"{name}.csv").to_dict("records"))
            source_importance = contained_path(cfg["paths"]["metrics"]) / run["exp_id"] / "feature_importance.csv"
            imp = pd.read_csv(source_importance)
            total = imp.gain.sum()
            imp["gain_share"] = imp.gain / total if total > 0 else 0.
            importance.extend({**context, **row} for row in imp.to_dict("records"))
            if run["target_mapping"]:
                statistics.append({**context, **{k: v for k, v in run["target_mapping"]["summary"].items()
                                                if k not in {"spec"}},
                                   "key_digest": run["target_mapping"]["key_digest"],
                                   "mapping": run["target_mapping"]["mapping"]})
    table = pd.DataFrame(comparison)
    means = table[["mean_ic_mean", "mean_annual_excess"]].to_numpy()
    table["pareto_ic_excess"] = [not any(np.all(b >= a) and np.any(b > a) for b in means) for a in means]
    baseline = table.loc[table.target == "Y0"].iloc[0]
    table["delta_predictive_vs_Y0"] = table.predictive_mean - baseline.predictive_mean
    table["delta_2023_predictive_vs_Y0"] = table.predictive_2023 - baseline.predictive_2023
    for name, frame in [("comparison", table), ("folds", pd.DataFrame(folds)), ("monthly", pd.DataFrame(months)),
                         ("quarterly", pd.DataFrame(quarters)), ("agreement", pd.DataFrame(agreement_rows)),
                         ("target_statistics", pd.DataFrame(statistics)), ("importance", pd.DataFrame(importance))]:
        save_csv(experiments / f"{prefix}_{name}.csv", frame)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = dict(zip(table.target, plt.get_cmap("tab10").colors[:len(table)]))
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = np.arange(len(table))
    for i, year in enumerate([2021, 2022, 2023]):
        frame = pd.DataFrame(folds)
        values = frame[frame.fold == f"wf{year}"].set_index("target").reindex(table.target).predictive_score
        ax.bar(x + (i-1)*.24, values, .24, label=str(year))
    ax.set_xticks(x, table.target)
    ax.set_ylabel("Raw PredictiveScore (0.4 IC + 0.3 excess)")
    ax.set_title("S005: fixed T030 parameters, training targets/losses")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False)
    fig.savefig(figures / f"{prefix}_year_scores.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 5.5), constrained_layout=True)
    for row in table.to_dict("records"):
        ax.scatter(row["mean_ic_mean"], row["mean_annual_excess"], s=110 if row["target"] in selected else 60,
                   color=palette[row["target"]], edgecolors="black" if row["pareto_ic_excess"] else "none")
        ax.annotate(row["target"] + (" *" if row["target"] in selected else ""),
                    (row["mean_ic_mean"], row["mean_annual_excess"]), xytext=(6, 7), textcoords="offset points")
    ax.set_xlabel("Mean daily Rank IC (three-fold mean)"); ax.set_ylabel("Annual excess (three-fold mean)")
    ax.set_title("Raw IC/excess; * = frozen S006 partner; border = Pareto point")
    ax.margins(.18)
    fig.savefig(figures / f"{prefix}_pareto.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    monthly = pd.DataFrame(months)
    for ax, year in zip(axes, [2021, 2022, 2023]):
        for target in ["Y0"] + sorted(selected):
            data = monthly[(monthly.target == target) & (monthly.fold == f"wf{year}")]
            ax.plot(np.arange(1, len(data)+1), data.predictive_score, label=target, color=palette[target])
        ax.set_title(str(year)); ax.set_ylabel("PredictiveScore"); ax.set_xticks(range(1, 13)); ax.grid(alpha=.2)
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.2), frameon=False)
    axes[-1].set_xlabel("Month (annual predictions, no refit)")
    fig.savefig(figures / f"{prefix}_monthly.png", dpi=150); plt.close(fig)
    published_figures = []
    for source in sorted(figures.glob("*.png")):
        destination = report_dir / "figures" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        published_figures.append(destination)
    report = report_dir / f"alpha_research_{meta['study_id']}.md"
    best = max(results[1:], key=lambda r: r["predictive_mean"])
    raw_reference = results[0]
    transformed_match = all(a["target_mapping"]["summary"]["target_digest"] == b["target_mapping"]["summary"]["target_digest"]
        for a, b in zip(results[3]["folds"], results[4]["folds"]))
    lines = [f"# {meta['study_id']}：训练目标与损失研究", "", f"完成时间：{timestamp()}（Asia/Shanghai）。",
        "", "## 协议与来源", "",
        f"实现提交 `{meta['git']['commit']}`；协议摘要 `{meta['protocol_digest']}`；源码摘要 `{meta['source_digest']}`。",
        f"参照为冻结 S003 T030，选择摘要 `{meta['source_selection_digest']}`；S004 控制器选择保持 `{meta['controller_selection_digest']}`。",
        "固定 V1 40 特征、T030 的结构/正则/采样参数、800 轮、seed=42、CPU 8 线程；每次独立 Dataset，无验证早停。",
        "训练为 2018–2020 / 2021 / 2022，分别验证 2021 / 2022 / 2023。先屏蔽跨界标签，再删除缺 y 或缺价监督样本，再构造目标。",
        "原始预测覆盖 4650 只股票的全部键，缺价行输出 0，按 trade_date、ts_code 排序；验证始终使用原始 y_ret_1d。2024 本轮未训练或评分。",
        "", "## 目标与原始预测比较", "",
        "Y0 复用原始收益 L2；Y1 为平均并列日百分位；Y2 为每日线性 1%/99% 缩尾；Y3 为 Y2 的 ddof=0 日标准化（零标准差输出 0）；Y4 与 Y3 目标完全相同、Huber alpha=0.9；Y5 为原始收益 L1。",
        "Y1/Y3/Y4 的输出是排序信号，不能解释成收益百分比。目标统计只作用于监督 y，不进入 X 或验证预测处理。",
        "研究初筛使用 **PredictiveScore = 0.4×IC + 0.3×年化超额**，它不是官方综合分；各原始预测同时公开全部八项官方指标。",
        "", table_markdown(table, ["target", "transform", "objective", "predictive_mean", "predictive_worst",
            "predictive_2023", "mean_ic_mean", "mean_annual_excess", "mean_mean_turnover", "raw_final_mean", "selected_for_S006"]),
        "", "### 逐年官方指标", "",
        table_markdown(pd.DataFrame(folds), ["target", "fold", "ic_mean", "annual_excess", "mean_turnover", "final_score", "predictive_score"]),
        "", "## 归因与互补性", "",
        f"新目标最高原始 PredictiveScore 为 {best['id']} 的 {best['predictive_mean']:.10f}，相对 Y0 {raw_reference['predictive_mean']:.10f} 变化 {best['predictive_mean']-raw_reference['predictive_mean']:+.10f}；2023 变化 {best['predictive_2023']-raw_reference['predictive_2023']:+.10f}。",
        "IC/超额二维非支配点为 " + "、".join(table.loc[table.pareto_ic_excess, "target"]) + "；它用于解释，不改变预定选优规则。",
        "与 T030 的相关性为全市场当日平均百分位的 Pearson 相关（即 Spearman）；Top 重合只删除涨停，使用官方非稳定降序排序，不读取 y 或缺失掩码。",
        "", table_markdown(table, ["target", "rank_correlation", "eligible_top_overlap", "eligible_top_jaccard", "mean_exact_ties"]),
        "", f"Y3/Y4 三折训练目标摘要完全相同：{transformed_match}；差异隔离为损失变化。相同原始收益下的 Y5/Y0 对照隔离为 L1/L2。",
        "目标尺度变化时固定 lambda 等正则参数的相对作用也会变化；本轮只能说明固定 T030 预算下的结果。收益改善与 IC 改善分别报告，较低相关性不保证融合提分。",
        "", "## S006 交接与阶段结论", "",
        "按三折原始 PredictiveScore 均值选择两个新目标，同分容差 1e-6 内优先最差折、编号；冻结伙伴为 **" + "、".join(sorted(selected)) + "**。",
        "这是下一阶段融合的研究伙伴，不是正式替换方案。本轮没有对新目标重扫 band，也未启动 S006 或 2024 确认；完整官方分和弱年门槛在 S006 判断。正式研究参照继续保留 S003。",
        "成员 B 接续独立核对训练日期、目标统计、损失参数、完整预测与原始评分；目前不声称已完成 B 复核。",
        "", "## 实际核对与产物", "",
        f"15 次新模型训练 / 15 条新增真实记录；Y0 三折复用不记新训练。官方八指标最大差 {audit['max_official_difference']:.3e}；模型重载最大差 {audit['max_reload_difference']:.3e}；保存 CSV 排名与并列组一致。",
        f"12 份唯一训练目标映射，全部实际训练前缀重放通过；Y3/Y4 共用映射。跨界剔除标签为 3853 / 4170 / 4280，相同监督样本数为 2594048 / 3568878 / 4597785。",
        f"原 {audit['initial_records']} 条实验记录保持不变，新增 {audit['new_records']} 条，共 {audit['total_records']} 条；原始数据、官方附件、S003/S004 来源和特征缓存不变。未新增或运行测试套件。",
        f"源码与数值协议先提交推送后执行。选择 canonical digest `{selection['selection_digest']}`；完整 SHA 索引见 `alpha_research_{meta['study_id']}_artifacts.json`。",
        "", "- `experiments/alpha_S005_{comparison,folds,monthly,quarterly,agreement,target_statistics,importance}.csv`：小型指标、非支配与解释表。",
        "- `outputs/metrics/research_studies/S005/targets/`：按键的 raw y / transformed y / training_position、每日统计、摘要与因果前缀核对。",
        "- `outputs/models/S005_Yx_wf20xx_raw/`：模型、配置、运行清单；`outputs/predictions/`：全量原始预测。",
        "- `outputs/metrics/S005_Yx_wf20xx_raw/`：官方八指标、日/月/季度、模型重要性、按日预测相关性与实际 Top 重合。大型产物保留本地忽略目录。",
        "", "## 图表", ""]
    for path in published_figures:
        lines += [f"![{path.stem}](figures/{path.name})", ""]
    lines += ["## API 依据", "", "损失与 alpha 支持核对于 2026-10-08，见 [LightGBM 4.7.0 官方参数](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html#objective-parameters)。这些参数支持事实不构成效果保证。"]
    write_text(report, "\n".join(lines))
    selection_path = report_dir / f"alpha_research_{meta['study_id']}_selection.json"
    save_json(selection_path, selection)
    pub_files = [report, selection_path, *published_figures, *sorted(experiments.glob(f"{prefix}_*.csv"))]
    audit["published_files"] = {relative(p): {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in pub_files}
    save_json(root / "audit.json", audit)
    save_json(report_dir / f"alpha_research_{meta['study_id']}_artifacts.json", audit)


def run_study(cfg, config_path, study_id, owner, resume):
    if not re.fullmatch(r"S005(?:_[A-Za-z0-9_-]+)?", study_id):
        raise ValueError("This entry point implements only S005.")
    fixed_protocol = protocol(cfg)
    root = contained_path(cfg["paths"]["research_studies"]) / study_id
    revision = git_state()
    if revision["branch"] != "main":
        raise ValueError("Work must be on user-authorized main.")
    if not resume:
        if revision["dirty"] or root.exists():
            raise ValueError("Start from committed clean main and a new study identity.")
        origin = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
        if origin != revision["commit"]:
            raise ValueError("Push the fixed protocol to origin/main before training.")
        if shutil.disk_usage(ROOT).free < 15 * 2**30:
            raise ValueError("At least 15 GiB is needed for complete model/target artifacts.")
        root.mkdir(parents=True)
    elif not (root / "study.json").exists():
        raise ValueError("No study metadata exists to resume.")
    log = RunLog(root / "run.log")
    with study_lock(root):
        try:
            source_selection, checked, sources = verify_inputs(cfg)
            if resume:
                meta = read_json(root / "study.json")
                if meta["status"] == "complete":
                    raise ValueError("Completed S005 cannot be rerun or reselected.")
                assert_identity(cfg, meta)
                if any(checked[k] != meta[k] for k in checked):
                    raise ValueError("Frozen S003/S004 input identity changed.")
            else:
                meta = {"schema": 1, "study_id": study_id, "owner": owner, "status": "running",
                        "started_at": timestamp(), "git": revision, "source": code_hashes(),
                        "protocol": fixed_protocol, "protocol_digest": digest(fixed_protocol),
                        "base_spec": source_selection["winner"]["spec"],
                        "source_selection_digest": source_selection["selection_digest"],
                        "initial_records": read_records(contained_path(cfg["paths"]["experiment_log"])),
                        "environment": {"python": sys.version, "platform": platform.platform(),
                            "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}, **checked}
                meta["source_digest"] = digest(meta["source"])
                save_json(root / "study.json", meta)
                write_text(root / "config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            folds, cache = prepare_target_folds(cfg, root, meta, sources, log)
            meta["cache"] = cache
            save_json(root / "study.json", meta)
            y0 = {"id": "Y0", "transform": "raw", "objective": "regression", "metric": "l2"}
            references = [add_diagnostics(cfg, root, fold, source, source["raw"], "Y0")
                          for fold, source in zip(folds, sources)]
            results = [summarize(y0, references)]
            for target in cfg["target_research"]["targets"]:
                assert_identity(cfg, meta)
                candidate_file = root / "candidates" / f"{target['id']}.json"
                log(f"target {target['id']}: fixed {target['objective']} / {target['transform']}")
                runs = [run_target(cfg, config_path, root, meta, fold, source, target, log)
                        for fold, source in zip(folds, sources)]
                result = summarize(target, runs)
                if candidate_file.exists() and read_json(candidate_file) != result:
                    raise ValueError("Previously completed target changed.")
                save_json(candidate_file, result)
                results.append(result)
                meta.update(completed_targets=len(results)-1, completed_fits=3*(len(results)-1),
                            last_target=target["id"], updated_at=timestamp())
                save_json(root / "study.json", meta)
                log(f"COMPLETE {target['id']}: raw predictive={result['predictive_mean']:.10f}; "
                    f"raw official={result['raw_final_mean']:.10f}; 2023 predictive={result['predictive_2023']:.10f}")
            ranked = rank_candidates(results[1:], cfg["target_research"]["selection_tolerance"])
            partners = []
            for chosen in ranked[:cfg["target_research"]["partner_count"]]:
                partners.append({k: v for k, v in chosen.items() if k != "folds"})
                partners[-1]["runs"] = {run["split"]["fold"]: {"exp_id": run["exp_id"],
                    "prediction": run["prediction"], "prediction_sha256": run["artifact_hashes"][run["prediction"]],
                    "model": run["model"], "model_sha256": run["artifact_hashes"][run["model"]],
                    "spec": run["spec"], "target_mapping": run["target_mapping"]} for run in chosen["folds"]}
            selection = {"schema": 1, "study_id": study_id, "created_at": timestamp(), "git": meta["git"],
                "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
                "source_selection_digest": meta["source_selection_digest"],
                "controller_selection_digest": meta["controller_selection_digest"],
                "partners": partners, "baseline": {k: v for k, v in results[0].items() if k != "folds"},
                "rule": fixed_protocol["selection"], "evaluated_years": [2021, 2022, 2023],
                "new_model_fits": 15, "row_order": ["trade_date", "ts_code"],
                "formal_adoption": False, "next_phase": "S006; frozen partners only; no 2024 evaluation yet"}
            selection["selection_digest"] = digest(selection)
            save_json(root / "selection.json", selection)
            assert_identity(cfg, meta, original_files=True)
            records = read_records(contained_path(cfg["paths"]["experiment_log"]))
            new_runs = [run for r in results[1:] for run in r["folds"]]
            if (records[:len(meta["initial_records"])] != meta["initial_records"]
                    or len(records) != len(meta["initial_records"]) + 15
                    or {r["exp_id"] for r in records[len(meta["initial_records"]):]} != {r["exp_id"] for r in new_runs}):
                raise ValueError("Old records changed or new records differ from exactly 15 valid fits.")
            maps = {r["target_mapping"]["mapping"]: r["target_mapping"] for r in new_runs}
            if len(maps) != 12:
                raise ValueError("The five targets should share exactly 12 unique fold/transform maps.")
            for run in new_runs:
                manifest = contained_path(cfg["paths"]["models"]) / run["exp_id"] / "run.json"
                if read_json(manifest) != run:
                    raise ValueError("Passed run manifest changed.")
                for path, expected in run["artifact_hashes"].items():
                    if sha256(contained_path(path)) != expected:
                        raise ValueError(f"Passed model/prediction artifact changed: {path}")
                ensure_record(cfg, run, manifest)
            for mapped in maps.values():
                if sha256(contained_path(mapped["manifest"])) != mapped["manifest_sha256"]:
                    raise ValueError("Target manifest changed.")
                for path, expected in mapped["artifact_hashes"].items():
                    if sha256(contained_path(path)) != expected:
                        raise ValueError("Target mapping/statistics changed.")
            audit = {"status": "passed", "study_id": study_id, "time": timestamp(),
                "implementation_commit": meta["git"]["commit"], "source_digest": meta["source_digest"],
                "protocol_digest": meta["protocol_digest"], "selection_digest": selection["selection_digest"],
                "complete_targets": 5, "failed_fits": 0, "new_model_fits": 15, "reused_reference_folds": 3,
                "evaluated_years": [2021, 2022, 2023], "unique_target_maps": 12,
                "target_prefix_replays": [m["prefix_replay"] for m in maps.values()],
                "initial_records": len(meta["initial_records"]), "new_records": 15, "total_records": len(records),
                "previous_records_unchanged": True, "original_files_unchanged": True,
                "frozen_S003_S004_and_X_cache_unchanged": True,
                "max_official_difference": max(r["official_max_difference"] for r in new_runs),
                "max_reload_difference": max(r["reload_max_difference"] for r in new_runs),
                "max_csv_difference": max(r["csv_max_difference"] for r in new_runs),
                "all_saved_csv_orders_and_ties_match": True, "data_provenance": meta["data"],
                "frozen_input_hashes": meta["frozen_files"], "target_maps": list(maps.values()),
                "runs": [{"exp_id": r["exp_id"], "manifest": relative(contained_path(cfg["paths"]["models"]) / r["exp_id"] / "run.json"),
                    "manifest_sha256": sha256(contained_path(cfg["paths"]["models"]) / r["exp_id"] / "run.json"),
                    "artifact_hashes": r["artifact_hashes"]} for r in new_runs]}
            publish(cfg, meta, root, results, selection, audit)
            meta.update(status="complete", completed_at=timestamp(), selection_digest=selection["selection_digest"],
                        new_model_fits=15, selected_partners=[p["id"] for p in partners])
            save_json(root / "study.json", meta)
            log(f"S005 complete: 15 fits, 12 maps; selected={[p['id'] for p in partners]}; "
                f"official_max_difference={audit['max_official_difference']:.2e}")
        except Exception:
            if (root / "study.json").exists():
                meta = read_json(root / "study.json")
                if meta["status"] != "complete":
                    meta.update(status="interrupted", error=traceback.format_exc(), interrupted_at=timestamp())
                    save_json(root / "study.json", meta)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study-id", default="S005")
    parser.add_argument("--owner", default="A")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg, config_path = load_config(args.config)
    run_study(cfg, config_path, args.study_id, args.owner, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

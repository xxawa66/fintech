"""S004: reuse frozen T030 predictions and score a bounded controller matrix.

Only 2021--2023 are read by this entry point. It never fits a model. Results
that are identical in complete order/tie groups are evaluated once and aliased.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.load_data import KEYS
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame, load_validation_frame
from src.evaluation.research_diagnostics import monthly_metrics
from src.evaluation.turnover_controllers import (ControllerInput, ControllerResult, holdings_diagnostics,
    official_top, order_fingerprint, prepare_controller_input, transform)
from src.models.baseline import RunLog
from src.models.optuna_tuning import ensure_record, provenance, read_json, save_json, study_lock
from src.utils.experiments import read_records
from src.utils.project import ROOT, git_state, load_config, sha256, timestamp
from src.utils.research_cache import code_hashes, contained_path, digest
from src.utils.tuning_cache import canonical

PACKAGES = ["numpy", "pandas", "scipy", "PyYAML", "matplotlib"]


@dataclass
class ControllerFold:
    definition: dict
    data: ControllerInput
    truth: pd.DataFrame
    source: dict


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((text.rstrip() + "\n").encode("utf-8"))


def is_canonical(frame: pd.DataFrame) -> bool:
    normalized = frame[KEYS].astype({"ts_code": "str", "trade_date": "int64"}).reset_index(drop=True)
    ordered = normalized.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    return normalized.equals(ordered)


def protocol(cfg: dict) -> dict:
    section = cfg["alpha_research"]
    result = {"schema": 1, "alpha_research": section,
              "paths": {k: cfg["paths"][k] for k in ["train", "test_x", "models", "metrics",
                  "predictions", "figures", "experiment_log", "research_studies", "research_reports"]},
              "features": cfg["features"], "score_tolerance": cfg["baseline"]["score_tolerance"],
              "score_formula": "0.4*ic_mean+0.3*annual_excess+0.3*(1-mean_turnover)"}
    if (section["phase"] != "S004" or section["initial_state"] != "cold_start"
            or section["row_order"] != ["trade_date", "ts_code"]
            or section["candidate_budget"] != 67 or len(section["folds"]) != 3
            or [f["name"] for f in section["folds"]] != ["wf2021", "wf2022", "wf2023"]
            or section["folds"] != cfg["optuna"]["folds"]
            or cfg["features"]["expected_count"] != 40):
        raise ValueError("S004 protocol differs from the approved 2021--2023 plan.")
    return result


def candidates(cfg: dict) -> list[dict]:
    settings = cfg["alpha_research"]["controllers"]
    result = [{"method": "raw"}]
    result += [{"method": method, "keep_q": q} for method in ["band", "band_minimal"]
               for q in settings["keep_qs"]]
    result += [{"method": "bonus", "beta": beta} for beta in settings["bonuses"]]
    result += [{"method": "gap", "delta": delta, "rho": rho}
               for delta in settings["gaps"] for rho in settings["replacement_caps"]]
    if len(result) != 67 or len({digest(s) for s in result}) != 67:
        raise ValueError("The predeclared matrix must contain 67 distinct parameter configurations.")
    return [{"id": f"C{i:03d}", "family": "band" if s["method"].startswith("band") else s["method"],
             "spec": s} for i, s in enumerate(result)]


def verify_sources(cfg: dict) -> tuple[dict, dict, list[dict]]:
    selection_path = contained_path(cfg["alpha_research"]["source_selection"])
    index_path = contained_path(cfg["alpha_research"]["source_artifacts"])
    selection, index = read_json(selection_path), read_json(index_path)
    payload = {k: v for k, v in selection.items() if k != "selection_digest"}
    if digest(payload) != selection["selection_digest"] or selection["winner"]["name"] != "T030":
        raise ValueError("The frozen S003 selection digest or selected model changed.")
    if index["selection_digest"] != selection["selection_digest"] or index["status"] != "passed":
        raise ValueError("The S003 artifact index does not match its frozen selection.")
    data = provenance(cfg)
    if data != selection["data"]:
        raise ValueError("Original data or official attachments differ from S003.")
    definitions = cfg["alpha_research"]["folds"]
    sources = []
    run_index = {r["exp_id"]: r for r in index["runs"]}
    for definition in definitions:
        name = definition["name"]
        entry = selection["winner"]["folds"][name]
        for layer in ["raw", "band"]:
            original = entry[layer]
            frozen = run_index[original["exp_id"]]
            manifest = contained_path(frozen["manifest"])
            if sha256(manifest) != frozen["manifest_sha256"]:
                raise ValueError(f"Frozen source manifest changed: {manifest}")
            actual = read_json(manifest)
            if actual != original or actual["status"] != "passed":
                raise ValueError("Source manifest and selection payload disagree.")
            for path, expected in original["artifact_hashes"].items():
                if sha256(contained_path(path)) != expected:
                    raise ValueError(f"Frozen source artifact changed: {path}")
        sources.append({"definition": definition, "raw": entry["raw"], "band": entry["band"]})
    frozen_files = {relative(selection_path): sha256(selection_path), relative(index_path): sha256(index_path)}
    for source in sources:
        for layer in ["raw", "band"]:
            run = source[layer]
            frozen_files.update(run["artifact_hashes"])
            manifest = contained_path(cfg["paths"]["models"]) / run["exp_id"] / "run.json"
            frozen_files[relative(manifest)] = sha256(manifest)
    return selection, {"data": data, "frozen_files": frozen_files}, sources


def prepare_sources(sources: list[dict], log) -> list[ControllerFold]:
    result = []
    for source in sources:
        definition, original = source["definition"], source["raw"]
        scored = load_validation_frame(contained_path(original["prediction"]), contained_path(original["labels"]))
        if not is_canonical(scored):
            raise ValueError("Frozen source prediction/label order is not canonical.")
        if (len(scored) != original["validation_rows"] or scored.ts_code.nunique() != 4650
                or not scored.trade_date.between(*definition["valid"]).all()):
            raise ValueError("Source keys do not match their declared complete validation fold.")
        truth = canonical(scored[KEYS + ["y_ret_1d", "flag_limit_up"]])
        inputs = prepare_controller_input(scored[KEYS + ["pred", "flag_limit_up"]])
        check_predictions(inputs.known[KEYS + ["pred"]], truth)
        observed = evaluate_frame(scored)
        if max(abs(observed[k] - original["metrics"][k]) for k in OFFICIAL_METRICS) > 1e-10:
            raise ValueError("Frozen raw source metrics no longer reproduce.")
        result.append(ControllerFold(definition, inputs, truth, source))
        log(f"verified {definition['name']}: rows={len(truth):,}; frozen raw and band SHA passed")
    return result


def assert_identity(cfg: dict, meta: dict, *, original_files=False) -> None:
    if (digest(protocol(cfg)) != meta["protocol_digest"] or code_hashes() != meta["source"]
            or {n: importlib.metadata.version(n) for n in PACKAGES} != meta["environment"]["packages"]):
        raise ValueError("Study code, protocol or packages changed; preserve this study and use a new ID.")
    if original_files:
        if provenance(cfg) != meta["data"]:
            raise ValueError("Original data/official files changed during S004.")
        for path, expected in meta["frozen_files"].items():
            if sha256(contained_path(path)) != expected:
                raise ValueError(f"Frozen input artifact changed: {path}")


def quarter_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    temporary = daily.copy()
    dates = temporary.trade_date.astype(str)
    temporary["quarter"] = dates.str[:4] + "Q" + ((dates.str[4:6].astype(int) - 1) // 3 + 1).astype(str)
    rows = []
    for period, group in temporary.groupby("quarter"):
        # Reuse the monthly formulas on an artificial one-period date. Signals
        # and turnover were calculated on the full fold, with no state restart.
        artificial = group.copy()
        artificial["trade_date"] = 20000101
        row = monthly_metrics(artificial).iloc[0].to_dict()
        row.pop("month")
        rows.append({"quarter": period, **row})
    return pd.DataFrame(rows)


def ensure_passed_record(cfg, info, manifest):
    # The shared writer refuses overwriting any existing experiment record.
    ensure_record(cfg, info, manifest)


def load_passed(cfg, exp_id, spec):
    manifest = contained_path(cfg["paths"]["models"]) / exp_id / "run.json"
    if not manifest.exists():
        return None
    info = read_json(manifest)
    if info["spec"] != spec:
        raise ValueError("Existing derived run has a different controller.")
    if info["status"] != "passed":
        raise ValueError("A failed or interrupted run requires a new experiment ID; it is not overwritten.")
    for path, expected in info["artifact_hashes"].items():
        if sha256(contained_path(path)) != expected:
            raise ValueError(f"Passed S004 artifact changed: {path}")
    ensure_passed_record(cfg, info, manifest)
    return info


def run_fold(cfg, config_path, meta, fold: ControllerFold, candidate, transformed: ControllerResult, log):
    exp_id = f"{meta['study_id']}_{candidate['id']}_{fold.definition['name']}"
    spec = {**candidate["spec"], "initial_state": "cold_start", "encoding": transformed.encoding}
    reused = load_passed(cfg, exp_id, spec)
    if reused:
        log(f"reuse passed {exp_id}")
        return reused
    started = time.perf_counter()
    paths = {k: contained_path(cfg["paths"][k]) / exp_id for k in ["models", "metrics", "predictions"]}
    for directory in paths.values():
        if directory.exists():
            raise ValueError(f"Incomplete derived artifacts are preserved: {directory}")
        directory.mkdir(parents=True)
    manifest = paths["models"] / "run.json"
    original = fold.source["raw"]
    info = {"exp_id": exp_id, "status": "running", "started_at": timestamp(), "owner": meta["owner"],
            "git": meta["git"], "config_path": relative(config_path), "family": "T030+controller",
            "spec": spec, "candidate": candidate["id"], "controller_family": candidate["family"],
            "source_exp_id": original["exp_id"], "source_prediction": original["prediction"],
            "source_prediction_sha256": sha256(contained_path(original["prediction"])),
            "source_model": original["model"], "source_model_sha256": sha256(contained_path(original["model"])),
            "split": original["split"], "training_samples": original["training_samples"],
            "new_model_fits": 0, "features": "baseline_v1 (40)",
            "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
            "labels": original["labels"], "fallback_rows_in_source": original["fallback_rows"]}
    save_json(manifest, info)
    write_text(paths["models"] / "config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    try:
        pred = fold.data.known[KEYS].copy()
        pred["pred"] = transformed.scores.ravel()
        check_predictions(pred, fold.truth)
        prediction = paths["predictions"] / f"valid_{fold.definition['valid'][0] // 10000}.csv"
        save_csv(prediction, pred)
        del pred
        scored = load_validation_frame(prediction, contained_path(original["labels"]))
        check_predictions(scored[KEYS + ["pred"]], fold.truth)
        if not is_canonical(scored):
            raise ValueError("Saved scoring frame is not canonical.")
        metrics = evaluate_frame(scored)
        differences = compare_official(prediction, fold.truth, metrics, cfg["baseline"]["score_tolerance"])
        daily = daily_metrics(scored)
        actual_scores = scored.pred.to_numpy().reshape(fold.data.raw.shape)
        diagnostic, stocks, histogram, summary, top = holdings_diagnostics(
            fold.data, actual_scores, transformed.operations, transformed.intended_top)
        input_order = order_fingerprint(transformed.scores)
        saved_order = order_fingerprint(actual_scores)
        if input_order != saved_order:
            raise ValueError("CSV round trip changed ranks or tie groups.")
        save_csv(paths["metrics"] / "daily_metrics.csv", daily)
        save_csv(paths["metrics"] / "monthly_metrics.csv", monthly_metrics(daily))
        save_csv(paths["metrics"] / "quarterly_metrics.csv", quarter_metrics(daily))
        save_csv(paths["metrics"] / "holdings_daily.csv", diagnostic)
        save_csv(paths["metrics"] / "holdings_stocks.csv", stocks)
        save_csv(paths["metrics"] / "holding_length_histogram.csv", histogram)
        save_json(paths["metrics"] / "holdings_summary.json", summary)
        save_json(paths["metrics"] / "metrics.json", metrics)
        np.savez_compressed(paths["metrics"] / "top_sets.npz", dates=fold.data.dates,
                            stocks=fold.data.stocks, top=top)
        info.update(status="passed", finished_at=timestamp(), metrics=metrics,
                    official_differences=differences, official_max_difference=max(differences.values()),
                    validation_rows=len(scored), validation_days=len(daily), prediction=relative(prediction),
                    duration_seconds=time.perf_counter() - started, order_fingerprint=saved_order,
                    holdings_summary=summary,
                    notes=f"{meta['study_id']}; {candidate['id']}; {candidate['family']}; derived from frozen T030; "
                          "no new model fit; cold_start; 2021-2023 only; canonical rows; saved-file official check")
        if candidate["spec"]["method"] == "raw":
            info["source_metric_max_difference"] = max(abs(metrics[k] - original["metrics"][k]) for k in OFFICIAL_METRICS)
            if info["source_metric_max_difference"] > 1e-10:
                raise ValueError("Raw derived reference differs from its frozen S003 metrics.")
        if candidate["spec"]["method"] == "band" and candidate["spec"]["keep_q"] == meta["baseline_keep_q"]:
            info["source_metric_max_difference"] = max(abs(metrics[k] - fold.source["band"]["metrics"][k]) for k in OFFICIAL_METRICS)
            if info["source_metric_max_difference"] > 1e-10:
                raise ValueError("The q* baseline differs from its frozen S003 metrics.")
        files = list(paths["metrics"].glob("*")) + [prediction, paths["models"] / "config.yaml"]
        info["artifact_hashes"] = {relative(path): sha256(path) for path in files if path.is_file()}
        save_json(manifest, info)
        ensure_passed_record(cfg, info, manifest)
        log(f"passed {exp_id}: score={metrics['final_score']:.10f}; IC={metrics['ic_mean']:.6f}; "
            f"excess={metrics['annual_excess']:.6f}; turn={metrics['mean_turnover']:.6f}; "
            f"official_diff={info['official_max_difference']:.2e}; seconds={info['duration_seconds']:.1f}")
        return info
    except Exception:
        info.update(status="failed", failed_at=timestamp(), error=traceback.format_exc())
        save_json(manifest, info)
        raise


def candidate_summary(candidate, runs, fingerprints, alias_of=None):
    values = np.array([r["metrics"]["final_score"] for r in runs])
    return {**candidate, "status": "COMPLETE", "alias_of": alias_of, "folds": runs,
            "order_fingerprints": fingerprints, "cv_mean": float(values.mean()),
            "cv_std": float(values.std(ddof=0)), "cv_worst": float(values.min()),
            "mean_ic": float(np.mean([r["metrics"]["ic_mean"] for r in runs])),
            "mean_excess": float(np.mean([r["metrics"]["annual_excess"] for r in runs])),
            "mean_turnover": float(np.mean([r["metrics"]["mean_turnover"] for r in runs])),
            "scores": {r["split"]["fold"]: r["metrics"]["final_score"] for r in runs}}


def ranking(results, tol):
    remaining = list(results)
    ranked = []
    while remaining:
        best_mean = max(r["cv_mean"] for r in remaining)
        tied = [r for r in remaining if best_mean - r["cv_mean"] <= tol]
        chosen = min(tied, key=lambda r: (-r["cv_worst"], r["id"]))
        ranked.append(chosen)
        remaining.remove(chosen)
    return ranked


def comparison(results, baseline):
    rows = []
    for r in results:
        changes = {"ic_component_change": .4 * (r["mean_ic"] - baseline["mean_ic"]),
                   "excess_component_change": .3 * (r["mean_excess"] - baseline["mean_excess"]),
                   "stability_component_change": -.3 * (r["mean_turnover"] - baseline["mean_turnover"])}
        if abs(sum(changes.values()) - (r["cv_mean"] - baseline["cv_mean"])) > 1e-10:
            raise ValueError("CV score attribution does not add up.")
        rows.append({"candidate": r["id"], "family": r["family"], "method": r["spec"]["method"],
                     "params": json.dumps(r["spec"], sort_keys=True), "alias_of": r["alias_of"],
                     **{k: r[k] for k in ["cv_mean", "cv_std", "cv_worst", "mean_ic", "mean_excess", "mean_turnover"]},
                     **r["scores"], "cv_delta_vs_S003": r["cv_mean"] - baseline["cv_mean"], **changes})
    table = pd.DataFrame(rows)
    values = table[["mean_ic", "mean_excess", "mean_turnover"]].to_numpy() * [1, 1, -1]
    nondominated = []
    for row in values:
        dominated = np.all(values >= row - 1e-12, axis=1) & np.any(values > row + 1e-12, axis=1)
        nondominated.append(not dominated.any())
    table["pareto"] = nondominated
    return table


def table_markdown(frame, columns):
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("|" + "|".join(f"{v:.10f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + "|")
    return lines


def publish(cfg, meta, root, results, selection, audit):
    baseline = next(r for r in results if r["spec"] == {"method": "band", "keep_q": meta["baseline_keep_q"]})
    table = comparison(results, baseline)
    table = table.sort_values("cv_mean", ascending=False)
    public = contained_path(cfg["paths"]["research_reports"])
    experiments = contained_path(cfg["paths"]["experiment_log"]).parent
    prefix = f"alpha_{meta['study_id']}"
    save_csv(root / "comparison.csv", table)
    save_csv(experiments / f"{prefix}_comparison.csv", table)
    folds, monthly, quarterly, holdings = [], [], [], []
    for r in results:
        for run in r["folds"]:
            name = run["split"]["fold"]
            common = {"candidate": r["id"], "family": r["family"], "fold": name,
                      "artifact_exp_id": run["exp_id"], "alias_of": r["alias_of"]}
            folds.append({**common, **run["metrics"], "official_max_difference": run["official_max_difference"]})
            directory = contained_path(cfg["paths"]["metrics"]) / run["exp_id"]
            monthly.append(pd.read_csv(directory / "monthly_metrics.csv").assign(**common))
            quarterly.append(pd.read_csv(directory / "quarterly_metrics.csv").assign(**common))
            holdings.append({**common, **run["holdings_summary"]})
    for suffix, frame in [("folds", pd.DataFrame(folds)), ("monthly", pd.concat(monthly, ignore_index=True)),
                          ("quarterly", pd.concat(quarterly, ignore_index=True)), ("holdings", pd.DataFrame(holdings))]:
        save_csv(experiments / f"{prefix}_{suffix}.csv", frame)
    selection_file = public / f"alpha_research_{meta['study_id']}_selection.json"
    save_json(selection_file, selection)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = contained_path(cfg["paths"]["figures"]) / meta["study_id"]
    figures.mkdir(parents=True, exist_ok=True)
    palette = {"raw": "#7a7a7a", "band": "#3572a5", "bonus": "#cc7b23", "gap": "#279b72"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, group in table.groupby("family"):
        ax.scatter(group.mean_turnover, .4 * group.mean_ic + .3 * group.mean_excess,
                   label=family, color=palette[family], alpha=.75)
    ax.scatter(baseline["mean_turnover"], .4 * baseline["mean_ic"] + .3 * baseline["mean_excess"],
               marker="*", s=180, color="black", label="S003 reference")
    for chosen in selection["controllers"]:
        ax.annotate(chosen["id"], (chosen["mean_turnover"], .4 * chosen["mean_ic"] + .3 * chosen["mean_excess"]),
                    xytext=(5, 6), textcoords="offset points")
    ax.axvspan(.02, .2, color="grey", alpha=.08)
    ax.set(xlabel="Mean official Jaccard turnover", ylabel="0.4 IC + 0.3 annual excess (research diagnostic)",
           title="T030: predictive contribution and turnover, 2021-2023")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "tradeoff.png", dpi=180); plt.close(fig)
    selected_ids = [baseline["id"]] + [r["id"] for r in selection["controllers"] if r["id"] != baseline["id"]]
    chosen_rows = table.set_index("candidate").loc[selected_ids]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    for i, (candidate, row) in enumerate(chosen_rows.iterrows()):
        ax.bar(x + (i - (len(chosen_rows)-1)/2) * .23,
               [row["wf2021"], row["wf2022"], row["wf2023"]], .23, label=candidate)
    ax.set(xticks=x, xticklabels=["2021", "2022", "2023"], ylabel="Official final score",
           title="Frozen research controllers and S003 reference")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "year_scores.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, column in enumerate(["ic_component_change", "excess_component_change", "stability_component_change"]):
        ax.bar(np.arange(len(chosen_rows)) + (i-1)*.24, chosen_rows[column], .24,
               label=["IC", "Excess", "Stability"][i])
    ax.axhline(0, color="black", linewidth=.6)
    ax.set(xticks=np.arange(len(chosen_rows)), xticklabels=selected_ids,
           ylabel="Weighted CV score change vs S003", title="Which scoring component changed?")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "attribution.png", dpi=180); plt.close(fig)
    public_figures = public / "figures"
    public_figures.mkdir(parents=True, exist_ok=True)
    for source in figures.glob("*.png"):
        shutil.copyfile(source, public_figures / f"{prefix}_{source.name}")
    top_table = table.head(10)
    lines = [f"# {meta['study_id']}：固定 T030 的换手控制研究", "", f"完成时间：{timestamp()}", "",
             "## 执行范围", "",
             "复用 S003 T030 的 2021 / 2022 / 2023 原始预测；每折冷启动，共用同一配置。",
             "旧 band、同一实际 Top 集合的按日编码、留仓奖励、排名差与主动替换上限均按预定网格执行。",
             f"预定配置 {len(results)}，完整排序等价去重后 {audit['unique_candidates']}；",
             f"实际落盘并由官方评分器核对 {audit['unique_fold_runs']} 个完整折；新增模型训练 **0**。",
             "2024 未参与本阶段计算或选择；S005 / S006 尚未执行。", "",
             f"实现提交：`{meta['git']['commit']}`；代码摘要：`{meta['source_digest']}`；",
             f"协议摘要：`{meta['protocol_digest']}`；控制器锁定摘要：`{selection['selection_digest']}`。", "",
             "## 原始信号与同层参照", ""]
    raw = next(r for r in results if r["spec"]["method"] == "raw")
    reference_table = table.set_index("candidate").loc[[raw["id"], baseline["id"]]].reset_index()
    lines += table_markdown(reference_table, ["candidate", "mean_ic", "mean_excess", "mean_turnover", "cv_mean", "wf2023"])
    lines += ["", "## 最高均分配置（包括排序等价映射）", ""]
    lines += table_markdown(top_table, ["candidate", "method", "cv_mean", "cv_std", "cv_worst", "wf2021", "wf2022", "wf2023"])
    lines += ["", "全部参数、等价映射、逐折八指标和月度 / 季度表见 `experiments/alpha_S004_*`。", "",
              "## 冻结交给 S006 的两种家族", ""]
    for r in selection["controllers"]:
        lines += [f"### {r['id']}：{r['family']}", "", "```json", json.dumps(r["spec"], ensure_ascii=False, indent=2),
                  "```", "", f"三折均分 {r['cv_mean']:.10f}；最差折 {r['cv_worst']:.10f}；",
                  f"相对 S003 均分变化 {r['cv_mean'] - baseline['cv_mean']:+.10f}。", ""]
        row = table[table.candidate == r["id"]].iloc[0]
        lines += [f"加权 IC / 超额 / 稳定性变化：{row.ic_component_change:+.10f} / "
                  f"{row.excess_component_change:+.10f} / {row.stability_component_change:+.10f}。", "",
                  f"三折最终候选的历史替换条件满足：**{r['passes_cv_replacement_gate']}**。",
                  "这里只冻结供后续目标 / 融合研究使用的控制器，不变更正式推荐模型。", ""]
    lines += ["## 留仓与编码诊断", ""]
    hold_table = pd.DataFrame(holdings)
    hold_table = hold_table[hold_table.candidate.isin(selected_ids)]
    lines += table_markdown(hold_table, ["candidate", "fold", "mean_spell_days", "p95_spell_days", "max_spell_days",
                    "mean_selected_raw_rank", "mean_raw_top_retained_fraction", "mean_selected_below_top20_fraction"])
    lines += ["", "持仓段在折末右删失，停留天数按交易观察计算；较长停留本身不代表收益提高。",
              "同集合编码以旧 band 的实际官方换手 Top 集合为参照，不假设其内部选择等于输出集合。",
              "留仓奖励先计算 rank+bonus，再保持非并列关系、按股票代码打破精确并列并编码严格次序。",
              "因此需把全市场 IC 的变化与 Top 收益的变化分别解释。", "",
              "## 核对与解释边界", "",
              f"官方八指标最大差 {audit['max_official_difference']:.3e}；原始信号 / q* band 参照最大差 "
              f"{audit['max_reference_difference']:.3e}；保存文件的 Top 集合全部与预期一致。",
              f"原有 {audit['initial_records']} 条实验记录保持不变，新增 {audit['new_records']} 条派生记录，"
              f"共享表共 {audit['total_records']} 条。",
              "冻结控制器已用真实三折的前半段重放核对因果性。原始数据、官方附件和已冻结 S003 输入未改动。",
              "排序 / 并列等价配置只做一次真实落盘评分，映射在比较表中公布；其参数配置仍完整保留。",
              "本轮为有限网格上的历史研究，没有证明全局最优或隐藏测试得分。未新增或运行测试套件。", "",
              "## 图表", "",
              f"![预测贡献与换手](figures/{prefix}_tradeoff.png)", "",
              f"![跨年分数](figures/{prefix}_year_scores.png)", "",
              f"![分项归因](figures/{prefix}_attribution.png)", "",
              "## 后续交接", "",
              "成员 B 可从选择清单、逐折表和产物索引独立重算；其新一轮复核仍待完成。",
              "S005 接续固定特征 / 参数的目标研究；S006 使用本次两种控制器以及 S003 q* band，",
              "最终完整方案冻结、推送后才执行一次 2024 历史确认。"]
    report = public / f"alpha_research_{meta['study_id']}.md"
    write_text(report, "\n".join(lines))
    published = list(experiments.glob(f"{prefix}_*.csv")) + list(public_figures.glob(f"{prefix}_*.png")) + [report, selection_file]
    audit["published_files"] = {relative(p): {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in published}
    save_json(root / "audit.json", audit)
    save_json(public / f"alpha_research_{meta['study_id']}_artifacts.json", audit)


def run_study(cfg, config_path, study_id, owner, resume):
    if not re.fullmatch(r"S004(?:_[A-Za-z0-9_-]+)?", study_id):
        raise ValueError("This entry point implements only S004; use an S004 study identity.")
    fixed_protocol = protocol(cfg)
    matrix = candidates(cfg)
    root = contained_path(cfg["paths"]["research_studies"]) / study_id
    revision = git_state()
    if revision["branch"] != "main":
        raise ValueError("The user requires implementation and research on main.")
    if not resume and revision["dirty"]:
        raise ValueError("Commit/push the implementation and fixed protocol before starting S004.")
    if not resume and root.exists():
        raise ValueError("Study artifacts already exist; preserve them and use --resume or a new identity.")
    if resume and not (root / "study.json").exists():
        raise ValueError("No study metadata exists to resume.")
    if not resume:
        origin = __import__("subprocess").check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
        if origin != revision["commit"]:
            raise ValueError("Push the clean implementation to origin/main before executing.")
        if shutil.disk_usage(ROOT).free < 15 * 2**30:
            raise ValueError("At least 15 GiB free space is required for complete derived artifacts.")
        root.mkdir(parents=True)
    log = RunLog(root / "run.log")
    with study_lock(root):
        try:
            source_selection, checked, sources = verify_sources(cfg)
            if resume:
                meta = read_json(root / "study.json")
                if meta["status"] == "complete":
                    raise ValueError("S004 is complete; do not rerun completed selection/publication.")
                assert_identity(cfg, meta)
                if checked["data"] != meta["data"] or checked["frozen_files"] != meta["frozen_files"]:
                    raise ValueError("Frozen S003 input identity changed during resume.")
            else:
                meta = {"schema": 1, "study_id": study_id, "owner": owner, "status": "running",
                        "started_at": timestamp(), "git": revision, "source": code_hashes(),
                        "protocol": fixed_protocol, "protocol_digest": digest(fixed_protocol),
                        "baseline_keep_q": source_selection["winner"]["spec"]["keep_q"],
                        "source_selection_digest": source_selection["selection_digest"],
                        "initial_records": read_records(contained_path(cfg["paths"]["experiment_log"])),
                        "environment": {"python": sys.version, "platform": platform.platform(),
                                        "packages": {n: importlib.metadata.version(n) for n in PACKAGES}},
                        "matrix": matrix, **checked}
                meta["source_digest"] = digest(meta["source"])
                save_json(root / "study.json", meta)
                write_text(root / "config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            folds = prepare_sources(sources, log)
            results, registry = [], {}
            for candidate in matrix:
                assert_identity(cfg, meta)
                candidate_file = root / "candidates" / f"{candidate['id']}.json"
                if candidate_file.exists():
                    completed = read_json(candidate_file)
                    if completed["spec"] != candidate["spec"] or completed["status"] != "COMPLETE":
                        raise ValueError("Existing candidate differs from its completed protocol.")
                    for run in completed["folds"]:
                        load_passed(cfg, run["exp_id"], run["spec"])
                    results.append(completed)
                    registry.setdefault(tuple(completed["order_fingerprints"]), completed)
                    log(f"reuse candidate {candidate['id']}")
                    continue
                log(f"candidate {candidate['id']} ({len(results)+1}/67): {json.dumps(candidate['spec'], sort_keys=True)}")
                outputs = [transform(fold.data, candidate["spec"]) for fold in folds]
                fingerprints = [order_fingerprint(output.scores) for output in outputs]
                equivalent = registry.get(tuple(fingerprints))
                if equivalent:
                    completed = candidate_summary(candidate, equivalent["folds"], fingerprints, equivalent["id"])
                    log(f"order/tie-equivalent {candidate['id']} -> {equivalent['id']}; no duplicate scoring/record")
                else:
                    runs = [run_fold(cfg, config_path, meta, fold, candidate, output, log)
                            for fold, output in zip(folds, outputs)]
                    completed = candidate_summary(candidate, runs, fingerprints)
                    registry[tuple(fingerprints)] = completed
                results.append(completed)
                save_json(candidate_file, completed)
                meta.update(completed_candidates=len(results), last_candidate=candidate["id"], updated_at=timestamp())
                save_json(root / "study.json", meta)
                log(f"completed {candidate['id']}: CV={completed['cv_mean']:.10f}; worst={completed['cv_worst']:.10f}")
                del outputs
                gc.collect()
            baseline = next(r for r in results if r["spec"] == {"method": "band", "keep_q": meta["baseline_keep_q"]})
            if abs(baseline["cv_mean"] - source_selection["winner"]["objective"]) > 1e-10:
                raise ValueError("S003 baseline CV no longer reproduces.")
            ranked = ranking(results, cfg["alpha_research"]["selection_tolerance"])
            selected, families, selected_orders = [], set(), set()
            for r in ranked:
                order = tuple(r["order_fingerprints"])
                if r["family"] == "raw" or r["family"] in families or order in selected_orders:
                    continue
                chosen = {k: v for k, v in r.items() if k != "folds"}
                tol = cfg["alpha_research"]["selection_tolerance"]
                chosen["passes_cv_replacement_gate"] = (r["cv_mean"] > baseline["cv_mean"] + tol
                    and r["cv_worst"] >= baseline["cv_worst"] - tol
                    and r["scores"]["wf2023"] >= baseline["scores"]["wf2023"] - tol)
                chosen["runs"] = {run["split"]["fold"]: {"exp_id": run["exp_id"], "prediction": run["prediction"],
                    "sha256": run["artifact_hashes"][run["prediction"]]} for run in r["folds"]}
                selected.append(chosen); families.add(r["family"]); selected_orders.add(order)
                if len(selected) == 2:
                    break
            prefix_replays = []
            for chosen in selected:
                original_result = next(r for r in results if r["id"] == chosen["id"])
                for fold, run in zip(folds, original_result["folds"]):
                    count = len(fold.data.dates) // 2
                    stop = fold.data.dates[count-1]
                    prefix = prepare_controller_input(fold.data.known[fold.data.known.trade_date <= stop])
                    replay = transform(prefix, chosen["spec"])
                    saved = pd.read_csv(contained_path(run["prediction"]))
                    saved = saved[saved.trade_date <= stop].pred.to_numpy().reshape(prefix.raw.shape)
                    if order_fingerprint(replay.scores) != order_fingerprint(saved):
                        raise ValueError("A frozen controller failed actual-data prefix causality replay.")
                    prefix_replays.append({"candidate": chosen["id"], "fold": fold.definition["name"],
                                           "end": int(stop), "rows": prefix.known.shape[0], "passed": True})
            selection = {"schema": 1, "study_id": study_id, "created_at": timestamp(), "git": meta["git"],
                         "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
                         "source_selection_digest": meta["source_selection_digest"], "controllers": selected,
                         "baseline": {k: v for k, v in baseline.items() if k != "folds"},
                         "highest_mean_unrestricted": {k: v for k, v in max(results, key=lambda r:r["cv_mean"]).items() if k != "folds"},
                         "rule": "Highest CV mean per distinct nonraw family, then worst fold and predeclared ID; "
                                 "order-equivalent configurations are evaluated once; freeze two for S006, not formal adoption.",
                         "initial_state": "cold_start", "row_order": ["trade_date", "ts_code"],
                         "prefix_replays": prefix_replays, "evaluated_years": [2021, 2022, 2023], "new_model_fits": 0}
            selection["selection_digest"] = digest(selection)
            save_json(root / "selection.json", selection)
            assert_identity(cfg, meta, original_files=True)
            records = read_records(contained_path(cfg["paths"]["experiment_log"]))
            if records[:len(meta["initial_records"])] != meta["initial_records"]:
                raise ValueError("Prior experiment records changed.")
            unique_runs = {run["exp_id"]: run for r in results for run in r["folds"]}
            for run in unique_runs.values():
                load_passed(cfg, run["exp_id"], run["spec"])
            new_records = [r for r in records if r["exp_id"].startswith(study_id + "_")]
            if len(new_records) != len(unique_runs):
                raise ValueError("Derived experiment record count does not match passed artifacts.")
            audit = {"status": "passed", "study_id": study_id, "time": timestamp(),
                     "source_digest": meta["source_digest"], "protocol_digest": meta["protocol_digest"],
                     "implementation_commit": meta["git"]["commit"], "selection_digest": selection["selection_digest"],
                     "configured_candidates": len(matrix), "complete_candidates": len(results),
                     "failed_candidates": 0, "unique_candidates": len(registry), "unique_fold_runs": len(unique_runs),
                     "new_model_fits": 0, "evaluated_years": [2021, 2022, 2023],
                     "initial_records": len(meta["initial_records"]), "new_records": len(new_records), "total_records": len(records),
                     "previous_records_unchanged": True, "original_files_unchanged": True, "frozen_S003_sources_unchanged": True,
                     "max_official_difference": max(r["official_max_difference"] for r in unique_runs.values()),
                     "max_reference_difference": max(r.get("source_metric_max_difference",0.) for r in unique_runs.values()),
                     "all_saved_top_sets_match": True, "prefix_replays": prefix_replays,
                     "data_provenance": meta["data"], "frozen_input_hashes": meta["frozen_files"],
                     "runs": [{"exp_id": r["exp_id"], "manifest": relative(contained_path(cfg["paths"]["models"]) / r["exp_id"] / "run.json"),
                               "manifest_sha256": sha256(contained_path(cfg["paths"]["models"]) / r["exp_id"] / "run.json"),
                               "artifact_hashes": r["artifact_hashes"]} for r in unique_runs.values()],
                     "aliases": {r["id"]: r["alias_of"] for r in results if r["alias_of"]}}
            publish(cfg, meta, root, results, selection, audit)
            meta.update(status="complete", completed_at=timestamp(), selection_digest=selection["selection_digest"],
                        unique_candidates=len(registry), unique_fold_runs=len(unique_runs), new_model_fits=0)
            save_json(root / "study.json", meta)
            log(f"S004 complete: configured={len(matrix)}; unique={len(registry)}; folds={len(unique_runs)}; "
                f"best_CV={max(r['cv_mean'] for r in results):.10f}; controllers={[r['id'] for r in selected]}")
        except Exception:
            if (root / "study.json").exists():
                saved = read_json(root / "study.json")
                if saved["status"] != "complete":
                    saved.update(status="interrupted", error=traceback.format_exc(), interrupted_at=timestamp())
                    save_json(root / "study.json", saved)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="S004 frozen-prediction controller research")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study-id", default="S004")
    parser.add_argument("--phase", choices=["controllers"], required=True)
    parser.add_argument("--owner", default="A")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg, path = load_config(args.config)
    run_study(cfg, path, args.study_id, args.owner, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

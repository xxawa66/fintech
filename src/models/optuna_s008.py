"""S008：第二轮 Optuna——model search 与 turnover search 分离。

《下一步》第二十一节：第二轮 Optuna 要重新搜索模型参数，且把 model search 和
turnover search 分开，不是把 50 trials 机械增加到 200。依据：S003 联合搜索的
fANOVA 显示 ``band_keep_q`` 占参数方差 75.3%、``learning_rate`` 仅 19.2%——
模型参数从未在冻结的换手层下被独立、干净地搜索过。

协议（复用 2026-10-09 S007 管线模式，已在 S007 四折逐位复现冻结 T030）：

- **特征固定**：40 列 baseline_v1（S007 市场状态特征臂未通过成功标准，不纳入）。
- **换手层冻结**：keep_q = T030 q*（从 S003 selection 读取），全程不参与搜索；
  S004/S006/LH014 已证放松 keep_q 或叠加控制器层不可靠，本轮不再开这一维。
- **Phase A（search）**：TPE 50 trials（同 S003 预算，不放大），搜索空间 =
  S003 search_space 去掉 ``band_keep_q``；目标 = 三折 band(q*) 官方综合分等权
  均值；trial 0 为 T030 锚点（必须逐折复现冻结 raw/band 值 1e-9 内）。
  三折 wf2021/2022/2023 扩展窗、冷启动、无早停、无 pruning、行序 (date, code)。
- **选择规则**：三折均值最大者胜；平局取更早 trial。锁定后才有 2024。
- **confirm2024**：锁定 winner 后评估一次（对齐 S007 纪律：搜索折不含 2024）；
  替换门槛沿用 S006：confirm2024 band(q*) ≥ T030 的 0.3873823479，否则 retain。
  成功标准（《下一步》第二十节）：2023 折 0.35710 → 0.38+ 且 2024 折保持 0.38+。
- **Phase B（turnover scan）**：对 winner 的原始预测做 keep_q 网格（不重训），
  覆盖 q* 邻域、放宽档与 q=1（无带）参照；confirm2024 同网格作描述性对照。
- **不触碰测试期**（2025–2026 已不干净）。

用法（s007_wt 根目录，干净 main）::

    python -m src.models.optuna_s008           # 全流程：search→winner→confirm→scan
    python -m src.models.optuna_s008 --resume  # 断点续跑（sqlite study + metrics 缓存）

产物：outputs/metrics/research_studies/S008/、docs/optuna_s008.md、
experiments/optuna_s008_{trials,scan}.csv、experiments/optuna_s008_summary.json。
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from src.data.clean_data import clean_history
from src.data.load_data import KEYS, load_training_data
from src.data.make_dataset import attach_labels, training_arrays
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame
from src.evaluation.research_diagnostics import monthly_metrics
from src.evaluation.turnover import band_scores
from src.evaluation.validation import Fold, split_train_valid
from src.features.build_features import feature_names
from src.models.baseline import RunLog, source_provenance
from src.models.lightgbm_model import train_model
from src.models.market_regime_research import (frozen_references, load_v1_features)
from src.models.optuna_tuning import save_json
from src.utils.experiments import append_record, read_records
from src.utils.project import ROOT, git_state, load_config, project_path, timestamp
from src.utils.research_cache import digest

STUDY_ID = "S008"
SEARCH_FOLDS = ["wf2021", "wf2022", "wf2023"]
BASELINE_TOLERANCE = 1.0e-9
# Phase B 换手层网格：q* 邻域 + 放宽档 + q=1（等价无带）。只评分，不重训。
SCAN_EXTRA_QS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0]


class InvalidTrial(ValueError):
    """非有限预测/评分的 trial 记为 FAIL，而不是赋一个假分。"""


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_base_dataset(cfg: dict, provenance: dict, log) -> pd.DataFrame:
    """40 特征 + 标签 + quote_valid + flag_limit_up（不含市场特征，行序=原始面板）。"""
    raw = load_training_data(project_path(cfg["paths"]["train"]))
    if len(raw) != 7_900_350:
        raise ValueError("S008 requires the complete original training CSV.")
    raw = raw[raw.trade_date.between(20180102, 20241231)]
    raw, _ = clean_history(raw)
    features = load_v1_features(cfg, provenance, raw, log)
    dataset = attach_labels(raw, features)
    dataset["flag_limit_up"] = raw["flag_limit_up"].to_numpy().astype("int8")
    del raw, features
    gc.collect()
    log(f"dataset ready: {len(dataset):,} rows, 40 V1 features, labels attached")
    return dataset


def fold_definitions(cfg: dict) -> list[Fold]:
    items = list(cfg["optuna"]["folds"]) + [cfg["optuna"]["confirm"]]
    return [Fold(item["name"], *item["train"], *item["valid"]) for item in items]


def sample_model_spec(trial, cfg: dict, keep_q: float) -> dict:
    """S003 search_space 去掉 band_keep_q；keep_q 冻结。"""
    space = {k: v for k, v in cfg["optuna"]["search_space"].items() if k != "band_keep_q"}
    sampled = {}
    for name, s in space.items():
        if isinstance(s, list):
            sampled[name] = trial.suggest_categorical(name, s)
        elif name in {"num_leaves", "min_data_in_leaf"}:
            sampled[name] = trial.suggest_int(name, **s)
        else:
            sampled[name] = trial.suggest_float(name, **s)
    params = copy.deepcopy(cfg["baseline"]["model"]["params"])
    params.update({k: v for k, v in sampled.items() if k != "num_boost_round"})
    if params["max_depth"] > 0:
        params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"])
    params["bagging_freq"] = 1 if params["bagging_fraction"] < 1 else 0
    return {"sampled_params": sampled, "model_params": params,
            "rounds": int(sampled["num_boost_round"]), "keep_q": float(keep_q)}


def anchor_spec(cfg: dict, keep_q: float) -> dict:
    """T030 winner 的模型参数（锚点 trial），keep_q 同样冻结。"""
    winner = frozen_references()[0]["spec"]
    sampled = {k: v for k, v in winner["sampled_params"].items() if k != "band_keep_q"}
    params = copy.deepcopy(winner["model_params"])
    return {"sampled_params": sampled, "model_params": params,
            "rounds": int(winner["rounds"]), "keep_q": float(keep_q)}


def evaluate_fold(cfg: dict, dataset: pd.DataFrame, columns: list[str], fold: Fold,
                  spec: dict, out_dir: Path, log, *,
                  save_artifacts: bool = False, external_check: bool = False) -> dict:
    """单折：切分 → 训练 → raw 与 band(q*) 官方口径评分（metrics 缓存可复用）。"""
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        saved = read_json(metrics_path)
        if saved.get("spec_digest") == digest(spec):
            log(f"reuse {relative(out_dir)}")
            return saved
        raise ValueError(f"Existing S008 metrics disagree with current spec: {out_dir}")
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

    def progress(message: str) -> None:
        done = int(message.split()[2].split("/")[0])
        if done % 200 == 0 or done == spec["rounds"]:
            log(f"{relative(out_dir)}: {message}")

    log(f"fit {relative(out_dir)}: rounds={spec['rounds']}, leaves={spec['model_params']['num_leaves']}")
    model = train_model(x_train, y_train, spec["model_params"], spec["rounds"], progress)
    values = np.full(len(valid), 0.0, dtype="float64")
    values[eligible] = model.predict(valid.loc[eligible, columns],
                                     num_threads=spec["model_params"]["num_threads"])
    importance = None
    if save_artifacts:
        importance = pd.DataFrame({"feature": columns,
                                   "gain": model.feature_importance(importance_type="gain"),
                                   "split": model.feature_importance(importance_type="split")})
    del model, x_train, y_train
    gc.collect()
    if not np.isfinite(values[eligible]).all():
        raise InvalidTrial("Model predictions are not finite.")
    pred = valid[KEYS].assign(pred=values).sort_values(
        ["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    del values, valid
    check_predictions(pred, truth)
    scored = pred.merge(truth, on=KEYS, validate="one_to_one")
    raw_metrics = evaluate_frame(scored)
    if not all(np.isfinite(raw_metrics[k]) for k in OFFICIAL_METRICS):
        raise InvalidTrial("Official score contains non-finite metrics.")
    differences = {}
    if save_artifacts or external_check:
        raw_path = out_dir / "prediction_raw.csv"
        pred.to_csv(raw_path, index=False)
        if external_check:
            differences["raw"] = compare_official(raw_path, truth, raw_metrics,
                                                  cfg["baseline"]["score_tolerance"])
    layers = {"raw": raw_metrics}
    keep_q = spec["keep_q"]
    banded = scored[KEYS + ["pred", "flag_limit_up"]].copy()
    banded["pred"] = band_scores(banded, keep_q).to_numpy()
    band_pred = banded[KEYS + ["pred"]]
    check_predictions(band_pred, truth)
    band_scored = band_pred.merge(truth, on=KEYS, validate="one_to_one")
    band_metrics = evaluate_frame(band_scored)
    if not all(np.isfinite(band_metrics[k]) for k in OFFICIAL_METRICS):
        raise InvalidTrial("Official score contains non-finite metrics.")
    if save_artifacts or external_check:
        band_path = out_dir / "prediction_band_qstar.csv"
        band_pred.to_csv(band_path, index=False)
        if external_check:
            differences["band_qstar"] = compare_official(band_path, truth, band_metrics,
                                                         cfg["baseline"]["score_tolerance"])
    layers["band_qstar"] = band_metrics
    if save_artifacts:
        daily = daily_metrics(scored)
        daily.to_csv(out_dir / "daily_metrics.csv", index=False)
        monthly_metrics(daily).to_csv(out_dir / "monthly_metrics.csv", index=False)
        if importance is not None:
            importance.to_csv(out_dir / "feature_importance.csv", index=False)
    result = {"study_id": STUDY_ID, "fold": fold.name,
              "fold_window": [fold.train_start, fold.train_end, fold.valid_start, fold.valid_end],
              "split": split, "supervision": supervision, "spec_digest": digest(spec),
              "metrics": layers,
              "official_differences": differences,
              "duration_seconds": time.perf_counter() - started,
              "finished_at": timestamp()}
    save_json(metrics_path, result)
    log(f"PASS {relative(out_dir)}: raw={raw_metrics['final_score']:.10f}, "
        f"band q*={band_metrics['final_score']:.10f}")
    del scored, pred, truth, banded, band_pred, band_scored
    gc.collect()
    return result


def verify_anchor(entries: dict, frozen: dict) -> None:
    """锚点 trial 必须逐折复现 S003 冻结 T030 的 raw / band(q*) 官方八指标。"""
    for fold_name, saved in entries.items():
        for layer, frozen_layer in [("raw", "raw"), ("band_qstar", "band")]:
            reference = frozen[fold_name][frozen_layer]
            for metric in OFFICIAL_METRICS:
                difference = abs(saved["metrics"][layer][metric] - reference[metric])
                if difference > BASELINE_TOLERANCE:
                    raise AssertionError(
                        f"anchor {fold_name} {layer} {metric} differs from frozen T030 "
                        f"by {difference:.3e}")


def write_trials_table(study, root: Path) -> pd.DataFrame:
    rows = []
    for t in study.get_trials(deepcopy=False):
        if t.state == TrialState.WAITING:
            continue
        row = {"trial": t.number, "state": t.state.name, "objective": t.value, **t.params,
               "error": t.user_attrs.get("error", "")}
        entries = t.user_attrs.get("entries", {})
        for name, saved in entries.items():
            for layer in ["raw", "band_qstar"]:
                row.update({f"{name}_{layer}_{k}": saved["metrics"][layer][k]
                            for k in OFFICIAL_METRICS})
        if len(entries) == len(SEARCH_FOLDS):
            values = [entries[f]["metrics"]["band_qstar"]["final_score"] for f in SEARCH_FOLDS]
            row.update(mean_band_qstar=float(np.mean(values)),
                       worst_band_qstar=float(np.min(values)))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(root / "trials.csv", index=False)
    return table


def turnover_scan(cfg: dict, scored: pd.DataFrame, keep_qs: list[float]) -> list[dict]:
    """对已合并 (pred, truth) 的帧按多档 keep_q 重编码 band 并评分（不重训）。"""
    rows = []
    base = scored[KEYS + ["pred", "flag_limit_up"]]
    for keep_q in keep_qs:
        banded = base.copy()
        banded["pred"] = band_scores(banded, keep_q).to_numpy()
        metrics = evaluate_frame(banded.merge(scored[KEYS + ["y_ret_1d"]],
                                              on=KEYS, validate="one_to_one"))
        rows.append({"keep_q": keep_q, **{k: metrics[k] for k in OFFICIAL_METRICS}})
    return rows


def append_winner_records(cfg: dict, winner_number: int, results: dict) -> None:
    log_path = project_path(cfg["paths"]["experiment_log"])
    existing = {r["exp_id"] for r in read_records(log_path)}
    revision = git_state()
    for fold_name, entry in results.items():
        exp_id = f"{STUDY_ID}_W_T{winner_number:03d}_{fold_name}"
        if exp_id in existing:
            continue
        metrics = entry["saved"]["metrics"]["band_qstar"]
        record = {"exp_id": exp_id, "date": entry["saved"]["finished_at"], "owner": "A",
                  "git_commit": revision["commit"], "config_path": "configs/project.yaml",
                  "features": "baseline_v1 (40)",
                  "model": f"LightGBM T{winner_number:03d} spec + band keep_q=q*",
                  "params": json.dumps(entry["spec"], sort_keys=True),
                  "train_period": f"{entry['fold'].train_start}-{entry['fold'].train_end}",
                  "valid_period": f"{entry['fold'].valid_start}-{entry['fold'].valid_end}",
                  **{k: metrics[k] for k in OFFICIAL_METRICS},
                  "artifact_path": relative(Path(entry["out_dir"])),
                  "notes": "S008 separated search winner; band q* layer; confirm2024 is the "
                           "confirm fold"}
        append_record(log_path, record)


def write_report(cfg: dict, root: Path, trials: pd.DataFrame, anchor_ok: bool,
                 winner_number: int, winner_spec: dict, winner_entries: dict,
                 confirm_entry: dict, band01_winner: dict, anchor_confirm: dict,
                 scan_tables: dict, importance: dict) -> None:
    anchor_frozen = frozen_references()
    search = trials[trials.state == "COMPLETE"].copy()
    lines = ["# S008：分离式第二轮 Optuna（model search 冻结换手层）", "",
             f"更新时间：{timestamp()}", "",
             "## 协议", "",
             "《下一步》第二十一节：第二轮 Optuna 把 model search 与 turnover search 分开，"
             "不放大 trial 预算。S003 联合搜索 fANOVA：band_keep_q 75.3%、learning_rate 19.2%，"
             "模型参数从未在冻结换手层下独立搜索。", "",
             f"- 特征：40 列 baseline_v1；换手层冻结 keep_q = T030 q*（不参与搜索）。",
             f"- Phase A：TPE {cfg['optuna']['trial_budget']} trials（S003 同预算），搜索空间 = S003 "
             "search_space 去掉 band_keep_q；目标 = 三折 band(q*) 官方综合分均值；"
             "trial 0 = T030 锚点。",
             "- Phase B：winner 原始预测上 keep_q 网格重编码（不重训），confirm2024 同网格对照。",
             "- confirm2024 锁定后评估；替换门槛 = 2024 band(q*) ≥ 0.3873823479（S006 规则）。",
             "- 成功标准（《下一步》第二十节）：2023 → 0.38+ 且 2024 保持 0.38+。", "",
             f"锚点复现：{'通过（raw/band q* 八指标逐折 1e-9 内）' if anchor_ok else '未通过'}。", "",
             "## Top 10 trials（band q*，三折均值）", "",
             "|trial|均值|最差折|2021|2022|2023|learning_rate|num_leaves|max_depth|min_data_in_leaf|rounds|",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    top = search.sort_values("mean_band_qstar", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"|T{int(row.trial):03d}|{row.mean_band_qstar:.10f}|{row.worst_band_qstar:.10f}|"
            f"{row['wf2021_band_qstar_final_score']:.10f}|{row['wf2022_band_qstar_final_score']:.10f}|"
            f"{row['wf2023_band_qstar_final_score']:.10f}|{row.learning_rate:.6g}|{int(row.num_leaves)}|"
            f"{int(row.max_depth)}|{int(row.min_data_in_leaf)}|{int(row.num_boost_round)}|")
    lines += ["", f"## Winner T{winner_number:03d} vs T030（band q*）", "",
              "|折|T030 冻结|S008 winner|Δ|ΔIC|Δ超额|Δ换手|", "|---|---:|---:|---:|---:|---:|---:|"]
    for fold_name in SEARCH_FOLDS:
        base = anchor_frozen[1][fold_name]["band"]
        mine = winner_entries[fold_name]["saved"]["metrics"]["band_qstar"]
        lines.append(f"|{fold_name}|{base['final_score']:.10f}|{mine['final_score']:.10f}|"
                     f"{mine['final_score'] - base['final_score']:+.10f}|"
                     f"{mine['ic_mean'] - base['ic_mean']:+.10f}|"
                     f"{mine['annual_excess'] - base['annual_excess']:+.10f}|"
                     f"{mine['mean_turnover'] - base['mean_turnover']:+.10f}|")
    confirm_metrics = confirm_entry["saved"]["metrics"]["band_qstar"]
    gate = 0.3873823479464019
    verdict_replace = confirm_metrics["final_score"] >= gate
    lines += ["", "## confirm2024", "",
              f"|方案|IC|年化超额|换手|综合分|", "|---|---:|---:|---:|---:|"]
    for label, metrics in [("T030 冻结", anchor_frozen[1]["confirm2024"]["band"]),
                           (f"S008 winner T{winner_number:03d}", confirm_metrics),
                           ("winner band 0.1（诊断）", band01_winner),
                           ("T030 band 0.1（诊断）", anchor_confirm)]:
        lines.append(f"|{label}|{metrics['ic_mean']:.10f}|{metrics['annual_excess']:.10f}|"
                     f"{metrics['mean_turnover']:.10f}|{metrics['final_score']:.10f}|")
    wf2023 = winner_entries["wf2023"]["saved"]["metrics"]["band_qstar"]["final_score"]
    lines += ["", "## 判定", "",
              f"- 替换门槛（confirm2024 band q* ≥ {gate:.10f}）："
              f"{'通过' if verdict_replace else '未通过 ⇒ retain T030'}。",
              f"- 成功标准 2023 ≥ 0.38：{'通过' if wf2023 >= 0.38 else '未通过'}"
              f"（winner 2023 = {wf2023:.10f}）。",
              f"- 成功标准 2024 ≥ 0.38：{'通过' if confirm_metrics['final_score'] >= 0.38 else '未通过'}"
              f"（confirm2024 = {confirm_metrics['final_score']:.10f}）。", "",
              "## Phase B：winner 的换手层网格（band 重编码，不重训）", "",
              "|keep_q|" + "|".join(SEARCH_FOLDS + ["confirm2024"]) + "|",
              "|---:|" + "---:|" * (len(SEARCH_FOLDS) + 1)]
    scan_by_q = {}
    for fold_name, rows in scan_tables.items():
        for row in rows:
            scan_by_q.setdefault(row["keep_q"], {})[fold_name] = row["final_score"]
    for keep_q, per_fold in scan_by_q.items():
        lines.append(f"|{keep_q:.10g}|" +
                     "|".join(f"{per_fold.get(f, float('nan')):.10f}" for f in SEARCH_FOLDS + ["confirm2024"]) + "|")
    lines += ["", "## 参数重要性（fANOVA，本轮观测）", ""]
    for name, value in importance.items():
        lines.append(f"- {name}: {value:.4f}")
    lines += ["", "## 产物与解释边界", "",
              f"研究产物：{relative(root)}；trials 表：experiments/optuna_s008_trials.csv；"
              "scan 表：experiments/optuna_s008_scan.csv。",
              "trials 阶段评分只用进程内 evaluate_frame（与官方评分器同实现），"
              "锚点与 winner 的 raw/band 另做外部评分器核对（1e-10）。",
              "测试期未参与本轮；2024 已被历史研究反复使用，非干净留出集。",
              "50 trials 不保证全局最优；换手层只在 q* 与网格档上评分，"
              "未做平滑/其他控制器（S004/LH014 已证其不可靠）。", ""]
    report = contained_path(cfg["paths"]["research_reports"]) / "optuna_s008.md"
    report.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S008 separated Optuna study")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg, _ = load_config(args.config)
    revision = git_state()
    if revision["branch"] != "main" or revision["dirty"]:
        raise ValueError("S008 must run from a clean main working tree.")
    winner_spec_frozen, frozen = frozen_references()
    keep_q = float(winner_spec_frozen["spec"]["keep_q"])
    columns = feature_names(cfg["features"])
    if len(columns) != 40:
        raise ValueError("S008 expects the frozen 40-feature V1.")
    root = contained_path(cfg["paths"]["research_studies"]) / STUDY_ID
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "process.lock"
    if lock.exists():
        raise RuntimeError("S008 study lock exists; inspect before rerunning.")
    lock.write_text(json.dumps({"pid": __import__("os").getpid(), "started_at": timestamp()}),
                    encoding="utf-8")
    try:
        log = RunLog(root / "run.log")
        provenance = source_provenance(cfg)
        dataset = build_base_dataset(cfg, provenance, log)
        folds = fold_definitions(cfg)
        folds_by_name = {f.name: f for f in folds}
        sampler = optuna.samplers.TPESampler(seed=cfg["optuna"]["seed"],
                    n_startup_trials=cfg["optuna"]["startup_trials"], multivariate=True)
        study = optuna.create_study(study_name=STUDY_ID, direction="maximize",
                    storage="sqlite:///" + (root / "study.db").as_posix(),
                    sampler=sampler, pruner=optuna.pruners.NopPruner(),
                    load_if_exists=args.resume)
        if not args.resume:
            anchor = anchor_spec(cfg, keep_q)
            study.enqueue_trial({k: v for k, v in anchor["sampled_params"].items()},
                                user_attrs={"role": "reference_T030"})
        else:
            for t in study.get_trials(deepcopy=False):
                if t.state == TrialState.RUNNING:
                    t.set_user_attr("error", "interrupted by resume")
                    study.tell(t.number, state=TrialState.FAIL)
        budget = cfg["optuna"]["trial_budget"]
        anchor_entries = None
        while True:
            finished = [t for t in study.get_trials(deepcopy=False) if t.state.is_finished()]
            if len(finished) >= budget:
                break
            trial = study.ask()
            spec = sample_model_spec(trial, cfg, keep_q)
            if trial.number == 0 and spec != anchor_spec(cfg, keep_q):
                raise ValueError("Enqueued anchor trial 0 does not reproduce the T030 spec.")
            trial.set_user_attr("spec", spec)
            log(f"TRIAL {trial.number + 1}/{budget}: rounds={spec['rounds']}, "
                f"leaves={spec['model_params']['num_leaves']}, lr={spec['model_params']['learning_rate']:.6g}")
            try:
                entries = {}
                for fold_name in SEARCH_FOLDS:
                    out_dir = root / "trials" / f"T{trial.number:03d}" / fold_name
                    entries[fold_name] = evaluate_fold(
                        cfg, dataset, columns, folds_by_name[fold_name], spec, out_dir, log)
                trial.set_user_attr("entries", entries)
                objective = float(np.mean([entries[f]["metrics"]["band_qstar"]["final_score"]
                                           for f in SEARCH_FOLDS]))
                study.tell(trial, objective)
                log(f"TRIAL_COMPLETE {trial.number}: mean={objective:.10f}")
                if trial.number == 0:
                    anchor_entries = entries
                    verify_anchor(entries, frozen)
                    log("anchor reproduces frozen T030 within 1e-9 on all search folds")
            except InvalidTrial as error:
                trial.set_user_attr("error", str(error))
                study.tell(trial, state=TrialState.FAIL)
                log(f"TRIAL_FAIL {trial.number}: {error}")
            write_trials_table(study, root)
        trials = write_trials_table(study, root)
        complete = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
        if not complete or complete[0].number != 0:
            raise ValueError("Anchor trial 0 must be COMPLETE.")
        winner = max(complete, key=lambda t: (t.value, -t.number))
        winner_number = winner.number
        winner_spec = winner.user_attrs["spec"]
        anchor_ok = True
        importance = optuna.importance.get_param_importances(study,
                    evaluator=optuna.importance.FanovaImportanceEvaluator(seed=cfg["optuna"]["seed"]))
        save_json(root / "parameter_importance.json", importance)
        selection = {"study_id": STUDY_ID, "created_at": timestamp(), "keep_q_frozen": keep_q,
                     "winner_trial": winner_number, "winner_spec": winner_spec,
                     "winner_objective": winner.value,
                     "trials_completed": len(complete)}
        save_json(root / "selection.json", selection)
        log(f"SELECTION: winner T{winner_number:03d}, mean={winner.value:.10f}")
        # ---- winner 最终产物（含预测落盘 + 外部核对）与 confirm2024 ----
        winner_results = {}
        for fold_name in SEARCH_FOLDS + ["confirm2024"]:
            out_dir = root / "winner" / f"T{winner_number:03d}" / fold_name
            saved = evaluate_fold(cfg, dataset, columns, folds_by_name[fold_name],
                                  winner_spec, out_dir, log, save_artifacts=True,
                                  external_check=True)
            winner_results[fold_name] = {"fold": folds_by_name[fold_name], "spec": winner_spec,
                                         "saved": saved, "out_dir": out_dir}
        confirm_entry = winner_results["confirm2024"]["saved"]
        # winner 的 band 0.1 诊断层（confirm 折）
        pred_path = root / "winner" / f"T{winner_number:03d}" / "confirm2024" / "prediction_raw.csv"
        raw_pred = pd.read_csv(pred_path)
        truth_confirm = None
        train, valid, split = split_train_valid(dataset, folds_by_name["confirm2024"])
        truth_confirm = valid[KEYS + ["y_ret_1d", "flag_limit_up"]].copy()
        truth_confirm["flag_limit_up"] = truth_confirm.flag_limit_up.astype("int8")
        truth_confirm = truth_confirm.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
        del train, valid
        gc.collect()
        known = raw_pred.merge(truth_confirm[KEYS + ["flag_limit_up"]], on=KEYS, validate="one_to_one")
        banded = known[KEYS + ["pred", "flag_limit_up"]].copy()
        banded["pred"] = band_scores(banded, 0.1).to_numpy()
        band01_scored = banded.merge(truth_confirm[KEYS + ["y_ret_1d"]], on=KEYS, validate="one_to_one")
        band01_metrics = evaluate_frame(band01_scored)
        # T030 的 2024 band 0.1 诊断层已有官方产物（S003 confirmation）：直接引用冻结值
        anchor_confirm = frozen_references()[1]["comparison"]["selected_model_band01"]
        # ---- Phase B：换手层网格 ----
        scan_tables = {}
        for fold_name in SEARCH_FOLDS + ["confirm2024"]:
            pred = pd.read_csv(root / "winner" / f"T{winner_number:03d}" / fold_name
                               / "prediction_raw.csv")
            train, valid, _ = split_train_valid(dataset, folds_by_name[fold_name])
            truth = valid[KEYS + ["y_ret_1d", "flag_limit_up"]].copy()
            truth["flag_limit_up"] = truth.flag_limit_up.astype("int8")
            del train, valid
            gc.collect()
            truth = truth.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
            scored = pred.merge(truth, on=KEYS, validate="one_to_one")
            scan_tables[fold_name] = turnover_scan(cfg, scored, [keep_q] + SCAN_EXTRA_QS)
            del scored
            gc.collect()
        scan_frame = pd.DataFrame([
            {"fold": fold_name, **row} for fold_name, rows in scan_tables.items() for row in rows])
        scan_frame.to_csv(ROOT / "experiments" / "optuna_s008_scan.csv", index=False)
        trials.to_csv(ROOT / "experiments" / "optuna_s008_trials.csv", index=False)
        append_winner_records(cfg, winner_number,
                              {k: v for k, v in winner_results.items()})
        summary = {"study_id": STUDY_ID, "created_at": timestamp(), "keep_q_frozen": keep_q,
                   "winner_trial": winner_number, "winner_spec": winner_spec,
                   "winner_objective": winner.value,
                   "winner_folds": {f: winner_results[f]["saved"]["metrics"] for f in winner_results},
                   "band01_confirm_winner": band01_metrics,
                   "scan": scan_tables,
                   "parameter_importance": importance}
        save_json(ROOT / "experiments" / "optuna_s008_summary.json", summary)
        write_report(cfg, root, trials, anchor_ok, winner_number, winner_spec,
                     winner_results, confirm_entry, band01_metrics, anchor_confirm,
                     scan_tables, importance)
        log(f"S008 COMPLETE: winner T{winner_number:03d}, "
            f"confirm2024={confirm_entry['metrics']['band_qstar']['final_score']:.10f}")
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""把长周期预测接入留仓带（band）后按官方 8 指标评分（事后重算，不重训）。

背景：``H20_F2T1`` / ``H20_F2T2`` 的预测已经存在（``LH003/{fold}/raw_predictions.parquet``），
本模块只做「接进留仓带 + 官方口径评分」，不重新训练任何模型。

- ``keep_q`` 默认直读 T030 冻结 spec（``docs/optuna_tuning_S003_selection.json`` 的
  ``winner.spec.keep_q``，= 0.0022778298112255263），不重新扫描；接口留了 ``--keep-q``。
- 两种接入方式（区别只在“验证期首日之后还用不用这个模型的 pred”）：
  - ``whole``：整个验证期都用该模型的 pred —— 等价于「该模型 + band」，与 T030
    归档里的 band 分数同口径可比；
  - ``seed`` ：只有验证期**首日**用该模型的 pred 选初始 Top 1/10，其余交易日仍用
    1 日基线 H01 的 pred —— 即仓库既定接入路径（首日换成长期预测，之后靠留仓带
    维持），机制上不需要任何新代码。
- 评分一律走 ``src/evaluation/official_eval.py``（官方口径逐字一致）。band 只读
  ``pred`` 与 ``flag_limit_up``，不含未来信息。
- 自检：``H01`` 的 ``whole`` 结果必须逐位复现 S003 归档里同折的 band 指标
  （fold1 ``wf2023`` / fold2 ``confirm2024``），否则说明链路有问题。

CLI::

    python -m src.long_horizon.band_eval --study LH003 \\
        --models H20_F2T1,H20_F2T2 --modes whole,seed
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import valid_quote
from src.data.load_data import KEYS
from src.evaluation.official_eval import evaluate_frame
from src.evaluation.turnover import band_scores
from src.evaluation.validation import get_folds
from src.utils.project import ROOT, load_config, project_path, write_json

PRICE_COLUMNS = ["open", "high", "low", "close"]
DEFAULT_MODELS = ("H20_F2T1", "H20_F2T2")
# S003 归档里与两折对应的记录位置（用于「H01 + band」的自检对照）
S003_REFERENCE = {
    "fold1": ("docs/optuna_tuning_S003_selection.json", "wf2023"),
    "fold2": ("docs/optuna_tuning_S003_confirmation.json", "confirm2024"),
}
METRIC_KEYS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_valid_panel(csv_path: Path, valid_start: int, valid_end: int,
                     chunksize: int = 1_000_000) -> pd.DataFrame:
    """分块读原始训练集，只保留验证期行（避免一次载入 790 万行占内存）。

    清洗口径与 ``src/data/clean_data.py`` 一致：价格列的非有限值置 NaN、
    ``quote_valid`` = 四个价格列全部非空且为正。
    """
    usecols = KEYS + PRICE_COLUMNS + ["flag_limit_up", "y_ret_1d"]
    dtypes = {"ts_code": "str", "trade_date": "int64", "flag_limit_up": "int64",
              "y_ret_1d": "float64", **{c: "float64" for c in PRICE_COLUMNS}}
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize,
                             dtype=dtypes):
        mask = (chunk["trade_date"] >= valid_start) & (chunk["trade_date"] <= valid_end)
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        for col in PRICE_COLUMNS:
            values = sub[col].to_numpy(dtype="float64")
            if np.isinf(values).any():
                sub[col] = np.where(np.isinf(values), np.nan, values)
        parts.append(sub)
    if not parts:
        raise ValueError(f"验证期 {valid_start}-{valid_end} 在 {csv_path} 中没有数据")
    panel = pd.concat(parts, ignore_index=True)
    panel["quote_valid"] = valid_quote(panel)
    panel = panel.sort_values(KEYS, kind="stable").reset_index(drop=True)
    return panel


def build_pred(frame: pd.DataFrame, model: str, mode: str) -> np.ndarray:
    """按接入方式拼出验证期的 pred。

    - ``whole``：全程用 ``model`` 的预测；
    - ``seed`` ：验证期首日（面板第一个交易日）用 ``model``，其余日回落到 ``H01``。
    """
    p_model = frame[model].to_numpy(dtype="float64")
    if mode == "whole":
        return p_model
    if mode != "seed":
        raise ValueError(f"未知 mode {mode}")
    base = frame["H01"].to_numpy(dtype="float64")
    first_day = int(np.min(frame["trade_date"].to_numpy()))
    return np.where(frame["trade_date"].to_numpy() == first_day, p_model, base)


def _fill(pred: np.ndarray) -> np.ndarray:
    good = np.isfinite(pred)
    if good.all():
        return pred
    return np.where(good, pred, np.nanmin(pred[good]) - 1.0)


def score_modes(frame: pd.DataFrame, pred: np.ndarray, keep_q: float) -> dict:
    """返回未加 band（raw）与加 band 后的官方 4 项核心指标。"""
    scored = frame[KEYS + ["y_ret_1d", "flag_limit_up"]].assign(pred=_fill(pred))
    out: dict[str, float] = {}
    raw = evaluate_frame(scored)
    for key in METRIC_KEYS:
        out[f"raw_{key}"] = float(raw[key])
    banded = scored.assign(
        pred=_fill(band_scores(scored, keep_q).to_numpy(dtype="float64")))
    band = evaluate_frame(banded)
    for key in METRIC_KEYS:
        out[key] = float(band[key])
    del scored, banded
    return out


def reference_band(fold_name: str) -> dict | None:
    """读 S003 归档里同折的 band 指标（自检用）。"""
    spec = S003_REFERENCE.get(fold_name)
    if spec is None:
        return None
    path, key = spec
    if not (ROOT / path).exists():
        return None
    data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if isinstance(data.get("winner"), dict) and "folds" in data["winner"]:
        metrics = data["winner"]["folds"][key]["band"]["metrics"]
    else:
        metrics = data["candidate"]["band"]["metrics"]
    return {k: float(metrics[k]) for k in METRIC_KEYS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="长周期预测接入留仓带（band）后的官方评分（事后重算）")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study", default="LH003", help="预测产物子目录名")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--modes", default="whole,seed")
    parser.add_argument("--folds", default="fold1,fold2")
    parser.add_argument("--keep-q", type=float, default=None,
                        help="留仓带阈值；默认直读 T030 冻结 spec")
    args = parser.parse_args(argv)

    cfg, _ = load_config(args.config)
    settings = cfg["long_horizon"]
    spec_path = ROOT / settings["spec_source"]
    selection = json.loads(spec_path.read_text(encoding="utf-8"))
    keep_q = float(selection["winner"]["spec"]["keep_q"]) \
        if args.keep_q is None else float(args.keep_q)
    log(f"keep_q = {keep_q!r}（来源：{settings['spec_source']} · winner.spec.keep_q）")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if "H01" not in models:
        models = ["H01"] + models           # H01 既是自检也是 seed 的基准
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    wanted = [f.strip() for f in args.folds.split(",") if f.strip()]
    folds = [f for f in get_folds() if f.name in wanted]
    if not folds:
        raise ValueError(f"未找到折 {wanted}（configs/project.yaml validation.folds）")

    csv_path = project_path(cfg["paths"]["train"])
    out_root = project_path("outputs/long_horizon") / args.study
    rows: list[dict] = []
    refs: dict[str, dict] = {}

    for fold in folds:
        log(f"{fold.name}: 读取验证期 {fold.valid_start}–{fold.valid_end} 的原始行 ...")
        panel = read_valid_panel(csv_path, fold.valid_start, fold.valid_end)
        log(f"  {len(panel):,} 行 / {panel['trade_date'].nunique()} 个交易日 / "
            f"{panel['ts_code'].nunique()} 只")

        pred_path = out_root / fold.name / "raw_predictions.parquet"
        if not pred_path.exists():
            raise FileNotFoundError(f"缺少预测文件 {pred_path}")
        pred_frame = pd.read_parquet(pred_path)
        pred_frame["ts_code"] = pred_frame["ts_code"].astype(str)
        pred_frame["trade_date"] = pred_frame["trade_date"].astype("int64")
        use = [c for c in dict.fromkeys(models + ["H01"]) if c in pred_frame.columns]
        missing = [m for m in models if m not in pred_frame.columns]
        if missing:
            raise KeyError(f"{pred_path} 缺少模型列 {missing}；现有 {list(pred_frame.columns)}")
        merged = panel.merge(pred_frame[KEYS + use], on=KEYS, how="left",
                             validate="one_to_one")
        nan_rows = int(merged[use].isna().any(axis=1).sum())
        if nan_rows:
            log(f"  注意：{nan_rows:,} 行没有预测（将被填成最小值）")
        del panel, pred_frame
        gc.collect()

        ref = reference_band(fold.name)
        refs[fold.name] = ref or {}
        for model in models:
            for mode in modes:
                if mode == "seed" and model == "H01":
                    continue
                pred = build_pred(merged, model, mode)
                metrics = score_modes(merged, pred, keep_q)
                row = {"study": args.study, "fold": fold.name, "model": model,
                       "mode": mode, "keep_q": keep_q, **metrics}
                if model == "H01" and mode == "whole" and ref:
                    row["s003_band_final"] = ref["final_score"]
                    row["band_final_diff_vs_s003"] = abs(
                        metrics["final_score"] - ref["final_score"])
                rows.append(row)
                log(f"  {fold.name} {model} [{mode}]：band final {metrics['final_score']:.10f} "
                    f"(IC {metrics['ic_mean']:+.6f} / 超额 {metrics['annual_excess']:+.6f} / "
                    f"换手 {metrics['mean_turnover']:.6f}) | raw final "
                    f"{metrics['raw_final_score']:.6f}")
        del merged
        gc.collect()

    frame = pd.DataFrame(rows)
    out_csv = ROOT / "experiments" / f"{args.study}_band_eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    write_json(out_root / "band_eval_report.json", json.loads(json.dumps({
        "study": args.study, "keep_q": keep_q, "models": models, "modes": modes,
        "spec_source": settings["spec_source"], "s003_reference": refs,
        "rows": rows, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, default=float)))

    pd.set_option("display.width", 240)
    print(f"\n===== {args.study} 接入留仓带（keep_q={keep_q:.10f}）官方评分 =====")
    print(frame[["fold", "model", "mode", "ic_mean", "annual_excess",
                 "mean_turnover", "final_score",
                 "raw_final_score"]].round(6).to_string(index=False))
    pivot = frame.pivot_table(index=["model", "mode"], columns="fold",
                              values="final_score")
    print("\n----- final_score（行 = 模型 × 接入方式，列 = 折）-----")
    print(pivot.round(6).to_string())

    check = frame[(frame.model == "H01") & (frame["mode"] == "whole")]
    for _, r in check.iterrows():
        ref = refs.get(r["fold"], {}).get("final_score")
        if ref is None:
            continue
        diff = abs(r["final_score"] - ref)
        flag = "逐位复现" if diff < 1e-10 else f"差异 {diff:.3e}（需排查）"
        print(f"自检 {r['fold']}：H01+band {r['final_score']:.10f} vs S003 归档 "
              f"{ref:.10f} → {flag}")
    print(f"\n汇总已写入 {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

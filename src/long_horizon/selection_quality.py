"""Top 组选股质量的**宽容度 / 下行风险**口径（事后重算，不涉及训练）。

**统一评估周期（重要）**：所有模型——不论它预测的是 1 日还是 30 日——选出的
top1/10 组一律用**同一个长周期实际收益**（``--eval-horizons``，默认 30 日）来
评估。理由是这里的任务目标是「找出长期有超额的股票」，而不是「做题的周期与
打分的周期一致」：若 H01 只跟实际 1 日收益比、H30 只跟实际 30 日收益比，两者
的基准不同，无法回答「谁更能选出长期赢家」。统一周期后所有模型落在**同一天
集合**（能算出 30 日收益的那些交易日）上比较，口径严格可比。

背景：留仓带（T030 ``keep_q≈0.0023``）把首日进入 Top 1/10 的股票几乎冻结整折，
换手仅 0.018——选错很难换出。因此真正要问的不是「预测组与实际 top10% 严丝合缝
重合多少」，而是：

1. **宽容度**：预测选出的股票即使没落在实际 top10%，只要还在前列（top15/20/30）
   就影响有限；
2. **下行风险**：最怕的是「刚选进去就立刻跌下去」——即预测组的实际排名很靠后。

本模块在已有预测（``raw_predictions.parquet``）上逐日重算下列指标，全部为
**排名口径**（不看收益率数值本身），随机选择下的理论基线一并给出：

- ``hit_{k}``  ：预测 top1/10 组中，实际排名落在 top k% 以内的比例。
  ``hit_10`` 即严格重合率。**随机基线 = k**（如 hit_15 ≈ 0.15）。
- ``mean_actual_pct``：预测组成员在当日实际收益序中的**平均百分位**，
  取值 (0,1]，随机 ≈ 0.50，完全命中 ≈ 0.95。
- ``worst_decile_rate`` / ``bottom_half_rate`` / ``bottom_20_rate``：
  预测组成员的实际百分位落在 <0.10 / <0.50 / <0.20 的比例（**越低越好**），
  随机基线分别 = 0.10 / 0.50 / 0.20。
- ``pred_excess_h``：预测组在**评估周期**上的实际收益均值 − 全市场均值（等权），
  随机 ≈ 0；``oracle_excess_h`` 为实际 top1/10 组的同口径值（上界参考）。

用法（仓库根目录）::

    python -m src.long_horizon.selection_quality --study LH002 --eval-horizons 30
    python -m src.long_horizon.selection_quality --study LH003 --folds fold1 \\
        --eval-horizons 5,10,20,30
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.data.clean_data import clean_history, valid_quote
from src.data.load_data import KEYS, load_training_data
from src.evaluation.official_eval import TOP_MIN_VALID
from src.evaluation.validation import get_folds, split_train_valid
from src.long_horizon.labels import (forward_labels, label_name,
                                     verify_against_official)
from src.utils.project import ROOT, load_config, project_path, write_json

# 宽容度档位与随机基线（随机抽取 1/10 的组，落在实际 top k% 内的期望 = k）
TOLERANCES = (0.10, 0.15, 0.20, 0.30)
# 下行风险档位（实际百分位低于该值视为「掉队」）
DOWNSIDE = (0.10, 0.20, 0.50)


def top_group_quality(y: np.ndarray, pred: np.ndarray, dates: np.ndarray,
                      eligible: np.ndarray, flag: np.ndarray,
                      k_frac: float = 0.10,
                      tolerances=TOLERANCES,
                      downside=DOWNSIDE,
                      min_valid: int = TOP_MIN_VALID) -> dict:
    """逐日评估「按预测取 top k_frac 组」的排名质量（宽容度 + 下行风险）。

    投资域与官方换手候选一致：报价有效、非涨停、且实际 h 日收益可算。
    组大小 k = floor(N * k_frac)（N 为该日投资域股票数）。
    所有指标先逐日计算、再对天数取均值（等权交易日）。
    """
    ok = eligible & (flag == 0) & np.isfinite(y)
    hit = {f"hit_{int(t * 100)}": [] for t in tolerances}
    pct_mean: list[float] = []
    pct_median: list[float] = []
    down = {f"downside_{int(d * 100)}": [] for d in downside}
    pred_exc: list[float] = []
    oracle_exc: list[float] = []
    sizes: list[int] = []
    for day in np.unique(dates):
        rows = ok & (dates == day)
        n = int(rows.sum())
        if n < min_valid:
            continue
        idx = np.flatnonzero(rows)
        a = y[idx]
        k = max(int(n * k_frac), 1)
        pred_rank = np.argsort(-pred[idx], kind="stable")
        actual_rank = np.argsort(-a, kind="stable")
        pred_top = pred_rank[:k]
        actual_top = actual_rank[:k]
        # 实际百分位（并列取平均秩），越接近 1 越好
        pct = rankdata(a, method="average") / n
        p = pct[pred_top]
        for t in tolerances:
            kt = max(int(n * t), 1)
            hit[f"hit_{int(t * 100)}"].append(
                float(np.isin(pred_top, actual_rank[:kt]).mean()))
        pct_mean.append(float(p.mean()))
        pct_median.append(float(np.median(p)))
        for d in downside:
            down[f"downside_{int(d * 100)}"].append(float((p < d).mean()))
        market = float(a.mean())
        pred_exc.append(float(a[pred_top].mean() - market))
        oracle_exc.append(float(a[actual_top].mean() - market))
        sizes.append(k)
    out: dict[str, float | int] = {}
    for name, vals in {**hit, **down}.items():
        arr = np.asarray(vals, dtype="float64")
        out[name] = float(arr.mean()) if len(arr) else float("nan")
    out["mean_actual_pct"] = float(np.mean(pct_mean)) if pct_mean else float("nan")
    out["median_actual_pct"] = float(np.mean(pct_median)) if pct_median else float("nan")
    out["pred_excess_h"] = float(np.mean(pred_exc)) if pred_exc else float("nan")
    out["oracle_excess_h"] = float(np.mean(oracle_exc)) if oracle_exc else float("nan")
    out["quality_days"] = len(sizes)
    out["quality_group_size"] = int(np.mean(sizes)) if sizes else 0
    # 相对随机基线的超额（pp）
    for t in tolerances:
        key = f"hit_{int(t * 100)}"
        out[f"{key}_lift"] = out[key] - t
    for d in downside:
        key = f"downside_{int(d * 100)}"
        out[f"{key}_lift"] = out[key] - d
    out["mean_actual_pct_lift"] = out["mean_actual_pct"] - 0.5
    return out


def _prediction_columns(pred: pd.DataFrame) -> list[str]:
    return [c for c in pred.columns if c not in ("ts_code", "trade_date")]


def _model_horizon(column: str) -> int | None:
    """从预测列名解析模型自身的预测周期（H01 / H30 / H05_F2T1 ...）。"""
    m = re.match(r"^H(\d+)", str(column))
    return int(m.group(1)) if m else None


def _eval_label(eval_horizon: int) -> str:
    return "y_ret_1d" if eval_horizon == 1 else label_name(eval_horizon)


def evaluate_study(study: str, folds: list[str], cfg_path: str = "configs/project.yaml",
                   eval_horizons: list[int] | None = None) -> pd.DataFrame:
    """在已有 raw 预测上重算宽容度/下行指标，返回长表。

    ``eval_horizons`` 为**统一评估周期**：每个模型的 top1/10 组都按这些周期的
    实际收益评估（默认 ``[30]``），与模型自身预测的周期无关。
    """
    cfg, _ = load_config(cfg_path)
    settings = cfg["long_horizon"]
    horizons = [int(h) for h in settings["horizons"]]
    eval_horizons = [int(h) for h in (eval_horizons or settings.get("eval_horizons", [30]))]
    all_periods = sorted(set([1] + horizons + eval_horizons))
    out_root = project_path("outputs/long_horizon") / study
    if not out_root.exists():
        raise FileNotFoundError(f"未找到产物目录 {out_root}")

    print(f"读取训练集并清洗（重算标签，不再训练）；统一评估周期 {eval_horizons} ...",
          flush=True)
    raw = load_training_data(project_path(cfg["paths"]["train"]))
    raw, _ = clean_history(raw)
    labels = forward_labels(raw, all_periods)
    verify = verify_against_official(labels[label_name(1)].to_numpy(),
                                     raw["y_ret_1d"], raw["trade_date"].to_numpy())
    print(f"  h=1 标签口径校验通过：{verify}", flush=True)
    panel = pd.DataFrame({
        "ts_code": raw["ts_code"].astype(str).to_numpy(),
        "trade_date": raw["trade_date"].to_numpy(),
        "quote_valid": valid_quote(raw).to_numpy(),
        "flag_limit_up": raw["flag_limit_up"].to_numpy(),
        "y_ret_1d": raw["y_ret_1d"].to_numpy(dtype="float64"),
    })
    for h in all_periods[1:]:
        panel[label_name(h)] = labels[label_name(h)].to_numpy()
    del raw, labels
    gc.collect()

    wanted = [f.strip() for f in folds if f.strip()]
    rows: list[dict] = []
    for fold in [f for f in get_folds() if f.name in wanted]:
        pred_path = out_root / fold.name / "raw_predictions.parquet"
        if not pred_path.exists():
            print(f"  跳过 {fold.name}：无 {pred_path.name}", flush=True)
            continue
        pred = pd.read_parquet(pred_path)
        pred["ts_code"] = pred["ts_code"].astype(str)
        pred["trade_date"] = pred["trade_date"].astype(int)
        _, valid_rows, _ = split_train_valid(panel, fold)
        merged = valid_rows.merge(pred, on=["ts_code", "trade_date"], how="left")
        dates = merged["trade_date"].to_numpy()
        elig = merged["quote_valid"].to_numpy()
        flag = merged["flag_limit_up"].to_numpy()
        for col in _prediction_columns(pred):
            model_h = _model_horizon(col)
            p = merged[col].to_numpy(dtype="float64")
            good = np.isfinite(p)
            if not good.all():
                p = np.where(good, p, np.nanmin(p[good]) - 1.0)
            for eh in eval_horizons:
                y = merged[_eval_label(eh)].to_numpy(dtype="float64")
                metrics = top_group_quality(y, p, dates, elig, flag)
                rows.append({"study": study, "fold": fold.name, "model": col,
                             "model_horizon": model_h, "eval_horizon": eh,
                             **metrics})
                print(f"  {fold.name} {col}（模型 h={model_h}）评估周期 {eh}d："
                      f"hit_10 {metrics['hit_10']:.4f} | hit_15 {metrics['hit_15']:.4f} | "
                      f"hit_20 {metrics['hit_20']:.4f} | "
                      f"平均实际百分位 {metrics['mean_actual_pct']:.4f} | "
                      f"跌破后 50% {metrics['downside_50']:.4f} | "
                      f"组超额 {metrics['pred_excess_h']:+.4f}", flush=True)
        del merged, pred
        gc.collect()
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Top 组选股质量：宽容度 + 下行风险（事后重算，统一评估周期）")
    parser.add_argument("--study", required=True, help="产物子目录名，如 LH002 / LH003")
    parser.add_argument("--folds", default="fold1,fold2")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--eval-horizons", default="30",
                        help="统一评估周期，逗号分隔（所有模型都用它评估，默认 30）")
    parser.add_argument("--tag", default=None, help="输出文件名后缀，默认随评估周期")
    args = parser.parse_args(argv)

    eval_horizons = [int(x) for x in str(args.eval_horizons).split(",") if x.strip()]
    frame = evaluate_study(args.study, args.folds.split(","), args.config, eval_horizons)
    if frame.empty:
        print("无可用预测，未产出结果。")
        return 1
    tag = args.tag or ("h" + "-".join(str(h) for h in eval_horizons))
    out_csv = ROOT / "experiments" / f"{args.study}_selection_quality_eval{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)

    pd.set_option("display.width", 260)
    cols = ["fold", "model", "model_horizon", "eval_horizon", "hit_10", "hit_15", "hit_20",
            "hit_30", "mean_actual_pct", "downside_50", "downside_20", "downside_10",
            "pred_excess_h", "oracle_excess_h"]
    print(f"\n===== {args.study} 选股质量（宽容度 + 下行风险，统一评估周期 {eval_horizons}）=====")
    print(frame[cols].round(4).to_string(index=False))
    print("\n随机基线：hit_10=0.100 hit_15=0.150 hit_20=0.200 hit_30=0.300 | "
          "mean_actual_pct=0.500 | downside_50=0.500 downside_20=0.200 downside_10=0.100")

    summary = {
        "study": args.study,
        "eval_horizons": eval_horizons,
        "baselines": {"hit_10": 0.10, "hit_15": 0.15, "hit_20": 0.20, "hit_30": 0.30,
                      "mean_actual_pct": 0.5, "downside_50": 0.5,
                      "downside_20": 0.2, "downside_10": 0.1},
        "rows": frame.to_dict(orient="records"),
    }
    write_json(project_path("outputs/long_horizon") / args.study / f"selection_quality_eval{tag}.json",
               json.loads(json.dumps(summary, default=float)))
    print(f"汇总已写入 {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

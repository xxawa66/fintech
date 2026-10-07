"""成员 B 独立复核 S002：复现 A 的两折分数 + 扫描融合权重 / 留仓带参数。

只读取 tmp/review_s002_b/artifacts/<fold>/ 下自建的组件预测，评分一律复用
src/evaluation/official_eval.py（官方口径）。不写共享实验表。

用法::

    python src/evaluation/s002_review/score_grid.py --fold fold1 --tag full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_data import KEYS                              # noqa: E402
from src.evaluation.official_eval import (                       # noqa: E402
    OFFICIAL_METRICS, evaluate_frame, load_validation_frame)
from src.evaluation.turnover import band_scores, smooth_scores   # noqa: E402
from src.models.ensemble import rank_blend                       # noqa: E402
from src.utils.project import write_json                         # noqa: E402

ART = ROOT / "tmp" / "review_s002_b" / "artifacts"

# 从 A 写下的正式实验表读取对照记录，避免手抄数字
A_ROWS = {
    "fold1": {"L1": "S002_screen_L1", "L1_band": "S002_screen_L1_band",
              "R2": "S002_screen_R2", "R2_band": "S002_screen_R2_band",
              "F025": "S002_screen_F025", "F025_band": "S002_screen_F025_band"},
    "fold2": {"L1": "S002_confirm_L1", "L1_band": "S002_confirm_L1_band",
              "R2": "S002_confirm_R2", "R2_band": "S002_confirm_R2_band",
              "F025": "S002_confirm_F025", "F025_band": "S002_confirm_F025_band"},
}
SHORT = ["ic_mean", "annual_excess", "mean_turnover", "final_score"]


def a_records(fold: str) -> pd.DataFrame:
    """按短键（L1 / L1_band / R2 / …）索引 A 写下的正式实验记录。"""
    log = pd.read_csv(ROOT / "experiments" / "experiment_log.csv").set_index("exp_id")
    rows = A_ROWS[fold]
    missing = set(rows.values()) - set(log.index)
    if missing:
        raise ValueError(f"共享实验表缺少记录: {sorted(missing)}")
    return log.loc[list(rows.values())].set_axis(list(rows.keys()))


def scored_frame(pred_frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = pred_frame.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    if len(df) != len(pred_frame):
        raise ValueError("预测与标签的键集合不一致")
    return df[KEYS + ["pred", "y_ret_1d", "flag_limit_up"]]


def banded(scored: pd.DataFrame, keep_q: float) -> pd.DataFrame:
    return scored.assign(pred=band_scores(scored, keep_q).to_numpy())


def smoothed(scored: pd.DataFrame, alpha: float) -> pd.DataFrame:
    return scored.assign(pred=smooth_scores(scored, alpha).to_numpy())


def brief(metrics: dict) -> list[float]:
    return [metrics[k] for k in SHORT]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["fold1", "fold2"])
    parser.add_argument("--tag", default="full")
    args = parser.parse_args(argv)

    root = ART / args.fold
    lgb_frame = pd.read_csv(root / "l1_valid.csv")
    ridge_frame = pd.read_csv(root / "r2_valid.csv")
    labels = pd.read_csv(root / "validation_labels.csv")
    scored = {"L1": scored_frame(lgb_frame, labels), "R2": scored_frame(ridge_frame, labels)}
    print(f"[{args.fold}] 键 {len(labels):,}，交易日 {labels.trade_date.nunique()}", flush=True)

    timings = {}

    def timeit(name, fn):
        step = time.perf_counter()
        value = fn()
        timings[name] = time.perf_counter() - step
        return value

    def blend(w: float) -> pd.DataFrame:
        frame = rank_blend([lgb_frame, ridge_frame], [w, 1.0 - w])
        return scored_frame(frame, labels)

    # ---------- 1. 复现：组件与 F025，两层（raw / band 0.1） ----------
    recorded = a_records(args.fold)
    repro: dict[str, dict] = {}
    for name, df in scored.items():
        repro[name] = {"raw": brief(evaluate_frame(df)),
                       "band": brief(evaluate_frame(banded(df, 0.1)))}
    f025 = timeit("blend_F025", lambda: blend(0.25))
    repro["F025"] = {"raw": brief(evaluate_frame(f025)),
                     "band": brief(evaluate_frame(banded(f025, 0.1)))}

    print(f"\n=== {args.fold} 复现：本机重建 vs A 的实验表记录（IC / 超额 / 换手 / 综合分）===")
    comparisons = []
    for name in ["L1", "R2", "F025"]:
        for layer in ["raw", "band"]:
            row_id = f"{name}_band" if layer == "band" else name
            mine = repro[name][layer]
            theirs = [float(recorded.loc[row_id, k]) for k in SHORT]
            delta = [m - t for m, t in zip(mine, theirs)]
            comparisons.append({"scheme": name, "layer": layer, "record": row_id,
                                **{f"mine_{k}": v for k, v in zip(SHORT, mine)},
                                **{f"a_{k}": v for k, v in zip(SHORT, theirs)},
                                **{f"delta_{k}": v for k, v in zip(SHORT, delta)}})
            print(f"{name:>4} {layer:>4} 本机 {mine[0]:.6f} {mine[1]:.6f} {mine[2]:.6f} {mine[3]:.10f} | "
                  f"A {theirs[0]:.6f} {theirs[1]:.6f} {theirs[2]:.6f} {theirs[3]:.10f} | "
                  f"Δ分 {delta[3]:+.3e}")
    compare_table = pd.DataFrame(comparisons)
    compare_table.to_csv(root / "repro_comparison.csv", index=False)
    worst = float(compare_table[[f"delta_{k}" for k in SHORT]].abs().to_numpy().max())
    print(f"最大单项差异 {worst:.3e}" + ("（逐位一致）" if worst < 1e-9 else "（存在差异，需排查）"))

    if args.tag == "quick":
        write_json(root / f"repro_{args.tag}.json",
                   {"fold": args.fold, "repro": repro, "max_abs_difference": worst,
                    "timings": timings})
        return 0

    # ---------- 2. 权重网格（keep_q = 0.1） ----------
    weights = [round(i * 0.05, 2) for i in range(21)]
    rows = []
    for w in weights:
        df = blend(w)
        raw = evaluate_frame(df)
        band = evaluate_frame(banded(df, 0.1))
        rows.append({"weight_l1": w, **{f"raw_{k}": raw[k] for k in OFFICIAL_METRICS},
                     **{f"band_{k}": band[k] for k in OFFICIAL_METRICS}})
    weight_table = pd.DataFrame(rows)
    weight_table.to_csv(root / "grid_weights.csv", index=False)
    best_w = weight_table.loc[weight_table["band_final_score"].idxmax()]
    print(f"\n权重网格（keep_q=0.1）最优 w_L1={best_w['weight_l1']:.2f}，"
          f"band 综合分 {best_w['band_final_score']:.10f}")

    # ---------- 3. keep_q 网格（若干权重） ----------
    keep_qs = [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.001]
    probe_weights = sorted({0.0, 0.25, 1.0, float(best_w["weight_l1"])})
    rows = []
    for w in probe_weights:
        df = blend(w)
        for keep_q in keep_qs:
            metrics = evaluate_frame(banded(df, keep_q))
            rows.append({"weight_l1": w, "keep_q": keep_q,
                         **{k: metrics[k] for k in OFFICIAL_METRICS}})
    keep_table = pd.DataFrame(rows)
    raw_ic = weight_table.set_index("weight_l1")["raw_ic_mean"]
    keep_table["ic_retention"] = keep_table.apply(
        lambda r: r["ic_mean"] / float(raw_ic.loc[r["weight_l1"]])
        if r["weight_l1"] in raw_ic.index else np.nan, axis=1)
    keep_table.to_csv(root / "grid_keep_q.csv", index=False)
    top = keep_table.sort_values("final_score", ascending=False).head(8)
    print("\n(权重 × keep_q) 前八：")
    print(top[["weight_l1", "keep_q", "ic_mean", "annual_excess", "mean_turnover",
               "final_score"]].to_string(index=False))

    # ---------- 4. 组合：排名平滑 + 留仓带 ----------
    rows = []
    for w in [0.25, float(best_w["weight_l1"])]:
        df0 = blend(w)
        for alpha in [1.0, 0.9, 0.8, 0.7]:
            for keep_q in [0.10, 0.05]:
                df = df0 if alpha == 1.0 else smoothed(df0, alpha)
                metrics = evaluate_frame(banded(df, keep_q))
                rows.append({"weight_l1": w, "smooth_alpha": alpha, "keep_q": keep_q,
                             **{k: metrics[k] for k in OFFICIAL_METRICS}})
    combo = pd.DataFrame(rows)
    combo.to_csv(root / "grid_smooth_band.csv", index=False)
    print("\n平滑 + 留仓带组合：")
    print(combo[["weight_l1", "smooth_alpha", "keep_q", "ic_mean", "annual_excess",
                  "mean_turnover", "final_score"]].to_string(index=False))

    write_json(root / f"repro_{args.tag}.json",
               {"fold": args.fold, "repro": repro, "max_abs_difference": worst,
                "best_weight": float(best_w["weight_l1"]),
                "best_weight_band_score": float(best_w["band_final_score"]),
                "timings": timings})
    print(f"\n产物写入 {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

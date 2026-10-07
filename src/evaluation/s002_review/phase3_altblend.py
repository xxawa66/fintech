"""成员 B 复核 S002（第三阶段）：排名融合 vs z-score 连续融合的对照。

起因（复核发现）：``rank_blend`` 把每只股票的当日百分位排名加权求和，排名取值是
``k/N`` 的倍数，两个排名相加后落到 ``0.25/N`` 的格点上，**融合分出现大量精确并列**
（fold1 实测 139,561 对，占 12.4%，单日峰值 725 对）。官方 Top 1/10 边界落进并列组时，
取哪几只由非稳定排序决定——分数里因此混进一段与模型无关的任意性，且融合的有效信息量
被量化损耗。

本脚本用**当日截面 z-score 连续融合**作对照：先对每个组件的原始预测做当日标准化，再
按同一组权重线性加权（连续值，几乎不产生并列），其余环节（留仓带、官方评分）不变。
两折都跑，严格按 A 的纪律：fold1 选点、fold2 确认。

用法::

    python src/evaluation/s002_review/phase3_altblend.py --fold fold1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.official_eval import OFFICIAL_METRICS, evaluate_frame    # noqa: E402
from src.models.ensemble import align_predictions                            # noqa: E402
from src.utils.project import write_json                                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_ops import Panel                                                   # noqa: E402

ART = ROOT / "tmp" / "review_s002_b" / "artifacts"
WEIGHTS = [round(i * 0.05, 2) for i in range(21)]
SHORT = ["ic_mean", "annual_excess", "mean_turnover", "final_score"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tie_stats(values: np.ndarray, day_index: np.ndarray) -> dict:
    """统计逐日精确并列（同一交易日内的重复取值）。"""
    dup = pd.Series(values).groupby(day_index).apply(lambda s: int(s.duplicated().sum()))
    return {"tied_pairs": int(dup.sum()),
            "tied_ratio": float(dup.sum() / len(values)),
            "peak_day_tied_pairs": int(dup.max())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["fold1", "fold2"])
    args = parser.parse_args(argv)
    root = ART / args.fold
    t0 = time.perf_counter()

    lgb = pd.read_csv(root / "l1_valid.csv")
    ridge = pd.read_csv(root / "r2_valid.csv")
    labels = pd.read_csv(root / "validation_labels.csv")
    keys, values = align_predictions([lgb, ridge])
    panel = Panel(keys, labels)
    days = panel.keys["trade_date"].to_numpy()
    frame = panel.frame(np.zeros(panel.D * panel.S))

    def score(pred) -> dict:
        frame["pred"] = np.asarray(pred, dtype=float)
        return evaluate_frame(frame)

    # 两种组件的当日截面标准化：排名（A 的方法）/ z-score（本对照）
    ranks = pd.DataFrame(values).groupby(days).rank(method="average", pct=True).to_numpy()
    zdf = pd.DataFrame(values)
    z = ((zdf - zdf.groupby(days).transform("mean"))
         / zdf.groupby(days).transform("std").replace(0.0, np.nan)).to_numpy()

    log(f"[{args.fold}] 融合并列结构：")
    for name, mat in [("排名 0.25/0.75", ranks @ np.array([0.25, 0.75])),
                      ("z-score 0.25/0.75", z @ np.array([0.25, 0.75]))]:
        s = tie_stats(mat, days)
        log(f"    {name:<20} 并列对 {s['tied_pairs']:>8,}（{s['tied_ratio']:.2%}），"
            f"单日峰值 {s['peak_day_tied_pairs']:>4}")

    rows = []
    for method, mat in [("rank", ranks), ("zscore", z)]:
        for w in WEIGHTS:
            pred = mat @ np.array([w, 1.0 - w])
            raw = score(pred)
            band = score(panel.apply(pred, "band", 0.1))
            rows.append({"method": method, "weight_l1": w,
                         **{f"raw_{k}": raw[k] for k in OFFICIAL_METRICS},
                         **{f"band_{k}": band[k] for k in OFFICIAL_METRICS}})
        log(f"[{args.fold}] {method} 21 点网格完成")
    table = pd.DataFrame(rows)
    table.to_csv(root / "grid_altblend.csv", index=False)

    best = {}
    for method in ("rank", "zscore"):
        sub = table[table["method"] == method]
        b = sub.loc[sub["band_final_score"].idxmax()]
        best[method] = {"weight_l1": float(b["weight_l1"]),
                        "band_final_score": float(b["band_final_score"]),
                        "band_ic_mean": float(b["band_ic_mean"]),
                        "band_annual_excess": float(b["band_annual_excess"]),
                        "band_mean_turnover": float(b["band_mean_turnover"]),
                        "raw_ic_mean": float(sub.loc[sub["weight_l1"] == b["weight_l1"],
                                                     "raw_ic_mean"].iloc[0])}
        log(f"[{args.fold}] {method:<7} 最优 w_L1={b['weight_l1']:.2f} → band "
            f"{b['band_final_score']:.10f}（IC {b['band_ic_mean']:.6f} / 超额 "
            f"{b['band_annual_excess']:.6f} / 换手 {b['band_mean_turnover']:.6f}）")

    a_rank = table[(table["method"] == "rank") & (table["weight_l1"] == 0.25)].iloc[0]
    a_z = table[(table["method"] == "zscore") & (table["weight_l1"] == 0.25)].iloc[0]
    log(f"[{args.fold}] 同为 w=0.25/keep_q=0.1：排名融合 {a_rank['band_final_score']:.10f} "
        f"vs z-score 融合 {a_z['band_final_score']:.10f}（Δ "
        f"{a_z['band_final_score'] - a_rank['band_final_score']:+.10f}）")

    write_json(root / "phase3_summary.json", {
        "fold": args.fold,
        "best_by_method": best,
        "at_w025_keepq01": {
            "rank": {k: float(a_rank["band_" + k]) for k in SHORT},
            "zscore": {k: float(a_z["band_" + k]) for k in SHORT},
        },
        "elapsed_seconds": time.perf_counter() - t0,
    })
    log(f"[{args.fold}] 完成，用时 {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

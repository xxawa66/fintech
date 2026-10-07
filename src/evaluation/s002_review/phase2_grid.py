"""成员 B 复核 S002（第二阶段）：稠密面板加速的参数网格，两折各跑一遍。

与第一版 ``score_grid.py`` 的区别：

* 融合权重网格从 A 的 5 点 ``{0, .25, .5, .75, 1}`` 扩到 21 点（步长 0.05）；
* 在同一融合预测上重扫 ``keep_q``（A 把 0.1 写死在协议里，没有在融合预测上重扫）；
* 加扫 ``排名平滑 × 留仓带`` 组合；
* 变换全部走 ``fast_ops.Panel``（截面排名缓存），评分仍是官方口径
  ``src.evaluation.official_eval.evaluate_frame``，且启动时与 ``src`` 实现自检。

只读本机自建组件 ``artifacts/<fold>/``，不写共享实验表、不覆盖 A 的产物。

用法::

    python src/evaluation/s002_review/phase2_grid.py --fold fold1
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

from src.evaluation.official_eval import OFFICIAL_METRICS, evaluate_frame   # noqa: E402
from src.models.ensemble import align_predictions                           # noqa: E402
from src.utils.project import write_json                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_ops import Panel, selfcheck                                       # noqa: E402

ART = ROOT / "tmp" / "review_s002_b" / "artifacts"

# A 写进正式实验表的记录（按 exp_id 取，避免手抄数字）
A_ROWS = {
    "fold1": {"L1": "S002_screen_L1", "L1_band": "S002_screen_L1_band",
              "R2": "S002_screen_R2", "R2_band": "S002_screen_R2_band",
              "F025": "S002_screen_F025", "F025_band": "S002_screen_F025_band"},
    "fold2": {"L1": "S002_confirm_L1", "L1_band": "S002_confirm_L1_band",
              "R2": "S002_confirm_R2", "R2_band": "S002_confirm_R2_band",
              "F025": "S002_confirm_F025", "F025_band": "S002_confirm_F025_band"},
}
SHORT = ["ic_mean", "annual_excess", "mean_turnover", "final_score"]
BAND_KEEP_Q = 0.1          # 团队锁定值（configs/project.yaml: evaluation.band.keep_q）
WEIGHTS = [round(i * 0.05, 2) for i in range(21)]
KEEP_SWEEP_WEIGHTS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
KEEP_QS = [0.30, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
COMBO_WEIGHTS = [0.25, 0.35]
COMBO_ALPHAS = [0.9, 0.8, 0.7]
COMBO_KEEP_QS = [0.10, 0.05]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Scorer:
    """复用同一个长表 DataFrame，只换 ``pred`` 列，避免重复分配 110 万行。"""

    def __init__(self, panel: Panel):
        self.panel = panel
        self.frame = panel.frame(np.zeros(self.panel.D * self.panel.S))
        self.n_calls = 0

    def __call__(self, pred) -> dict:
        self.frame["pred"] = np.asarray(pred, dtype=float)
        self.n_calls += 1
        return evaluate_frame(self.frame)


def a_records(fold: str) -> pd.DataFrame:
    log_df = pd.read_csv(ROOT / "experiments" / "experiment_log.csv").set_index("exp_id")
    rows = A_ROWS[fold]
    missing = set(rows.values()) - set(log_df.index)
    if missing:
        raise ValueError(f"共享实验表缺少记录: {sorted(missing)}")
    return log_df.loc[list(rows.values())].set_axis(list(rows.keys()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["fold1", "fold2"])
    args = parser.parse_args(argv)
    root = ART / args.fold
    t_start = time.perf_counter()

    log(f"[{args.fold}] 读入自建组件")
    lgb = pd.read_csv(root / "l1_valid.csv")
    ridge = pd.read_csv(root / "r2_valid.csv")
    labels = pd.read_csv(root / "validation_labels.csv")
    keys, values = align_predictions([lgb, ridge])
    panel = Panel(keys, labels)
    if not (np.array_equal(keys["trade_date"].to_numpy(), panel.keys["trade_date"].to_numpy())
            and np.array_equal(keys["ts_code"].to_numpy(), panel.keys["ts_code"].to_numpy())):
        raise ValueError("面板行序与 align_predictions 的键序不一致")
    log(f"[{args.fold}] 面板 {panel.D} 交易日 × {panel.S} 股 = {panel.D * panel.S:,} 行")

    # 融合排名只算一次；之后任意权重的融合分都是矩阵乘法
    ranks = pd.DataFrame(values).groupby(keys["trade_date"].to_numpy()).rank(
        method="average", pct=True).to_numpy()
    raw_component_rank = {"L1": ranks[:, 0], "R2": ranks[:, 1]}

    def blend(w: float) -> np.ndarray:
        return ranks @ np.array([w, 1.0 - w])

    # 与 rank_blend 对齐（w=0.25 必须逐位一致）
    from src.models.ensemble import rank_blend
    ref = rank_blend([lgb, ridge], [0.25, 0.75])
    ref_sorted = ref.sort_values(["trade_date", "ts_code"], kind="stable",
                                 ignore_index=True)["pred"].to_numpy()
    blend_gap = float(np.abs(blend(0.25) - ref_sorted).max())
    log(f"[{args.fold}] 稠密融合与 rank_blend 最大差 {blend_gap:.3e}"
        + ("（一致）" if blend_gap == 0 else "（不一致，需排查）"))

    score = Scorer(panel)
    checks = selfcheck(panel, blend(0.25))
    log(f"[{args.fold}] 变换自检：{len(checks)} 项，最大差 "
        f"{max(c['max_abs_diff'] for c in checks):.3e}（对 src 实现）")

    recorded = a_records(args.fold)

    # ---------- 1. 复现比对 ----------
    log(f"[{args.fold}] 复现比对")
    repro_rows = []
    for name, pred in [("L1", raw_component_rank["L1"]), ("R2", raw_component_rank["R2"]),
                       ("F025", blend(0.25))]:
        layers = {"raw": pred, "band": panel.apply(pred, "band", BAND_KEEP_Q)}
        for layer, p in layers.items():
            mine = score(p)
            row_id = f"{name}_band" if layer == "band" else name
            theirs = [float(recorded.loc[row_id, k]) for k in SHORT]
            repro_rows.append({
                "scheme": name, "layer": layer, "record": row_id,
                **{f"mine_{k}": mine[k] for k in SHORT},
                **{f"a_{k}": v for k, v in zip(SHORT, theirs)},
                **{f"delta_{k}": mine[k] - v for k, v in zip(SHORT, theirs)},
            })
    repro = pd.DataFrame(repro_rows)
    repro.to_csv(root / "repro_comparison.csv", index=False)
    for _, r in repro.iterrows():
        log(f"    {r['scheme']:>4} {r['layer']:>4} 本机 {r['mine_ic_mean']:.6f} "
            f"{r['mine_annual_excess']:.6f} {r['mine_mean_turnover']:.6f} "
            f"{r['mine_final_score']:.10f} | A {r['a_final_score']:.10f} | "
            f"Δ分 {r['delta_final_score']:+.3e}")
    worst = float(repro[[f"delta_{k}" for k in SHORT]].abs().to_numpy().max())

    # ---------- 2. 权重网格（21 点，raw 与 band(0.1) 两层） ----------
    log(f"[{args.fold}] 权重网格 {len(WEIGHTS)} 点 × 2 层")
    rows = []
    for w in WEIGHTS:
        p = blend(w)
        raw = score(p)
        band = score(panel.apply(p, "band", BAND_KEEP_Q))
        rows.append({"weight_l1": w, **{f"raw_{k}": raw[k] for k in OFFICIAL_METRICS},
                     **{f"band_{k}": band[k] for k in OFFICIAL_METRICS}})
    grid_w = pd.DataFrame(rows)
    grid_w.to_csv(root / "grid_weights.csv", index=False)
    best_w = grid_w.loc[grid_w["band_final_score"].idxmax()]
    a_w = grid_w.loc[grid_w["weight_l1"] == 0.25].iloc[0]
    log(f"[{args.fold}] 权重网格最优 w_L1={best_w['weight_l1']:.2f} "
        f"band 综合分 {best_w['band_final_score']:.10f}；"
        f"A 选的 0.25 为 {a_w['band_final_score']:.10f}（差 "
        f"{best_w['band_final_score'] - a_w['band_final_score']:+.10f}）")

    # ---------- 3. keep_q 重扫（针对融合预测） ----------
    log(f"[{args.fold}] keep_q 重扫：{len(KEEP_SWEEP_WEIGHTS)} 权重 × {len(KEEP_QS)} 档")
    rows = []
    for w in KEEP_SWEEP_WEIGHTS:
        p = blend(w)
        cache = panel.ranks(p)
        raw_ic = score(p)["ic_mean"]
        rows.append({"weight_l1": w, "keep_q": np.nan, "layer": "raw", "ic_retention": 1.0,
                     "ic_mean": raw_ic, "annual_excess": np.nan,
                     "mean_turnover": np.nan, "final_score": np.nan})
        for q in KEEP_QS:
            m = score(panel.apply(p, "band", q, cache))
            rows.append({"weight_l1": w, "keep_q": q, "layer": "band",
                         "ic_retention": m["ic_mean"] / raw_ic,
                         **{k: m[k] for k in OFFICIAL_METRICS}})
    grid_k = pd.DataFrame(rows)
    grid_k.to_csv(root / "grid_keep_q.csv", index=False)
    band_only = grid_k[grid_k["layer"] == "band"]
    best_k = band_only.loc[band_only["final_score"].idxmax()]
    a_k = band_only[(band_only["weight_l1"] == 0.25) & (band_only["keep_q"] == BAND_KEEP_Q)].iloc[0]
    log(f"[{args.fold}] keep_q 重扫最优 w={best_k['weight_l1']:.2f}/keep_q={best_k['keep_q']:.2f} "
        f"→ {best_k['final_score']:.10f}（IC {best_k['ic_mean']:.6f} 保留率 "
        f"{best_k['ic_retention']:.3f}，换手 {best_k['mean_turnover']:.6f}）；"
        f"A 的 w=0.25/keep_q=0.10 → {a_k['final_score']:.10f}")
    top = band_only.sort_values("final_score", ascending=False).head(10)
    log("    前十分档：\n" + top[["weight_l1", "keep_q", "ic_mean", "ic_retention",
                                  "annual_excess", "mean_turnover", "final_score"]]
        .to_string(index=False))

    # ---------- 4. 排名平滑 × 留仓带 ----------
    log(f"[{args.fold}] 平滑 × 留仓带组合")
    rows = []
    for w in COMBO_WEIGHTS:
        p = blend(w)
        raw_ic = score(p)["ic_mean"]
        for alpha in [1.0] + COMBO_ALPHAS:
            base = p if alpha == 1.0 else panel.smooth(p, alpha)
            for q in COMBO_KEEP_QS:
                m = score(panel.apply(base, "band", q))
                rows.append({"weight_l1": w, "smooth_alpha": alpha, "keep_q": q,
                             "ic_retention": m["ic_mean"] / raw_ic,
                             **{k: m[k] for k in OFFICIAL_METRICS}})
    combo = pd.DataFrame(rows)
    combo.to_csv(root / "grid_smooth_band.csv", index=False)
    log("    组合表：\n" + combo[["weight_l1", "smooth_alpha", "keep_q", "ic_mean",
                                  "ic_retention", "annual_excess", "mean_turnover",
                                  "final_score"]].to_string(index=False))

    elapsed = time.perf_counter() - t_start
    write_json(root / "phase2_summary.json", {
        "fold": args.fold,
        "panel": {"days": panel.D, "stocks": panel.S, "rows": panel.D * panel.S},
        "blend_gap_vs_rank_blend": blend_gap,
        "fast_ops_selfcheck": checks,
        "max_abs_repro_difference": worst,
        "n_evaluate_calls": score.n_calls,
        "best_weight_grid": {"weight_l1": float(best_w["weight_l1"]),
                             "band_final_score": float(best_w["band_final_score"])},
        "best_keep_q": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in best_k.items() if k in ("weight_l1", "keep_q",
                                                            "final_score", "ic_mean",
                                                            "ic_retention",
                                                            "mean_turnover",
                                                            "annual_excess")},
        "elapsed_seconds": elapsed,
    })
    log(f"[{args.fold}] 完成，{score.n_calls} 次官方评分，用时 {elapsed:.1f}s，产物 {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

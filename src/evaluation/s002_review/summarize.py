"""成员 B 复核 S002（汇总）：把两折的参数网格拉平成候选阶梯，按 A 的纪律
（fold1 选点、fold2 确认）给出结论，并对齐 A 的提分归因口径。

同时计算一个"冻结组合"退化基准：把首日排名原样复制到每一天（换手≈0，几乎零预测
能力），用来判断低 keep_q 的提分是不是在向"零换手套利"角落漂移。

用法::

    python src/evaluation/s002_review/summarize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_data import KEYS                                    # noqa: E402
from src.evaluation.official_eval import evaluate_frame                # noqa: E402
from src.models.ensemble import align_predictions                      # noqa: E402
from src.utils.project import write_json                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_ops import Panel                                             # noqa: E402

ART = ROOT / "tmp" / "review_s002_b" / "artifacts"
FOLDS = ["fold1", "fold2"]
SHORT = ["ic_mean", "annual_excess", "mean_turnover", "final_score"]


def load(fold: str) -> dict[str, pd.DataFrame]:
    root = ART / fold
    return {name: pd.read_csv(root / f"grid_{name}.csv")
            for name in ["weights", "keep_q", "smooth_band", "altblend"]}


def frozen_benchmark(fold: str) -> dict:
    """把首日排名复制到每一天：换手≈0 的退化提交，给低换手优化一个上限参照。"""
    root = ART / fold
    lgb = pd.read_csv(root / "l1_valid.csv")
    ridge = pd.read_csv(root / "r2_valid.csv")
    labels = pd.read_csv(root / "validation_labels.csv")
    keys, values = align_predictions([lgb, ridge])
    panel = Panel(keys, labels)
    ranks = pd.DataFrame(values).groupby(panel.keys["trade_date"].to_numpy()).rank(
        method="average", pct=True).to_numpy()
    fused = ranks @ np.array([0.25, 0.75])
    P = panel.pack(fused)
    frozen = np.repeat(P[:1], panel.D, axis=0)
    m = evaluate_frame(panel.frame(panel.unpack(frozen)))
    return {k: m[k] for k in SHORT}


def mask(*conditions) -> np.ndarray:
    """把一串 ``np.isclose`` 结果与 Series 条件合并成纯 numpy 布尔掩码。"""
    out = None
    for c in conditions:
        arr = np.asarray(c) if not isinstance(c, pd.Series) else c.to_numpy()
        out = arr if out is None else (out & arr)
    return out


def pick(table: pd.DataFrame, key: str, value) -> pd.Series:
    sub = table[np.isclose(table[key].to_numpy(dtype=float), float(value))]
    if sub.empty:
        raise KeyError(f"网格缺少 {key}={value}")
    return sub.iloc[0]


def main() -> int:
    grids = {f: load(f) for f in FOLDS}
    rows = []

    def add(label, family, w, alpha, kq, note=""):
        entry = {"label": label, "family": family, "weight_l1": w,
                 "smooth_alpha": alpha, "keep_q": kq, "note": note}
        for fold in FOLDS:
            g = grids[fold]
            if family == "smooth_band":
                r = g["smooth_band"]
                row = r[mask(np.isclose(r.weight_l1, w), np.isclose(r.smooth_alpha, alpha),
                             np.isclose(r.keep_q, kq))].iloc[0]
            elif family == "keep_q":
                r = g["keep_q"]
                row = r[mask(np.isclose(r.weight_l1, w), np.isclose(r.keep_q, kq),
                             r.layer == "band")].iloc[0]
            else:  # weights 网格固定 keep_q=0.1、无平滑
                r = g["weights"]
                row = r[mask(np.isclose(r.weight_l1, w))].iloc[0]
            for k in SHORT:
                col = f"band_{k}" if family == "weights" else k
                entry[f"{fold}_{k}"] = float(row[col])
        entry["mean_final"] = float(np.mean([entry[f"{f}_final_score"] for f in FOLDS]))
        entry["min_final"] = float(np.min([entry[f"{f}_final_score"] for f in FOLDS]))
        rows.append(entry)
        return entry

    # ---- A 锁定的方案与三档候选 ----
    add("C0 A 锁定 w=.25 kq=.10", "weights", 0.25, 1.0, None, "A 的 F025")
    add("C1 只改权重 w=.35 kq=.10", "weights", 0.35, 1.0, None, "fold1 权重网格最优")
    add("C2 只降 keep_q w=.25 kq=.05", "keep_q", 0.25, 1.0, 0.05, "")

    # ---- fold1 上按不同约束挑出的最优点 ----
    kq1 = grids["fold1"]["keep_q"]
    kq1b = kq1[kq1.layer == "band"]
    best_kq = kq1b.loc[kq1b.final_score.idxmax()]
    add(f"C3 fold1 keep_q 全局最优 w={best_kq.weight_l1:.2f} kq={best_kq.keep_q:.2f}",
        "keep_q", float(best_kq.weight_l1), 1.0, float(best_kq.keep_q),
        "未做 IC 保留率约束")
    hi = kq1b[kq1b.ic_retention >= 0.97]
    best_hi = hi.loc[hi.final_score.idxmax()]
    add(f"C4 fold1 高 IC 保留率最优 w={best_hi.weight_l1:.2f} kq={best_hi.keep_q:.2f}",
        "keep_q", float(best_hi.weight_l1), 1.0, float(best_hi.keep_q),
        "IC 保留率 ≥ 97%")
    sb1 = grids["fold1"]["smooth_band"]
    best_sb = sb1.loc[sb1.final_score.idxmax()]
    add(f"C5 fold1 组合最优 w={best_sb.weight_l1:.2f} a={best_sb.smooth_alpha} kq={best_sb.keep_q:.2f}",
        "smooth_band", float(best_sb.weight_l1), float(best_sb.smooth_alpha),
        float(best_sb.keep_q), "平滑+留仓带")
    sb2 = grids["fold2"]["smooth_band"]
    best_sb2 = sb2.loc[sb2.final_score.idxmax()]
    add(f"C6 fold2 组合最优 w={best_sb2.weight_l1:.2f} a={best_sb2.smooth_alpha} kq={best_sb2.keep_q:.2f}",
        "smooth_band", float(best_sb2.weight_l1), float(best_sb2.smooth_alpha),
        float(best_sb2.keep_q), "仅供对照，不参与选择")

    # ---- 推荐候选：在 A 的协议内只做"权重修正 + 排名平滑"，keep_q 保持协议值 ----
    add("R1 推荐 w=.35 a=0.9 kq=.10", "smooth_band", 0.35, 0.9, 0.10,
        "权重修正 + 排名平滑；keep_q 保持协议 0.1")
    add("R2 备选 w=.35 a=0.9 kq=.05", "smooth_band", 0.35, 0.9, 0.05,
        "再压一档换手，IC 保留率降至约 0.94")

    ladder = pd.DataFrame(rows)
    ladder.to_csv(ROOT / "tmp" / "review_s002_b" / "candidate_ladder.csv", index=False)

    base = ladder.iloc[0]
    pd.set_option("display.width", 240)
    print("=== 候选阶梯（两折，官方口径；同一套自建组件与评分代码）===")
    cols = ["label"] + [f"{f}_{k}" for f in FOLDS for k in SHORT] + ["mean_final", "min_final"]
    print(ladder[cols].to_string(index=False))

    print("\n=== 相对 C0（A 锁定）的提分归因（IC 项 / 超额项 / 低换手项 / 合计）===")
    attribution = []
    for _, r in ladder.iloc[1:].iterrows():
        for fold in FOLDS:
            d_ic = r[f"{fold}_ic_mean"] - base[f"{fold}_ic_mean"]
            d_ex = r[f"{fold}_annual_excess"] - base[f"{fold}_annual_excess"]
            d_tu = r[f"{fold}_mean_turnover"] - base[f"{fold}_mean_turnover"]
            attribution.append({
                "label": r["label"], "fold": fold,
                "IC项 x0.4": 0.4 * d_ic, "超额项 x0.3": 0.3 * d_ex,
                "低换手项 x0.3": -0.3 * d_tu,
                "合计": r[f"{fold}_final_score"] - base[f"{fold}_final_score"]})
    att = pd.DataFrame(attribution)
    att.to_csv(ROOT / "tmp" / "review_s002_b" / "candidate_attribution.csv", index=False)
    print(att.to_string(index=False, float_format=lambda v: f"{v:+.6f}"))

    print("\n=== 冻结组合退化基准（首日排名复制到每一天）===")
    frozen = {f: frozen_benchmark(f) for f in FOLDS}
    for f in FOLDS:
        m = frozen[f]
        print(f"  {f}: IC {m['ic_mean']:.6f} / 超额 {m['annual_excess']:.6f} / "
              f"换手 {m['mean_turnover']:.6f} / 综合分 {m['final_score']:.10f}")

    print("\n=== 权重网格最优（keep_q=0.1，两层）===")
    for fold in FOLDS:
        g = grids[fold]["weights"]
        bw = g.loc[g.band_final_score.idxmax()]
        a = g[np.isclose(g.weight_l1, 0.25)].iloc[0]
        print(f"  {fold}: 最优 w={bw.weight_l1:.2f} band {bw.band_final_score:.10f}；"
              f"w=0.25 → {a.band_final_score:.10f}（差 {bw.band_final_score - a.band_final_score:+.10f}）")

    print("\n=== 排名融合 vs z-score 融合（同点 w=0.25 / keep_q=0.1）===")
    for fold in FOLDS:
        g = grids[fold]["altblend"]
        rk = g[(g.method == "rank") & np.isclose(g.weight_l1, 0.25)].iloc[0]
        zs = g[(g.method == "zscore") & np.isclose(g.weight_l1, 0.25)].iloc[0]
        print(f"  {fold}: rank {rk.band_final_score:.10f} vs zscore {zs.band_final_score:.10f}"
              f"（Δ {zs.band_final_score - rk.band_final_score:+.10f}）")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    write_json(ROOT / "tmp" / "review_s002_b" / "review_summary.json", clean({
        "ladder": ladder.to_dict(orient="records"),
        "attribution": att.to_dict(orient="records"),
        "frozen_benchmark": frozen,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

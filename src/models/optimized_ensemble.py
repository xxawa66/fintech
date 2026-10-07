"""R1 优化候选：S002 组件的加权排名融合 + 排名平滑 + 冷启动留仓带（成员 B 复核建议）。

背景、证据与两折结果见 ``docs/model_research_S002_review.md``。本模块只做**信号组装**
与可选评分：读取已训练好的组件预测（S002 的 L1 / R2），按 ``configs/project.yaml`` 的
``optimized_candidate`` 参数构造 R1 信号并落盘；训练仍由既有 S002 流程负责。它**不接入**
``src/models/model_research.py`` 的筛选 / 确认状态机（那里的 ``validate_protocol`` 固定
了 5 点权重网格与 band(0.1)），因此在成员 A 确认前不会影响任何冻结产物。

信号链（三段，均为已知 X / pred 的确定性变换，不读 ``y_ret_1d``，无未来信息）::

    pred_raw    = rank_blend([L1, R2], [0.35, 0.65])     # 当日截面 pct 排名加权融合
    pred_smooth = smooth_scores(pred_raw, alpha=0.9)     # 排名指数平滑
    pred_out    = band_scores(pred_smooth, keep_q=0.1)   # 首日冷启动留仓带

用法（在仓库根目录运行）::

    python -m src.models.optimized_ensemble \
        --lightgbm outputs/predictions/S002_screen_L1/valid_2023.csv \
        --ridge    outputs/predictions/S002_screen_R2/valid_2023.csv \
        --labels   outputs/metrics/S002_screen_L1/validation_labels.csv \
        --out      outputs/predictions/S002_R1/valid_2023.csv

口径核对::

    python -m src.models.optimized_ensemble --selfcheck

在合成面板上断言 ``alpha = 1`` 时两段串联逐项复现纯留仓带（8 项官方指标一致）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.official_eval import OFFICIAL_METRICS, evaluate_frame
from src.evaluation.turnover import band_scores, smooth_band_scores
from src.models.ensemble import rank_blend
from src.utils.project import load_config, write_json


def signal_params(cfg: dict) -> dict:
    """从配置读取并校验 R1 参数。"""
    candidate = cfg["optimized_candidate"]
    weights = [float(w) for w in candidate["blend_weights"]]
    alpha = float(candidate["smooth_alpha"])
    keep_q = float(candidate["keep_q"])
    if len(weights) != 2 or min(weights) < 0 or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("optimized_candidate.blend_weights 必须是两个非负且和为 1 的权重。")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"smooth_alpha 必须在 (0, 1] 内，收到 {alpha}")
    if not 0.0 <= keep_q <= 1.0:
        raise ValueError(f"keep_q 必须在 [0, 1] 内，收到 {keep_q}")
    return {"name": candidate["name"], "weights": weights, "alpha": alpha, "keep_q": keep_q,
            "initial_state": candidate["initial_state"], "method": candidate["method"]}


def build_signal(lightgbm: pd.DataFrame, ridge: pd.DataFrame, labels: pd.DataFrame,
                 params: dict) -> pd.DataFrame:
    """融合 → 平滑 → 留仓带，返回 ``keys + pred``（口径与 ``band_scores`` 输出一致）。

    ``labels`` 需含 ``flag_limit_up``（当日已知 X 字段，留仓带据此确定候选集合）；
    ``y_ret_1d`` 只在调用方评分时使用，本函数不读取。返回行序为 ``(trade_date, ts_code)``
    规范序。
    """
    fused = rank_blend([lightgbm, ridge], params["weights"])
    missing = {"y_ret_1d", "flag_limit_up"} - set(labels.columns)
    if missing:
        raise ValueError(f"标签文件缺少列: {sorted(missing)}")
    scored = fused.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    if len(scored) != len(fused):
        raise ValueError(f"标签未覆盖全部融合键：{len(scored):,} / {len(fused):,}")
    out = scored[KEYS].copy()
    out["pred"] = smooth_band_scores(scored, params["alpha"], params["keep_q"]).to_numpy()
    return out.sort_values(KEYS, kind="stable").reset_index(drop=True)


def selfcheck(tol: float = 1e-12) -> None:
    """合成面板上核对 ``alpha = 1`` 时两段串联退化为纯留仓带（官方 8 指标一致）。"""
    rng = np.random.default_rng(11)
    n_stocks, n_days = 150, 30
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    records = []
    for day in range(n_days):
        date = 20240101 + day
        limit_up = rng.random(n_stocks) < 0.03
        pred = rng.normal(0.0, 1.0, n_stocks)
        for i in range(n_stocks):
            records.append({"ts_code": codes[i], "trade_date": date,
                            "pred": float(pred[i]), "flag_limit_up": int(limit_up[i])})
    scored = pd.DataFrame(records)
    scored["y_ret_1d"] = rng.normal(0.0, 0.02, len(scored))

    worst_metric, worst_pred = 0.0, 0.0
    for keep_q in (1.0, 0.8, 0.5, 0.1, 0.05):
        combined = smooth_band_scores(scored, 1.0, keep_q).to_numpy(dtype=float)
        plain = band_scores(scored, keep_q).to_numpy(dtype=float)
        worst_pred = max(worst_pred, float(np.nanmax(np.abs(combined - plain))))
        left = evaluate_frame(scored.assign(pred=combined))
        right = evaluate_frame(scored.assign(pred=plain))
        worst_metric = max(worst_metric, max(abs(left[k] - right[k]) for k in OFFICIAL_METRICS))
        if worst_metric >= tol:
            raise AssertionError(f"alpha=1 未退化为 band：keep_q={keep_q} 指标差 {worst_metric:.3e}")
    print(f"selfcheck PASS：alpha=1 与 band_scores 的 8 项指标最大差 {worst_metric:.3e}"
          f"（pred 数值最大差 {worst_pred:.3e}，仅编码尺度差异）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R1 优化候选信号组装（官方口径评分）")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--lightgbm", help="L1 组件预测 CSV（ts_code, trade_date, pred）")
    parser.add_argument("--ridge", help="R2 组件预测 CSV（ts_code, trade_date, pred）")
    parser.add_argument("--labels", help="标签 CSV（含 y_ret_1d, flag_limit_up）")
    parser.add_argument("--out", help="R1 信号输出 CSV（ts_code, trade_date, pred）")
    parser.add_argument("--metrics-out", help="可选：官方口径评分 JSON 输出路径")
    parser.add_argument("--selfcheck", action="store_true", help="合成口径核对，通过后退出")
    args = parser.parse_args(argv)

    if args.selfcheck:
        selfcheck()
        return 0
    missing = [n for n in ("lightgbm", "ridge", "labels", "out") if not getattr(args, n)]
    if missing:
        parser.error("需要 --lightgbm / --ridge / --labels / --out 或 --selfcheck；"
                     f"缺少 {', '.join('--' + m for m in missing)}")

    cfg, _ = load_config(args.config)
    params = signal_params(cfg)
    signal = build_signal(pd.read_csv(args.lightgbm), pd.read_csv(args.ridge),
                          pd.read_csv(args.labels), params)

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"{destination} 已存在；保留既有信号，另选输出路径。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    signal.to_csv(destination, index=False)

    merged = signal.merge(pd.read_csv(args.labels), on=KEYS, how="inner", validate="one_to_one")
    metrics = evaluate_frame(merged)
    if args.metrics_out:
        write_json(Path(args.metrics_out), metrics)

    print(f"===== {params['name']} 信号（官方口径）=====")
    print(f"  参数：权重 {params['weights']} / alpha {params['alpha']} / keep_q {params['keep_q']}"
          f" / {params['initial_state']}")
    for key in OFFICIAL_METRICS:
        print(f"  {key:12s} {metrics[key]: .10f}")
    print(f"已写入 {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

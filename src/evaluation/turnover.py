"""换手优化：预测平滑、留仓带与参数扫描（成员 B）。

官方 ``final_score = 0.4·IC + 0.3·AnnualExcess + 0.3·(1−Turnover)``，其中换手为
Top 集合的逐日 Jaccard 距离（``official_eval`` 口径：剔除涨停、不要求标签、取前
10%）。Day 10 的诊断显示换手几乎全部来自 90% 切线附近的排名抖动——被换出股票的
昨日排名分位与新进股票的今日排名分位都是 0.948。本模块实现两种抑制手段：

1. ``smooth``——**预测平滑（Day 11–12 预设方案）**。每日 pred 先转成截面百分位
   rank，再按股票递推指数平滑：

       Rank_t = 当日全市场 pred 的百分位排名（groupby(trade_date).rank(pct=True)）
       S_t    = alpha · Rank_t + (1 - alpha) · S_{t-1}      （S_1 = Rank_1）

   ``alpha = 1`` 时 S 是 rank 的严格单调变换，8 项官方指标应逐位复现基线（本模块
   的自检）。``alpha`` 越小换手越低，代价是排序对新信息反应变慢。

2. ``band``——**留仓带 / hysteresis（本模块补充方案）**。只对“切线附近”动手：
   昨日在榜且今日排名分位仍不低于 ``keep_q`` 的股票保留，其余按当日排名补足前十
   分之一。切线以下的排序完全不动，因此 IC 与超额的损失远小于同换手水平的平滑。

3. ``smooth_band``——**两段串联（成员 B 复核 S002 时提出，见
   ``docs/model_research_S002_review.md``）**。先按 ``alpha`` 做排名平滑，再在平滑后
   的排名上做留仓带。两段读的是同一个 ``pred``，``alpha = 1`` 时逐位退化为 ``band``
   （截面 pct 排名幂等）。它不改变任何评分口径，仍是官方 8 指标。

三种方法都只改变提交的 ``pred`` 排序，评分一律复用 ``src/evaluation/official_eval.py``
（官方口径逐字一致）。优化目标是 **final_score 最大**，不是换手最小。

CLI 示例::

    python -m src.evaluation.turnover --method smooth \
        --pred outputs/predictions/E001_baseline_repeat/valid_2024.csv \
        --labels outputs/metrics/E001_baseline_repeat/validation_labels.csv \
        --out-dir outputs/metrics/turnover_scan_E001_fold2

    python -m src.evaluation.turnover --method band \
        --pred outputs/predictions/E001_baseline_repeat/valid_2024.csv \
        --labels outputs/metrics/E001_baseline_repeat/validation_labels.csv \
        --out-dir outputs/metrics/turnover_band_E001_fold2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import load_scored
from src.evaluation.official_eval import OFFICIAL_METRICS, TOP_MIN_VALID, daily_metrics, evaluate_frame

DEFAULT_ALPHAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
DEFAULT_KEEP_QS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01)
TOP_FRACTION = 10  # 官方取前 1/10


def _wide_rank(scored: pd.DataFrame, eligible_only: bool) -> pd.DataFrame:
    """把 pred 的截面百分位排名摊成 (trade_date × ts_code) 宽表。

    ``eligible_only=True`` 时只在 ``flag_limit_up == 0`` 的股票内排名——官方换手
    与 Top 超额均以该集合为门槛；``False`` 时对当日全部有预测的股票排名（用于保持
    非入选股票的排序不被扭曲，避免影响 IC）。两种模式都对齐到全体股票列，未参与
    排名的位置为 NaN（调用方以 NaN 判定“当日不可选”）。
    """
    full = scored.pivot(index="trade_date", columns="ts_code", values="pred").sort_index()
    df = scored if not eligible_only else scored[scored["flag_limit_up"] == 0]
    rank = df.groupby("trade_date")["pred"].rank(pct=True)
    wide = df.assign(_r=rank).pivot(
        index="trade_date", columns="ts_code", values="_r").sort_index()
    return wide.reindex(index=full.index, columns=full.columns)


def _to_long(scored: pd.DataFrame, wide: pd.DataFrame, name: str) -> pd.Series:
    """把宽表按 ``scored`` 的行序展开回同长度的 Series。"""
    long = wide.rename_axis("trade_date").reset_index().melt(
        id_vars="trade_date", var_name="ts_code", value_name=name)
    merged = scored.merge(
        long, on=["trade_date", "ts_code"], how="left", validate="many_to_one")
    return merged[name].astype(float)


def smooth_scores(scored: pd.DataFrame, alpha: float) -> pd.Series:
    """预测平滑：截面 rank 的指数移动平均，返回与 scored 等长等序的 S。

    ``scored`` 需含 ``ts_code, trade_date, pred``。递推按交易日升序逐日进行；
    某只股票当日 rank 缺失时 S 重置为缺失（不向前携带），验证期为完全平衡面板，
    实际不触发。
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha 必须在 (0, 1] 内，收到 {alpha}")
    wide = _wide_rank(scored, eligible_only=False)
    vals = wide.to_numpy(dtype=float)
    out = np.empty_like(vals)
    out[0] = vals[0]
    for t in range(1, len(vals)):
        cur = vals[t]
        # 当日 rank 缺失则 S 缺失；否则正常递推
        out[t] = np.where(np.isnan(cur), np.nan, alpha * cur + (1 - alpha) * out[t - 1])
    return _to_long(scored, pd.DataFrame(out, index=wide.index, columns=wide.columns), "smoothed")


def band_scores(scored: pd.DataFrame, keep_q: float = 0.80,
                hold_pred: np.ndarray | None = None) -> pd.Series:
    """留仓带：只抑制切线附近的抖动，返回编码后的 pred。

    ``hold_pred`` 只改变**留仓判定与裁剪的参照分数**，``None``（默认）时与历史版本
    逐位一致：

    - ``None``：判定/裁剪用传入 ``pred`` 在候选集合内的分位，``delta = 1 - keep_q``；
    - 提供时：判定/裁剪改用 ``hold_pred`` 的候选内分位（例如 S003 的原始 ``H01``
      预测，而非上层重排映射后的分数），**补足新成员仍按传入 ``pred`` 排序**，从而
      保留重排对新进名单的决定权。此时 ``delta`` 换成 ``1 - min(留仓股全市场分位)``：
      判定源与编码源解耦后 ``1 - keep_q`` 不再能保证留仓集合高于落选股。

    注意 ``delta`` 的具体取值不影响官方三项指标——官方用 Spearman IC 且 Top 组只由
    排序决定，故只要留仓集合严格高于落选候选即可。这带来一个自检点：当
    ``hold_pred`` 与 ``pred`` 相同时（例如都是 ``H01``），两种路径必须逐位一致。

    口径与 ``official_eval`` 的换手组一致：候选集合为当日剔除涨停的股票，规模为
    其前 1/10（不足 ``TOP_MIN_VALID`` 只时不建仓）。首日无历史持仓，直接取当日前
    1/10，与基线一致。

    编码方式保证 **官方评分器选出的 Top 1/10 恰好等于留仓集合**，同时把排序扰动
    限制到最小：留仓股票保持自己的全市场排名分位，未留仓**候选集合内**的股票统一
    下移 ``delta``，涨停股不参与候选、分数不动。``delta`` 只需大到让最弱的留仓股
    仍然领先于最强的落选股，取 ``1 - keep_q``（再留 1e-9 的余量），因此

    - 留仓集合必然占据前 1/10（落选股最高只有 ``keep_q`` 分位）；
    - 段内顺序全部随原始排名单调，``delta`` 越小、对整体排序的扰动越小；
    - 唯一被改变的相对顺序是“排名已经掉下去、但仍在留仓带上的股票被抬回 Top”，
      即策略本身的设计意图；
    - ``keep_q = 1.0`` 时 ``delta ≈ 1e-9``，远小于满分位间距（1/4650），不产生任何
      名次交换，指标逐位复现基线（本模块自检点）。
    """
    if not 0.0 <= keep_q <= 1.0:
        raise ValueError(f"keep_q 必须在 [0, 1] 内，收到 {keep_q}")
    rank_elig = _wide_rank(scored, eligible_only=True)
    rank_full = _wide_rank(scored, eligible_only=False).reindex(
        index=rank_elig.index, columns=rank_elig.columns)
    dates = rank_elig.index
    rf = rank_full.to_numpy(dtype=float)
    out = rf.copy()
    vals = rank_elig.to_numpy(dtype=float)
    # 判定分数：默认与编码源同为传入 pred；提供 hold_pred 时改用它的候选内分位
    if hold_pred is None:
        hold_vals = vals
        fixed_delta: float | None = 1.0 - keep_q + 1e-9
    else:
        hold_rank = _wide_rank(
            scored.assign(pred=np.asarray(hold_pred, dtype="float64")),
            eligible_only=True).reindex(index=dates, columns=rank_elig.columns)
        hold_vals = hold_rank.to_numpy(dtype=float)
        fixed_delta = None
    prev_top: list[int] = []
    for t in range(len(dates)):
        row = vals[t]
        hrow = hold_vals[t]
        ok = ~np.isnan(row)
        n_elig = int(ok.sum())
        if n_elig < TOP_MIN_VALID:
            prev_top = []
            continue
        n_top = max(n_elig // TOP_FRACTION, 1)
        # 留仓：昨日在榜 + 今日仍可选 + 判定分数分位不低于 keep_q（NaN 视为不合格）
        keep = [i for i in prev_top if ok[i] and hrow[i] >= keep_q]
        if len(keep) > n_top:
            keep = [keep[i] for i in np.argsort(-hrow[keep])[:n_top]]
        if len(keep) < n_top:
            # 补足新成员：仍按传入 pred（上层重排结果）排序
            chosen = set(keep)
            for idx in np.argsort(-np.where(ok, row, -np.inf)):
                if len(keep) >= n_top:
                    break
                if idx not in chosen:
                    keep.append(int(idx))
                    chosen.add(int(idx))
        drop = np.array([i for i in np.flatnonzero(ok) if i not in set(keep)], dtype=int)
        if fixed_delta is None:
            # 判定源与编码源解耦：delta 只要让最弱留仓股仍高于最强落选股即可
            delta = 1.0 - float(np.nanmin(rf[t, keep])) + 1e-9
        else:
            delta = fixed_delta
        out[t, drop] = rf[t, drop] - delta
        prev_top = keep
    return _to_long(scored, pd.DataFrame(out, index=dates, columns=rank_elig.columns), "band")


def smooth_band_scores(scored: pd.DataFrame, alpha: float, keep_q: float) -> pd.Series:
    """两段串联：先排名平滑，再留仓带，返回编码后的 pred（与 ``band_scores`` 同语义）。

    串联顺序有实质含义——平滑先把截面排名压成对新信息反应更慢的时序信号，留仓带再
    只对“切线附近”动手，于是同换手下 IC 保留率高于任一单段（S002 两折实证，见
    ``docs/model_research_S002_review.md``）。

    与单段实现的关系：

    - ``alpha = 1`` 时 ``smooth_scores`` 返回 pred 的截面 pct 排名，而 pct 排名幂等
      （对 pct 排名再排名得到同一数值），``band_scores`` 内部重算的宽表排名因此与
      传入值逐位相同，整体退化为 ``band_scores(scored, keep_q)``（调用方自检点）；
    - 两段都只读 ``pred`` 与 ``flag_limit_up``，不接触 ``y_ret_1d``，无未来信息；
    - 首日冷启动：平滑的 ``S_1 = Rank_1``、留仓带首日无历史持仓，与单段实现一致。
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha 必须在 (0, 1] 内，收到 {alpha}")
    if not 0.0 <= keep_q <= 1.0:
        raise ValueError(f"keep_q 必须在 [0, 1] 内，收到 {keep_q}")
    smoothed = smooth_scores(scored, alpha).to_numpy(dtype=float)
    return band_scores(scored.assign(pred=smoothed), keep_q)


def _score_with(scored: pd.DataFrame, transformed: pd.Series) -> tuple[dict, pd.DataFrame]:
    df = scored.assign(pred=transformed)
    return evaluate_frame(df), daily_metrics(df)


def evaluate_smoothed(scored: pd.DataFrame, alpha: float) -> tuple[dict, pd.DataFrame]:
    """给定 alpha 返回官方 8 项指标与逐日指标表。"""
    return _score_with(scored, smooth_scores(scored, alpha))


def evaluate_band(scored: pd.DataFrame, keep_q: float) -> tuple[dict, pd.DataFrame]:
    """给定 keep_q 返回官方 8 项指标与逐日指标表。"""
    return _score_with(scored, band_scores(scored, keep_q))


def evaluate_smooth_band(scored: pd.DataFrame, alpha: float,
                         keep_q: float) -> tuple[dict, pd.DataFrame]:
    """给定 (alpha, keep_q) 返回两段串联后的官方 8 项指标与逐日指标表。"""
    return _score_with(scored, smooth_band_scores(scored, alpha, keep_q))


def _scan(scored: pd.DataFrame, values, transform, name: str,
          selfcheck_value: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """通用扫描：``transform(scored, value)`` 返回新的 pred，逐点按官方口径评分。

    自检点 ``selfcheck_value`` 的 8 项指标必须逐位复现基线（未变换）评分。
    """
    base = evaluate_frame(scored)
    rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    for value in values:
        metrics, daily = _score_with(scored, transform(scored, value))
        if abs(value - selfcheck_value) < 1e-12:
            for key in OFFICIAL_METRICS:
                diff = abs(metrics[key] - base[key])
                assert diff < 1e-12, (
                    f"{name}={value} 自检失败：{key} 差异 {diff:.3e}（实现有误）")
        rows.append({name: value, **{k: metrics[k] for k in OFFICIAL_METRICS}})
        daily_frames.append(daily.assign(**{name: value}))
    return pd.DataFrame(rows), pd.concat(daily_frames, ignore_index=True)


def scan_alphas(scored: pd.DataFrame, alphas=DEFAULT_ALPHAS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """扫描 alpha，返回 (汇总表, 逐日指标 long 表)。

    自检：alpha=1.0 的全部官方指标必须与未平滑评分逐位一致（单调变换不变性），
    否则抛出 AssertionError。
    """
    return _scan(scored, alphas, smooth_scores, "alpha", 1.0)


def scan_bands(scored: pd.DataFrame, keep_qs=DEFAULT_KEEP_QS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """扫描 keep_q，返回 (汇总表, 逐日指标 long 表)。

    自检：``keep_q = 1.0`` 时下移量 ``delta ≈ 1e-9``，远小于满分位间距
    （1/4650），编码退化为恒等变换，不发生任何名次交换，Top 集合与基线逐日
    相同，8 项官方指标应逐位复现基线（机理同 ``band_scores`` 的 docstring）。
    """
    return _scan(scored, keep_qs, band_scores, "keep_q", 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="换手优化参数扫描（官方口径评分）")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--method", choices=("smooth", "band"), default="smooth",
                        help="smooth=预测平滑（alpha），band=留仓带（keep_q）")
    parser.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS),
                        help="method=smooth 时的 alpha 列表，逗号分隔")
    parser.add_argument("--keep-qs", default=",".join(str(q) for q in DEFAULT_KEEP_QS),
                        help="method=band 时的 keep_q 列表，逗号分隔")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    scored = load_scored(args.pred, args.labels)
    if args.method == "smooth":
        values = tuple(float(a) for a in str(args.alphas).split(","))
        summary, daily_long = scan_alphas(scored, values)
        key = "alpha"
    else:
        values = tuple(float(q) for q in str(args.keep_qs).split(","))
        summary, daily_long = scan_bands(scored, values)
        key = "keep_q"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "scan_summary.csv", index=False)
    daily_long.to_csv(out_dir / "scan_daily.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"===== {args.method} 扫描（官方口径）=====")
    print(summary.to_string(index=False))
    best = summary.loc[summary["final_score"].idxmax()]
    print(f"\nfinal_score 最大的 {key} = {best[key]}："
          f"IC {best['ic_mean']:.6f} / 超额 {best['annual_excess']:.6f} / "
          f"换手 {best['mean_turnover']:.6f} / final {best['final_score']:.10f}")
    print(f"\n已写入 {out_dir / 'scan_summary.csv'} 与 {out_dir / 'scan_daily.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

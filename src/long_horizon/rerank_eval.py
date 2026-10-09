"""顶部候选池二次重排：一个模型划池、另一个在池内精选（事后重算，不重训）。

背景（2026-10-08 测试期复盘，`docs/long_horizon_LH003.md` 之后的追加分析）：
``H20_F2T2``（长特征 59 列 × 20 日截面百分位目标）在测试期跑不赢 T030 基线，
归因显示**亏的全部在年化超额项**，IC 反而是正的：

    whole  IC 项 +0.0068 / 超额项 −0.0402 / 稳定项 +0.0023

raw 口径更直白：``H20_F2T2`` 全截面 IC 0.0775 > H01 的 0.0755，但 Top 1/10
年化超额 0.1617 不到 H01（0.3709）的一半 ⇒ 20 日 rank 目标把预测力**平摊到
整个截面**，没有集中在最顶部。

本模块检验一个对应假设：**两个模型分工**——H01 擅长找顶部候选（Top 超额高），
``H20_F2T2`` 的粗排信息更全（IC 更高）。那就先用 ``primary`` 划出前 ``q`` 比例
的候选池，**只在池内**用 ``refiner`` 重排，池外顺序一点不动。

- 池 = 当日候选集合（``flag_limit_up == 0``，与官方换手/超额口径一致）内
  ``primary`` 排名前 ``q`` 比例，规模 ``max(round(m*q), 1)``；
- 分数映射保证**池内整体高于池外**（池内 ∈ (1−q, 1]，池外 ∈ [0, 1−q)），
  因此 ``q ≥ 0.1`` 时官方「取前 1/10」必然完全落在池内；
- 池外保持 ``primary`` 的相对顺序（单调压缩，不改变池外彼此的名次）。

只在验证折上做，判据是 **Top 1/10 的实现年化超额**与 band 后的 `final_score`，
**不看 IC**（IC 与超额脱节正是这次的死因）。

折来源（``--fold-source``）= ``validation``（两折 2023/2024）或
``optuna``（四折 wf2021 / wf2022 / wf2023 / confirm2024）。四折用来回答
「q 是否只在 2023–2024 上成立」——2021–2022 是**没有参与过 q 选择**的两折。

自检（必须通过，否则抛错）：
1. ``q = 0`` 退化为 ``primary`` 原样 ⇒ 官方 raw 指标逐位复现 S003 归档；
2. ``H01 + band(T030 冻结 keep_q)`` 必须逐位复现 S003 归档对应折的 band 指标。

CLI::

    python -m src.long_horizon.rerank_eval --study LH003 \\
        --directions H01>H20_F2T2,H20_F2T2>H01 --folds fold1,fold2
    python -m src.long_horizon.rerank_eval --study LH006 --fold-source optuna \\
        --folds wf2021,wf2022,wf2023,confirm2024 --out-tag LH005_4fold
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
from src.evaluation.official_eval import TOP_MIN_VALID, evaluate_frame
from src.evaluation.turnover import _to_long, _wide_rank, band_scores
from src.evaluation.validation import get_folds
from src.long_horizon.run import get_optuna_folds
from src.utils.project import ROOT, load_config, project_path, write_json

PRICE_COLUMNS = ["open", "high", "low", "close"]
DEFAULT_DIRECTIONS = ("H01>H20_F2T2", "H20_F2T2>H01")
DEFAULT_QS = (0.10, 0.15, 0.20, 0.30, 0.50, 1.00)
METRIC_KEYS = ("ic_mean", "annual_excess", "top1_annual_ret",
               "mean_turnover", "final_score")
# S003 归档里与各验证折对应的记录位置（自检对照）。
# 值 = (归档文件, 键)；键在 winner.folds 下取，或（确认折）走 candidate 分支。
S003_SELECTION = "docs/optuna_tuning_S003_selection.json"
S003_CONFIRMATION = "docs/optuna_tuning_S003_confirmation.json"
S003_REFERENCE = {
    "fold1": (S003_SELECTION, "wf2023"),
    "wf2021": (S003_SELECTION, "wf2021"),
    "wf2022": (S003_SELECTION, "wf2022"),
    "wf2023": (S003_SELECTION, "wf2023"),
    "fold2": (S003_CONFIRMATION, "confirm2024"),
    "confirm2024": (S003_CONFIRMATION, "confirm2024"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_valid_panel(csv_path: Path, valid_start: int, valid_end: int,
                     chunksize: int = 1_000_000) -> pd.DataFrame:
    """分块读原始训练集，只保留验证期行（清洗口径同 ``src/data/clean_data.py``）。"""
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


def _fill(pred: np.ndarray) -> np.ndarray:
    """把 NaN 填成最小值 − 1（与 band_eval 同口径，保证排序稳定落到底部）。"""
    good = np.isfinite(pred)
    if good.all():
        return pred
    return np.where(good, pred, np.nanmin(pred[good]) - 1.0)


def pool_rerank(pred_primary: np.ndarray, pred_refiner: np.ndarray,
                flag_limit_up: np.ndarray, dates: np.ndarray,
                q: float) -> np.ndarray:
    """池外保持 ``primary`` 顺序、池内按 ``refiner`` 排序，返回新 pred（等长）。

    池 = 当日候选集合（``flag_limit_up == 0``）内 primary 排名前 ``q`` 比例。
    ``q <= 0`` 直接返回 primary 原样（自检锚点）；``q >= 1`` 时池覆盖全部候选，
    池外只剩不能上榜的涨停股。
    """
    if q <= 0:
        return pred_primary.copy()
    out = np.full(len(pred_primary), np.nan, dtype="float64")
    order = np.argsort(dates, kind="stable")
    uniq, starts = np.unique(dates[order], return_index=True)
    starts = np.append(starts, len(order))
    kept_total = 0
    for k in range(len(uniq)):
        idx = order[starts[k]:starts[k + 1]]
        a = pred_primary[idx]
        b = pred_refiner[idx]
        elig = flag_limit_up[idx] == 0
        m = int(elig.sum())
        if m < TOP_MIN_VALID:
            out[idx] = a                      # 当日样本不足，不建仓，原样返回
            continue
        e_pos = np.flatnonzero(elig)
        n_pool = int(round(m * min(q, 1.0)))
        n_pool = max(min(n_pool, m), 1)
        # 池：primary 在候选集合内降序的前 n_pool 只
        pool_pos = e_pos[np.argsort(-a[e_pos], kind="stable")[:n_pool]]
        in_pool = np.zeros(len(idx), dtype=bool)
        in_pool[pool_pos] = True
        score = np.empty(len(idx), dtype="float64")
        # 池内：按 refiner 降序，映射到 (1−q, 1]
        pb = b[pool_pos]
        pool_sorted = pool_pos[np.argsort(-pb, kind="stable")]
        score[pool_sorted] = (1.0 - q) + q * (1.0 - np.arange(n_pool) / n_pool)
        # 池外（含涨停股）：保持 primary 顺序，映射到 [0, 1−q)
        outer = np.flatnonzero(~in_pool)
        ao = a[outer]
        outer_sorted = outer[np.argsort(-ao, kind="stable")]
        n_out = len(outer)
        score[outer_sorted] = (1.0 - q) * (1.0 - (np.arange(n_out) + 1) / n_out)
        out[idx] = score
        kept_total += n_pool
    if kept_total == 0:
        raise ValueError("所有交易日候选样本均不足 TOP_MIN_VALID，无法评分")
    return out


def score_variants(frame: pd.DataFrame, pred: np.ndarray, keep_q: float,
                   hold_pred: np.ndarray | None = None) -> dict:
    """返回未加 band（raw）与加 band 后的官方核心指标。

    ``hold_pred`` 透传给 ``band_scores``：提供时留仓判定/裁剪改用它的候选内分位
    （例如未加工的 S003 ``H01`` 原始预测），而不是 ``pred``（重排映射后的分数）。
    """
    scored = frame[KEYS + ["y_ret_1d", "flag_limit_up"]].assign(pred=_fill(pred))
    out: dict[str, float] = {}
    raw = evaluate_frame(scored)
    for key in METRIC_KEYS:
        out[f"raw_{key}"] = float(raw[key])
    banded = scored.assign(pred=_fill(
        band_scores(scored, keep_q, hold_pred=hold_pred).to_numpy(dtype="float64")))
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


def parse_directions(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ">" not in item:
            raise ValueError(f"方向格式应为 primary>refiner，收到 {item!r}")
        primary, refiner = (s.strip() for s in item.split(">", 1))
        out.append((primary, refiner))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="顶部候选池二次重排（不重训，事后重算官方口径评分）")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study", default="LH003", help="预测产物子目录名")
    parser.add_argument("--directions", default=",".join(DEFAULT_DIRECTIONS),
                        help="primary>refiner 列表，逗号分隔")
    parser.add_argument("--qs", default=",".join(str(q) for q in DEFAULT_QS),
                        help="候选池比例列表（池 = primary 前 q 比例）")
    parser.add_argument("--folds", default="fold1,fold2")
    parser.add_argument("--fold-source", default="validation",
                        choices=["validation", "optuna"],
                        help="validation=两折(fold1/fold2)；optuna=四折(wf2021–wf2023+confirm2024)")
    parser.add_argument("--out-tag", default="LH005",
                        help="产物前缀：experiments/<out-tag>_rerank_eval.csv、"
                             "outputs/long_horizon/<out-tag>/")
    parser.add_argument("--keep-q", type=float, default=None,
                        help="留仓带阈值；默认直读 T030 冻结 spec")
    parser.add_argument("--hold-source", default="mapped",
                        choices=["mapped", "h01", "primary", "combo"],
                        help="留仓判定的参照分数：mapped=重排映射后的 pred（历史行为）；"
                             "h01=未加工的 S003/H01 原始预测；primary=该方向 primary "
                             "模型的原始预测；combo=primary 与 refiner 候选内分位的算术"
                             "平均（每天独立重算的组合排名）。补足新成员一律按 pred 排序")
    args = parser.parse_args(argv)

    cfg, _ = load_config(args.config)
    settings = cfg["long_horizon"]
    selection = json.loads((ROOT / settings["spec_source"]).read_text(encoding="utf-8"))
    keep_q = (float(selection["winner"]["spec"]["keep_q"]) if args.keep_q is None
              else float(args.keep_q))
    log(f"keep_q = {keep_q!r}（来源：{settings['spec_source']} · winner.spec.keep_q）")

    directions = parse_directions(args.directions)
    qs = [float(q) for q in str(args.qs).split(",") if q.strip()]
    wanted = [f.strip() for f in args.folds.split(",") if f.strip()]
    fold_pool = get_optuna_folds(cfg) if args.fold_source == "optuna" else get_folds()
    folds = [f for f in fold_pool if f.name in wanted]
    if not folds:
        raise ValueError(f"未找到折 {wanted}（fold_source={args.fold_source}）")
    models = sorted({m for pair in directions for m in pair})
    hold_source = args.hold_source
    fetch = sorted(set(models) | ({"H01"} if hold_source == "h01" else set()))
    log(f"hold_source = {hold_source}（留仓判定参照；其余折参数不变）")
    if hold_source == "primary":
        log(f"  提示：池 = primary 前 q 比例，故持仓股的 primary 候选内分位恒 ≥ 1−q"
            f"（当前 max q={max(qs):.2f} ⇒ ≥{1 - max(qs):.4f}）；"
            f"keep_q ≤ {1 - max(qs):.4f} 时判定恒真、结果应与 mapped 逐位一致，"
            "要真正触发必须 keep_q > 1−q")

    csv_path = project_path(cfg["paths"]["train"])
    out_root = project_path("outputs/long_horizon") / args.out_tag
    rows: list[dict] = []
    refs: dict[str, dict] = {}

    for fold in folds:
        log(f"{fold.name}: 读取验证期 {fold.valid_start}–{fold.valid_end} 的原始行 ...")
        panel = read_valid_panel(csv_path, fold.valid_start, fold.valid_end)
        log(f"  {len(panel):,} 行 / {panel['trade_date'].nunique()} 个交易日 / "
            f"{panel['ts_code'].nunique()} 只")

        pred_path = (project_path("outputs/long_horizon") / args.study
                     / fold.name / "raw_predictions.parquet")
        if not pred_path.exists():
            raise FileNotFoundError(f"缺少预测文件 {pred_path}")
        pred_frame = pd.read_parquet(pred_path)
        pred_frame["ts_code"] = pred_frame["ts_code"].astype(str)
        pred_frame["trade_date"] = pred_frame["trade_date"].astype("int64")
        missing = [m for m in fetch if m not in pred_frame.columns]
        if missing:
            raise KeyError(f"{pred_path} 缺少模型列 {missing}；"
                           f"现有 {list(pred_frame.columns)}")
        merged = panel.merge(pred_frame[KEYS + fetch], on=KEYS, how="left",
                             validate="one_to_one")
        del panel, pred_frame
        gc.collect()

        dates = merged["trade_date"].to_numpy()
        flag = merged["flag_limit_up"].to_numpy()
        # 留仓判定参照：h01 时用未加工的 S003 原始预测（含 NaN，NaN 视为不可留）；
        # primary 时逐方向取该方向 primary 的原始预测，故在方向循环内赋值。
        hold_arr = (merged["H01"].to_numpy(dtype="float64")
                    if hold_source == "h01" else None)
        refs[fold.name] = reference_band(fold.name) or {}

        # --- 自检 1：q = 0 退化为 primary 原样，raw 指标须与归档逐位一致 ---
        for primary, _ in directions:
            base = _fill(merged[primary].to_numpy(dtype="float64"))
            identity = pool_rerank(base, base, flag, dates, 0.0)
            if not np.array_equal(identity, base):
                raise AssertionError(f"自检失败：q=0 未退化为 {primary} 原样")
        log(f"  自检 1 通过：q=0 逐位退化为 primary 原样（{models}）")

        # --- 自检 2：H01 + band 必须逐位复现 S003 归档 ---
        if "H01" in models:
            h01 = merged["H01"].to_numpy(dtype="float64")
            check = score_variants(merged, h01, keep_q)
            ref = refs[fold.name]
            if ref:
                diff = abs(check["final_score"] - ref["final_score"])
                flag_txt = "逐位复现" if diff < 1e-10 else f"差异 {diff:.3e}（需排查）"
                log(f"  自检 2：H01+band {check['final_score']:.10f} vs S003 归档 "
                    f"{ref['final_score']:.10f} → {flag_txt}")
                if diff >= 1e-10:
                    raise AssertionError(f"自检失败：H01+band 与 S003 归档差 {diff:.3e}")
            rows.append({"study": args.study, "fold": fold.name,
                         "primary": "H01", "refiner": "-", "q": 0.0,
                         "hold_source": "mapped",
                         "note": "baseline_H01（= T030 监督路径）", **check})

        # --- 自检 3：hold_pred 与 pred 同源时，解耦路径须与默认路径近似一致 ---
        # 二者 keep 集合完全相同，唯一的差别是 delta：默认用 1-keep_q（近似，留仓股
        # 全市场分位一旦低于 keep_q 就会有落选股挤进 Top），解耦用 1-min(留仓股分位)
        # （精确）。故存在 ~1e-6 量级的固有差异，方向恒为解耦路径略高。
        if hold_source == "h01":
            plain = score_variants(merged, hold_arr, keep_q)
            decoupled = score_variants(merged, hold_arr, keep_q, hold_pred=hold_arr)
            diff = abs(plain["final_score"] - decoupled["final_score"])
            log(f"  自检 3：H01 作判定源 {decoupled['final_score']:.10f} "
                f"vs 默认路径 {plain['final_score']:.10f} → 差 {diff:.3e}"
                f"（{'一致' if diff < 1e-10 else 'delta 近似差异，符合预期'}）")
            if diff >= 1e-4:
                raise AssertionError(f"自检失败：解耦路径与默认路径差 {diff:.3e}（>1e-4）")

        # --- 主循环：方向 × q ---
        for primary, refiner in directions:
            pa = _fill(merged[primary].to_numpy(dtype="float64"))
            pb = _fill(merged[refiner].to_numpy(dtype="float64"))
            # primary 判定源：判定用该方向 primary 的原始预测（与名单同源，但未经
            # 分段映射，分位尺度不失真）；combo：两模型候选内分位的算术平均，即
            # 「每天都从零重算的组合预测排名」；mapped 时 dir_hold=None，h01 沿用折级值
            dir_hold = hold_arr
            if hold_source == "primary":
                dir_hold = merged[primary].to_numpy(dtype="float64")
            elif hold_source == "combo":
                basef = merged[KEYS + ["y_ret_1d", "flag_limit_up"]]
                r1 = _wide_rank(basef.assign(pred=pa), eligible_only=True)
                r2 = _wide_rank(basef.assign(pred=pb), eligible_only=True)
                combo_long = _to_long(basef, (r1 + r2) / 2.0, "combo")
                dir_hold = combo_long.to_numpy(dtype="float64")
            for q in qs:
                pred = pool_rerank(pa, pb, flag, dates, q)
                metrics = score_variants(merged, pred, keep_q, hold_pred=dir_hold)
                rows.append({"study": args.study, "fold": fold.name,
                             "primary": primary, "refiner": refiner, "q": q,
                             "hold_source": hold_source,
                             "note": "", **metrics})
                log(f"  {fold.name} {primary}>{refiner} q={q:.2f}："
                    f"band final {metrics['final_score']:.10f} "
                    f"(超额 {metrics['annual_excess']:+.6f} / "
                    f"top1 年化 {metrics['top1_annual_ret']:+.6f} / "
                    f"换手 {metrics['mean_turnover']:.6f}) | "
                    f"raw final {metrics['raw_final_score']:.6f}")
        del merged
        gc.collect()

    frame = pd.DataFrame(rows)
    out_csv = ROOT / "experiments" / f"{args.out_tag}_rerank_eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "rerank_report.json", json.loads(json.dumps({
        "study": args.study, "keep_q": keep_q, "directions": directions,
        "hold_source": hold_source, "qs": qs, "spec_source": settings["spec_source"],
        "fold_notes": {f: "baseline_H01" for f in refs},
        "s003_reference": refs, "rows": rows,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, default=float)))

    pd.set_option("display.width", 240)
    show = ["fold", "primary", "refiner", "q", "ic_mean", "annual_excess",
            "top1_annual_ret", "mean_turnover", "final_score"]
    print(f"\n===== {args.out_tag} 候选池二次重排"
          f"（band keep_q={keep_q:.10f} · hold_source={hold_source}）=====")
    print(frame[show].round(6).to_string(index=False))

    n_fold = len(folds)
    mean_label, worst_label = f"{n_fold}折均值", f"{n_fold}折最差"
    main_rows = frame[frame["note"] == ""].copy()
    if len(main_rows):
        pivot = main_rows.pivot_table(index=["primary", "refiner", "q"],
                                      columns="fold", values="final_score")
        pivot = pivot.reindex(columns=[f.name for f in folds])
        pivot[mean_label] = pivot.mean(axis=1)
        pivot[worst_label] = pivot.min(axis=1)
        print(f"\n----- band final_score（行 = 方向 × 池比例，列 = 折）-----")
        print(pivot.round(6).to_string())
        best = pivot[mean_label].idxmax()
        print(f"\n{mean_label}最高：{best} → {pivot.loc[best, mean_label]:.6f}")
    print(f"\n汇总已写入 {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

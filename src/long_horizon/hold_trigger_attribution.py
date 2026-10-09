"""留仓判定的触发归因：不同判定源 × keep_q 下，band 究竟为什么踢人。

背景
----
``pool_rerank`` 把池内分数抬到 ``(1−q, 1]``，池内 = primary 候选内前 q 比例。
因此**任何**持仓股（Top 1/10 ⊂ 池内）在两个判定源下都有一个分位下界：

- ``mapped``  ：编码分位，池内下界 = 1−q
- ``primary`` ：primary 原始分位，池内下界同样是 1−q（池就是这么划的）

⇒ ``keep_q ≤ 1−q`` 时判定恒真，band 只剩「涨停/停牌」一个换手来源。
要让它真正触发，``keep_q`` 必须 > 1−q。本探针量化各档位的触发量。

口径与 ``band_scores`` 的逐日循环完全一致（候选集合剔涨停、规模取前 1/10），
额外记录被剔除股票的**原因**：``liq`` = 今日不可交易（涨停/停牌/无价），
``q`` = 判定分位跌破 keep_q。换手按留仓集合（= 最终 Top 1/10）的 Jaccard 距离。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.official_eval import TOP_MIN_VALID
from src.evaluation.turnover import TOP_FRACTION, _wide_rank
from src.data.load_data import KEYS
from src.long_horizon.rerank_eval import _fill, pool_rerank, read_valid_panel
from src.utils.project import load_config, project_path

ROOT = Path(__file__).resolve().parents[2]
KQ_GRID = (0.0022778298112255263, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.65,
           0.80, 0.875, 0.95)


def simulate(rank_elig: np.ndarray, hold_rank: np.ndarray, keep_q: float) -> dict:
    """照抄 band_scores 的逐日循环，加归因。返回汇总与逐日记录。"""
    n_days = rank_elig.shape[0]
    prev_top: list[int] = []
    rows: list[dict] = []
    for t in range(n_days):
        row = rank_elig[t]
        hrow = hold_rank[t]
        ok = ~np.isnan(row)
        n_elig = int(ok.sum())
        if n_elig < TOP_MIN_VALID:
            prev_top = []
            rows.append({"t": t, "n_elig": n_elig, "drop_liq": 0, "drop_q": 0,
                         "kept": 0, "n_top": 0, "turnover": np.nan})
            continue
        n_top = max(n_elig // TOP_FRACTION, 1)
        drop_liq = [i for i in prev_top if not ok[i]]
        drop_q = [i for i in prev_top if ok[i] and not (hrow[i] >= keep_q)]
        keep = [i for i in prev_top if ok[i] and hrow[i] >= keep_q]
        if len(keep) > n_top:
            keep = [keep[i] for i in np.argsort(-hrow[keep])[:n_top]]
        if len(keep) < n_top:
            chosen = set(keep)
            for idx in np.argsort(-np.where(ok, row, -np.inf)):
                if len(keep) >= n_top:
                    break
                if idx not in chosen:
                    keep.append(int(idx))
                    chosen.add(int(idx))
        cur = set(keep)
        prev = set(prev_top)
        to = (np.nan if not prev else 1.0 - len(cur & prev) / len(cur | prev))
        rows.append({"t": t, "n_elig": n_elig, "drop_liq": len(drop_liq),
                     "drop_q": len(drop_q), "kept": len(cur), "n_top": n_top,
                     "turnover": to})
        prev_top = keep
    frame = pd.DataFrame(rows)
    frame["keep_q"] = keep_q
    live = frame[frame["n_elig"] >= TOP_MIN_VALID]
    return {
        "keep_q": keep_q,
        "days": int(len(live)),
        "drop_liq_total": int(live["drop_liq"].sum()),
        "drop_q_total": int(live["drop_q"].sum()),
        "drop_q_per_day": float(live["drop_q"].mean()),
        "drop_liq_per_day": float(live["drop_liq"].mean()),
        "kept_per_day": float(live["kept"].mean()),
        "turnover_mean": float(live["turnover"].mean()),
        "days_with_q_drop": int((live["drop_q"] > 0).sum()),
    }, frame


def hold_distribution(elig: np.ndarray, sources: dict[str, np.ndarray],
                      keep_q: float = 0.0022778298112255263) -> dict:
    """固定一个几乎不触发的 keep_q，逐日记录**持仓股**在各判定源下的分位分布。

    这是回答「判定源到底能不能把持仓股的分位拉开」的直接证据：如果某个判定源
    下持仓股的分位仍然全部堆在高位，那它的 keep_q 门槛就必然只能设在高位。
    """
    n_days = elig.shape[0]
    prev_top: list[int] = []
    acc: dict[str, list[np.ndarray]] = {k: [] for k in sources}
    for t in range(n_days):
        row = elig[t]
        ok = ~np.isnan(row)
        if int(ok.sum()) < TOP_MIN_VALID:
            prev_top = []
            continue
        n_top = max(int(ok.sum()) // TOP_FRACTION, 1)
        keep = [i for i in prev_top if ok[i] and row[i] >= keep_q]
        if len(keep) < n_top:
            chosen = set(keep)
            for idx in np.argsort(-np.where(ok, row, -np.inf)):
                if len(keep) >= n_top:
                    break
                if idx not in chosen:
                    keep.append(int(idx))
                    chosen.add(int(idx))
        if prev_top:
            for name, arr in sources.items():
                v = arr[t][prev_top]
                v = v[~np.isnan(v)]
                if len(v):
                    acc[name].append(v)
        prev_top = keep
    return {k: np.concatenate(v) for k, v in acc.items() if v}


def describe(name: str, values: np.ndarray) -> dict:
    qs = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 100])
    return {"source": name, "n": int(values.size),
            "min": qs[0], "p01": qs[1], "p05": qs[2], "p25": qs[3],
            "median": qs[4], "p75": qs[5], "p95": qs[6], "max": qs[7],
            "share_lt_0.80": float((values < 0.80).mean()),
            "share_lt_0.50": float((values < 0.50).mean()),
            "share_lt_0.20": float((values < 0.20).mean()),
            "share_lt_0.05": float((values < 0.05).mean())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="留仓判定触发归因探针")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--study", default="LH003")
    parser.add_argument("--fold", default="wf2021")
    parser.add_argument("--primary", default="H05_F2T2")
    parser.add_argument("--refiner", default="H20_F2T2")
    parser.add_argument("--q", type=float, default=0.20)
    parser.add_argument("--valid-start", type=int, default=None)
    parser.add_argument("--valid-end", type=int, default=None)
    parser.add_argument("--sources", default="mapped,primary,refiner,combo")
    parser.add_argument("--out-tag", default="hold_trigger_attribution")
    args = parser.parse_args(argv)

    cfg, _ = load_config(args.config)
    fold_valid = {"wf2021": (20210101, 20211231), "wf2022": (20220101, 20221231),
                  "wf2023": (20230101, 20231231), "confirm2024": (20240101, 20241231),
                  "fold1": (20230101, 20231231), "fold2": (20240101, 20241231)}
    v0, v1 = (args.valid_start, args.valid_end)
    if v0 is None or v1 is None:
        v0, v1 = fold_valid.get(args.fold, (20210101, 20211231))

    panel = read_valid_panel(project_path(cfg["paths"]["train"]), v0, v1)
    pred = pd.read_parquet(project_path("outputs/long_horizon") / args.study
                           / args.fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    cols = [args.primary, args.refiner]
    merged = panel.merge(pred[KEYS + cols], on=KEYS, how="left",
                         validate="one_to_one")
    del panel, pred

    dates = merged["trade_date"].to_numpy()
    flag = merged["flag_limit_up"].to_numpy()
    pa = _fill(merged[args.primary].to_numpy(dtype="float64"))
    pb = _fill(merged[args.refiner].to_numpy(dtype="float64"))
    reranked = _fill(pool_rerank(pa, pb, flag, dates, args.q))
    base = merged[KEYS + ["y_ret_1d", "flag_limit_up"]]

    rank_elig = _wide_rank(base.assign(pred=reranked), eligible_only=True)
    hold_rank = _wide_rank(base.assign(pred=pa), eligible_only=True).reindex(
        index=rank_elig.index, columns=rank_elig.columns)
    ref_rank = _wide_rank(base.assign(pred=pb), eligible_only=True).reindex(
        index=rank_elig.index, columns=rank_elig.columns)
    # 组合预测：primary 与 refiner 的候选内分位算术平均（真正交织，无分段硬切）
    combo_rank = (hold_rank + ref_rank) / 2.0
    elig = rank_elig.to_numpy(dtype="float64")
    hold = hold_rank.to_numpy(dtype="float64")
    refn = ref_rank.to_numpy(dtype="float64")
    comb = combo_rank.to_numpy(dtype="float64")
    source_arrays = {"mapped": elig, "primary": hold, "refiner": refn,
                     "combo": comb}

    print(f"折 {args.fold}（{v0}–{v1}）| {args.primary}>{args.refiner} q={args.q:.2f}")
    print(f"交易日 {elig.shape[0]} | 股票 {elig.shape[1]}")
    print(f"提示：池内下界 = 1−q = {1 - args.q:.4f}，keep_q ≤ 该值时判定恒真\n")

    # --- 核心诊断：持仓股在各判定源下到底落在什么分位 ---
    dist = hold_distribution(elig, source_arrays)
    print("===== 持仓股在各判定源下的候选内分位分布（keep_q≈0 冻结，几乎不触发）=====")
    desc = pd.DataFrame([describe(k, v) for k, v in dist.items()])
    print(desc.round(6).to_string(index=False))
    print()

    summaries = []
    for src in [s.strip() for s in args.sources.split(",") if s.strip()]:
        hr = source_arrays[src]
        print(f"===== 判定源 = {src} =====")
        rows = []
        for kq in KQ_GRID:
            summ, _ = simulate(elig, hr, kq)
            summ["source"] = src
            rows.append(summ)
        t = pd.DataFrame(rows)[["source", "keep_q", "days", "drop_liq_total",
                                "drop_q_total", "drop_liq_per_day",
                                "drop_q_per_day", "days_with_q_drop",
                                "kept_per_day", "turnover_mean"]]
        t["drop_q_share"] = (t["drop_q_total"]
                             / (t["drop_q_total"] + t["drop_liq_total"]).replace(0, np.nan))
        print(t.round(6).to_string(index=False))
        print()
        summaries.append(t)

    out = pd.concat(summaries, ignore_index=True)
    out_csv = ROOT / "experiments" / f"{args.out_tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"已写出 {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

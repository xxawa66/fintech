"""探针：两个问题一起答。

Q1（放宽换手）——band 的 keep_q 放宽后，持仓更快跟随模型更新，
   超额项能否向「滚动持有口径」的优势靠拢？扫 keep_q。
Q2（H20 兜底）——把排序权完全交给 primary（短周期强项），
   refiner（H20）只做「否决」：池内 refiner 分位 < tau 的降到池外。

veto 模式（`pool_veto`）与 `pool_rerank` 的区别：
   pool_rerank：池内改由 refiner 排序     ← 现方案，H20 会推翻 H05 的排序
   pool_veto  ：池内仍由 primary 排序，refiner 只把「长期看坏」的挑出去
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.official_eval import TOP_MIN_VALID, evaluate_frame
from src.evaluation.turnover import band_scores
from src.long_horizon.rerank_eval import _fill, pool_rerank, read_valid_panel
from src.long_horizon.run import get_optuna_folds
from src.utils.project import load_config, project_path

cfg, _ = load_config("configs/project.yaml")
TRAIN = project_path(cfg["paths"]["train"])
FOLD_NAMES = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KEEP0 = 0.0022778298112255263
METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")


def pool_veto(pred_primary, pred_refiner, flag_limit_up, dates, q, tau):
    """池 = primary 前 q 比例；池内保持 primary 排序，refiner 分位 < tau 降到池外。"""
    out = np.full(len(pred_primary), np.nan, dtype="float64")
    order = np.argsort(dates, kind="stable")
    uniq, starts = np.unique(dates[order], return_index=True)
    starts = np.append(starts, len(order))
    lo = 1.0 - q
    for k in range(len(uniq)):
        idx = order[starts[k]:starts[k + 1]]
        a = pred_primary[idx]
        b = pred_refiner[idx]
        elig = flag_limit_up[idx] == 0
        m = int(elig.sum())
        if m < TOP_MIN_VALID:
            out[idx] = a
            continue
        e_pos = np.flatnonzero(elig)
        n_pool = max(min(int(round(m * min(q, 1.0))), m), 1)
        pool_pos = e_pos[np.argsort(-a[e_pos], kind="stable")[:n_pool]]
        in_pool = np.zeros(len(idx), dtype=bool)
        in_pool[pool_pos] = True
        # refiner 在当日候选集合内的分位（1 = 最好）
        ranks = np.argsort(np.argsort(b[e_pos], kind="stable"), kind="stable")
        pct = np.zeros(len(idx), dtype="float64")
        pct[e_pos] = (ranks + 1) / m
        passed = in_pool & (pct >= tau)
        # 通过：主榜 (1-q, 1]，按 primary 降序
        pp = np.flatnonzero(passed)
        ps = pp[np.argsort(-a[pp], kind="stable")]
        n = len(ps)
        if n:
            out[idx[ps]] = lo + (1.0 - lo) * (1.0 - np.arange(n) / n)
        # 未通过：副榜 [0, 1-q)，按 primary 降序
        rp = np.flatnonzero(~passed)
        rs = rp[np.argsort(-a[rp], kind="stable")]
        nr = len(rs)
        out[idx[rs]] = lo * (1.0 - (np.arange(nr) + 1) / nr)
    return out


def band_metrics(frame, pred, keep_q):
    scored = frame[KEYS + ["y_ret_1d", "flag_limit_up"]].assign(pred=_fill(pred))
    banded = scored.assign(pred=_fill(band_scores(scored, keep_q).to_numpy("float64")))
    res = evaluate_frame(banded)
    return {k: float(res[k]) for k in METRICS}


def main() -> None:
    folds_all = {f.name: f for f in get_optuna_folds(cfg)}
    rows = []
    KQS = [KEEP0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50]
    RQ = [0.20, 0.30, 0.50]
    TAUS = [0.2, 0.3, 0.5]
    PAIRS = [("H05_F2T2", "H20_F2T2"), ("H05_F2T2", "H05_F1T2")]

    for fn in FOLD_NAMES:
        fold = folds_all[fn]
        panel = read_valid_panel(TRAIN, fold.valid_start, fold.valid_end)
        pred_frame = pd.read_parquet(
            f"outputs/long_horizon/LH003/{fn}/raw_predictions.parquet")
        pred_frame["ts_code"] = pred_frame["ts_code"].astype(str)
        pred_frame["trade_date"] = pred_frame["trade_date"].astype("int64")
        cols = sorted({m for p in PAIRS for m in p})
        merged = panel.merge(pred_frame[KEYS + cols], on=KEYS, how="left",
                             validate="one_to_one")
        del panel, pred_frame
        dates = merged["trade_date"].to_numpy()
        flag = merged["flag_limit_up"].to_numpy()
        print(f"[{fn}] {len(merged):,} 行 / {merged.trade_date.nunique()} 日", flush=True)

        for pa_name, pb_name in PAIRS:
            pa = _fill(merged[pa_name].to_numpy("float64"))
            pb = _fill(merged[pb_name].to_numpy("float64"))
            # --- Q1: keep_q 扫描（在 pool_rerank 的固定 q 上）---
            for q in (0.20, 1.00):
                vec = pool_rerank(pa, pb, flag, dates, q)
                for kq in KQS:
                    m = band_metrics(merged, vec, kq)
                    rows.append({"fold": fn, "block": "keepq", "primary": pa_name,
                                 "refiner": pb_name, "q": q, "tau": np.nan,
                                 "keep_q": kq, **m})
                    print(f"   keepq {pa_name}>{pb_name} q={q:.2f} kq={kq:.6f} "
                          f"final {m['final_score']:.6f} (超额 {m['annual_excess']:+.4f} "
                          f"换手 {m['mean_turnover']:.4f} IC {m['ic_mean']:.4f})", flush=True)
            # --- Q2: veto ---
            for q in RQ:
                for tau in TAUS:
                    vec = pool_veto(pa, pb, flag, dates, q, tau)
                    m = band_metrics(merged, vec, KEEP0)
                    rows.append({"fold": fn, "block": "veto", "primary": pa_name,
                                 "refiner": pb_name, "q": q, "tau": tau,
                                 "keep_q": KEEP0, **m})
                    print(f"   veto  {pa_name}>{pb_name} q={q:.2f} tau={tau:.1f} "
                          f"final {m['final_score']:.6f} (超额 {m['annual_excess']:+.4f} "
                          f"换手 {m['mean_turnover']:.4f} IC {m['ic_mean']:.4f})", flush=True)
        del merged

    df = pd.DataFrame(rows)
    df.to_csv("outputs/long_horizon/_probe_keepq_veto.csv", index=False)

    pd.set_option("display.width", 280)
    print("\n================ Q1: keep_q 扫描（四折均值）================")
    k = df[df.block == "keepq"]
    piv = k.pivot_table(index=["primary", "refiner", "q", "keep_q"],
                        columns="fold", values="final_score")
    piv["均值"] = piv.mean(axis=1)
    piv["最差"] = piv[[c for c in FOLD_NAMES if c in piv.columns]].min(axis=1)
    print(piv.round(6).to_string())
    ex = k.pivot_table(index=["primary", "refiner", "q", "keep_q"],
                       columns="fold", values="annual_excess")
    ex["均值"] = ex.mean(axis=1)
    to = k.pivot_table(index=["primary", "refiner", "q", "keep_q"],
                       columns="fold", values="mean_turnover")
    to["均值"] = to.mean(axis=1)
    print("\n--- 对应年化超额（四折均值）---")
    print(ex[["均值"]].round(5).to_string())
    print("\n--- 对应换手（四折均值）---")
    print(to[["均值"]].round(5).to_string())

    print("\n================ Q2: veto（四折均值，keep_q 冻结）================")
    v = df[df.block == "veto"]
    pv = v.pivot_table(index=["primary", "refiner", "q", "tau"],
                       columns="fold", values="final_score")
    pv["均值"] = pv.mean(axis=1)
    pv["最差"] = pv[[c for c in FOLD_NAMES if c in pv.columns]].min(axis=1)
    print(pv.round(6).to_string())
    pve = v.pivot_table(index=["primary", "refiner", "q", "tau"], columns="fold",
                        values="annual_excess")
    pve["均值"] = pve.mean(axis=1)
    print("\n--- 对应年化超额（四折均值）---")
    print(pve[["均值"]].round(5).to_string())


if __name__ == "__main__":
    main()

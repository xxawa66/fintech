"""LH028-b —— 「不可交易 IC」是模型属性还是常数？

LH028-a 发现：官方 IC 域（valid_ic）含当日涨停股，而超额/换手域都剔涨停
⇒ 官方 IC 与非涨停域 IC 之间存在一个差额，且 TS_K40 的自然排序已经落在这条轴的
**最优端**（强行顶到顶反而 ΔIC −0.000181，四折同号）⇒ 这条轴饱和。

剩下真正有意义的问题是：**这个差额本身是不是模型依赖的？**
- 若是常数 ⇒ 对所有模型一视同仁，只是评分函数的口径偏移，无选型价值。
- 若因模型而异 ⇒ 「可投资域 IC」是一个从未被用过的、有经济含义的选型指标：
  它衡量「模型的预测能力里有多少能真正落到能买的股票上」，正对应赛题那句
  「使统计意义上的预测准确真正转化为投资实践中的有效决策」。

本脚本对 H01 / H05_F2T2 / LH019 mix / TS_K100 / TS_K80 / TS_K40 六个信号，
逐折给出 official IC 与 investable IC 及其差。
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = (
    _P(_os.environ["FINTECH_ROOT"])
    if _os.environ.get("FINTECH_ROOT")
    else next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
)
_OUT = _P(_os.environ.get("LH028_DIR", _REPO / "experiments" / "LH028"))
_OUT.mkdir(parents=True, exist_ok=True)

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

REPO = _REPO
CACHE = REPO / "outputs/long_horizon"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/LH023"))
sys.path.insert(0, str(REPO / "research/LH025"))

from _probe_dgtw import pct_rank_2d  # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, to_wide  # noqa: E402
import probe_band_freeze as L25  # noqa: E402

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KQ = 0.0022778298112255263
IC_MIN_VALID = 30
ORDER = ["H01", "H05_F2T2", "LH019_mix", "TS_K100", "TS_K80", "TS_K40"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ic_pair(P: np.ndarray, panel: Panel):
    """返回 (官方 IC 序列, 可投资域 IC 序列)。"""
    T = panel.T
    off = np.full(T, np.nan)
    inv = np.full(T, np.nan)
    for t in range(T):
        specs = ((off, None), (inv, ~panel.LU[t]))
        for dst, extra in specs:
            m = panel.valid_ic[t] if extra is None else (panel.valid_ic[t] & extra)
            if int(m.sum()) < IC_MIN_VALID:
                continue
            rp = rankdata(P[t][m])
            ry = rankdata(panel.Y[t][m])
            if rp.std() == 0 or ry.std() == 0:
                continue
            dst[t] = float(np.corrcoef(rp, ry)[0, 1])
    return off, inv


def main() -> int:
    t0 = time.time()
    dw = pd.read_csv(REPO / "experiments/LH019_deploy_weights.csv")
    row = dw[np.isclose(dw.lam, 0.5)].iloc[0]
    weights = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in str(row["w"]).split(";")}
    log(f"部署权重 {weights}")

    rows = []
    for fold in FOLDS:
        df = load_fold(fold)
        pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
        pred["ts_code"] = pred["ts_code"].astype(str)
        pred["trade_date"] = pred["trade_date"].astype("int64")
        extra = [m for m in weights if m in pred.columns and m not in df.columns]
        if extra:
            df = df.drop(columns=[c for c in extra if c in df.columns]).merge(
                pred[["ts_code", "trade_date"] + extra], on=["ts_code", "trade_date"], how="left")
            df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
        panel = Panel(df)
        raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in weights}
        pct = {m: pct_rank_2d(P) for m, P in raw.items()}
        mix = np.nansum(np.stack([w * pct[m] for m, w in weights.items()]), axis=0)
        pr_mix = pct_rank_2d(mix)

        variants = {
            "H01": raw["H01"],
            "H05_F2T2": raw["H05_F2T2"],
            "LH019_mix": mix,
            "TS_K100": L25.two_stage(pr_mix, pct["H01"], 100.0),
            "TS_K80": L25.two_stage(pr_mix, pct["H01"], 80.0),
            "TS_K40": L25.two_stage(pr_mix, pct["H01"], 40.0),
        }
        for tag, P0 in variants.items():
            Pb, _ = fast_band(P0, panel.LU, KQ)
            off, inv = ic_pair(Pb, panel)
            rows.append(dict(fold=fold, tag=tag,
                             ic_off=float(np.nanmean(off)), ic_inv=float(np.nanmean(inv)),
                             gap=float(np.nanmean(off - inv)),
                             n_days=int((~np.isnan(off) & ~np.isnan(inv)).sum())))
        log(f"  {fold} done")

    M = pd.DataFrame(rows)
    M.to_csv(_OUT / "investable_ic_by_model.csv", index=False)

    pd.set_option("display.width", 400)
    for k in ["ic_off", "ic_inv", "gap"]:
        t = M.pivot(index="tag", columns="fold", values=k).reindex(ORDER)[FOLDS]
        t["mean"] = t.mean(axis=1)
        t["sd"] = t[FOLDS].std(axis=1)
        print(f"\n=== {k} ===")
        print(t.round(5).to_string())

    print("\n=== 逐折 gap 的跨模型极差（模型依赖性的尺度）===")
    g = M.pivot(index="tag", columns="fold", values="gap")[FOLDS]
    print(pd.DataFrame({"min": g.min(), "max": g.max(),
                        "range": g.max() - g.min(), "mean": g.mean()}).round(5).to_string())
    print(f"\n用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

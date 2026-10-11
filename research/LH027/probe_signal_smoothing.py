"""LH027-B —— 稳定性应该从信号层取得，而不是从决策层冻结

## 要检验的机制命题

当前 TS_K40 的低换手（T=0.0184）**不是信号稳定带来的**，而是 band 的出局阈值
q*=0.0023 造成的决策层迟滞：一只在榜股要跌到全市场最后 0.23% 才被换掉，名单近乎冻结。

LH025 的三条实测把这件事钉死了：

1. 关掉出局规则 ⇒ IC / α / E / profile 四项同降。名单已经在靠惯性，不是靠信号。
2. 沿 lkeep 轴 Δα/ΔT ≈ 1.0——压掉多少换手，就卖掉多少层内选股能力。
3. q=0.08 处 Δα = +0.0643（四折同号 t=2.55）、ΔIC = +0.0028（四折同号 t=3.08），
   代价是 ΔT = +0.0631、Δprofile = −0.0366、Δfinal = −0.0086。

⇒ 决策层迟滞是一种**用 α 换换手**的交易，且官方评分奖励这种交易。

## 本实验的做法

把稳定性搬到**信号层**：先对每个模型的每日预测百分位做时间平滑（EMA / SMA），
再原样跑两阶段 + band。决策规则一点不改，因此名单**始终贴（平滑后的）信号**。

    baseline : pct[m] 直接使用          → 稳定性来自 band 冻结（决策层）
    smoothed : EMA/SMA(pct[m])          → 稳定性来自信念平滑（信号层）

若命题成立，应观察到：平滑后可以在**高得多的 q**（灵敏出局）下维持基线换手，
同时拿到 q=0.08 那一档的 α。

## 判据
- 主判据：在 T ≤ 基线换手（0.0184）×1.15 的配置里，α 的最高值 vs 基线 0.18886
- 副判据：Δα / ΔIC / Δprofile / ΔT 的四折配对 Δ/t 与逐折符号
- 可外推口径 = 0.4·IC + 0.3·α + 0.3·(1−T)（不含 profile）
- 净 α 口径（LH027-A）：α − 504·T/(2−T)·c，c 取 0.0010 / 0.0025 / 0.0040

**全程只用 2021–2024 四个 CV 折。**
"""
from __future__ import annotations

import os as _os
import sys
import time
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = (
    _P(_os.environ["FINTECH_ROOT"])
    if _os.environ.get("FINTECH_ROOT")
    else next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
)
_OUT = _P(_os.environ.get("LH027_DIR", _REPO / "experiments" / "LH027"))
_OUT.mkdir(parents=True, exist_ok=True)

os = _os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "research/LH023"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _probe_dgtw import decompose, pct_rank_2d, stratify  # noqa: E402
from _probe_e_attr import load_factors, top_sets, wide_of  # noqa: E402
from _probe_root_cause import Panel, fast_band, holding_stats, load_fold, score, to_wide  # noqa: E402

REPO = _REPO
CACHE = REPO / "outputs/long_horizon"
OUT = _OUT
FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KQ = 0.0022778298112255263
K_STAGE1 = 40.0
ANN = 252
NBIN = 10
LAYER = "vol60"

# (kind, param)：kind='none' 基线；'ema' 的 param 是惯性 ρ（越大越平滑）；'sma' 的 param 是窗口天数
SMOOTH_GRID = [("none", 0.0), ("ema", 0.30), ("ema", 0.50), ("ema", 0.70), ("ema", 0.85),
               ("sma", 3.0), ("sma", 5.0), ("sma", 10.0)]
Q_GRID = [KQ, 0.01, 0.02, 0.04, 0.08]
COSTS = [0.0010, 0.0025, 0.0040]
# 注：曾用「前 30 日置 NaN」做 warm-up 对照，构造无效——首日名单为空集⇒Jaccard=1.0
# 直接把均值拉高。EMA 的 warm-up 长度 ≈ 1/alpha ≤ 7 日，相对 240 日的折长可忽略。

ARCHIVE = {  # experiments/LH020_twostage_folds.csv 的 TS_K40 行
    "wf2021": dict(ic=0.081156, excess=0.257336, turnover=0.022824,
                   final=0.402816, alpha=0.256086, profile=-0.009986),
    "wf2022": dict(ic=0.097536, excess=0.207012, turnover=0.016528,
                   final=0.396160, alpha=0.150050, profile=0.051173),
    "wf2023": dict(ic=0.069012, excess=0.138517, turnover=0.012252,
                   final=0.365484, alpha=0.140308, profile=-0.006093),
    "confirm2024": dict(ic=0.097487, excess=0.230244, turnover=0.021913,
                        final=0.401494, alpha=0.209000, profile=0.018148),
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def smooth_wide(P: np.ndarray, kind: str, p: float) -> np.ndarray:
    """沿时间轴（行）逐列平滑。NaN 不参与递推（ewm ignore_na / rolling min_periods=1）。"""
    if kind == "none":
        return P
    df = pd.DataFrame(P)
    if kind == "ema":
        return df.ewm(alpha=1.0 - p, ignore_na=True, min_periods=1).mean().to_numpy()
    if kind == "sma":
        return df.rolling(int(p), min_periods=1).mean().to_numpy()
    raise ValueError(kind)


def two_stage(pr_lh: np.ndarray, pr_h1: np.ndarray, K: float) -> np.ndarray:
    """候选池 = 长周期 mix top K%；池内按 H01 重排（与 LH025 同构）。"""
    from scipy.stats import rankdata
    out = np.full(pr_lh.shape, np.nan)
    for t in range(pr_lh.shape[0]):
        a, b = pr_lh[t], pr_h1[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        bb = np.where(cand & np.isfinite(b), b, np.nan)
        v = bb[cand & np.isfinite(b)]
        lo, hi = v.min(), v.max()
        within = np.where(np.isfinite(bb), (bb - lo) / (hi - lo), 0.5) if hi > lo \
            else np.where(np.isfinite(bb), 0.5, 0.5)
        out[t] = np.where(cand, 1.0 + within, np.where(m, a, np.nan))
    return out


def load_weights() -> dict:
    dw = pd.read_csv(REPO / "experiments/LH019_deploy_weights.csv")
    row = dw[np.isclose(dw.lam, 0.5)].iloc[0]
    return {kv.split(":")[0]: float(kv.split(":")[1]) for kv in str(row["w"]).split(";")}


def ann_turn(t):
    return 504.0 * np.asarray(t, float) / (2.0 - np.asarray(t, float))


def main() -> int:
    t0 = time.time()
    weights = load_weights()
    log(f"部署权重 {weights}")
    fdf = load_factors(20210101, 20241231)

    rows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
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
        W = wide_of(fdf, panel)
        raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in weights}
        pct = {m: pct_rank_2d(P) for m, P in raw.items()}

        for kind, p in SMOOTH_GRID:
            sm = {m: smooth_wide(P, kind, p) for m, P in pct.items()}
            mix = np.nansum(np.stack([w * sm[m] for m, w in weights.items()]), axis=0)
            for warm in [0]:
                m2 = mix.copy()
                P0 = two_stage(pct_rank_2d(m2), sm["H01"], K_STAGE1)
                for q in Q_GRID:
                    tag = f"{kind}{p:g}|q{q:g}"
                    Pb, _ = fast_band(P0, panel.LU, q)
                    r = score(Pb, panel)
                    S = top_sets(Pb, panel)
                    d = decompose(W, panel, S, [LAYER], [NBIN], fold, tag)
                    alpha = float(d["alpha"].mean()) * ANN
                    prof = float(d["profile"].mean()) * ANN
                    hs = holding_stats(r["_S"])
                    T = float(r["mean_turnover"])
                    rows.append(dict(fold=fold, smooth=kind, sparam=p, warm=warm, q=q, tag=tag,
                                     ic=r["ic_mean"], excess=r["annual_excess"], turnover=T,
                                     final=r["final_score"], alpha=alpha, profile=prof,
                                     noprofile=0.4 * r["ic_mean"] + 0.3 * alpha + 0.3 * (1 - T),
                                     hold_days=hs["avg_hold_days"],
                                     aturn=float(ann_turn(T)),
                                     **{f"net{c:.4f}": alpha - float(ann_turn(T)) * c
                                        for c in COSTS}))
        log(f"  {fold} done  ({time.time()-t0:.0f}s)")

    M = pd.DataFrame(rows)
    M.to_csv(OUT / "signal_smoothing_grid.csv", index=False)

    # ---- 自检：none + q* 必须复现归档 TS_K40 ----
    print("\n=== 自检：none|q* vs 归档 TS_K40 ===")
    bad = 0
    for fold in FOLDS:
        got = M[(M.fold == fold) & (M.smooth == "none") & (M.warm == 0)
                & np.isclose(M.q, KQ)].iloc[0]
        for k, v in ARCHIVE[fold].items():
            if abs(float(got[k]) - v) > 1e-5:
                bad += 1
                print(f"  FAIL {fold} {k}: {float(got[k]):.6f} vs {v:.6f}")
    print(f"  四折 × 6 项 = 24 项，不一致 {bad} 项 -> {'PASS' if bad == 0 else 'FAIL'}")

    cols = ["ic", "alpha", "profile", "excess", "turnover", "final", "noprofile",
            "hold_days"] + [f"net{c:.4f}" for c in COSTS]
    base = M[(M.smooth == "none") & (M.warm == 0) & np.isclose(M.q, KQ)].set_index("fold")
    bt = float(base["turnover"].mean())
    print(f"\n基线（部署点）四折均值:")
    print(base[cols].mean().round(5).to_string())

    print(f"\n=== 主判据：T ≤ 基线×1.15 = {bt*1.15:.5f} 的配置里 α 最高者 ===")
    sub = M[(M.warm == 0) & (M["turnover"] <= bt * 1.15)]
    g = sub.groupby(["smooth", "sparam", "q"])[cols].mean()
    print(g.sort_values("alpha", ascending=False).head(12).round(5).to_string())

    print("\n=== 全网格 α 最高 10 ===")
    g2 = M[M.warm == 0].groupby(["smooth", "sparam", "q"])[cols].mean()
    print(g2.sort_values("alpha", ascending=False).head(10).round(5).to_string())

    print("\n=== 逐 smooth 档、q* 下的对照（换手能否降下来）===")
    for (k, p), gg in M[(M.warm == 0)].groupby(["smooth", "sparam"]):
        r = gg[gg.q == KQ].iloc[0]
        print(f"  {k}{p:g} @ q*:  T={r['turnover']:.5f} alpha={r['alpha']:.5f} "
              f"ic={r['ic']:.5f} hold={r['hold_days']:.1f}d")

    pass
    pass
    pass
    pass
    pass
    pass

    print(f"\n[{time.time()-t0:.0f}s] done -> {OUT/'signal_smoothing_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

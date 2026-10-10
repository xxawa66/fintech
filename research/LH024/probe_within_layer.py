"""TS_K40 之后的可行性探针：层内(within-layer)排序值不值得付重训代价。

背景
----
LH023 判定参数面封闭（λ/K/keep_q/w_F1 四条轴四次留一折选参 Δ 全负；74 个 λ×K
组合无一达 0.4；α 当优化目标失败，其权重剖面跨折秩相关仅 +0.196）。剩余唯一理论
通道是把长周期模型的**训练标签**从「全池 rank」换成「分层内 rank」，让梯度直接指
向 α 的定义。本脚本不重训，只回答两个可判定问题：

A. 层内排序本身能拿多少 α、会不会炸换手？
   把现有信号的排序按「层内化程度」beta 插值：
       pred(beta) = (1 - beta) * 全池百分位 + beta * 层内百分位
   beta=0 退化回原信号，beta=1 是纯层内排序（等价于把层间共同成分从排序里剔除）。
   逐 beta 跑 band + 官方三项 + DGTW(vol60 十层) α/profile 分解。

B. 「层内化程度」这条轴，比「权重」轴更可优化吗？
   对同一个 beta 网格算剖面跨折秩相关（4 折两两，共 6 对），与 LH023 §7.2 的
   α 剖面 rho=+0.196 / IC rho=+0.961 对照。剖面稳 ⇒ 该轴可优化（有内部极值可
   找且可重复）；剖面乱 ⇒ 与 w_F1 同病，重训也没用。

判据：四折 + 配对 Δ/t + 逐折符号。**全程只用 2021-2024 CV 折，不读 2025 以后数据。**

自检：beta=0 / H01 的四折 final 必须命中归档值 0.379801（LH014_h01_kq0023）。
"""
from __future__ import annotations

# --- LH024 archival shim: 按脚本位置反查仓库根与数据目录，与当前工作目录无关 ---
import os as _os
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = (
    _P(_os.environ["FINTECH_ROOT"])
    if _os.environ.get("FINTECH_ROOT")
    else next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
)
_LH024_DIR = _P(_os.environ.get("LH024_DIR", _REPO / "experiments" / "LH024"))
if _LH024_DIR.is_dir():
    _os.chdir(_LH024_DIR)  # 脚本内部使用 cv_design_audit/ 相对路径
# ---------------------------------------------------------------------------

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
REPO = _REPO
CACHE = REPO / "outputs/long_horizon"
OUT = _LH024_DIR / "cv_design_audit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/LH023"))

from _probe_dgtw import decompose, pct_rank_2d, stratify  # noqa: E402
from _probe_e_attr import load_factors, top_sets, wide_of  # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide  # noqa: E402

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
BETAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
KQ = 0.0022778298112255263
ANN = 252
NBIN = 10
LAYER = "vol60"
DEPLOY = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
SELFCHECK_FINAL = 0.379801


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pct_rank_row(x: np.ndarray) -> np.ndarray:
    """单行百分位（含并列取平均），NaN 原样返回。"""
    out = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    n = int(m.sum())
    if n < 2:
        return out
    v = x[m]
    o = np.argsort(v, kind="mergesort")
    rk = np.empty(n)
    rk[o] = np.arange(n, dtype=float)
    s = v[o]
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        rk[o[i:j]] = (i + j - 1) / 2.0
        i = j
    out[m] = rk / (n - 1)
    return out


def within_layer_pct(P: np.ndarray, L: np.ndarray, nbin: int = NBIN) -> np.ndarray:
    """逐日、逐层（按 L 分 nbin 层）在层内做百分位。层内没有有效值的股票保持 NaN。

    结果落在 [0,1]，但层间不再有偏好：全池取 top 10% 时各层的入选比例大致相同，
    即名单的层暴露 ≈ 全池层分布 ⇒ profile 按构造被压到 ≈0。
    """
    T = P.shape[0]
    out = np.full(P.shape, np.nan)
    for t in range(T):
        p, l = P[t], L[t]
        b = stratify(l, nbin)
        for k in range(nbin):
            m = (b == k) & np.isfinite(p)
            if m.sum() < 20:
                continue
            out[t, m] = pct_rank_row(p[m])
    return out


def fold_block(fold: str, fdf: pd.DataFrame) -> list[dict]:
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    extra = [m for m in ["H01", "H05_F1T2", "H05_F2T2"]
             if m in pred.columns and m not in df.columns]
    if extra:
        df = df.drop(columns=[c for c in extra if c in df.columns]).merge(
            pred[["ts_code", "trade_date"] + extra], on=["ts_code", "trade_date"], how="left")
        df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    vol = W[LAYER]
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in DEPLOY}
    pct = {m: pct_rank_2d(P) for m, P in raw.items()}
    mix = np.nansum(np.stack([w * pct[m] for m, w in DEPLOY.items()]), axis=0)

    base = {"H01": pct["H01"], "mix": pct_rank_2d(mix)}
    rows = []
    for name, G in base.items():
        Lw = within_layer_pct(G, vol)
        okb = np.isfinite(Lw) & np.isfinite(G)
        for beta in BETAS:
            P0 = G.copy()
            P0[okb] = (1.0 - beta) * G[okb] + beta * Lw[okb]
            P, _ = fast_band(P0, panel.LU, KQ)
            r = score(P, panel)
            S = top_sets(P, panel)
            d = decompose(W, panel, S, [LAYER], [NBIN], f"{fold}|{name}", f"b{beta:g}")
            alpha = float(d["alpha"].mean()) * ANN
            prof = float(d["profile"].mean()) * ANN
            rows.append(dict(
                fold=fold, name=name, beta=beta,
                ic=r["ic_mean"], excess=r["annual_excess"], turnover=r["mean_turnover"],
                final=r["final_score"], alpha=alpha, profile=prof,
                noprofile=0.4 * r["ic_mean"] + 0.3 * alpha + 0.3 * (1 - r["mean_turnover"]),
            ))
            log(f"  {fold} {name:4s} b={beta:<4g} IC {r['ic_mean']:.4f} E {r['annual_excess']:.4f} "
                f"T {r['mean_turnover']:.4f} F {r['final_score']:.4f} "
                f"a {alpha:.4f} p {prof:+.4f} np {rows[-1]['noprofile']:.4f}")
    return rows


def paired(w: pd.Series, base_beta: float, metric: str) -> dict:
    """相对 beta=base 的配对差：均值、t=mean/(sd/2)（n=4）、逐折符号。w 以 beta 为索引。"""
    d = (w - w.loc[base_beta]).drop(index=base_beta)
    sd = float(np.std(d.to_numpy(), ddof=1))
    return {"metric": metric, "delta": float(d.mean()),
            "t": float(d.mean() / (sd / 2)) if sd > 0 else np.nan,
            "signs": "".join("+" if v > 0 else "-" for v in d.to_numpy())}


def main() -> int:
    t0 = time.time()
    fdf = load_factors(20210101, 20241231)
    rows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        rows.extend(fold_block(fold, fdf))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "within_layer_grid.csv", index=False)

    pd.set_option("display.width", 320)
    print("\n=== 四折均值（beta 前沿）===")
    for name in ["H01", "mix"]:
        sub = M[M.name == name]
        g = sub.groupby("beta")[["ic", "excess", "turnover", "final", "alpha", "profile", "noprofile"]].mean()
        print(f"\n--- {name} ---")
        print(g.round(4).to_string())

    print("\n=== 配对 Δ（vs beta=0），H01 ===")
    rows_p = []
    for name in ["H01", "mix"]:
        sub = M[M.name == name]
        for metric in ["ic", "alpha", "profile", "turnover", "noprofile"]:
            w = sub.pivot_table(index="beta", values=metric, aggfunc="mean")[metric]
            r = paired(w, 0.0, metric)
            r["name"] = name
            rows_p.append(r)
    P = pd.DataFrame(rows_p)[["name", "metric", "delta", "t", "signs"]]
    print(P.round(6).to_string(index=False))
    P.to_csv(OUT / "within_layer_paired.csv", index=False)

    print("\n=== 剖面跨折秩相关（4 折两两，6 对）===")
    rows_c = []
    for name in ["H01", "mix"]:
        sub = M[M.name == name]
        for metric in ["ic", "alpha", "noprofile", "profile"]:
            w = sub.pivot_table(index="fold", columns="beta", values=metric)
            w = w.loc[FOLDS]
            cors = []
            for i in range(len(FOLDS)):
                for j in range(i + 1, len(FOLDS)):
                    cors.append(float(spearmanr(w.iloc[i], w.iloc[j]).statistic))
            rows_c.append(dict(name=name, metric=metric, mean=float(np.mean(cors)),
                               min=float(np.min(cors)), max=float(np.max(cors)), n_neg=int(sum(c < 0 for c in cors))))
    C = pd.DataFrame(rows_c)
    print(C.round(3).to_string(index=False))
    C.to_csv(OUT / "within_layer_crossfold.csv", index=False)

    sc = float(M[(M.name == "H01") & (M.beta == 0.0)]["final"].mean())
    print(f"\n[自检] H01 beta=0 四折 final = {sc:.6f}  归档 = {SELFCHECK_FINAL:.6f}  "
          f"差 {abs(sc - SELFCHECK_FINAL):.2e}  -> {'PASS' if abs(sc - SELFCHECK_FINAL) < 1e-5 else 'FAIL'}")
    log(f"完成，用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

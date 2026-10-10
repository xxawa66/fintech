"""把 TS_K40 的四个「具体数字」放到同一套标准下重审：F1 权重 0.1204、K=40、keep_q=0.0023。

K 与 λ 此前已经按此标准做过（cv_lambda_k_opt.py + forward_select.py + stability_check.py）。
这里补上缺的两条轴：
  [A] keep_q 轴（此前 cv_design_audit.py 扫过，但那次没有 alpha/profile，走不了可外推口径）
  [B] F1 权重轴（此前只知道「去掉 F1 是噪声」，没扫过完整权重轴）
并对三条轴（含已知的 K）统一做：
  1. jackknife argmax 稳定性（去掉一折后 argmax 是否仍命中部署点）
  2. 1·se 平台宽度（|Δ| <= 一个配对标准误的点都算平台）
  3. 留一折选参（三折 argmax 在未见折上 vs 部署点的差，Δ<0 = 加密/挑选有害）

不读任何 2025 年以后的数据。
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT_FIN = Path(os.environ.get("FINTECH_ROOT", _HERE.parents[1]))   # 仓库根
CACHE = Path(os.environ.get("LH_CACHE", ROOT_FIN / "outputs" / "long_horizon"))
ROOT = os.environ.get("LH_TESTDIR", str(ROOT_FIN.parent.parent / "test_y_2025_2026"))
for p in (str(_HERE), str(CACHE), str(ROOT_FIN), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import time

import numpy as np
import pandas as pd

import cv_lambda_k_opt as clk
from cv_lambda_k_opt import FOLDS, Evaluator, available, pct_rank_2d, two_stage
from _probe_e_attr import load_factors

DEPLOY = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
OUT = clk.OUT


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def np_of(df):
    return 0.4 * df.ic + 0.3 * df.alpha + 0.3 * (1 - df.turnover)


# --------------------------------------------------------------- 三条轴定义
KQ_GRID = [0.0005, 0.0010, 0.002278, 0.0040, 0.0060, 0.0100, 0.0200, 0.0300, 0.0600]
F1_GRID = [0.00, 0.06, 0.10, 0.1204, 0.15, 0.20, 0.25, 0.35, 0.50]

DEP = {"keep_q": 0.002278, "F1 权重": 0.1204, "K": 40.0}

K_GRID_FULL = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
               60.0, 70.0, 85.0, 100.0]
K_GRID_COARSE = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0,
                 85.0, 100.0]


def _w_deploy(sigs):
    w = np.array([DEPLOY.get(s, 0.0) for s in sigs])
    return w / w.sum()


def run_k(E, sigs):
    log("[C] K 轴（deploy 权重 / λ=0.5 / keep_q 部署值，含 35 与 45）")
    w = _w_deploy(sigs)
    rows = []
    for K in K_GRID_FULL:
        for held in FOLDS:
            mix = pct_rank_2d(E.mix(held, w))
            rows.append(dict(axis="K", x=K, **E.run(
                held, two_stage(mix, E.get(held)["wide"]["H01"], K), f"K{K}")))
        log(f"    K={K:<6} 均值 {np_of(pd.DataFrame(rows[-4:])).mean():.6f}")
    return pd.DataFrame(rows)


def run_keepq(E, sigs):
    log("[A] keep_q 轴（deploy 权重 / λ=0.5 / K=40 固定）")
    w = np.array([DEPLOY.get(s, 0.0) for s in sigs])
    rows = []
    for kq in KQ_GRID:
        clk.KQ = kq
        for held in FOLDS:
            mix = pct_rank_2d(E.mix(held, w))
            rows.append(dict(axis="keep_q", x=kq, **E.run(
                held, two_stage(mix, E.get(held)["wide"]["H01"], 40.0), f"kq{kq}")))
        log(f"    keep_q={kq:<8} 均值 {np_of(pd.DataFrame(rows[-4:])).mean():.6f}")
    clk.KQ = DEP["keep_q"]
    return pd.DataFrame(rows)


def run_f1(E, sigs):
    log("[B] F1 权重轴（H01=0.5 / K=40 / keep_q 部署值）")
    rows = []
    for f1 in F1_GRID:
        wt = {"H01": 0.5, "H05_F1T2": f1, "H05_F2T2": 0.5 - f1}
        w = np.array([wt.get(s, 0.0) for s in sigs])
        for held in FOLDS:
            mix = pct_rank_2d(E.mix(held, w))
            rows.append(dict(axis="F1 权重", x=f1, **E.run(
                held, two_stage(mix, E.get(held)["wide"]["H01"], 40.0), f"f1_{f1}")))
        log(f"    w_F1={f1:<8} 均值 {np_of(pd.DataFrame(rows[-4:])).mean():.6f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 统一三项分析
def analyze(piv, mean, dep_x, label):
    """piv: index=x, columns=FOLDS, values=可外推口径。"""
    print(f"\n{'-' * 82}")
    print(f"{label}   部署取值 = {dep_x}")
    print("-" * 82)

    xs = list(mean.index)
    base = piv.loc[dep_x].to_numpy()

    # ---- 1. 配对 Δ / t / 符号 ----
    print(f"  {'取值':>10}{'可外推':>11}{'Δ vs 部署':>12}{'t':>8}   逐折符号")
    for x in xs:
        d = piv.loc[x].to_numpy() - base
        sd = d.std(ddof=1)
        t = d.mean() / (sd / 2) if sd > 0 else np.nan
        mark = "  <-部署" if x == dep_x else ""
        print(f"  {x:>10.6g}{mean[x]:>11.6f}{d.mean():>+12.6f}{t:>+8.2f}   "
              + "".join("+" if v > 0 else "-" for v in d) + mark)

    # ---- 2. 1·se 平台 ----
    ses = []
    for x in xs:
        if x == dep_x:
            continue
        d = piv.loc[x].to_numpy() - base
        sd = d.std(ddof=1)
        ses.append(sd / 2 if sd > 0 else 0.0)
    se = float(np.median(ses))
    plat = [x for x in xs if abs(mean[x] - mean[dep_x]) <= se]
    print(f"\n  配对标准误中位数 {se:.6f}  ⇒  1·se 平台 = "
          f"[{min(plat):.6g}, {max(plat):.6g}]  共 {len(plat)}/{len(xs)} 个网格点")
    print(f"  峰 argmax = {mean.idxmax():.6g}（均值 {mean.max():.6f}）"
          f"{'   ==部署点' if mean.idxmax() == dep_x else '   ≠部署点'}")

    # ---- 3. jackknife argmax ----
    hits, picks = [], []
    for held in FOLDS:
        sub = piv.drop(columns=[held]).mean(axis=1)
        pk = sub.idxmax()
        picks.append(pk)
        hits.append(abs(pk - dep_x) < 1e-12)
    print(f"  留一折 argmax  = " + " / ".join(f"{p:.6g}" for p in picks)
          + f"    命中部署点 {sum(hits)}/4")

    # ---- 4. 留一折选参 ----
    deltas, chosen = [], []
    for held in FOLDS:
        sub = piv.drop(columns=[held]).mean(axis=1)
        pk = sub.idxmax()
        chosen.append(pk)
        deltas.append(piv.loc[pk, held] - piv.loc[dep_x, held])
    d = np.array(deltas)
    sd = d.std(ddof=1)
    t = d.mean() / (sd / 2) if sd > 0 else np.nan
    print("  留一折选参 Δ（三折 argmax 在未见折上 − 部署点）：")
    print("    " + "  ".join(f"{c:.6g}→{v:+.6f}" for c, v in zip(chosen, deltas)))
    print(f"    均值 {d.mean():+.6f}   t {t:+.2f}   逐折符号 "
          + "".join("+" if v > 0 else "-" for v in d)
          + ("   ⇒ 加密/挑选有害" if d.mean() < 0 else "   ⇒ 挑选有效"))
    return dict(se=se, plateau=(min(plat), max(plat)), argmax=mean.idxmax(),
                jackknife_hits=sum(hits), lofo_delta=float(d.mean()), lofo_t=t)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-k", action="store_true",
                    help="重跑 K 轴（含 35/45 加密点），默认读已有 kgrid_lam0.5.csv")
    ap.add_argument("--k-only", action="store_true", help="只跑 K 轴，跳过另外两条")
    args = ap.parse_args()

    sigs = available()
    fdf = load_factors(20210101, 20241231)
    E = Evaluator(sigs, fdf)

    if args.k_only:
        R = run_k(E, sigs)
        R["np"] = np_of(R)
        R.to_csv(OUT / "k_grid35_45.csv", index=False)
        log(f"已写入 {OUT / 'k_grid35_45.csv'}")
        piv = R.pivot_table(index="x", columns="fold", values="np")[FOLDS]
        sf = analyze(piv, piv.mean(axis=1), DEP["K"],
                     "K 轴 · 加密网格（含 35 / 45，deploy 权重）")
        pc = piv.loc[[x for x in K_GRID_COARSE if x in piv.index]]
        sc = analyze(pc, pc.mean(axis=1), DEP["K"],
                     "K 轴 · 只保留步长 10 的粗网格（同一批数据，便于对比）")
        print("\n  ⇒ 加密（加入 35/45）前后对比：")
        print(f"     留一折 argmax 命中部署点  {sc['jackknife_hits']}/4  →  "
              f"{sf['jackknife_hits']}/4")
        print(f"     留一折选参 Δ              {sc['lofo_delta']:+.6f}  →  "
              f"{sf['lofo_delta']:+.6f}")
        return 0

    parts = [run_keepq(E, sigs), run_f1(E, sigs)]

    summary = {}
    if args.with_k:
        Rk = run_k(E, sigs)
        R = pd.concat(parts + [Rk], ignore_index=True)
        R["np"] = np_of(R)
        R.to_csv(OUT / "three_numbers_grid.csv", index=False)
        log(f"已写入 {OUT / 'three_numbers_grid.csv'}")

        piv = Rk.pivot_table(index="x", columns="fold", values="np")[FOLDS]
        summary["K"] = analyze(piv, piv.mean(axis=1), DEP["K"],
                               "K 轴 · 加密网格（含 35 / 45，deploy 权重）")

        pc = piv.loc[[x for x in K_GRID_COARSE if x in piv.index]]
        sc = analyze(pc, pc.mean(axis=1), DEP["K"],
                     "K 轴 · 只保留步长 10 的粗网格（同一批数据，便于对比）")
        print("\n  ⇒ 加密（加入 35/45）前后的对比：")
        print(f"     留一折 argmax 命中部署点  {sc['jackknife_hits']}/4  →  "
              f"{summary['K']['jackknife_hits']}/4")
        print(f"     留一折选参 Δ              {sc['lofo_delta']:+.6f}  →  "
              f"{summary['K']['lofo_delta']:+.6f}")
    else:
        R = pd.concat(parts, ignore_index=True)
        R["np"] = np_of(R)
        R.to_csv(OUT / "three_numbers_grid.csv", index=False)

        src = pd.read_csv(OUT / "kgrid_lam0.5.csv")
        Rk = src.copy()
        Rk["np"] = np_of(Rk)
        piv = Rk.pivot_table(index="K", columns="fold", values="np")[FOLDS]
        summary["K"] = analyze(piv, piv.mean(axis=1), DEP["K"],
                               "K 轴（来自 kgrid_lam0.5.csv，LOFO 权重，步长 10）")

    for axis in ("keep_q", "F1 权重"):
        s = R[R.axis == axis]
        piv = s.pivot_table(index="x", columns="fold", values="np")[FOLDS]
        summary[axis] = analyze(piv, piv.mean(axis=1), DEP[axis], f"{axis} 轴")

    print("\n" + "=" * 82)
    print("汇总：四个「具体数字」现在各自能被四折定到什么程度")
    print("=" * 82)
    print(f"  {'参数':<12}{'部署值':>12}{'峰':>12}{'1·se 平台':>26}"
          f"{'留一折argmax':>14}{'留一折选参Δ':>14}")
    for axis, info in summary.items():
        lo, hi = info["plateau"]
        plat_s = f"[{lo:.6g}, {hi:.6g}]"
        hits_s = str(info["jackknife_hits"]) + "/4"
        print(f"  {axis:<12}{DEP[axis]:>12.6g}{info['argmax']:>12.6g}"
              f"{plat_s:>26}{hits_s:>14}{info['lofo_delta']:>+14.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

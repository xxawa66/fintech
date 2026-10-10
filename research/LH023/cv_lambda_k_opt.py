"""只用四折（2021-2024），依次优化两阶段模型 TS_K(λ, K) 的两个超参。

背景
----
TS_K40 的阶段 1 用 LH019 的 λ=0.5 融合权重做粗筛，但 LH019 已经写明
「没有任何 CV 数据能决定 λ」。而上一轮实测 λ 对 α 的影响达 0.023
（λ=0.5 α 0.1435 vs 不要 H01 α 0.1201），两阶段的目的恰恰是拿 α
⇒ λ 很可能是两阶段里**没被优化过**的那个旋钮。

定义
----
    mix  = λ·pct(H01) + (1−λ)·Σ_{k≠H01} v_k·pct(信号k)
    v    = 在训练折上以「逐日 Pearson 相关」为目标的 IC 最优解
    K    = 阶段 1 候选池宽度（%），阶段 2 池内按 H01 重排，最后加 band(q*)

λ=1 退化为纯 H01；K=100 也退化为纯 H01（但走两阶段路径，可作自检）。

优化顺序与诚实的口径
--------------------
- **权重一律用 LOFO**：每个留出折的权重在另外 3 折上求解，绝不在留出折上拟合。
  ⇒ 报出的 IC / α 是真正的样本外数字，不含权重调优的乐观偏差。
- λ 用**等式约束** w[H01] = λ（LH019 用的是 ≥ λ；等式才能干净地扫前沿）。
- 主口径 = 可外推的 0.4·IC + 0.3·α + 0.3·(1−T)（不含 profile，理由见 LH020）。
  同时报官方 final 作对照。

不读任何 2025 年以后的数据。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 必须在 import numpy 之前：多线程 BLAS 的浮点累加顺序不确定，会让 SLSQP
# 收敛到略微不同的权重，进而使同一配置的两次运行结果差到 3e-4
# （已实测：多线程两次 0.385601 / 0.385326，单线程两次均为 0.385601）。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from scipy.optimize import minimize                                 # noqa: E402

# 路径一律由本文件位置推导，不写死任何个人机器路径（AGENTS.md §14）。
# 需要放在别处时用环境变量覆盖：FINTECH_ROOT / LH_CACHE / LH_OUT。
_HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FINTECH_ROOT", _HERE.parents[1]))      # 仓库根
CACHE = Path(os.environ.get("LH_CACHE", ROOT / "outputs" / "long_horizon"))
for _p in (str(_HERE), str(ROOT), str(CACHE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _probe_dgtw import decompose, pct_rank_2d                      # noqa: E402
from _probe_e_attr import load_factors, top_sets, wide_of           # noqa: E402
from _probe_icopt import available, prepare                          # noqa: E402
from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide  # noqa: E402
from _probe_twostage import two_stage                                # noqa: E402

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KQ = 0.0022778298112255263
ANN = 252
OUT = Path(os.environ.get("LH_OUT", ROOT / "experiments" / "LH023"))
OUT.mkdir(parents=True, exist_ok=True)

LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K_GRID = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 85.0, 100.0]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------- LOFO 权重
def solve_eq(train_folds, sigs, lam, seed=0):
    """在 train_folds 上求 IC 最优权重，约束 w[H01] = lam（等式）。"""
    K = len(sigs)
    U = np.vstack([prepare(f, sigs)["u"] for f in train_folds])
    C = np.vstack([prepare(f, sigs)["C"] for f in train_folds]).reshape(-1, K, K)
    SB = np.concatenate([prepare(f, sigs)["sb"] for f in train_folds])
    i0 = sigs.index("H01")

    def neg(w):
        num = U @ w
        den = np.maximum(np.sqrt(np.einsum("i,kij,j->k", w, C, w)) * SB, 1e-12)
        return -float(np.mean(num / den))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w: w[i0] - lam}]
    rng = np.random.default_rng(seed)
    best, bestv = None, np.inf
    starts = []
    for _ in range(14):
        x = rng.dirichlet(np.ones(K))
        x[i0] = lam
        rest = np.delete(x, i0)
        rest = rest / max(rest.sum(), 1e-12) * (1 - lam)
        x = np.insert(rest, i0, lam)
        starts.append(x)
    starts.append(np.insert(np.full(K - 1, (1 - lam) / (K - 1)), i0, lam))
    for x0 in starts:
        try:
            r = minimize(neg, x0, method="SLSQP", bounds=[(0.0, 1.0)] * K,
                         constraints=cons, options=dict(maxiter=500, ftol=1e-12))
        except Exception:
            continue
        if np.isfinite(r.fun) and r.fun < bestv:
            best, bestv = r.x, r.fun
    if best is None:
        best = np.full(K, (1 - lam) / (K - 1))
        best[i0] = lam
    best = np.clip(best, 0, None)
    return best / best.sum()


# --------------------------------------------------------------- 评估
class Evaluator:
    """缓存每折的 panel / 宽矩阵 / 因子宽矩阵 / pct 排名，避免重复 I/O。"""

    def __init__(self, sigs, fdf):
        self.sigs = sigs
        self.fdf = fdf
        self.d = {}

    def get(self, fold):
        if fold in self.d:
            return self.d[fold]
        p = prepare(fold, self.sigs)
        rec = dict(panel=p["panel"], wide=p["wide"], W=wide_of(self.fdf, p["panel"]))
        self.d[fold] = rec
        return rec

    def mix(self, fold, w):
        pct = self.get(fold)["wide"]
        out = np.zeros_like(pct[self.sigs[0]])
        for k, s in enumerate(self.sigs):
            if w[k] > 1e-6:
                out = out + w[k] * pct[s]
        return out

    def run(self, fold, P0, tag):
        p = self.get(fold)
        panel = p["panel"]
        Pb, _ = fast_band(P0, panel.LU, KQ)
        r = score(Pb, panel)
        S = top_sets(Pb, panel)
        d = decompose(p["W"], panel, S, ["vol60"], [10], fold, tag)
        return dict(fold=fold, tag=tag, ic=r["ic_mean"], excess=r["annual_excess"],
                    turnover=r["mean_turnover"], final=r["final_score"],
                    alpha=float(d["alpha"].mean()) * ANN,
                    profile=float(d["profile"].mean()) * ANN)


def paired(p, ref, keys=None):
    """相对 ref 的配对检验（四折配对；t = mean/(sd/2)）。"""
    base = p.loc[ref].to_numpy()
    print(f"    {'tag':<16}{'Δ均值':>12}{'t':>8}   逐折符号")
    for k in (keys if keys is not None else p.index):
        if k == ref:
            continue
        d = p.loc[k].to_numpy() - base
        sd = d.std(ddof=1)
        t = d.mean() / (sd / 2) if sd > 0 else np.nan
        print(f"    {k:<16}{d.mean():>+12.6f}{t:>+8.2f}   " +
              "".join("+" if x > 0 else "-" for x in d))


def report(R, title, order=None):
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)
    p = R.pivot_table(index="tag", columns="fold", values="noprofile")[FOLDS]
    mean = p.mean(axis=1).sort_values(ascending=False)
    print(f"  {'tag':<16}{'可外推':>10}{'IC':>9}{'α':>9}{'profile':>9}"
          f"{'E':>9}{'换手':>9}{'final':>9}   逐折(可外推)")
    for k in mean.index:
        g = R[R["tag"] == k]
        fm = float(g["final"].mean())
        print(f"  {k:<16}{mean[k]:>10.6f}{g['ic'].mean():>9.4f}{g['alpha'].mean():>9.4f}"
              f"{g['profile'].mean():>9.4f}{g['excess'].mean():>9.4f}"
              f"{g['turnover'].mean():>9.4f}{fm:>9.4f}   " +
              " ".join(f"{p.loc[k, f]:.4f}" for f in FOLDS))
    return p


# --------------------------------------------------------------- Part A
def part_a(E, sigs, suffix=""):
    log(f"[A] λ 扫描（K=40 固定）{'  [精细网格]' if suffix else ''}")
    rows = []
    for lam in LAM_GRID:
        t0 = time.time()
        for held in FOLDS:
            train = [f for f in FOLDS if f != held]
            w = solve_eq(train, sigs, lam)
            mix = pct_rank_2d(E.mix(held, w))
            tag = f"lam{lam:.2f}"
            rows.append(dict(
                lam=lam, K=40.0, **E.run(held, two_stage(mix, E.get(held)["wide"]["H01"], 40.0), tag),
                w=";".join(f"{s}:{v:.4f}" for s, v in zip(sigs, w) if v > 1e-3)))
        log(f"    λ={lam:.1f}  {time.time()-t0:.0f}s")
    R = pd.DataFrame(rows)
    R["noprofile"] = 0.4 * R.ic + 0.3 * R.alpha + 0.3 * (1 - R.turnover)
    R.to_csv(OUT / f"lamgrid{suffix}.csv", index=False)

    p = report(R, "[A] λ 前沿（可外推口径降序）")
    print("\n  配对检验 vs λ=0.5（当前部署）")
    paired(p, f"lam{0.5:.2f}", [t for t in p.index if t.startswith("lam") and t != f"lam{0.5:.2f}"])

    print("\n  LOFO 解出的权重（每留出折一行，只列权重 >0.001）")
    for _, r in R.iterrows():
        print(f"    λ={r.lam:.1f}  {r.fold:<12s} {r.w}")
    return R


# --------------------------------------------------------------- Part B
def part_b(E, sigs, lam_star, suf=""):
    log(f"[B] K 扫描（λ={lam_star} 固定）")
    rows = []
    for K in K_GRID:
        t0 = time.time()
        for held in FOLDS:
            train = [f for f in FOLDS if f != held]
            w = solve_eq(train, sigs, lam_star)
            mix = pct_rank_2d(E.mix(held, w))
            tag = f"K{K:.0f}"
            rows.append(dict(lam=lam_star, K=K,
                             **E.run(held, two_stage(mix, E.get(held)["wide"]["H01"], K), tag),
                             w=";".join(f"{s}:{v:.4f}" for s, v in zip(sigs, w) if v > 1e-3)))
        log(f"    K={K:.0f}  {time.time()-t0:.0f}s")
    R = pd.DataFrame(rows)
    R["noprofile"] = 0.4 * R.ic + 0.3 * R.alpha + 0.3 * (1 - R.turnover)
    R.to_csv(OUT / f"kgrid_lam{lam_star:.2f}{suf}.csv", index=False)

    p = report(R, f"[B] K 前沿（λ={lam_star}，可外推口径降序）")
    best = p.mean(axis=1).idxmax()
    print(f"\n  均值最优 = {best}")
    print("\n  配对检验 vs K=40（当前部署）")
    paired(p, "K40", [t for t in p.index if t.startswith("K") and t != "K40"])
    print("\n  相邻档配对（检查是否是平台）")
    for i in range(len(K_GRID) - 1):
        a, b = f"K{K_GRID[i]:.0f}", f"K{K_GRID[i+1]:.0f}"
        d = p.loc[b].to_numpy() - p.loc[a].to_numpy()
        sd = d.std(ddof=1)
        t = d.mean() / (sd / 2) if sd > 0 else np.nan
        print(f"    {a}→{b}  Δ {d.mean():+.6f}  t {t:+6.2f}  " +
              "".join("+" if x > 0 else "-" for x in d))
    return R


# --------------------------------------------------------------- Part C
def part_c(E, sigs, lam_ref=0.5, suf=""):
    """交互检验：同一组 K 在另一个 λ 下扫一遍，看 K 的最优位置是否随 λ 移动。"""
    log(f"[C] 交互检验：K 前沿在 λ={lam_ref} 下重扫")
    rows = []
    for K in K_GRID:
        t0 = time.time()
        for held in FOLDS:
            train = [f for f in FOLDS if f != held]
            w = solve_eq(train, sigs, lam_ref)
            mix = pct_rank_2d(E.mix(held, w))
            tag = f"K{K:.0f}"
            rows.append(dict(lam=lam_ref, K=K,
                             **E.run(held, two_stage(mix, E.get(held)["wide"]["H01"], K), tag),
                             w=";".join(f"{s}:{v:.4f}" for s, v in zip(sigs, w) if v > 1e-3)))
        log(f"    K={K:.0f}  {time.time()-t0:.0f}s")
    R = pd.DataFrame(rows)
    R["noprofile"] = 0.4 * R.ic + 0.3 * R.alpha + 0.3 * (1 - R.turnover)
    R.to_csv(OUT / f"kgrid_lam{lam_ref:.2f}{suf}.csv", index=False)
    p = report(R, f"[C] K 前沿（λ={lam_ref}，可外推口径降序）")
    return R


# --------------------------------------------------------------- 基线
def baseline(E, sigs):
    log("[0] 基线：纯 H01 + band（λ=1 / K=100 应为同一条路径，作自检）")
    rows = []
    for held in FOLDS:
        p = E.get(held)
        Pb, _ = fast_band(p["wide"]["H01"], p["panel"].LU, KQ)
        r = score(Pb, p["panel"])
        S = top_sets(Pb, p["panel"])
        d = decompose(p["W"], p["panel"], S, ["vol60"], [10], held, "H01")
        rows.append(dict(fold=held, tag="H01", ic=r["ic_mean"], excess=r["annual_excess"],
                         turnover=r["mean_turnover"], final=r["final_score"],
                         alpha=float(d["alpha"].mean()) * ANN,
                         profile=float(d["profile"].mean()) * ANN, lam=1.0, K=100.0, w="H01:1"))
    R = pd.DataFrame(rows)
    R["noprofile"] = 0.4 * R.ic + 0.3 * R.alpha + 0.3 * (1 - R.turnover)
    print(f"    H01 基线  noprofile {R.noprofile.mean():.6f}  ic {R.ic.mean():.4f} "
          f"α {R.alpha.mean():.4f} T {R.turnover.mean():.4f} final {R.final.mean():.4f}")
    return R


class Joint:
    """把各 λ 的 K 网格拼成一张 λ×K 表，检查两个超参是否存在交互。"""

    @staticmethod
    def build(out: Path):
        fs = sorted(out.glob("kgrid_lam*.csv"))
        if not fs:
            return None
        R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        R = R.drop_duplicates(subset=["lam", "K", "fold"], keep="last")
        R.to_csv(out / "lambda_k_joint_grid.csv", index=False)
        p = R.pivot_table(index="lam", columns="K", values="noprofile")
        p = p.reindex(sorted(p.columns), axis=1)
        print("\n" + "=" * 82)
        print("[D] λ × K 联合网格（可外推口径，四折均值）")
        print("=" * 82)
        print("  λ\\K  " + "".join(f"{c:>8.0f}" for c in p.columns))
        for lam, r in p.iterrows():
            print(f"  {lam:<5.1f}" + "".join(f"{v:>8.4f}" for v in r.values) +
                  f"    最优K={r.idxmax():.0f}")
        print("\n  每个 λ 下的前三 K（含相邻差值）")
        for lam, r in p.iterrows():
            top = r.sort_values(ascending=False).head(3)
            print(f"    λ={lam:<4.1f}  " +
                  "  ".join(f"K{k:.0f}:{v:.4f}" for k, v in top.items()) +
                  f"    极差 {r.max() - r.min():.4f}")
        return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="0,A,B,C")
    ap.add_argument("--lam-grid", default=None,
                    help="覆盖 λ 网格，逗号分隔，例如 0.35,0.45,0.55,0.65")
    ap.add_argument("--lam-refs", default="0.2,0.8",
                    help="Part C 交互检验用的对照 λ，逗号分隔")
    ap.add_argument("--k-grid", default=None,
                    help="覆盖 K 网格，逗号分隔，例如 30,35,40,45,50")
    ap.add_argument("--lam-star", type=float, default=None,
                    help="Part B 用的 λ；默认取 Part A 的可外推口径均值最优")
    args = ap.parse_args()
    parts = [x.strip() for x in args.parts.split(",") if x.strip()]

    sigs = available()
    log(f"信号池 {len(sigs)}: {sigs}")
    global LAM_GRID, K_GRID
    if args.lam_grid:
        LAM_GRID = [float(x) for x in args.lam_grid.split(",")]
        log(f"λ 网格覆盖为 {LAM_GRID}")
    KSUF = ""
    if args.k_grid:
        K_GRID = [float(x) for x in args.k_grid.split(",")]
        KSUF = "_fine"
        log(f"K 网格覆盖为 {K_GRID}")
    for f in FOLDS:
        prepare(f, sigs)
    log("各折 Gram 矩阵就绪")

    fdf = load_factors(20210101, 20241231)
    E = Evaluator(sigs, fdf)

    if "0" in parts:
        B = baseline(E, sigs)
    if "A" in parts:
        RA = part_a(E, sigs, suffix="_fine" if args.lam_grid else "")
    if "B" in parts:
        lam_star = args.lam_star
        if lam_star is None:
            p = RA.pivot_table(index="tag", columns="fold", values="noprofile")[FOLDS]
            lam_star = float(p.mean(axis=1).idxmax().replace("lam", ""))
            log(f"Part A 均值最优 λ = {lam_star}")
        RB = part_b(E, sigs, lam_star, KSUF)
        p = RB.pivot_table(index="tag", columns="fold", values="noprofile")[FOLDS]
        # 相对 H01 基线的配对检验
        print("\n  相对 H01 基线的配对检验（可外推口径）")
        bmean = B.set_index("fold").loc[FOLDS, "noprofile"].to_numpy()
        for k in [f"K{K:.0f}" for K in K_GRID]:
            d = p.loc[k].to_numpy() - bmean
            sd = d.std(ddof=1)
            t = d.mean() / (sd / 2) if sd > 0 else np.nan
            print(f"    {k:<8} Δ {d.mean():+.6f}  t {t:+6.2f}  " +
                  "".join("+" if x > 0 else "-" for x in d))
    if "C" in parts:
        for lr in [float(x) for x in args.lam_refs.split(",") if x.strip()]:
            part_c(E, sigs, lr, KSUF)
    Joint().build(OUT)
    print(f"\n产出目录：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

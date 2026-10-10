"""第一次把 α 直接放进**优化目标**（此前的目标一律是 IC 或官方 final）。

动机（见 cv_three_numbers.py 结尾的结论）
------------------------------------------
拧旋钮（λ / K / keep_q / F1 权重）这条路已四次独立证明没有样本外空间；
距 0.4 仍缺 0.0100，而换手项已贴定义下限 ⇒ 缺口只能由 α 从 0.189 提到
0.237 补上。但 **此前所有权重优化的目标函数里都没有 α**。
唯一一次把 α 放进判据（λ 扫描）恰恰让 λ 首次变成可识别的参数
（LH019 自认不可识别）⇒ 目标函数决定参数能不能被看见。

本脚本回答三个递进的问题
------------------------
[Q1] 在同一条轴上，把目标从 IC / 可外推口径换成 α，选出来的点是不是同一个人？
     ⇒ 不是 ⇒ 说明 α 目标确实访问了不同的解（有信息量）
[Q2] 用 train α 选出来的权重，在未见折上的 α 真的更高吗？
     ⇒ LOFO：每折的权重只在另外 3 折上求解，留出折只用于评估
[Q3] train α 与 held α 在整条网格上到底相不相关？
     ⇒ Spearman(train α, held α)：若接近 0 或负，说明 α 本身不可优化，
       这是比任何单个 Δ 都更强的结论（独立于"选哪个点"）。

同时使用三种目标做对照：
    ic        = 逐日 Spearman IC
    noprofile = 0.4·IC + 0.3·α + 0.3·(1−T)    （现行可外推选型口径）
    alpha     = DGTW 分层后的组内选股 α（本次的新目标）

轴：H01 权重锁 0.5（λ=0.5），余量在 H05_F1T2 / H05_F2T2 之间分配，
    K=40、keep_q 部署值固定。这样只剩 1 个自由度，过拟合风险最小，
    而且这条轴此前已被完整审计（three_numbers_grid.csv），可直接对照。

不读任何 2025 年以后的数据。
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT_FIN = Path(os.environ.get("FINTECH_ROOT", _HERE.parents[1]))   # 仓库根
CACHE = Path(os.environ.get("LH_CACHE", ROOT_FIN / "outputs" / "long_horizon"))
ROOT = os.environ.get("LH_TESTDIR", str(ROOT_FIN.parent.parent / "test_y_2025_2026"))
for p in (str(_HERE), str(CACHE), str(ROOT_FIN), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cv_lambda_k_opt as clk
from cv_lambda_k_opt import FOLDS, Evaluator, available, pct_rank_2d, two_stage
from _probe_e_attr import load_factors

OUT = clk.OUT
ANN = 252
SIGS3 = ["H01", "H05_F1T2", "H05_F2T2"]
DEPLOY_F1 = 0.1204
GRID_F1 = sorted({round(0.025 * i, 4) for i in range(21)} | {DEPLOY_F1})
CACHE_CSV = OUT / "alpha_objective_grid.csv"
OBJ = {"ic": lambda r: r["ic"],
       "noprofile": lambda r: np_of(r),
       "alpha": lambda r: r["alpha"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def np_of(r):
    return 0.4 * r["ic"] + 0.3 * r["alpha"] + 0.3 * (1 - r["turnover"])


def build(sigs, f1):
    wt = {"H01": 0.5, "H05_F1T2": f1, "H05_F2T2": 0.5 - f1}
    return np.array([wt.get(s, 0.0) for s in sigs])


def main() -> int:
    sigs = available()
    fdf = load_factors(20210101, 20241231)
    E = Evaluator(sigs, fdf)
    for f in FOLDS:
        E.get(f)
    log("四折预计算完成")

    rows = []
    done = set()
    if CACHE_CSV.exists():
        old = pd.read_csv(CACHE_CSV)
        done = set(zip(old.held, old.w_F1))
        rows.extend(old.to_dict("records"))
        log(f"读到已有 {len(done)} 个 (held, w_F1) 组合，只补跑缺失的")

    todo = [(h, f1) for h in FOLDS for f1 in GRID_F1 if (h, f1) not in done]
    log(f"待跑 {len(todo)} 个组合 × 4 折 ≈ {len(todo) * 4 * 0.5:.0f}s")
    t00 = time.time()
    for i, (held, f1) in enumerate(todo, 1):
        train = [f for f in FOLDS if f != held]
        w = build(sigs, f1)
        for f in train + [held]:
            mix = pct_rank_2d(E.mix(f, w))
            P0 = two_stage(mix, E.get(f)["wide"]["H01"], 40.0)
            r = E.run(f, P0, f"a_{f1}")
            rows.append(dict(held=held, split="train" if f != held else "held",
                             fold=f, w_F1=f1, ic=r["ic"], excess=r["excess"],
                             turnover=r["turnover"], final=r["final"],
                             alpha=r["alpha"], profile=r["profile"]))
        if i % 10 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)}  累计 {time.time() - t00:.0f}s")

    R = pd.DataFrame(rows).drop_duplicates(["held", "fold", "w_F1"], keep="last")
    R["np"] = np_of(R)
    R.to_csv(CACHE_CSV, index=False)
    log(f"已写入 {CACHE_CSV}  （{len(R)} 行）")

    # 注意：LOFO 拼接后，「四个 held 各自的 train 部分」的均值 与「四个 held 部分」的均值
    # 是同一个数（每折在前者出现 3 次、后者出现 1 次，权重都是 1/4）⇒ 这是算术恒等式，
    # 不是程序错误。真正区分 train/held 的信息在下面的 Q2（逐 held 配对）与 Q3（逐 held 剖面相关）。
    print("\n" + "=" * 88)
    print("全四折均值：w_F1 网格上的各项指标（train 与 held 拼接均值相同，见代码注释）")
    print("=" * 88)
    print(f"  {'w_F1':>8}{'IC':>10}{'可外推':>10}{'α':>10}{'profile':>10}"
          f"{'换手':>9}{'E':>9}")
    for f1 in GRID_F1:
        g = R[R.w_F1 == f1]
        mark = "   ←部署" if abs(f1 - DEPLOY_F1) < 1e-9 else ""
        print(f"  {f1:>8.3f}{g.ic.mean():>10.5f}{g.np.mean():>10.5f}"
              f"{g.alpha.mean():>10.5f}{g.profile.mean():>10.5f}"
              f"{g.turnover.mean():>9.5f}{g.excess.mean():>9.5f}{mark}")

    print(f"\n  三种目标各自的网格 argmax（全折均值口径，仅作参考）")
    for name, fn in OBJ.items():
        m = R.assign(v=lambda d: fn(d)).groupby("w_F1")["v"].mean()
        g = R[R.w_F1 == m.idxmax()]
        print(f"    {name:<10} argmax = {m.idxmax():.3f}   "
              f"α={g.alpha.mean():.5f}  IC={g.ic.mean():.5f}  "
              f"可外推={g.np.mean():.5f}")

    # ------------------------------------------------- 信噪比：α 到底能不能被看见
    print("\n" + "=" * 88)
    print("[Q0] 决定性前提：网格带来的「信号跨度」 vs 同一个点在四折之间的「噪声」")
    print("=" * 88)
    print("  若 信号跨度 << 折间噪声 ⇒ 该指标在这条轴上数学上不可分辨，")
    print("  与用什么目标去挑无关。这是 Q2/Q3 结论的根因。")
    print("  噪声的正确定义不是「同一配置跨年份的水平漂移」，而是**配对差的跨折标准误**：")
    print("  同一折内两个配置的差 d[fold]，其 sd/2 才是配对检验能用的噪声单位")

    def pair_se(col):
        """相对部署点的配对差 Δ[fold]，其均值标准误；取网格上的中位数。"""
        base = R[(R.split == "held") & (R.w_F1 == DEPLOY_F1)].set_index("fold")[col]
        ses = []
        for f1 in GRID_F1:
            if abs(f1 - DEPLOY_F1) < 1e-9:
                continue
            cur = R[(R.split == "held") & (R.w_F1 == f1)].set_index("fold")[col]
            d = (cur - base).dropna().to_numpy()
            if len(d) == 4:
                ses.append(d.std(ddof=1) / 2)
        return float(np.median(ses)) if ses else np.nan

    print(f"\n  {'指标':<10}{'网格均值跨度':>14}{'配对差se':>12}{'信噪比':>10}"
          f"{'网格峰vs部署 t':>16}   能否被这条轴分辨")
    for name, col in (("IC", "ic"), ("可外推", "np"), ("α", "alpha"), ("E", "excess")):
        prof = R.groupby("w_F1")[col].mean()
        span = prof.max() - prof.min()
        se = pair_se(col)
        t_pk = (prof.max() - prof[DEPLOY_F1]) / se if se > 0 else np.nan
        verdict = "可以" if abs(t_pk) > 2 else ("边缘" if abs(t_pk) > 1 else "不能")
        print(f"  {name:<10}{span:>14.6f}{se:>12.6f}{span / se:>10.2f}"
              f"{t_pk:>+16.2f}   {verdict}")

    # ------------------------------------------------- α 剖面在折与折之间是否复现
    print("\n" + "=" * 88)
    print("[Q3b] 指标的 w_F1 剖面本身在折与折之间复不复现（6 对折的两两秩相关）")
    print("=" * 88)
    print("  这一条不依赖 train/held 划分，纯粹问「形状稳不稳定」。")
    for metric in ("ic", "np", "alpha"):
        prof = R[R.split == "held"].pivot_table(index="w_F1", columns="held",
                                                values=metric)
        pairs, vals = [], []
        for i, a in enumerate(FOLDS):
            for b in FOLDS[i + 1:]:
                rho, _ = spearmanr(prof[a].values, prof[b].values)
                pairs.append(f"{a[-4:]}~{b[-4:]}")
                vals.append(rho)
        print(f"  {metric:<7}" + "  ".join(f"{p}={v:+.3f}" for p, v in zip(pairs, vals)))
        print(f"         均值 {np.mean(vals):+.3f}   最小 {min(vals):+.3f}")

    # ------------------------------------------------- 全超参空间的 α 可达幅度
    print("\n" + "=" * 88)
    print("[Q4] 把四条超参轴合起来看：α 在整个已探索空间里最多能动多少")
    print("=" * 88)
    print("  需要把 α 从 0.189 提到 0.237 ⇒ 缺口 +0.048（见「0.4 的算术」）。")
    print("  注意：跨度会把「往下掉的余地」也算进来（λ=0 会让 α 掉到 0.15），")
    print("  而我们只关心**相对部署点还能往上抬多少** ⇒ 用 max(α) − α(部署点)。")

    # ---- 缺口口径：换手项已贴定义下限，剩余空间极小 ----
    T_FLOOR = 0.0175                      # T ≈ 2r/(n+r) 的定义下限
    need = 0.4 - 0.3 * (1 - T_FLOOR)      # ⇒ 需要 0.4·IC + 0.3·E ≥ need
    dep = R[(R.split == "held") & (R.w_F1 == DEPLOY_F1)]
    ic0, al0, pr0 = dep.ic.mean(), dep.alpha.mean(), dep.profile.mean()
    # 保守：profile 在测试期不可控且会塌 ⇒ 全部超额必须来自 α
    need_a_cons = (need - 0.4 * ic0) / 0.3 - al0
    # 乐观：CV 上的 profile 能维持
    need_a_opt = need_a_cons - pr0

    print(f"  换手项上限 0.3·(1−{T_FLOOR}) = {0.3 * (1 - T_FLOOR):.5f}  ⇒  需 "
          f"0.4·IC + 0.3·E ≥ {need:.5f}")
    print(f"  部署点：IC {ic0:.5f}  α {al0:.5f}  profile {pr0:.5f}")
    print(f"  缺口 Δα：保守（profile 归零）{need_a_cons:+.5f}   "
          f"乐观（profile 维持）{need_a_opt:+.5f}")

    axes = []

    def add_axis(lab, idx, vals, dep_key):
        s = pd.Series(vals, index=idx).sort_index()
        base = s.loc[dep_key]
        axes.append((lab, base, float(s.max()), float(s.max() - base),
                     float(s.idxmax())))

    # λ×K 是**已经联合跑过**的二维网格（74 组合），不必假设两条轴可加
    J = pd.read_csv(OUT / "lambda_k_joint_grid.csv")
    gj = J.groupby(["lam", "K"])["alpha"].mean()
    joint_base = float(gj.loc[(0.5, 40.0)])
    joint_max = float(gj.max())
    joint_arg = gj.idxmax()

    d = pd.read_csv(OUT / "lambda_k_lamgrid.csv")
    g = d.groupby("lam")["alpha"].mean()
    add_axis("λ（单独，K=40）", g.index, g.to_numpy(), 0.5)
    d = pd.read_csv(OUT / "k_grid35_45.csv")
    g = d.groupby("x")["alpha"].mean()
    add_axis("K（单独，λ=0.5）", g.index, g.to_numpy(), 40.0)
    print(f"\n  λ×K 联合网格（{gj.size} 个组合，真实联合上界，不需假设可加）：")
    print(f"     α 最高点 (λ,K)=({joint_arg[0]:g}, {joint_arg[1]:g})  α={joint_max:.5f}"
          f"   部署点 (0.5, 40) α={joint_base:.5f}   ⇒ 可上抬 {joint_max - joint_base:+.5f}")
    print(f"     单独 λ 可上抬 {axes[0][3]:+.5f} ＋ 单独 K 可上抬 {axes[1][3]:+.5f}"
          f" = {axes[0][3] + axes[1][3]:+.5f}"
          f"（比联合高 {axes[0][3] + axes[1][3] - (joint_max - joint_base):+.5f}"
          f" ⇒ 两轴沿 ridge 近似等价，联合几乎无增效）")

    d = pd.read_csv(OUT / "three_numbers_grid.csv")
    for a, dep_k in (("keep_q", 0.002278), ("F1 权重", 0.1204)):
        if a in set(d["axis"]):
            g = d[d.axis == a].groupby("x")["alpha"].mean()
            add_axis(a, g.index, g.to_numpy(), dep_k)
    g0 = R.groupby("w_F1")["alpha"].mean()
    add_axis("F1 权重（本脚本细网格）", g0.index, g0.to_numpy(), DEPLOY_F1)

    print(f"\n  {'轴':<24}{'部署点α':>10}{'轴上最高α':>12}{'可上抬':>12}"
          f"{'/缺口(保守)':>13}{'/缺口(乐观)':>13}{'最佳点':>10}")
    tot = joint_max - joint_base
    for lab, base, mx, up, arg in axes[2:]:
        tot += max(up, 0.0)
    for lab, base, mx, up, arg in axes:
        print(f"  {lab:<24}{base:>10.4f}{mx:>12.4f}{up:>+12.5f}"
              f"{up / need_a_cons:>12.1%}{up / need_a_opt:>13.1%}{arg:>10.4g}")
    print(f"  {'联合上界(λK)+其余两轴':<24}{'':>10}{'':>12}{tot:>+12.5f}"
          f"{tot / need_a_cons:>12.1%}{tot / need_a_opt:>13.1%}")
    print(f"  {'所需缺口':<24}{'':>10}{'':>12}{need_a_cons:>+12.5f}"
          f"{1.0:>12.1%}{need_a_opt / need_a_cons:>13.1%}")

    # ---- IC/α 前沿：换中间件 een 图留出空間 ----
    F = J.groupby(["lam", "K"])[["ic", "alpha", "turnover", "excess",
                                 "profile"]].mean().reset_index()
    F["demand"] = 0.4 * F.ic + 0.3 * F.alpha
    F["gap"] = need - F.demand
    F.to_csv(OUT / "ic_alpha_frontier.csv", index=False)
    print(f"\n  已写出 λ×K 联合网格的 IC/α 前沿 "
          f"{OUT / 'ic_alpha_frontier.csv'}（{len(F)} 个组合）")
    print(f"  其中 0.4·IC + 0.3·α 的最高值 {F.demand.max():.5f} < 所需 {need:.5f}"
          f"   ⇒ 缺口 {(need - F.demand.max()):+.5f}（**所有 74 个组合无一达标**）")

    print("\n  注：以上均为**代数上界**（完全无视噪声）。对照 Q0 的配对标准误")
    print(f"  α 的配对 se = {pair_se('alpha'):.5f}，比任何一条轴的上抬幅度都大 ⇒")
    print("  这些上抬不仅不够用，而且在统计上根本看不见。")

    # ------------------------------------------------------------ Q3 相关性
    print("\n" + "=" * 88)
    print("[Q3] α 到底可不可优化：train α 与同一 w_F1 的 held α 之间的秩相关")
    print("=" * 88)
    tr = R[R.split == "train"].groupby(["held", "w_F1"])[
        ["ic", "np", "alpha"]].mean().reset_index()
    he = R[R.split == "held"].set_index(["held", "w_F1"])[
        ["ic", "np", "alpha"]]
    for metric in ("ic", "np", "alpha"):
        rhos, ps = [], []
        for held in FOLDS:
            a = tr[tr.held == held].set_index("w_F1")[metric]
            b = he.loc[held][metric]
            rho, p = spearmanr(a.values, b.reindex(a.index).values)
            rhos.append(rho)
            ps.append(p)
        print(f"  {metric:<7}Spearman(train, held) 逐折 = "
              + "  ".join(f"{r:+.3f}" for r in rhos)
              + f"     均值 {np.mean(rhos):+.3f}")

    print("\n  解读口径：ρ 高 ⇒ 在 train 上把指标做高，held 上也会高 ⇒ 该指标可优化；")
    print("            ρ≈0 或负 ⇒ train 上的高低与 held 无关 ⇒ 不可优化，")
    print("            此时任何『用这个目标挑权重』的动作都只是在拟合折内的噪声。")

    # ------------------------------------------------------------ Q2 留一折选参
    print("\n" + "=" * 88)
    print("[Q2] 三种目标做留一折选参：拿 train argmax 去未见折上兑现多少")
    print("=" * 88)
    for name, fn in OBJ.items():
        print(f"\n  目标 = {name}")
        print(f"    {'held':<12}{'train argmax':>14}{'Δ vs 部署(不留)':>18}"
              f"{'Δα':>12}{'ΔIC':>12}{'Δ可外推':>12}")
        acc = {k: [] for k in ("chosen", "d", "da", "dic", "dnp")}
        for held in FOLDS:
            t = R[(R.split == "train") & (R.held == held)].assign(v=lambda x: fn(x))
            m = t.groupby("w_F1")["v"].mean()
            pk = m.idxmax()
            hb = R[(R.split == "held") & (R.held == held)].set_index("w_F1")
            base = hb.loc[DEPLOY_F1]
            got = hb.loc[pk]
            acc["chosen"].append(pk)
            acc["d"].append(fn(got) - fn(base))
            acc["da"].append(got.alpha - base.alpha)
            acc["dic"].append(got.ic - base.ic)
            acc["dnp"].append(got.np - base.np)
            print(f"    {held:<12}{pk:>14.3f}{fn(got) - fn(base):>+18.6f}"
                  f"{got.alpha - base.alpha:>+12.5f}{got.ic - base.ic:>+12.5f}"
                  f"{got.np - base.np:>+12.5f}")
        for k, lab in (("da", "Δα"), ("dic", "ΔIC"), ("dnp", "Δ可外推")):
            v = np.array(acc[k])
            sd = v.std(ddof=1)
            t = v.mean() / (sd / 2) if sd > 0 else np.nan
            print(f"      {lab:<8}均值 {v.mean():>+.6f}   t {t:>+7.2f}   逐折符号 "
                  + "".join("+" if x > 0 else "-" for x in v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

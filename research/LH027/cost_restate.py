"""
LH027-A —— 净收益口径下的机制重估（不重训，纯重算已有四折结果）

问题：官方 final = 0.4·IC + 0.3·E + 0.3·(1−T) 里的换手罚项 0.3·(1−T)，
      与真实交易成本是不是同一回事？若不是，官方评分会系统性地偏袒哪一类策略？

成本模型
--------
Jaccard 换手 T = |A△B| / |A∪B|。设名单规模 n，每日换出 r 只、换入 r 只，则
    |A△B| = 2r,  |A∪B| = n + r   ⇒   T = 2r/(n+r)  ⇒  r/n = T/(2−T)
（与 n 无关，这一点很关键：不需要知道每天的精确名单规模。）

年化双边换手率 = 252 · 2r/n = 504 · T/(2−T)
年化交易成本   = 504 · T/(2−T) · c          （c = 单边交易成本率）
净超额         = E − 504 · T/(2−T) · c

全流程只用 2021–2024 四个 CV 折。
"""
import numpy as np
import pandas as pd

ANN = 252.0


def ann_turnover(t):
    """Jaccard 换手 -> 年化双边换手率"""
    t = np.asarray(t, dtype=float)
    return 504.0 * t / (2.0 - t)


def net_excess(e, t, c):
    """年化超额 - 年化交易成本"""
    return np.asarray(e, dtype=float) - ann_turnover(t) * c


def implied_cost(slope):
    """把「官方/真实」对 T 的边际敏感度反解成单边成本率。
    d(cost)/dT = 504c · 2/(2−T)^2 ；在 T≈0.018 处 ≈ 256.7·c
    """
    return slope / 256.7


FILES = {
    "LH023_lamK": ("experiments/LH023/lambda_k_joint_grid.csv", ["lam", "K"]),
    "LH025_band": ("experiments/LH025/band_freeze_grid.csv", ["mode", "param"]),
    "LH020_fam": ("experiments/LH020_twostage_folds.csv", ["tag"]),
}
COSTS = [0.0010, 0.0025, 0.0040]      # 单边成本率三档：乐观 / 中性 / 悲观
FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]


def load(path, keys):
    d = pd.read_csv(path)
    d = d[d["fold"].isin(FOLDS)]
    g = d.groupby(keys)[["ic", "excess", "turnover", "final", "alpha", "profile"]].mean()
    g["n"] = d.groupby(keys).size()
    return g[g["n"] == len(FOLDS)]


def main():
    print("=" * 78)
    print("LH027-A  净收益口径重估")
    print("=" * 78)

    frames = {}
    for name, (path, keys) in FILES.items():
        try:
            frames[name] = load(path, keys)
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    for name, g in frames.items():
        g["aturn"] = ann_turnover(g["turnover"])
        for c in COSTS:
            g[f"net_{c:.4f}"] = net_excess(g["excess"], g["turnover"], c)
        print(f"\n### {name}  ({len(g)} 个配置, 四折均值) ###")

        dep = g.loc[g["final"].idxmax()]
        print(f"官方 final argmax : final={dep['final']:.5f}  IC={dep['ic']:.5f} "
              f"E={dep['excess']:.5f} T={dep['turnover']:.5f} alpha={dep['alpha']:.5f} "
              f"年化双边换手={dep['aturn']:.2f}")
        for c in COSTS:
            col = f"net_{c:.4f}"
            b = g.loc[g[col].idxmax()]
            same = "（同上）" if b.name == dep.name else ""
            print(f"  净超额 argmax c={c:.4f}: net={b[col]:.5f}  E={b['excess']:.5f} "
                  f"T={b['turnover']:.5f} alpha={b['alpha']:.5f} "
                  f"final={b['final']:.5f} {same}")

    # ---------- lkeep 轴：Δα/ΔT 与临界成本 ----------
    print("\n" + "=" * 78)
    print("沿 LH025 lkeep 轴的边际兑换率与临界成本")
    print("=" * 78)
    bf = frames.get("LH025_band")
    if bf is not None:
        lk = bf.loc["lkeep"].sort_index()
        base = bf.loc[("V0_base",)]
        print(f"{'param':>8} {'T':>8} {'alpha':>8} {'E':>8} {'final':>8} " +
              " ".join(f"net{c:.4f}" for c in COSTS))
        for p, r in lk.iterrows():
            print(f"{p:>8.4f} {r['turnover']:8.5f} {r['alpha']:8.5f} {r['excess']:8.5f} "
                  f"{r['final']:8.5f} " + " ".join(f"{r[f'net_{c:.4f}']:8.5f}" for c in COSTS))
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        print(f"{'deploy':>8} {base['turnover']:8.5f} {base['alpha']:8.5f} "
              f"{base['excess']:8.5f} {base['final']:8.5f} " +
              " ".join(f"{base[f'net_{c:.4f}']:8.5f}" for c in COSTS))

        print("\n  Δα/ΔT（相对部署点）:")
        for p, r in lk.iterrows():
            da = r["alpha"] - base["alpha"]
            de = r["excess"] - base["excess"]
            dt = r["turnover"] - base["turnover"]
            if abs(dt) < 1e-6:
                continue
            print(f"    q={p:<8.4f} ΔT={dt:+.5f} Δα={da:+.5f} Δα/ΔT={da/dt:7.3f} "
                  f"ΔE={de:+.5f} ΔE/ΔT={de/dt:7.3f} 临界 c*={implied_cost(da/dt):.5f}")

    # ---------- 无差异斜率：官方 vs 真实 ----------
    # 官方: dU = 0.3 dE − 0.3 dT  ⇒ 无差异斜率 dE/dT = 0.3/0.3 = 1.0
    # 真实: dU = 1.0 dE − 256.7c dT ⇒ 无差异斜率 dE/dT = 256.7c
    print("\n" + "=" * 78)
    print("无差异斜率：官方评分 vs 真实经济效用")
    print("=" * 78)
    print("  官方 U = 0.4IC + 0.3E + 0.3(1−T)  ⇒  dU = 0.3dE − 0.3dT  ⇒ 无差异斜率 dE/dT = 1.000")
    print("  真实 U = E − cost(T)              ⇒  dU = 1.0dE − 256.7c·dT ⇒ 无差异斜率 dE/dT = 256.7c")
    print("  官方的 1.000 对应真实成本 c* = 1.000/256.7 = "
          f"{1.0/256.7:.5f}（单边）\n")
    print("  真实单边成本参考：佣金 ~0.00025 + 印花税(双边摊薄 ~0.00025) + 冲击 0.0005~0.0015")
    print("                    ⇒ 约 0.0010 ~ 0.0020")
    print(f"  ⇒ 官方隐含 {1.0/256.7:.5f}，是真实区间的 "
          f"{(1.0/256.7)/0.0020:.1f}~{(1.0/256.7)/0.0010:.1f} 倍"
          " ⇒ 官方对换手的厌恶显著强于真实成本所要求的")
    print("     即：只要某策略的 ΔE/ΔT 落在 (256.7c, 1.000) 区间内，")
    print("     真实世界该做，官方评分却判它不该做。")

    # ---------- 前瞻口径：profile 归零（LH020 已证其会塌） ----------
    print("\n" + "=" * 78)
    print("前瞻口径：若 profile 断档（LH020 §3 已用半年频率验证），E' = α")
    print("=" * 78)
    for name, g in frames.items():
        gg = g.copy()
        for c in COSTS:
            gg[f"netA_{c:.4f}"] = gg["alpha"] - ann_turnover(gg["turnover"]) * c
        a = gg.loc[gg["final"].idxmax()]
        print(f"\n### {name} ###")
        print(f"  官方 final argmax : alpha={a['alpha']:.5f} T={a['turnover']:.5f} "
              f"E={a['excess']:.5f} profile={a['profile']:.5f} final={a['final']:.5f}")
        for c in COSTS:
            col = f"netA_{c:.4f}"
            b = gg.loc[gg[col].idxmax()]
            same = "（同上）" if b.name == a.name else ""
            print(f"    c={c:.4f} 净α argmax: netα={b[col]:.5f} alpha={b['alpha']:.5f} "
                  f"T={b['turnover']:.5f} E={b['excess']:.5f} final={b['final']:.5f} {same}")

    if bf is not None:
        lk = bf.loc["lkeep"].sort_index()
        if isinstance(base, pd.Series):
            b0 = base
        else:
            b0 = base.iloc[0]
        print("\n  lkeep 轴在前瞻口径下的边际（相对部署点）:")
        for p, r in lk.iterrows():
            da = r["alpha"] - b0["alpha"]
            dt = r["turnover"] - b0["turnover"]
            if abs(dt) < 1e-6:
                continue
            print(f"    q={p:<8.4f} ΔT={dt:+.5f} Δα={da:+.5f} Δα/ΔT={da/dt:7.3f} "
                  f"临界 c*={implied_cost(da/dt):.5f}  "
                  f"c=0.0025 时净 = {da - 256.7*0.0025*dt:+.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

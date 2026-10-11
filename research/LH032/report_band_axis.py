"""LH032 band 段轴分析报告：读取 band_axis_grid.csv，输出判据表。

用法
  python research/LH032/report_band_axis.py            # 默认 band_axis_grid.csv
  python research/LH032/report_band_axis.py ext        # 读 band_axis_grid_ext.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[2]
F = ["wf2021", "wf2022", "wf2023", "confirm2024"]
pd.set_option("display.width", 340)

suffix = sys.argv[1] if len(sys.argv) > 1 else ""
if suffix.endswith(".csv"):
    fn = ROOT / "experiments" / "LH032" / suffix
elif suffix:
    fn = ROOT / "experiments" / "LH032" / f"band_axis_grid_{suffix}.csv"
else:
    fn = ROOT / "experiments" / "LH032" / "band_axis_grid.csv"
M = pd.read_csv(fn)
B = M[(M.band == "H01") & (M.phi_amt == 0) & (M.phi_vol == 0)].set_index("fold").reindex(F)
if B.isna().all().all():
    # 扩展网格不含 H01 基线 ⇒ 回落到主表取部署点
    M0 = pd.read_csv(ROOT / "experiments" / "LH032" / "band_axis_grid.csv")
    B = M0[(M0.band == "H01") & (M0.phi_amt == 0) & (M0.phi_vol == 0)].set_index("fold").reindex(F)
    print(f"# 基线取自 band_axis_grid.csv（本文件无 H01 基线）")


def row(band, a, v, m=None):
    q = M[(M.band == band) & (np.isclose(M.phi_amt, a)) & (np.isclose(M.phi_vol, v))]
    q = q.set_index("fold").reindex(F)
    return q if m is None else q[m]


def df_(k):
    return row(*k, m="final") - B["final"]


KEYS = [tuple(x) for x in
        M[["band", "phi_amt", "phi_vol"]].drop_duplicates().itertuples(index=False, name=None)]

print(f"# 数据源 {fn.name}   配置数 {len(KEYS)}")
print()
print("=== 部署点 逐折（自检） ===")
print(B[["ic", "excess", "turnover", "final", "alpha_vol", "prof_amt", "hold_days"]].round(5).to_string())
print()

rows = []
for k in KEYS:
    dI = row(*k, m="ic") - B["ic"]
    dE = row(*k, m="excess") - B["excess"]
    dT = row(*k, m="turnover") - B["turnover"]
    dF = df_(k)
    dA = row(*k, m="alpha_vol") - B["alpha_vol"]
    dP = row(*k, m="prof_amt") - B["prof_amt"]
    dH = row(*k, m="hold_days") - B["hold_days"]
    rows.append(dict(band=k[0], a=k[1], v=k[2],
                     dIC=dI.mean(), dE=dE.mean(), dE_min=dE.min(),
                     dT=dT.mean(), dalpha=dA.mean(), dprof=dP.mean(), dhold=dH.mean(),
                     dfinal=dF.mean(), df_min=dF.min(),
                     sign="".join("+" if x > 0 else "-" for x in dF),
                     t=dF.mean() / (dF.std() / 2) if dF.std() > 0 else np.inf))
R = pd.DataFrame(rows)


def dominated(r):
    better = ((R.dT <= r.dT + 1e-12) & (R.dE >= r.dE - 1e-12) & (R.dIC >= r.dIC - 1e-12))
    same = (np.isclose(R.dT, r.dT) & np.isclose(R.dE, r.dE) & np.isclose(R.dIC, r.dIC))
    return bool((better & ~same).any())


R["pareto"] = [not dominated(r) for _, r in R.iterrows()]
print("=== 全表（按 Δfinal）===")
print(R.sort_values("dfinal", ascending=False).round(5).to_string(index=False))
print()
print("=== 帕累托前沿（ΔT 更小 + ΔE 更大 + ΔIC 更大）===")
print(R[R.pareto].sort_values("dfinal", ascending=False).round(5).to_string(index=False))
print()

if ("H01", -0.100, 0.040) in KEYS:
    m2 = df_(("H01", -0.100, 0.040))
    print("=== 配对 t：各配置 vs M2(H01,-0.100,+0.040) ===")
    for k in R.sort_values("dfinal", ascending=False).head(8).itertuples(index=False):
        o = df_((k.band, k.a, k.v))
        d = o - m2
        tt = ttest_rel(o, m2)
        print(f"  {k.band:9s}{k.a:+.3f}{k.v:+.3f}  "
              + " ".join(f"{x:+.5f}" for x in d)
              + f"  mean {d.mean():+.5f}  t {tt.statistic:+.2f}  p {tt.pvalue:.3f}"
              + f"  胜 {int((d > 0).sum())}/4")
    print()

print("=== LOFO：三折 argmax(Δfinal) -> 未见折 ===")
for h in F:
    tr = [f for f in F if f != h]
    best = max(KEYS, key=lambda k: df_(k)[tr].mean())
    vE = (row(*best, m="excess") - B["excess"])[h]
    vT = (row(*best, m="turnover") - B["turnover"])[h]
    print(f"  留出 {h:11s} 选中 band={best[0]:9s} a={best[1]:+.3f} v={best[2]:+.3f}"
          f" -> 未见折 Δfinal {df_(best)[h]:+.5f}  ΔE {vE:+.5f}  ΔT {vT:+.5f}")
print()
print("=== LOFO：以三折「最小折 Δfinal」为目标 ===")
for h in F:
    tr = [f for f in F if f != h]
    best = max(KEYS, key=lambda k: df_(k)[tr].min())
    print(f"  留出 {h:11s} 选中 band={best[0]:9s} a={best[1]:+.3f} v={best[2]:+.3f}"
          f" -> 未见折 Δfinal {df_(best)[h]:+.5f}")
print()
print("=== LOFO：固定 band 段，只选倾斜 ===")
for h in F:
    tr = [f for f in F if f != h]
    cand = [k for k in KEYS if k[0] == "H05_F2T2"]
    if cand:
        best = max(cand, key=lambda k: df_(k)[tr].mean())
        print(f"  留出 {h:11s} (band=H05_F2T2) 选中 a={best[1]:+.3f} v={best[2]:+.3f}"
              f" -> 未见折 Δfinal {df_(best)[h]:+.5f}")
print()

print("=== 层暴露：log_amt 十层占比（L0=最小成交额）===")
cols = [f"E_L{k}" for k in range(10)]
b0 = B[cols].mean()
print("  DEPLOY :", " ".join(f"{y:.3f}" for y in b0.values), f"| L0+L1 {b0['E_L0'] + b0['E_L1']:.3f}")
for r in R.sort_values("dfinal", ascending=False).head(6).itertuples(index=False):
    x = M[(M.band == r.band) & (np.isclose(M.phi_amt, r.a)) & (np.isclose(M.phi_vol, r.v))]
    x = x.set_index("fold").reindex(F)[cols].mean()
    print(f"  {r.band:9s}{r.a:+.3f}{r.v:+.3f}:", " ".join(f"{y:.3f}" for y in x.values),
          f"| L0+L1 {x['E_L0'] + x['E_L1']:.3f}")
print()
print("=== 层暴露：vol60 十层占比（L9=最高波动）===")
cols2 = [f"V_L{k}" for k in range(10)]
b1 = B[cols2].mean()
print("  DEPLOY :", " ".join(f"{y:.3f}" for y in b1.values), f"| L8+L9 {b1['V_L8'] + b1['V_L9']:.3f}")
for r in R.sort_values("dfinal", ascending=False).head(6).itertuples(index=False):
    x = M[(M.band == r.band) & (np.isclose(M.phi_amt, r.a)) & (np.isclose(M.phi_vol, r.v))]
    x = x.set_index("fold").reindex(F)[cols2].mean()
    print(f"  {r.band:9s}{r.a:+.3f}{r.v:+.3f}:", " ".join(f"{y:.3f}" for y in x.values),
          f"| L8+L9 {x['V_L8'] + x['V_L9']:.3f}")

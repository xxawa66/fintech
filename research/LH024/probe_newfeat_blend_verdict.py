"""三个判据量：折级聚合、留一窗、曲线平滑度。"""

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

import numpy as np, pandas as pd
from math import comb
W = pd.read_csv("cv_design_audit/newfeat_blend_windows.csv")
pd.set_option("display.width", 340)
W["fold"] = W.win.str.replace(r"[AB]$", "", regex=True)
base = W[W.w == 1.0].set_index("win")
for m in ("alpha", "ic", "turnover", "profile", "final"):
    W[f"d{m}"] = W[m] - W.win.map(base[m])
W["strict"] = 0.4 * W.dic - 0.3 * W.dturnover
W["withalpha"] = 0.4 * W.dic + 0.3 * W.dalpha - 0.3 * W.dturnover

print("=" * 80)
print("① 折级聚合（先 A/B 平均再算，n=4 折；半窗之间共享训练数据，不是独立样本）")
print("=" * 80)
f = W.groupby(["fold", "w"])[["dalpha", "dic", "strict"]].mean().reset_index()
for m in ("dalpha", "dic"):
    p = f.pivot_table(index="fold", columns="w", values=m)
    print(f"\n  --- 折级 {m} ---")
    print(p.round(5).to_string())
    pos = (p > 0).sum()
    print("    各 w 的正折数:", " ".join(f"{w}:{pos[w]}" for w in p.columns))

print("\n" + "=" * 80)
print("② 留一窗（LOOW）：w=0.6 的 with-alpha 均值，去掉任一窗后还剩多少（n=8 窗）")
print("=" * 80)
for wq in (0.4, 0.5, 0.6, 0.7):
    sub = W[W.w == wq].set_index("win")["withalpha"]
    print(f"  w={wq}: 全 8 窗均值 {sub.mean():+.6f} | 正 {int((sub>0).sum())}/8")
    for i in sub.index:
        rest = sub.drop(i)
        print(f"      去 {i:13s} → {rest.mean():+.6f}（正 {int((rest>0).sum())}/7）")

print("\n" + "=" * 80)
print("③ 曲线平滑度：Δα 相邻 w 的二阶差分幅度 vs 效应高度")
print("=" * 80)
p = W.pivot_table(index="win", columns="w", values="dalpha").sort_index()
d2 = p.diff(axis=1).diff(axis=1)
print(f"  |二阶差分| 均值 {np.nanmean(np.abs(d2.to_numpy())):.5f}  最大 {np.nanmax(np.abs(d2.to_numpy())):.5f}")
print(f"  效应高度（各窗 |Δα| 最大）均值 {np.nanmean(np.nanmax(np.abs(p.to_numpy()),axis=1)):.5f}")
print("  ⇒ 若二阶差分 ≈ 效应高度，说明曲线由噪声主导，内部极值不可信。")

print("\n" + "=" * 80)
print("④ 判据①②的直接回答")
print("=" * 80)
g = W.groupby(["fold", "w"])[["dalpha", "dic", "strict", "withalpha"]].mean().reset_index()
q24 = g[g.fold == "confirm2024"].set_index("w")["dalpha"]
q23 = g[g.fold == "wf2023"].set_index("w")["dalpha"]
print("  w    confirm2024Δα   wf2023Δα    ②成立?   8窗strict  8窗withalpha")
s8 = W.groupby("w")[["strict", "withalpha"]].mean()
for w in sorted(set(W.w)):
    ok = (q24[w] >= -0.005) and (q23[w] >= 0.02)
    print(f"  {w:.1f}   {q24[w]:+.5f}       {q23[w]:+.5f}    {'✓' if ok else ' '}      "
          f"{s8.loc[w,'strict']:+.5f}    {s8.loc[w,'withalpha']:+.5f}")

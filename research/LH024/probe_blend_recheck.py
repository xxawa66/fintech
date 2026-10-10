"""对「w 仍有可取之处」这条反驳的逐项核查（不重训，只读已落盘的表）。

用户的三条论点
--------------
① 2022 折的 α 是多少、相对纯 H01 下降百分之几？
② w ∈ [0.6, 0.8] 时，剔除 2023 后是否仍有提升？
③ 「2023 年的稳定性增强太多」——2023 折的提升是否等价于整体折间更稳？

口径固定：基线 = w=1（纯 H01）。Δα/ΔIC 全部是相对该基线的增量。
"""

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

import numpy as np
import pandas as pd

OUT = "cv_design_audit/"
ANN = 252
pd.set_option("display.width", 340)

W = pd.read_csv(OUT + "newfeat_blend_windows.csv")
D = pd.read_csv(OUT + "newfeat_blend_daily.csv")
M = pd.read_csv(OUT + "newfeat_blend_daymetrics.csv")
W["fold"] = W.win.str.replace(r"[AB]$", "", regex=True)
WS = sorted(W.w.unique())

base_alpha_fold = W[W.w == 1.0].groupby("fold")["alpha"].mean()

print("=" * 92)
print("① 逐折 α 绝对值（行=折，列=w）")
print("=" * 92)
piv = W.pivot_table(index="fold", columns="w", values="alpha")
print(piv.round(6).to_string())

print("\n   相对纯 H01（w=1）的变化百分比：")
pct = (piv.div(piv[1.0], axis=0) - 1.0) * 100
print(pct.round(2).to_string())

print("\n   2022 折：", "  ".join(
    f"w={w:.1f}: {piv.loc['wf2022', w]:.4f} ({pct.loc['wf2022', w]:+.1f}%)"
    for w in (1.0, 0.8, 0.7, 0.6, 0.5)))

print("\n" + "=" * 92)
print("② 剔除 2023 后是否还有提升（三条口径：折级 / 8 窗 / 逐日）")
print("=" * 92)
d = W.copy()
for m in ("alpha", "ic", "turnover"):
    d["d" + m] = d[m] - d.win.map(W[W.w == 1.0].set_index("win")[m])

fold_lv = d.groupby(["fold", "w"])[["dalpha", "dic", "dturnover"]].mean().reset_index()
print("\n  [a] 折级 Δα（n=4 折，半窗已先平均）")
print(fold_lv.pivot_table(index="fold", columns="w", values="dalpha").round(5).to_string())
keep = fold_lv[fold_lv.fold != "wf2023"].groupby("w")[["dalpha", "dic"]].mean()
print("\n  剔除 wf2023 后的 3 折均值：")
print(keep.round(6).to_string())

win_lv = d.groupby("w")[["dalpha", "dic"]].mean()
win_no23 = d[d.fold != "wf2023"].groupby("w")[["dalpha", "dic"]].mean()
print("\n  [b] 8 窗均值 vs 剔除 2023 的 6 窗")
cmp = pd.DataFrame({"8窗Δα": win_lv.dalpha, "剔23六窗Δα": win_no23.dalpha,
                    "8窗ΔIC": win_lv.dic, "剔23六窗ΔIC": win_no23.dic})
print(cmp.round(6).to_string())


def tt(x):
    x = pd.Series(x).dropna()
    n = len(x)
    return (x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 5 else np.nan), x.mean(), n


A = D.pivot_table(index=["win", "trade_date"], columns="w", values="alpha")
I = M.pivot_table(index=["win", "trade_date"], columns="w", values="ic")
is23 = A.index.get_level_values(0).to_series().str.contains("2023").to_numpy()
m23 = pd.Series(is23, index=A.index)
print("\n  [c] 逐日配对（681 天 / 剔 2023 的 511 天 / 仅 2023 的 170 天）")
rows = []
for w in WS[:-1]:
    r = {"w": w}
    for lbl, sel in (("all", None), ("ex23", ~m23.reindex(A.index).to_numpy()),
                     ("only23", m23.reindex(A.index).to_numpy())):
        da = (A[w] - A[1.0]).dropna()
        di = (I[w] - I[1.0]).dropna()
        if sel is not None:
            mk = pd.Series(sel, index=A.index)
            da = da[mk.reindex(da.index).to_numpy()]
            di = di[mk.reindex(di.index).to_numpy()]
        ta, ma, n = tt(da)
        ti, mi, _ = tt(di)
        r[f"Δα_{lbl}"] = ma * ANN
        r[f"tα_{lbl}"] = ta
        r[f"ΔIC_{lbl}"] = mi
        r[f"tIC_{lbl}"] = ti
    rows.append(r)
P = pd.DataFrame(rows)
print(P[["w", "Δα_all", "tα_all", "Δα_ex23", "tα_ex23", "Δα_only23", "tα_only23"]].round(6).to_string(index=False))
print()
print(P[["w", "ΔIC_all", "tIC_all", "ΔIC_ex23", "tIC_ex23"]].round(6).to_string(index=False))

print("\n" + "=" * 92)
print("③ 折间稳定性：跨 4 折 α 的标准差（越小越稳）")
print("=" * 92)
for w in (1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0):
    v = piv[w]
    print(f"  w={w:.1f}: 跨折 α 均值 {v.mean():.4f}   sd {v.std(ddof=1):.4f}   "
          f"最弱折 {v.min():.4f}（{v.idxmin()}）  强弱极差 {v.max()-v.min():.4f}")

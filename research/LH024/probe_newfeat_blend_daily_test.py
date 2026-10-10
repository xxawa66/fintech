"""逐日配对检验：每个 w 相对纯 H01 的 Δα / ΔIC，含剔除 2023。"""

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
D = pd.read_csv("cv_design_audit/newfeat_blend_daily.csv")
M = pd.read_csv("cv_design_audit/newfeat_blend_daymetrics.csv")
D["trade_date"] = D.trade_date.astype("int64"); M["trade_date"] = M.trade_date.astype("int64")
pd.set_option("display.width", 340)
ANN = 252

A = D.pivot_table(index=["win", "trade_date"], columns="w", values="alpha")
I = M.pivot_table(index=["win", "trade_date"], columns="w", values="ic")
T = M.pivot_table(index=["win", "trade_date"], columns="w", values="turnover")
ref = 1.0

def tst(x):
    x = x.dropna(); n = len(x)
    if n < 5: return np.nan, np.nan, n
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), x.mean(), n

print("=" * 88)
print("① 逐日配对（n=681 天，11 个窗口串起来）：Δα 相对纯 H01")
print("=" * 88)
print("  w     Δα 均值(年化)   t(全日)   |  剔除 2023 六窗: 年化    t    |  仅 2023 两窗: 年化    t")
d23 = A.index.get_level_values(0).to_series().str.contains("2023").to_numpy()
w23 = pd.Series(d23, index=A.index)
for w in sorted(set(D.w)):
    if w == ref: continue
    d = (A[w] - A[ref]).dropna()
    t_all, m_all, n = tst(d)
    d_x = d[~w23.reindex(d.index).to_numpy()]
    d_2 = d[w23.reindex(d.index).to_numpy()]
    t_x, m_x, nx = tst(d_x); t_2, m_2, n2 = tst(d_2)
    print(f"  {w:.1f}   {m_all*ANN:+.5f}      {t_all:+.2f}     |   {m_x*ANN:+.5f}   {t_x:+.2f}  |  {m_2*ANN:+.5f}   {t_2:+.2f}")

print("\n" + "=" * 88)
print("② 逐日配对：ΔIC 相对纯 H01（n=681 天）")
print("=" * 88)
print("  w     ΔIC 均值     t(全日)   |  剔除 2023: 均值      t     |  正日比(全日)")
for w in sorted(set(D.w)):
    if w == ref: continue
    d = (I[w] - I[ref]).dropna()
    t_all, m_all, n = tst(d)
    d_x = d[~w23.reindex(d.index).to_numpy()]
    t_x, m_x, nx = tst(d_x)
    print(f"  {w:.1f}   {m_all:+.6f}   {t_all:+.2f}     |  {m_x:+.6f}   {t_x:+.2f}  |  {(d>0).mean():.3f}")

print("\n" + "=" * 88)
print("③ w=0.6 峰的稳定性：逐窗口 Δα 的 t 与自由度")
print("=" * 88)
for wq in (0.4, 0.6):
    print(f"  --- w={wq} ---")
    for win in sorted(set(D.win)):
        d = (A[wq].xs(win) - A[ref].xs(win)).dropna()
        t_, m_, n_ = tst(d)
        print(f"     {win:13s} n={n_:3d} Δα年化 {m_*ANN:+.5f} t={t_:+.2f}")

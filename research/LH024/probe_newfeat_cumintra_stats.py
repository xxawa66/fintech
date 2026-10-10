"""cumintra / 非cumintra / 全8列 三条对照线的逐日配对检验（不重训，只读表）。

对照设计
--------
基线固定为 L2（40 列 V1，点式回归，同一候选池/训练行/协议）。
三条受控增量：
    L2C = L2 + cumintra{5,20,60}             （3 列，probe_newfeat_cumintra.py）
    L2N = L2 + gapspread20+corrmkt60/120+amihud20/60（5 列，probe_newfeat_nocum.py）
    L2X = L2 + 8 列（= L2C ∪ L2N，归档 newfeat_halves.csv）

每个对照报告 Δα / ΔIC 的逐日配对 t：
    全样本（8 窗串联，681 天）/ 剔除 2023 六窗（511 天）/ 仅 2023 两窗（170 天）
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


def tt(x):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), x.mean(), n


def load(pref):
    D = pd.read_csv(OUT + pref + "_daily.csv")
    M = pd.read_csv(OUT + pref + "_daymetrics.csv")
    for c in ("trade_date",):
        D[c] = D[c].astype("int64")
        if c in M:
            M[c] = M[c].astype("int64")
    return D, M


def pair_table(D, M, base, alt, label):
    A = D.pivot_table(index=["win", "trade_date"], columns="tag", values="alpha")
    I = M.pivot_table(index=["win", "trade_date"], columns="tag", values="ic")
    is23 = A.index.get_level_values(0).to_series().str.contains("2023").to_numpy()
    m23 = pd.Series(is23, index=A.index)
    rows = []
    for metric, T in (("d_alpha", A), ("d_ic", I)):
        d = (T[alt] - T[base]).dropna()
        ta, ma, n = tt(d)
        tx, mx, _ = tt(d[~m23.reindex(d.index).to_numpy()])
        t2, m2, _ = tt(d[m23.reindex(d.index).to_numpy()])
        rows.append(dict(compare=label, metric=metric, n=n,
                         all=(ma * ANN if metric == "d_alpha" else ma), t_all=ta,
                         ex2023=(mx * ANN if metric == "d_alpha" else mx), t_ex2023=tx,
                         only2023=(m2 * ANN if metric == "d_alpha" else m2), t_2023=t2))
    return rows


rows = []
Dc, Mc = load("newfeat_cumintra")
rows += pair_table(Dc, Mc, "L2", "L2C", "L2C(cumintra3) vs L2")

Dn, Mn = load("newfeat_nocum")
rows += pair_table(Dn, Mn, "L2", "L2N", "L2N(no-cumintra5) vs L2")

H = pd.read_csv(OUT + "newfeat_halves_daily.csv")
H["trade_date"] = H.trade_date.astype("int64")
A = H.pivot_table(index=["win", "trade_date"], columns="tag", values="alpha")
is23 = A.index.get_level_values(0).to_series().str.contains("2023").to_numpy()
m23 = pd.Series(is23, index=A.index)
for label, alt in (("L2X(all8) vs L2", "L2X"), ("L2C+L2N 加总校验", "L2X")):
    d = (A[alt] - A["L2"]).dropna()
    ta, ma, n = tt(d)
    tx, mx, _ = tt(d[~m23.reindex(d.index).to_numpy()])
    t2, m2, _ = tt(d[m23.reindex(d.index).to_numpy()])
    rows.append(dict(compare=label, metric="d_alpha", n=n, all=ma * ANN, t_all=ta,
                     ex2023=mx * ANN, t_ex2023=tx, only2023=m2 * ANN, t_2023=t2))
    if label.startswith("L2X(all8)"):
        break

P = pd.DataFrame(rows)
P.to_csv(OUT + "newfeat_cumintra_paired.csv", index=False)
print("=== 逐日配对（Δα 年化；ΔIC 原始单位）===")
print(P.round(6).to_string(index=False))

print("\n=== 三方对照汇总（8 窗均值的受控增量）===")
W = {
    "L2C(3列)": pd.read_csv(OUT + "newfeat_cumintra_halves.csv"),
    "L2N(5列)": pd.read_csv(OUT + "newfeat_nocum_halves.csv"),
    "L2X(8列)": pd.read_csv(OUT + "newfeat_halves.csv"),
}
out = []
for k, df in W.items():
    piv = df.pivot_table(index="win", columns="tag", values=["alpha", "ic", "turnover"]).reset_index()
    piv.columns = [f"{a}_{b}" if b else a for a, b in piv.columns]
    if k.startswith("L2X"):
        alt = "L2X"
    elif k.startswith("L2N"):
        alt = "L2N"
    else:
        alt = "L2C"
    da = (piv[f"alpha_{alt}"] - piv["alpha_L2"]).mean()
    di = (piv[f"ic_{alt}"] - piv["ic_L2"]).mean()
    dt = (piv[f"turnover_{alt}"] - piv["turnover_L2"]).mean()
    out.append(dict(arm=k, d_alpha=da, d_ic=di, d_turn=dt,
                    strict=0.4 * di - 0.3 * dt, with_alpha=0.4 * di - 0.3 * dt + 0.3 * da,
                    share_of_gap_strict=(0.4 * di - 0.3 * dt) / 0.0143))
print(pd.DataFrame(out).round(6).to_string(index=False))

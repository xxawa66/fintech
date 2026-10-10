"""8 窗的稳健性统计：逐日配对 t、留一窗、剔除 2023 后相关是否存活。"""

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
from pathlib import Path
OUT = Path("cv_design_audit")
R = pd.read_csv(OUT / "newfeat_halves.csv")
D = pd.read_csv(OUT / "newfeat_halves_daily.csv")
pd.set_option("display.width", 320)

# 窗口级
piv = R.pivot_table(index="win", columns="tag", values=["alpha", "ic", "profile", "final"])
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index().merge(R.drop_duplicates("win")[["win", "fold", "half"]], on="win")
piv["da"] = piv["alpha_L2X"] - piv["alpha_L2"]
piv["dic"] = piv["ic_L2X"] - piv["ic_L2"]
W = R.drop_duplicates("win").set_index("win")[["w_disp", "w_medvol", "w_volspread"]]
piv = piv.merge(W, left_on="win", right_index=True)

print("=" * 82)
print("① 逐日配对 t（n=680 天，把所有窗口串起来；比 4/8 个折点强得多）")
print("=" * 82)
WA = D.pivot_table(index=["win", "trade_date"], columns="tag", values="alpha")
d = (WA["L2X"] - WA["L2"]).dropna()
n = len(d); sd = d.std(ddof=1)
print(f"  Δα: n={n} mean {d.mean():+.6f} sd {sd:.6f} t={d.mean()/(sd/np.sqrt(n)):+.2f} "
      f"正日比 {(d>0).mean():.3f}")
for sub, lbl in ((d.index.get_level_values(0).str.contains("2023"), "仅 2023 两窗"),
                 (~d.index.get_level_values(0).str.contains("2023"), "剔除 2023 六窗")):
    x = d[sub]
    print(f"      {lbl}: n={len(x)} mean {x.mean():+.6f} 年化 {x.mean()*252:+.4f} "
          f"t={x.mean()/(x.std(ddof=1)/np.sqrt(len(x))):+.2f} 正日比 {(x>0).mean():.3f}")
from math import comb
k = int((piv["dic"] > 0).sum()); m = len(piv)
pv = sum(comb(m, i) for i in range(k, m + 1)) / 2 ** m
print(f"  ΔIC（只到窗口级）: n={m} 正 {k}/{m} 符号检验 p={pv:.4f} mean {piv['dic'].mean():+.6f}")

print("\n" + "=" * 82)
print("② 留一窗（LOOW）：8 窗里去掉任意一个，Δα 均值还剩多少")
print("=" * 82)
for i, r_ in piv.set_index("win").iterrows():
    rest = piv[piv.win != i]["da"]
    print(f"  去掉 {i:13s}(Δα {r_['da']:+.6f}) → 剩 7 窗均值 {rest.mean():+.6f} "
          f"（正 {int((rest>0).sum())}/7）")

print("\n" + "=" * 82)
print("③ 两个竞争解释：剔除 2023 两窗后，相关是否存活")
print("=" * 82)
for lbl, sub in (("全 8 窗", piv), ("剔除 2023 两窗", piv[piv.fold != "wf2023"])):
    print(f"  [{lbl}] n={len(sub)}")
    for name, x in (("H01 窗内 final", "final_H01"), ("H01 窗内 IC", "ic_H01"),
                    ("H01 窗内 α", "alpha_H01"), ("regime vol60 中位", "w_medvol"),
                    ("regime 截面离散度", "w_disp")):
        r = sub["da"].corr(sub[x], method="spearman")
        flag = "显著" if abs(r) > 0.707 else ""
        print(f"      corr(Δα, {name:16s}) = {r:+.3f} {flag}")

print("\n" + "=" * 82)
print("④ 2023H2 的弱是什么弱：逐地平窗的各周期模型 IC（horizon 越短越惨？）")
print("=" * 82)
REPO = _REPO
import sys
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "research" / "LH023"))
from _probe_root_cause import Panel, score, to_wide
from _probe_dgtw import pct_rank_2d
MS = ["H01", "H05_F2T2", "H10_F2T2", "H20_F2T2", "H30_F2T2"]
rec = []
for fold in ["wf2021", "wf2022", "wf2023", "confirm2024"]:
    c = pd.read_parquet(REPO / "outputs/long_horizon" / f"_probe_rc_{fold}.parquet")
    c["trade_date"] = c["trade_date"].astype("int64")
    c = c.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    p = Panel(c); dates = sorted(c.trade_date.unique())
    cuts = [(0.30, 0.60, "A"), (0.60, 1.00, "B")]
    for a, b, h in cuts:
        ev = np.arange(int(len(dates) * a), int(len(dates) * b))
        for m in MS:
            if m not in c.columns:
                continue
            ic = score(pct_rank_2d(to_wide(p, c[m].to_numpy(dtype="float64"))), p)["_ic"]
            rec.append(dict(win=f"{fold}{h}", fold=fold, half=h, model=m,
                            ic=float(np.nanmean(ic[ev]))))
IC = pd.DataFrame(rec).pivot_table(index=["win", "fold", "half"], columns="model",
                                   values="ic").reset_index()
IC["其余窗均值_H01"] = np.nan
base = IC[IC.fold != "wf2023"]
for i, r_ in IC.iterrows():
    IC.loc[i, "其余窗均值_H01"] = base["H01"].mean()
out = IC[["win"]].copy()
for m in MS:
    out[f"{m}_相对其余窗"] = IC[m] - base[m].mean()
print(out.round(5).to_string(index=False))

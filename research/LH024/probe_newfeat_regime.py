"""追问：wf2023 的 +0.152 Δα 是「补弱 regime」还是「暴露换了个方向」？

三个问题
1) 各折（评估窗口）的 regime 画像：截面离散度、vol60 中位、低波-高波价差。
2) 逐日 Δα = L2X − L2 是否与 regime 变量相关（389 天，独立于折数）。
3) H01 折内弱度 vs Δα 的关系符号。

注意：regime 标签用 y_ret_1d / vol60 构造，含同期信息 —— 这是**归因变量**，不是可交易信号。
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

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = _REPO
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "research" / "LH023"))
from _probe_root_cause import Panel  # noqa: E402
from _probe_e_attr import load_factors, wide_of  # noqa: E402
from _probe_dgtw import stratify  # noqa: E402

OUT = Path("cv_design_audit")
D = pd.read_csv(OUT / "newfeat_stage2_daily_split.csv").rename(columns={"label": "fold"})
D["trade_date"] = D["trade_date"].astype("int64")
R = pd.read_csv(OUT / "newfeat_stage2_split.csv")
pd.set_option("display.width", 340)

fdf = load_factors(20210101, 20241231)
cols = ["ts_code", "trade_date", "close", "y_ret_1d", "flag_limit_up"]
rows = []
for fold in ["wf2021", "wf2022", "wf2023", "confirm2024"]:
    c = pd.read_parquet(REPO / "outputs/long_horizon" / f"_probe_rc_{fold}.parquet", columns=cols)
    c["ts_code"] = c["ts_code"].astype(str); c["trade_date"] = c["trade_date"].astype("int64")
    c = c.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    p = Panel(c)
    Y = p.Y
    vol = wide_of(fdf, p)["vol60"]
    ev = D[D.fold == fold].trade_date.to_numpy()
    for t, dt in enumerate(p.dates):
        if dt not in set(ev):
            continue
        y = Y[t]; v = vol[t]
        m = np.isfinite(y)
        if m.sum() < 100:
            continue
        b = stratify(v, 10)
        lo = m & (b <= 2); hi = m & (b >= 7)
        rows.append(dict(fold=fold, trade_date=int(dt),
                         disp=float(np.nanstd(y[m])),
                         medvol=float(np.nanmedian(v[np.isfinite(v)])),
                         volspread=float(np.nanmean(y[lo]) - np.nanmean(y[hi])) if lo.sum() > 10 and hi.sum() > 10 else np.nan,
                         poolmean=float(np.nanmean(y[m]))))
G = pd.DataFrame(rows)
M = D.pivot_table(index=["fold", "trade_date"], columns="tag", values="alpha").reset_index()
M = M.merge(G, on=["fold", "trade_date"], how="inner")
M["da"] = M["L2X"] - M["L2"]
M["dic"] = (pd.read_csv(OUT / "newfeat_stage2_daymetrics_split.csv")
            .pivot_table(index=["fold", "trade_date"], columns="tag", values="ic")
            .reset_index().melt(id_vars=["fold", "trade_date"], var_name="tag", value_name="ic")
            .pivot_table(index=["fold", "trade_date"], columns="tag", values="ic")
            .reset_index().pipe(lambda x: x)[["fold", "trade_date"]]
            .merge(pd.read_csv(OUT / "newfeat_stage2_daymetrics_split.csv")
                   .pivot_table(index=["fold", "trade_date"], columns="tag", values="ic")
                   .reset_index(), on=["fold", "trade_date"], how="left")
            .pipe(lambda x: x["L2X"] - x["L2"]))
print("=" * 84)
print("① 各折 regime 画像（评估窗口均值）")
print("=" * 84)
g = M.groupby("fold")[["disp", "medvol", "volspread", "poolmean", "da", "dic"]].mean()
cnt = M.groupby("fold").size().rename("天数")
print(g.join(cnt).round(6).to_string())

print("\n" + "=" * 84)
print("② 逐日 Δα 与 regime 变量的相关（n=%d，跨折合并）" % len(M))
print("=" * 84)
for v in ("disp", "medvol", "volspread", "poolmean"):
    r_all = M["da"].corr(M[v], method="spearman")
    r_ic = M["dic"].corr(M[v], method="spearman")
    # 折内去均值（去掉折固定效应）
    z = M.copy()
    for c in ("da", "dic", v):
        z[c] = z[c] - z.groupby("fold")[c].transform("mean")
    r_in = z["da"].corr(z[v], method="spearman")
    r_in_ic = z["dic"].corr(z[v], method="spearman")
    print(f"  {v:10s} spearman(Δα) 全样本 {r_all:+.3f} / 折内去均值 {r_in:+.3f} | "
          f"spearman(ΔIC) 全样本 {r_ic:+.3f} / 折内 {r_in_ic:+.3f}")

print("\n" + "=" * 84)
print("③ 按 regime 变量分三档，看 Δα / ΔIC（日收益单位，×252 为年化）")
print("=" * 84)
for v in ("disp", "medvol", "volspread"):
    q = pd.qcut(M[v], 3, labels=["低", "中", "高"])
    t = M.groupby(q, observed=True).agg(天=("da", "size"), Δα=("da", "mean"), ΔIC=("dic", "mean"),
                                        Δα正日比=("da", lambda s: (s > 0).mean()))
    t["Δα年化"] = t["Δα"] * 252
    print(f"\n  --- 按 {v} 三档 ---")
    print(t.round(6).to_string())

print("\n" + "=" * 84)
print("④ 折级：H01 折内强度 vs L2X 相对 H01 的 Δα（n=4，看符号方向）")
print("=" * 84)
piv = R.pivot_table(index="fold", columns="tag", values="alpha")
piv["Δα_L2X"] = piv["L2X"] - piv["H01"]
print(piv.round(6).to_string())
print(f"\n  corr(H01折内α, Δα_L2X) = {piv['H01'].corr(piv['Δα_L2X']):+.3f}  "
      f"（n=4，|r| 需 >0.95 才显著；方向若为负＝「H01 越弱、新特征越有用」）")
for m in ("ic", "final"):
    p2 = R.pivot_table(index="fold", columns="tag", values=m)
    d = p2["L2X"] - p2["H01"]
    print(f"  corr(H01折内{m}, Δ{m}_L2X) = {p2['H01'].corr(d):+.3f}")
print("\n  逐折（相对 H01）：")
for f in piv.index:
    print(f"    {f:12s} H01α {piv.loc[f,'H01']:+.4f} → L2Xα {piv.loc[f,'L2X']:+.4f}  "
          f"Δα {piv.loc[f,'Δα_L2X']:+.4f}")

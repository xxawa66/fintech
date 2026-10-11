"""M8 选型逻辑链条的逐环节核验脚本（只用四折，不读测试期）。

每个环节打印该步所需的全部数字与出处文件，供 docs/long_horizon_LH034_m8_chain.md 引用。
用法：python research/LH032/verify_m8_chain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
F = ["wf2021", "wf2022", "wf2023", "confirm2024"]
pd.set_option("display.width", 400)


def load(rel: str) -> pd.DataFrame:
    return pd.read_csv(EXP / rel)


def by_tag(df: pd.DataFrame, tag: str, key: str = "tag") -> pd.DataFrame:
    return df[df[key] == tag].set_index("fold").reindex(F)


def sec(t: str) -> None:
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


def dev(x: pd.DataFrame, base: pd.DataFrame, cols=("excess", "alpha", "profile",
                                                   "turnover", "hold_days")) -> dict:
    """相对 base 的四折最大绝对偏差（暗线各项）。"""
    d = {}
    for c in cols:
        if c in x.columns and c in base.columns:
            d["max" + c] = float(np.abs(x[c] - base[c]).max())
    return d


# ---------------------------------------------------------------- 环节 0
sec("环节 0 · 评分恒等式与硬约束（部署点）")
cg = load("LH032/combo_cv_grid.csv")
D = by_tag(cg, "DEPLOY")
print(D[["ic", "excess", "turnover", "final"]].round(6).to_string())
ciE = 0.4 * D["ic"] + 0.3 * D["excess"]
turn = 0.3 * (1 - D["turnover"])
print(f"\n  0.4·IC + 0.3·E          = {ciE.mean():.6f}")
print(f"  换手项 0.3·(1-T)        = {turn.mean():.6f}")
print(f"  合计                    = {ciE.mean() + turn.mean():.6f}   final 均值 = {D['final'].mean():.6f}")
print(f"  恒等式闭合              = {ciE.mean() + turn.mean() - D['final'].mean():.2e}")
print(f"\n  换手项已占理论上限(0.3) {100 * turn.mean() / 0.3:.2f}%，剩余空间 {0.3 - turn.mean():.6f}")
need = 0.4 - turn.mean()
print(f"  ⇒ 在部署点换手水平上拿 final=0.400 需 0.4·IC+0.3·E ≥ {need:.6f}"
      f"；当前 {ciE.mean():.6f}，缺口 {need - ciE.mean():.6f}")

# ---------------------------------------------------------------- 环节 1
sec("环节 1 · band 留存规则轴已穷尽（LH025，36 配置 × 4 折）")
B = load("LH025/band_freeze_grid.csv")
g = B.groupby("tag")["final"].agg(["mean", "min"]).sort_values("mean", ascending=False)
print(f"  配置数 {B.tag.nunique()}，按 final 均值降序前 6：")
print(g.head(6).round(6).to_string())
top_tag = "V0_base|0.00227783"
print(f"\n  部署点 {top_tag}: final {g.loc[top_tag, 'mean']:.6f}，排名 "
      f"{int((g['mean'] > g.loc[top_tag, 'mean']).sum()) + 1}/{len(g)}")

# ---------------------------------------------------------------- 环节 2
sec("环节 2 · 三段自由度：drop 段是唯一零风险的段（LH031-d，free_ic_grid）")
Gdf = load("LH031/free_ic_grid.csv")
DEPlr = by_tag(Gdf, "DEPLOY")
G = Gdf[Gdf.tag.str.count(r"\|") == 2].copy()
G["seg_top"] = G.tag.str.split("|").str[0]
G["seg_drop"] = G.tag.str.split("|").str[1]
G["seg_np"] = G.tag.str.split("|").str[2]

rows = []
for t, gr in G.groupby("tag"):
    x = gr.set_index("fold").reindex(F)
    if x["final"].isna().any():
        continue
    d = dict(tag=t, top=x["seg_top"].iloc[0], drop=x["seg_drop"].iloc[0],
             np=x["seg_np"].iloc[0], dIC=float((x["ic"] - DEPlr["ic"]).mean()),
             dF=float((x["final"] - DEPlr["final"]).mean()))
    d.update(dev(x, DEPlr, ("excess", "alpha", "profile", "turnover", "hold_days")))
    rows.append(d)
Rc = pd.DataFrame(rows)
Rc = Rc[Rc.tag != "H01|H01|mix"]
cols = ["tag", "dIC", "maxexcess", "maxalpha", "maxprofile", "maxturnover",
        "maxhold_days", "dF"]

print("\n  (a) 只动 drop 段：top=H01 且 np=mix")
a = Rc[(Rc.top == "H01") & (Rc.np == "mix")]
print(a[cols].round(9).to_string(index=False))
print(f"  ⇒ 暗线各项偏差全部 = {max(a[c].max() for c in cols[2:7]):.1f}（机器零）")

b0 = Rc[(Rc["top"] == "H01") & (Rc["np"] == "mix")]
b = Rc[(Rc["top"] == "H01") & (Rc["np"] != "mix")]
print("\n  (b) 只动 nonpool 段：band 段保持 H01 且 np≠mix")
print(b[cols].round(9).to_string(index=False))
print(f"  ⇒ 暗线立刻被改动：max|ΔE| 达 {b['maxexcess'].max():.2e}（对照组 np=mix 为 "
      f"{b0['maxexcess'].max():.1f}）⇒ nonpool 段不是自由段")

print("\n  (c) max|ΔE| 是否只依赖 (top, np)、与 drop 无关")
pv = Rc.pivot_table(index=["top", "np"], columns="drop", values="maxexcess")
print(pv.round(6).to_string())
if pv.shape[1] > 1:
    spread = (pv.max(axis=1) - pv.min(axis=1)).max()
    print(f"  ⇒ 同行（同 top,np 换 drop）极差 max = {spread:.2e}")

print("\n  (d) np 段：mix → H05_F1T2 的 ΔIC 增量是否恒定")
for tp in sorted(Rc["top"].unique()):
    for dp in sorted(Rc["drop"].unique()):
        x = Rc[(Rc["top"] == tp) & (Rc["drop"] == dp) & (Rc["np"] == "H05_F1T2")]
        y = Rc[(Rc["top"] == tp) & (Rc["drop"] == dp) & (Rc["np"] == "mix")]
        if len(x) and len(y):
            print(f"    top={tp:9s} drop={dp:9s}  Δ {x.dIC.iloc[0] - y.dIC.iloc[0]:+.6f}")

# ---------------------------------------------------------------- 环节 3
sec("环节 3 · 明线信号选 rawmix（LH031-e，rawline_grid）")
W = load("LH031/rawline_grid.csv")
bw = by_tag(W, "DEPLOY")
rt = []
for t, gr in W.groupby("tag"):
    x = gr.set_index("fold").reindex(F)
    if x["final"].isna().any() or t in ("DEPLOY",):
        continue
    dI = x["ic"] - bw["ic"]
    row = dict(tag=t, dIC=float(dI.mean()),
               t_stat=float(dI.mean() / (dI.std() / 2)) if dI.std() > 0 else np.nan,
               dI_min=float(dI.min()),
               sign="".join("+" if v > 0 else "-" for v in dI),
               dF=float((x["final"] - bw["final"]).mean()))
    row.update(dev(x, bw, ("excess", "turnover", "hold_days")))
    rt.append(row)
Rw = pd.DataFrame(rt).sort_values("dIC", ascending=False)
show = ["tag", "dIC", "t_stat", "dI_min", "sign", "maxexcess", "maxturnover",
        "maxhold_days", "dF"]
print("  全部配置按 ΔIC 降序前 6：")
print(Rw[show].head(6).round(9).to_string(index=False))

ZERO = Rw[Rw.maxexcess < 1e-12]
NONZ = Rw[Rw.maxexcess >= 1e-12]
print(f"\n  硬约束“暗线逐位不变”(max|ΔE|=max|ΔT|=max|Δhold|=0) 下有 {len(ZERO)} 个可行解：")
print(ZERO[show].sort_values("dIC", ascending=False).round(9).to_string(index=False))
print(f"\n  被该约束排除的最高 ΔIC 配置（ΔIC 更大但暗线已动）：")
print(NONZ[show].sort_values("dIC", ascending=False).head(4).round(9).to_string(index=False))

print("\n  LOFO（可行解集合内，三折 argmax(ΔIC) → 未见折）")
keys = ZERO["tag"].tolist()
sub = W[W.tag.isin(keys)]
ser = {t: sub[sub.tag == t].set_index("fold").reindex(F)["ic"] - bw["ic"] for t in keys}
for h in F:
    tr = [f for f in F if f != h]
    best = max(keys, key=lambda t: ser[t][tr].mean())
    print(f"    留出 {h:11s} 选中 {best:12s} → 未见折 ΔIC {ser[best][h]:+.6f}")

# ---------------------------------------------------------------- 环节 4
sec("环节 4 · 倾斜必须落在 stage2（池内重排），stage1 无效（LH032 stage_tilt_grid）")
S = load("LH032/stage_tilt_grid.csv")
bs = by_tag(S, "DEPLOY")
sr = []
for t, gr in S.groupby("tag"):
    x = gr.set_index("fold").reindex(F)
    if x["final"].isna().any():
        continue
    sr.append(dict(tag=t, factor=gr["factor"].iloc[0], mode=gr["mode"].iloc[0],
                   phi=gr["phi"].iloc[0], dIC=float((x["ic"] - bs["ic"]).mean()),
                   dE=float((x["excess"] - bs["excess"]).mean()),
                   dA=float((x["alpha_vol"] - bs["alpha_vol"]).mean()),
                   dF=float((x["final"] - bs["final"]).mean())))
Rs = pd.DataFrame(sr)
comp = Rs[Rs.factor == "log_amt"].pivot_table(index="phi", columns="mode", values=["dE", "dA", "dF"])
print("  log_amt 单因子：stage1(候选池) vs stage2(池内重排)")
print(comp.round(5).to_string())

# ---------------------------------------------------------------- 环节 5
sec("环节 5 · 双因子倾斜参数的邻域（LH032 dual_tilt，band=H01 三段构造）")
DT = pd.concat([load("LH032/dual_tilt_grid.csv"), load("LH032/dual_tilt_grid_ext.csv")],
               ignore_index=True).drop_duplicates(subset=["fold", "phi_amt", "phi_vol"])
rows = []
for (aa, vv), gr in DT.groupby(["phi_amt", "phi_vol"]):
    x = gr.set_index("fold").reindex(F)
    if x["final"].isna().any():
        continue
    dE = x["excess"] - D["excess"]
    dA = x["alpha_vol"] - D["alpha_vol"]
    dF = x["final"] - D["final"]
    rows.append(dict(a=aa, v=vv, dIC=float((x["ic"] - D["ic"]).mean()),
                     dE=float(dE.mean()), dA=float(dA.mean()),
                     share=float(dA.mean() / dE.mean()) if abs(dE.mean()) > 1e-9 else np.nan,
                     dT=float((x["turnover"] - D["turnover"]).mean()),
                     dF=float(dF.mean()), dF_min=float(dF.min()), dF_sd=float(dF.std()),
                     t=float(dF.mean() / (dF.std() / 2)),
                     sign="".join("+" if q > 0 else "-" for q in dF)))
Rt = pd.DataFrame(rows).drop_duplicates(subset=["a", "v"])
m2row = Rt[(np.isclose(Rt.a, -0.100)) & (np.isclose(Rt.v, 0.040))].iloc[0]
print(f"  倾斜配置总数 {len(Rt)}")
print(f"\n  M2 参数 (-0.100,+0.040):  Δfinal {m2row.dF:+.5f} 排 {int((Rt.dF > m2row.dF).sum()) + 1}/{len(Rt)}"
      f"   α占比 {m2row.share:.3f} 排 {int((Rt.share > m2row.share).sum()) + 1}/{len(Rt)}")
print("\n  Δfinal 降序前 8：")
print(Rt.sort_values("dF", ascending=False).head(8).round(5).to_string(index=False))
print("\n  α 占比降序前 6（约束 ΔE>0.04，比值在 ΔE→0 时爆炸）：")
print(Rt[Rt.dE > 0.04].sort_values("share", ascending=False).head(6).round(5).to_string(index=False))
print("\n  α 占比降序前 6（再加 ΔIC>0 约束——否则会选到什么都不做的配置）：")
print(Rt[(Rt.dE > 0.04) & (Rt.dIC > 0)].sort_values("share", ascending=False).head(6).round(5).to_string(index=False))
cc = Rt[(Rt.dE > 0.04) & (Rt.dIC > 0)]
print(f"\n  α 占比 与 Δfinal 的相关系数 = {np.corrcoef(cc.share, cc.dF)[0, 1]:+.3f}"
      f"（负相关 ⇒ 该比值不能单独用于排序，只能在同等 ΔE 或绝对值口径下比较）")
print(f"  Δα（绝对值）与 Δfinal 的相关系数 = {np.corrcoef(cc.dA, cc.dF)[0, 1]:+.3f}")

# ---------------------------------------------------------------- 环节 6
sec("环节 6 · 组合顺序必须是 COMBO_B（明线在前、倾斜在后）")
for fn, lab in [("LH032/combo_cv_grid.csv", "φ=-0.100/+0.040"),
                ("LH032/combo_cv_grid_p125.csv", "φ=-0.125/+0.075")]:
    K = load(fn)
    T = by_tag(K, "TILT")
    print(f"\n  --- {lab} ---")
    for tag in ["TILT", "RAWMIX", "COMBO_A", "COMBO_B"]:
        x = by_tag(K, tag)
        dd = dict(dIC=float((x["ic"] - D["ic"]).mean()),
                  dE=float((x["excess"] - D["excess"]).mean()),
                  dA=float((x["alpha_vol"] - D["alpha_vol"]).mean()),
                  dT=float((x["turnover"] - D["turnover"]).mean()),
                  dF=float((x["final"] - D["final"]).mean()),
                  share=float((x["alpha_vol"] - D["alpha_vol"]).mean() /
                              (x["excess"] - D["excess"]).mean()))
        dd.update(dev(x, T, ("excess", "turnover", "hold_days", "alpha_vol")))
        print(f"    {tag:8s} ΔIC {dd['dIC']:+.6f}  ΔE {dd['dE']:+.6f}  Δα {dd['dA']:+.6f}"
              f"  ΔT {dd['dT']:+.6f}  Δfinal {dd['dF']:+.6f}  α占比 {dd['share']:.3f}"
              f"  | vs TILT: max|ΔE| {dd['maxexcess']:.2e} max|ΔT| {dd['maxturnover']:.2e}"
              f" max|Δhold| {dd['maxhold_days']:.3f}")

print("\n  明线增益是否与倾斜参数无关（COMBO_B − TILT）：")
for fn, lab in [("LH032/combo_cv_grid.csv", "φ=-0.100/+0.040"),
                ("LH032/combo_cv_grid_p125.csv", "φ=-0.125/+0.075")]:
    K = load(fn)
    T, CB = by_tag(K, "TILT"), by_tag(K, "COMBO_B")
    print(f"    {lab}: ΔIC {(CB['ic'] - T['ic']).mean():+.6f}  "
          f"max|ΔE| {np.abs(CB['excess'] - T['excess']).max():.2e}  "
          f"max|ΔT| {np.abs(CB['turnover'] - T['turnover']).max():.2e}  "
          f"max|Δhold| {np.abs(CB['hold_days'] - T['hold_days']).max():.3f}")

# ---------------------------------------------------------------- 环节 7
sec("环节 7 · 排除长周期 band 段信号（LH033 阴性结果，α 占比崩塌）")
M = load("LH032/band_axis_grid_all.csv")
rows = []
for (bb, aa, vv), gr in M.groupby(["band", "phi_amt", "phi_vol"]):
    x = gr.set_index("fold").reindex(F)
    if x["final"].isna().any():
        continue
    dE = (x["excess"] - D["excess"]).mean()
    dA = (x["alpha_vol"] - D["alpha_vol"]).mean()
    if abs(dE) < 0.04:
        continue
    rows.append(dict(band=bb, a=aa, v=vv, dIC=float((x["ic"] - D["ic"]).mean()),
                     dE=float(dE), dA=float(dA), share=float(dA / dE),
                     dT=float((x["turnover"] - D["turnover"]).mean()),
                     dF=float((x["final"] - D["final"]).mean())))
Rm = pd.DataFrame(rows)
print("  band 段信号 = H01（部署点/M2 路线）的 α 占比：")
print(Rm[Rm.band == "H01"].sort_values("share", ascending=False).head(4).round(4).to_string(index=False))
print("\n  F2T2 族沿 horizon（φ=-0.125/+0.075）：")
for bb, hh in [("H01", 1), ("H05_F2T2", 5), ("H10_F2T2", 10), ("H20_F2T2", 20), ("H30_F2T2", 30)]:
    r = Rm[(Rm.band == bb) & (np.isclose(Rm.a, -0.125)) & (np.isclose(Rm.v, 0.075))]
    if len(r):
        r = r.iloc[0]
        print(f"    h={hh:2d} {bb:9s} ΔE {r.dE:+.4f}  Δα {r.dA:+.4f}  α占比 {r.share:.3f}  Δfinal {r.dF:+.5f}")

# ---------------------------------------------------------------- 环节 8
sec("环节 8 · 稳定性指标失效的反例（C：四折稳、外推失败）")
C = M[(M.band == "H20_F2T2") & (np.isclose(M.phi_amt, -0.125)) &
      (np.isclose(M.phi_vol, 0.075))].set_index("fold").reindex(F)
M8 = by_tag(cg, "COMBO_B")
dC, d8 = C["final"] - D["final"], M8["final"] - D["final"]
dCc = C["alpha_vol"] - D["alpha_vol"]
d8c = M8["alpha_vol"] - D["alpha_vol"]
print(f"  C  : Δfinal min {dC.min():+.5f}  sd {dC.std():.5f}  t {dC.mean() / (dC.std() / 2):5.2f}"
      f"  α占比 {(dCc.mean() / (C['excess'] - D['excess']).mean()):.3f}")
print(f"  M8 : Δfinal min {d8.min():+.5f}  sd {d8.std():.5f}  t {d8.mean() / (d8.std() / 2):5.2f}"
      f"  α占比 {(d8c.mean() / (M8['excess'] - D['excess']).mean()):.3f}")
print("  ⇒ 所有稳定性指标都判 C 更稳，但只有 α 占比把两者分开")

# ---------------------------------------------------------------- 环节 9
sec("环节 9 · 最终对照表")
tab = []
K = load("LH032/combo_cv_grid.csv")
Kp = load("LH032/combo_cv_grid_p125.csv")
for nm, x in [("DEPLOY", D),
              ("TILT φ=-0.100/+0.040", by_tag(K, "TILT")),
              ("M8   = TILT + 明线", by_tag(K, "COMBO_B")),
              ("TILT φ=-0.125/+0.075", by_tag(Kp, "TILT")),
              ("M8p  = TILT + 明线", by_tag(Kp, "COMBO_B")),
              ("C (两段 + H20_F2T2)", C)]:
    dE = (x["excess"] - D["excess"]).mean()
    dA = (x["alpha_vol"] - D["alpha_vol"]).mean()
    dF = x["final"] - D["final"]
    tab.append(dict(cfg=nm, IC=float(x["ic"].mean()), E=float(x["excess"].mean()),
                    T=float(x["turnover"].mean()), final=float(x["final"].mean()),
                    dF=float(dF.mean()), dF_min=float(dF.min()), sd=float(dF.std()),
                    t=float(dF.mean() / (dF.std() / 2)) if dF.std() > 0 else np.nan,
                    dAlpha=float(dA), aShare=float(dA / dE) if abs(dE) > 1e-9 else np.nan,
                    hold=float(x["hold_days"].mean()),
                    sign="".join("+" if q > 0 else "-" for q in dF)))
print(pd.DataFrame(tab).round(5).to_string(index=False))

sys.exit(0)

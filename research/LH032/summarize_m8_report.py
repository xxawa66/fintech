"""汇总 M8 及其三个竞争候选的四折证据（报告用）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[2]
F = ["wf2021", "wf2022", "wf2023", "confirm2024"]


def k(df, t):
    return df[df.tag == t].set_index("fold").reindex(F)


K = pd.read_csv(ROOT / "experiments" / "LH032" / "combo_cv_grid.csv")
N = pd.read_csv(ROOT / "experiments" / "LH032" / "combo_cv_grid_np.csv")
M = pd.read_csv(ROOT / "experiments" / "LH032" / "band_axis_grid_all.csv")

D = k(K, "DEPLOY")
T = k(K, "TILT")
M8 = k(K, "COMBO_B")
COMA = k(K, "COMBO_A")
NP = k(N, "COMBO_B_NP")
RNP = k(N, "RAWMIX_NP")
M8_ = k(N, "COMBO_B")
C = M[(M.band == "H20_F2T2")
      & np.isclose(M.phi_amt, -0.125)
      & np.isclose(M.phi_vol, 0.075)].set_index("fold").reindex(F)

pd.set_option("display.width", 400)

print("=" * 100)
print("复现校验：新脚本 combo_cv_grid_np 的 DEPLOY/COMBO_B 应与原脚本逐位相同")
print("=" * 100)
for m in ["ic", "excess", "turnover", "final", "alpha_vol", "prof_amt", "hold_days"]:
    print("  %-10s DEPLOY %.1e   COMBO_B %.1e"
          % (m, np.abs(k(N, "DEPLOY")[m] - D[m]).max(), np.abs(M8_[m] - M8[m]).max()))

print()
print("=" * 100)
print("逐折 final")
print("=" * 100)
for nm, X in [("DEPLOY", D), ("M2  TILT", T), ("M8  COMBO_B", M8),
              ("M9  COMBO_A", COMA), ("M8_NP COMBO_B_NP", NP), ("C   H20_F2T2", C)]:
    vals = " ".join("%.5f" % x for x in X["final"])
    print("  %-18s %s  | mean %.5f" % (nm, vals, X["final"].mean()))

print()
print("=" * 100)
print("相对 DEPLOY 的增量（四折均值）")
print("=" * 100)
rows = []
for nm, X in [("M2  (TILT)", T), ("M8  (COMBO_B)", M8), ("M9  (COMBO_A)", COMA),
              ("M8_NP (同时铺np)", NP), ("C   (长周期band)", C)]:
    dE = (X["excess"] - D["excess"]).mean()
    dA = (X["alpha_vol"] - D["alpha_vol"]).mean()
    dF = X["final"] - D["final"]
    rows.append(dict(cfg=nm, dIC=(X["ic"] - D["ic"]).mean(), dE=dE, dA=dA,
                     dP=(X["prof_amt"] - D["prof_amt"]).mean(),
                     dT=(X["turnover"] - D["turnover"]).mean(),
                     share=dA / dE, dfinal=dF.mean(), df_min=dF.min(), sd=dF.std(),
                     t=dF.mean() / (dF.std() / 2), hold=X["hold_days"].mean(),
                     sign="".join("+" if x > 0 else "-" for x in dF)))
R = pd.DataFrame(rows)
print(R.round(5).to_string(index=False))

print()
print("=" * 100)
print("配对检验 vs M8")
print("=" * 100)
for nm, X in [("M9  COMBO_A", COMA), ("M8_NP", NP), ("C", C)]:
    d = X["final"] - M8["final"]
    tt = ttest_rel(X["final"], M8["final"])
    ds = " ".join("%+.5f" % x for x in d)
    print("  %-14s 逐折 %s | mean %+.5f  min %+.5f  t %+.2f  p %.4f  胜 %d/4"
          % (nm, ds, d.mean(), d.min(), tt.statistic, tt.pvalue, int((d > 0).sum())))

print()
print("=" * 100)
print("暗线偏移：相对各自纯倾斜基线 TILT(φ=-0.100/+0.040)")
print("=" * 100)
for nm, X in [("M8   COMBO_B", M8), ("M9   COMBO_A", COMA), ("M8_NP COMBO_B_NP", NP)]:
    print("  %-18s max|ΔE| %.2e  max|ΔT| %.2e  max|Δα| %.2e  max|Δhold| %.3f 天"
          % (nm, np.abs(X["excess"] - T["excess"]).max(),
             np.abs(X["turnover"] - T["turnover"]).max(),
             np.abs(X["alpha_vol"] - T["alpha_vol"]).max(),
             np.abs(X["hold_days"] - T["hold_days"]).max()))
print("  RAWMIX_NP 单独(vs DEPLOY) max|ΔE| %.2e  max|ΔT| %.2e  max|Δhold| %.4f 天  ΔIC %+.6f"
      % (np.abs(RNP["excess"] - D["excess"]).max(),
         np.abs(RNP["turnover"] - D["turnover"]).max(),
         np.abs(RNP["hold_days"] - D["hold_days"]).max(),
         (RNP["ic"] - D["ic"]).mean()))

print()
print("=" * 100)
print("微盘暴露（log_amt 十层，L0=最小成交额）")
print("=" * 100)
Lc = ["E_L%d" % j for j in range(10)]
for nm, X in [("DEPLOY", D), ("M8", M8), ("M9", COMA), ("M8_NP", NP)]:
    if Lc[0] in X.columns:
        x = X[Lc].mean()
        print("  %-8s %s | L0+L1 %.3f" % (nm, " ".join("%.3f" % x[c] for c in Lc),
                                       x["E_L0"] + x["E_L1"]))
    else:
        print("  %-8s n/a" % nm)

print()
print("=" * 100)
print("C 的最后一次对照组（外推失败样本）")
print("=" * 100)
d = C["final"] - M8["final"]
tt = ttest_rel(C["final"], M8["final"])
print("  C - M8 逐折 %s | mean %+.5f  t %+.2f  p %.4f  胜 %d/4"
      % (" ".join("%+.5f" % x for x in d), d.mean(), tt.statistic, tt.pvalue,
         int((d > 0).sum())))
print("  C 持股 %.1f 天 / M8 持股 %.1f 天 / DEPLOY %.1f 天"
      % (C["hold_days"].mean(), M8["hold_days"].mean(), D["hold_days"].mean()))
print("  C α占比 %.3f  vs  M8 α占比 %.3f"
      % ((C["alpha_vol"] - D["alpha_vol"]).mean() / (C["excess"] - D["excess"]).mean(),
         (M8["alpha_vol"] - D["alpha_vol"]).mean() / (M8["excess"] - D["excess"]).mean()))

if __name__ == "__main__":
    sys.exit(0)

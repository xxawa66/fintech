"""LH032 第七步：把双因子网格的 φ_amt 边界往更负处扩。

原网格 PHI_AMT 只到 −0.100，而 M2(φ_amt=−0.100) 恰好落在边界上 ⇒ LOFO 4/4
是在「截断网格」内成立的。本步把 φ_amt 扩到 −0.30/−0.20/−0.15/−0.125，并把
φ_vol 上界扩到 +0.100，检验：
  Q1 更负的 φ_amt 是否给出更高的 Δfinal / ΔE？
  Q2 若更高，最小折是否崩（单因子网格的经验：φ 越负 ΔE_min 越崩）？
  Q3 M2 在扩展后是否仍在帕累托前沿上、是否仍被 LOFO 选中？
纯训练期（四折 CV），不涉及测试集。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "LH032"))
sys.path.insert(0, str(ROOT / "research" / "LH023"))
sys.path.insert(0, str(ROOT / "outputs" / "long_horizon"))

import probe_dual_tilt as base                                       # noqa: E402
from _probe_e_attr import load_factors                               # noqa: E402

# 覆盖网格：更负的 φ_amt + 更大的 φ_vol
base.PHI_AMT = [-0.30, -0.20, -0.15, -0.125, -0.10]
base.PHI_VOL = [0.0, 0.02, 0.04, 0.075, 0.10]

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]


def log(s: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def main() -> int:
    log("读因子面板")
    fdf = load_factors(20210101, 20241231)
    parts = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        parts.append(base.fold_block(fold, fdf))
    M = pd.concat(parts, ignore_index=True)
    out = ROOT / "experiments" / "LH032" / "dual_tilt_grid_ext.csv"
    M.to_csv(out, index=False)
    log(f"saved {out}  rows={len(M)}")

    F = FOLDS
    b = M[M.tag == "DEPLOY"].set_index("fold").reindex(F)
    rows = []
    for (pa, pb), g in M[M.tag != "DEPLOY"].groupby(["phi_amt", "phi_vol"]):
        g = g.set_index("fold").reindex(F)
        dE = g["excess"] - b["excess"]
        dF = g["final"] - b["final"]
        dT = g["turnover"] - b["turnover"]
        dI = g["ic"] - b["ic"]
        dA = g["alpha_vol"] - b["alpha_vol"]
        dP = g["prof_amt"] - b["prof_amt"]
        rows.append(dict(a=pa, v=pb,
                         dE=dE.mean(), dE_min=dE.min(), dT=dT.mean(),
                         dIC=dI.mean(), dalpha=dA.mean(), dprof=dP.mean(),
                         dfinal=dF.mean(), df_min=dF.min(),
                         df_sd=dF.std(),
                         sign="".join("+" if x > 0 else "-" for x in dF),
                         Eall="Y" if (dE > 0).all() else "n"))
    R = pd.DataFrame(rows).set_index(["a", "v"]).sort_values("dfinal", ascending=False)
    pd.set_option("display.width", 300)
    print()
    print("=== 扩展网格：按 Δfinal 排序 ===")
    print(R.round(5).to_string())
    print()
    print("=== 四折 ΔE 全正者里 Δfinal 前 8 ===")
    print(R[R.Eall == "Y"].head(8).round(5).to_string())
    print()
    print("=== 帕累托前沿（ΔT 更小 且 ΔE 更大）===")

    def dominated(r):
        return ((R.dT <= r.dT + 1e-12) & (R.dE >= r.dE - 1e-12)
                & ~((np.isclose(R.dT, r.dT)) & (np.isclose(R.dE, r.dE)))).any()

    R["pareto"] = [not dominated(r) for _, r in R.iterrows()]
    print(R[R.pareto].sort_values("dT").round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

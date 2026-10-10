"""反证实验：把 cumintra 三列**排除**，只加剩下 5 列（受控，与 cumintra 脚本同构）。

问题
----
L2X(40+8) 的增益到底是不是 cumintra 扛的？
上一轮 probe_newfeat_cumintra.py 显示「单独加 cumintra 三列」几乎没有增益
（8 窗 Δα +0.00265 正 4/8、ΔIC +0.00035 正 5/8，折级 t 都不显著）。
本脚本给另一边：把 cumintra 拿掉、只加 5 列，看增益还剩多少。

设计（与 probe_newfeat_cumintra.py / probe_newfeat_halves.py 完全同构）
固定：候选池、训练行、T030 超参、500 轮、band、评分口径、8 个样本外窗口。
唯一变量 = 阶段2 输入列：
    L2   : 40 列（V1）
    L2N  : 40 + gapspread20 + corrmkt60/120 + amihud20/60 = 45 列
自检：本脚本的 L2 必须逐位复现归档 newfeat_halves.csv 的 L2 行。

产物：cv_design_audit/newfeat_nocum_halves.csv / _daily.csv / _daymetrics.csv
"""
from __future__ import annotations

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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve()
REPO = _REPO
OUT = _LH024_DIR / "cv_design_audit"
for _p in (str(REPO), str(REPO / "research" / "LH023"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_pairwise_rerank import (  # noqa: E402
    FEATS, FOLDS, build_fold, eval_arm, group_sizes, log, s2_of,
    stage2_encode, train_rows,
)
from probe_pairwise_newfeat import fit_l2, s2_from_booster  # noqa: E402
from probe_new_features import build_features, load_raw  # noqa: E402
from _probe_root_cause import Panel  # noqa: E402
from _probe_e_attr import load_factors, wide_of  # noqa: E402

COLS = ["gapspread20", "corrmkt60", "corrmkt120", "amihud20", "amihud60"]
ROUNDS = 500
CUT_A, CUT_B = 0.30, 0.60
ANN = 252


def build_subset(names):
    raw = load_raw()
    raw = raw.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel_full = Panel(raw)
    log(f"特征面板 {panel_full.T} 天 × {panel_full.N} 只")
    allfeat = build_features(panel_full, raw)
    missing = [nm for nm in names if nm not in allfeat]
    if missing:
        raise ValueError(f"build_features 缺列：{missing}")
    keep = {nm: allfeat[nm].astype("float32") for nm in names}
    del allfeat, raw
    gc.collect()
    return keep


def slice_cols(feats, panel, names) -> np.ndarray:
    cols = []
    for nm in names:
        A = feats[nm].reindex(index=panel.dates, columns=panel.codes).to_numpy(dtype="float32")
        cols.append(A[panel.ti, panel.si])
    return np.column_stack(cols).astype("float32", copy=False)


def run(fdf, folds, rounds: int):
    feats = build_subset(COLS)
    rows, dailies, dailym = [], [], []
    for fold in folds:
        b = build_fold(fold)
        W = wide_of(fdf, b["panel"])
        panel = b["panel"]
        T = panel.T
        X40 = b["X"]
        Xn = slice_cols(feats, panel, COLS)
        X45 = np.hstack([X40, Xn]).astype("float32", copy=False)

        for tag_w, cut, ev_cut in (("A", CUT_A, CUT_B), ("B", CUT_B, 1.0)):
            n_tr, n_ev = int(T * cut), int(T * ev_cut)
            days_ev = np.arange(n_tr, n_ev)
            idx = train_rows(b, np.arange(0, n_tr))
            ytr = b["long"]["y"][idx].astype("float64")
            grp = group_sizes(b["long"]["ti"][idx])
            win = f"{fold}{tag_w}"
            log(f"{win}: 训练 {len(idx):,} 行/{len(grp)} 天，评估 {len(days_ev)} 天")

            for arm, Xtr, Xfull, names in (
                ("L2", X40[idx], X40, FEATS),
                ("L2N", X45[idx], X45, FEATS + COLS),
            ):
                bo = fit_l2(Xtr, ytr, grp, names, rounds)
                s2 = s2_from_booster(b, Xfull, bo)
                row, d, dd = eval_arm(stage2_encode(b["pr_mix"], s2, b["cand"]), b, W, arm, days_ev)
                row.update(win=win, fold=fold, half=tag_w,
                           d0=int(panel.dates[days_ev[0]]), d1=int(panel.dates[days_ev[-1]]))
                rows.append(row)
                dailies.append(d.assign(win=win, tag=arm))
                dailym.append(dd.assign(win=win, tag=arm))
                log(f"  {win} {arm:4s}: IC {row['ic']:.6f} E {row['excess']:.6f} "
                    f"T {row['turnover']:.6f} a {row['alpha']:.6f} F {row['final']:.6f}")
        del X45, Xn
        gc.collect()

    R = pd.DataFrame(rows)
    pd.concat(dailies, ignore_index=True).to_csv(OUT / "newfeat_nocum_daily.csv", index=False)
    pd.concat(dailym, ignore_index=True).to_csv(OUT / "newfeat_nocum_daymetrics.csv", index=False)
    R.to_csv(OUT / "newfeat_nocum_halves.csv", index=False)
    return R


def report(R: pd.DataFrame) -> None:
    pd.set_option("display.width", 340)
    try:
        H = pd.read_csv(OUT / "newfeat_halves.csv")
        j = H[H.tag == "L2"].set_index("win")
        m = R[R.tag == "L2"].set_index("win")
        for c in ("ic", "alpha", "turnover", "final"):
            d = (m[c] - j[c]).abs().max()
            print(f"  自检 L2 {c:9s} max|Δ| = {d:.3e}")
            assert d < 1e-9
    except FileNotFoundError:
        pass

    piv = R.pivot_table(index="win", columns="tag", values=["alpha", "ic", "turnover"]).reset_index()
    piv.columns = [f"{a}_{b}" if b else a for a, b in piv.columns]
    piv["dalpha"] = piv["alpha_L2N"] - piv["alpha_L2"]
    piv["dic"] = piv["ic_L2N"] - piv["ic_L2"]
    piv["dturn"] = piv["turnover_L2N"] - piv["turnover_L2"]
    piv["strict"] = 0.4 * piv["dic"] - 0.3 * piv["dturn"]
    piv["withalpha"] = piv["strict"] + 0.3 * piv["dalpha"]
    print("\n=== 8 窗：L2N(45，去掉 cumintra) vs L2 ===")
    print(piv[["win", "dalpha", "dic", "dturn", "strict", "withalpha"]].round(6).to_string(index=False))
    print(f"\n  8 窗均值 Δα {piv['dalpha'].mean():+.6f}（正 {int((piv['dalpha']>0).sum())}/8）"
          f"  ΔIC {piv['dic'].mean():+.6f}（正 {int((piv['dic']>0).sum())}/8）"
          f"  strict {piv['strict'].mean():+.6f}  with-alpha {piv['withalpha'].mean():+.6f}")


if __name__ == "__main__":
    t0 = time.time()
    folds = [f.strip() for f in
             (sys.argv[1] if len(sys.argv) > 1 else ",".join(FOLDS)).split(",") if f.strip()]
    log("读因子面板...")
    fdf = load_factors(20210101, 20241231)
    R = run(fdf, folds, ROUNDS)
    report(R)
    log(f"完成，用时 {time.time()-t0:.0f}s")

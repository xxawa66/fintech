"""cumintra 单独提出来做：阶段2 只加 cumintra{5,20,60} 三列（受控对照）。

背景
----
方向二筛出的 8 个新特征里，`cumintra{5,20,60}`（累计日内收益，负向）是唯一
「单特征独立 α 为正 + profile 近零」的一族：
    α  +0.092 / +0.113 / +0.092      profile  −0.026 / −0.004 / +0.033
且与 vol60 秩相关仅 +0.028（几乎正交）。在 L2X 的 gain 榜上 cumintra5 排第 11。
其余 5 列各有瑕疵：
    corrmkt60/120  profile +0.075（α 最高但一半来自低波倾斜）
    amihud20/60    半 α 半 profile
    gapspread20    L2X 里 gain 排第 43（几乎没被用）
⇒ 问题：把 cumintra 三列单独加进阶段2，能不能干净地拿到（甚至超过）L2X 的增益？

受控设计（与 probe_newfeat_halves.py 完全同构）
-----------------------------------------------
固定：候选池（mix top 40%）、训练行（窗口内前段天数）、超参 T030、轮数 500、
      band(keep_q=q*)、评分口径、8 个样本外窗口
      （A：训 d[0,30%) 评估 d[30%,60%)；B：训 d[0,60%) 评估 d[60%,100%)）。
唯一变量 = 阶段2 的输入列：
    L2  : 40 列（V1 features.parquet）
    L2C : 40 + cumintra{5,20,60} = 43 列
L2X（40+8）不在本脚本重跑，直接读归档 newfeat_halves.csv 作参照。

自检：本脚本的 L2 必须逐位复现归档 newfeat_halves.csv 的 L2 行。

产物（仓库外）：cv_design_audit/newfeat_cumintra_halves.csv
                cv_design_audit/newfeat_cumintra_daily.csv
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

CUM = ["cumintra5", "cumintra20", "cumintra60"]
ROUNDS = 500
CUT_A, CUT_B = 0.30, 0.60
ANN = 252


def build_subset(names):
    """在全样本面板上一次算完新特征，再切片到折（滚动窗口需要折外历史）。"""
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
    feats = build_subset(CUM)
    rows, dailies, dailym, nans = [], [], [], []
    for fold in folds:
        b = build_fold(fold)
        W = wide_of(fdf, b["panel"])
        panel = b["panel"]
        T = panel.T
        X40 = b["X"]
        Xn = slice_cols(feats, panel, CUM)
        X43 = np.hstack([X40, Xn]).astype("float32", copy=False)

        for tag_w, cut, ev_cut in (("A", CUT_A, CUT_B), ("B", CUT_B, 1.0)):
            n_tr, n_ev = int(T * cut), int(T * ev_cut)
            days_ev = np.arange(n_tr, n_ev)
            idx = train_rows(b, np.arange(0, n_tr))
            ytr = b["long"]["y"][idx].astype("float64")
            grp = group_sizes(b["long"]["ti"][idx])
            assert grp.sum() == len(idx), (grp.sum(), len(idx))
            win = f"{fold}{tag_w}"
            nan_tr = {k: float(np.isnan(v[idx]).mean()) for k, v in zip(CUM, Xn.T)}
            nans.append(dict(win=win, **nan_tr))
            log(f"{win}: 训练 {len(idx):,} 行/{len(grp)} 天(d[0,{n_tr})) 评估 {len(days_ev)} 天；"
                f"cumintra 训练行 NaN {max(nan_tr.values()):.3f}(max)")

            for arm, Xtr, Xfull, names in (
                ("L2", X40[idx], X40, FEATS),
                ("L2C", X43[idx], X43, FEATS + CUM),
            ):
                t0 = time.time()
                bo = fit_l2(Xtr, ytr, grp, names, rounds)
                s2 = s2_from_booster(b, Xfull, bo)
                row, d, dd = eval_arm(stage2_encode(b["pr_mix"], s2, b["cand"]), b, W, arm, days_ev)
                row.update(win=win, fold=fold, half=tag_w,
                           d0=int(panel.dates[days_ev[0]]), d1=int(panel.dates[days_ev[-1]]))
                rows.append(row)
                dailies.append(d.assign(win=win, tag=arm))
                dailym.append(dd.assign(win=win, tag=arm))
                log(f"  {win} {arm:4s}: IC {row['ic']:.6f} E {row['excess']:.6f} "
                    f"T {row['turnover']:.6f} a {row['alpha']:.6f} prof {row['profile']:.6f} "
                    f"F {row['final']:.6f}  ({time.time()-t0:.0f}s)")
        del X43, Xn
        gc.collect()

    R = pd.DataFrame(rows)
    D = pd.concat(dailies, ignore_index=True)
    DM = pd.concat(dailym, ignore_index=True)
    R.to_csv(OUT / "newfeat_cumintra_halves.csv", index=False)
    D.to_csv(OUT / "newfeat_cumintra_daily.csv", index=False)
    DM.to_csv(OUT / "newfeat_cumintra_daymetrics.csv", index=False)
    pd.DataFrame(nans).to_csv(OUT / "newfeat_cumintra_nanfrac.csv", index=False)
    return R, D, DM


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def selfcheck(R: pd.DataFrame) -> None:
    print("\n=== 自检：本脚本 L2 应逐位复现归档 newfeat_halves.csv 的 L2 ===")
    try:
        H = pd.read_csv(OUT / "newfeat_halves.csv")
    except FileNotFoundError:
        print("  未找到 newfeat_halves.csv，跳过")
        return
    j = H[H.tag == "L2"].set_index("win")
    m = R[R.tag == "L2"].set_index("win")
    for c in ("ic", "alpha", "turnover", "final"):
        d = (m[c] - j[c]).abs().max()
        print(f"  L2 {c:9s} max|Δ| = {d:.3e}")
        assert d < 1e-9, f"L2 未复现归档：{c}"


def paired_fold(R: pd.DataFrame, metric: str) -> dict:
    piv = R.pivot_table(index="fold", columns="tag", values=metric)
    d = (piv["L2C"] - piv["L2"]).dropna()
    n = len(d)
    sd = float(np.std(d.to_numpy(), ddof=1))
    return dict(metric=metric, n=n, delta=float(d.mean()),
                t=float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan,
                signs="".join("+" if v > 0 else "-" for v in d))


def daily_paired(src: pd.DataFrame, metric: str, drop_wins=()) -> dict:
    piv = src[~src.win.isin(drop_wins)].pivot_table(index=["win", "trade_date"], columns="tag", values=metric)
    d = (piv["L2C"] - piv["L2"]).dropna()
    n = len(d)
    sd = float(np.std(d.to_numpy(), ddof=1))
    return dict(metric=metric, n=n, delta=float(d.mean()),
                t=float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan,
                posfrac=float((d > 0).mean()))


def report(R: pd.DataFrame, D: pd.DataFrame, DM: pd.DataFrame) -> None:
    pd.set_option("display.width", 340)
    piv = R.pivot_table(index=["win", "fold", "half", "d0", "d1"], columns="tag",
                        values=["ic", "alpha", "turnover", "profile", "final"]).reset_index()
    piv.columns = [f"{a}_{b}" if b else a for a, b in piv.columns]
    for m, ref in (("ic", "L2"), ("alpha", "L2"), ("turnover", "L2")):
        piv[f"d{m}"] = piv[f"{m}_L2C"] - piv[f"{m}_{ref}"]
    piv["strict"] = 0.4 * piv["dic"] - 0.3 * piv["dturnover"]
    piv["withalpha"] = piv["strict"] + 0.3 * piv["dalpha"]

    print("\n=== 8 窗：L2C vs L2（受控，只差 3 列 cumintra）===")
    cols = ["win", "alpha_L2", "alpha_L2C", "dalpha", "dic", "dturnover", "strict", "withalpha"]
    print(piv[cols].round(6).to_string(index=False))

    g = piv.groupby("win")[["dalpha", "dic", "dturnover", "strict", "withalpha"]].mean()
    print(f"\n  8 窗均值: Δα {g['dalpha'].mean():+.6f}（正 {int((g['dalpha']>0).sum())}/8） "
          f"ΔIC {g['dic'].mean():+.6f}（正 {int((g['dic']>0).sum())}/8） "
          f"ΔT {g['dturnover'].mean():+.6f}")
    print(f"          strict(0.4ΔIC−0.3ΔT) {g['strict'].mean():+.6f}   "
          f"with-alpha {g['withalpha'].mean():+.6f}")

    for lbl, sel in (("wf2023 两窗", g.index.str.startswith("wf2023")),
                     ("confirm2024 两窗", g.index.str.startswith("confirm")),
                     ("非2023 六窗", ~g.index.str.startswith("wf2023"))):
        s = g[sel]
        print(f"  {lbl:16s} Δα {s['dalpha'].mean():+.6f}  ΔIC {s['dic'].mean():+.6f}")

    print("\n=== 折级配对（n=4）===")
    for metric in ("ic", "alpha", "turnover", "profile", "final"):
        r = paired_fold(R, metric)
        print(f"  {metric:9s} Δ {r['delta']:+.6f}  t={r['t']:+.2f}  [{r['signs']}]")

    print("\n=== 逐日配对（8 窗串联）===")
    for src, metric in ((DM, "ic"), (D, "alpha"), (DM, "turnover")):
        a = daily_paired(src, metric)
        x = daily_paired(src, metric, drop_wins=("wf2023A", "wf2023B"))
        print(f"  {metric:9s} 全日 Δ {a['delta']:+.6f} t={a['t']:+.2f} 正日比 {a['posfrac']:.3f} | "
              f"剔除2023 Δ {x['delta']:+.6f} t={x['t']:+.2f}")

    print("\n=== 对照：归档 L2X（40+8）在同一受控下的表现 ===")
    try:
        H = pd.read_csv(OUT / "newfeat_halves.csv")
        p = H.pivot_table(index="win", columns="tag", values=["alpha", "ic", "turnover"]).reset_index()
        p.columns = [f"{a}_{b}" if b else a for a, b in p.columns]
        da = (p["alpha_L2X"] - p["alpha_L2"]).mean()
        di = (p["ic_L2X"] - p["ic_L2"]).mean()
        dt = (p["turnover_L2X"] - p["turnover_L2"]).mean()
        print(f"  L2X vs L2 : Δα {da:+.6f}  ΔIC {di:+.6f}  ΔT {dt:+.6f}  "
              f"strict {0.4*di-0.3*dt:+.6f}  with-alpha {0.4*di-0.3*dt+0.3*da:+.6f}")
        y23 = p.win.str.startswith("wf2023")
        da23 = (p["alpha_L2X"] - p["alpha_L2"])[y23].mean()
        dax = (p["alpha_L2X"] - p["alpha_L2"])[~y23].mean()
        print(f"  L2X Δα: wf2023 {da23:+.6f} / 非2023 {dax:+.6f}（2023 占比 "
              f"{da23/da if da else np.nan:.1%}）")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    t0 = time.time()
    folds = [f.strip() for f in
             (sys.argv[1] if len(sys.argv) > 1 else ",".join(FOLDS)).split(",") if f.strip()]
    log("读因子面板...")
    fdf = load_factors(20210101, 20241231)
    log(f"因子面板 {fdf.shape}")
    R, D, DM = run(fdf, folds, ROUNDS)
    selfcheck(R)
    report(R, D, DM)
    log(f"完成，用时 {time.time()-t0:.0f}s")

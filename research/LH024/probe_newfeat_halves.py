"""L2X 的 +0.152(wf2023) 是「补足弱折」还是「只在某种 regime 有效」？

背景
----
原 4 折设计（`probe_pairwise_newfeat.py`）每折只用「训 1–8 月 / 评估 8–12 月」，
折数 n=4。wf2023（评估 2023-08~12）同时是：
  · H01 最弱的折（IC 比其余三折低 0.0316、final 低 0.0303）
  · 四个窗口里 regime 最平的一折（截面离散度最低 0.0218、vol60 中位最低 0.0197）
两种解释在 n=4 上完全混淆，无法分辨，还可能都只是噪声。

本探针把每折再切出一个**独立的样本外窗口**：
  窗口 A：训练 day[0,30%)  → 评估 day[30%,60%)   （约 4 月下 ~ 8 月上）
  窗口 B：训练 day[0,60%)  → 评估 day[60%,100%)  （约 8 月上 ~ 12 月底，= 原设计）
⇒ 8 个样本外窗口、8 个 regime 状态，n 翻倍且**每折内部也有 regime 变化**。
受控不变：候选池、特征集（40 vs 48）、超参、band、评分口径；只改训练天数与评估窗口。

判据（预注册）
--------------
① 若 Δα 在 2023 两个半窗都为正，且随 regime（低离散度/低 vol60）单调 → 「regime 假说」成立，
   该效应可从市场状态预报；
② 若 Δα 只在 2023-08~12 为正 → 单窗口异常，方向关闭；
③ 用 8 点分别算 corr(Δα, H01 窗口内弱度) 与 corr(Δα, regime)，看哪个更能解释。

产物：cv_design_audit/newfeat_halves.csv、newfeat_halves_daily.csv
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
OUT.mkdir(parents=True, exist_ok=True)
for _p in (str(REPO), str(REPO / "research" / "LH023"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_pairwise_rerank import (  # noqa: E402
    FEATS, FOLDS, build_fold, eval_arm, group_sizes, log, s2_of,
    stage2_encode, train_rows,
)
from probe_pairwise_newfeat import (  # noqa: E402
    NEW8, build_new_features, fit_l2, s2_from_booster, slice_new,
)
from _probe_dgtw import stratify  # noqa: E402
from _probe_e_attr import load_factors, wide_of  # noqa: E402

ROUNDS = 500
CUT_A, CUT_B = 0.30, 0.60


def regime_of(panel, W) -> pd.DataFrame:
    """逐日 regime 变量（归因用，含同期信息，不是可交易信号）。"""
    Y, vol = panel.Y, W["vol60"]
    rows = []
    for t in range(panel.T):
        y, v = Y[t], vol[t]
        m = np.isfinite(y)
        if m.sum() < 100:
            continue
        b = stratify(v, 10)
        lo, hi = m & (b <= 2), m & (b >= 7)
        rows.append(dict(
            trade_date=int(panel.dates[t]),
            disp=float(np.nanstd(y[m])),
            medvol=float(np.nanmedian(v[np.isfinite(v)])),
            volspread=(float(np.nanmean(y[lo]) - np.nanmean(y[hi]))
                       if lo.sum() > 10 and hi.sum() > 10 else np.nan),
        ))
    return pd.DataFrame(rows)


def run(folds, rounds: int) -> pd.DataFrame:
    feats = build_new_features()
    rows, dailies, regs = [], [], []
    for fold in folds:
        b = build_fold(fold)
        W = wide_of(fdf, b["panel"])
        panel = b["panel"]
        T = panel.T
        regs.append(regime_of(panel, W).assign(fold=fold))

        X59 = b["X"]
        Xn, _ = slice_new(feats, panel)
        X67 = np.hstack([X59, Xn]).astype("float32", copy=False)
        s2h = s2_of(b, "H01")

        for tag_w, cut, ev_cut in (("A", CUT_A, CUT_B), ("B", CUT_B, 1.0)):
            n_tr, n_ev = int(T * cut), int(T * ev_cut)
            days_tr = np.arange(0, n_tr)
            days_ev = np.arange(n_tr, n_ev)
            idx = train_rows(b, days_tr)
            ytr = b["long"]["y"][idx].astype("float64")
            grp = group_sizes(b["long"]["ti"][idx])
            assert grp.sum() == len(idx)
            win = f"{fold}{tag_w}"
            log(f"{win}: 训练 {len(idx):,} 行/{len(grp)} 天(d[0,{n_tr})) "
                f"评估 d[{n_tr},{n_ev}) 共 {len(days_ev)} 天")

            for arm, Xtr, Xfull, names, use2 in (
                ("H01", None, None, None, False),
                ("L2", X59[idx], X59, FEATS, False),
                ("L2X", X67[idx], X67, FEATS + NEW8, True),
            ):
                if arm == "H01":
                    s2 = s2h
                else:
                    bo = fit_l2(Xtr, ytr, grp, names, rounds)
                    s2 = s2_from_booster(b, Xfull, bo)
                row, d, _ = eval_arm(stage2_encode(b["pr_mix"], s2, b["cand"]), b, W, arm, days_ev)
                row.update(win=win, fold=fold, half=tag_w,
                           d0=int(panel.dates[days_ev[0]]), d1=int(panel.dates[days_ev[-1]]))
                rows.append(row); dailies.append(d.assign(win=win, tag=arm))
                log(f"  {win} {arm:4s}: IC {row['ic']:.6f} E {row['excess']:.6f} "
                    f"T {row['turnover']:.6f} a {row['alpha']:.6f} prof {row['profile']:.6f} "
                    f"F {row['final']:.6f}")
        del X67, Xn, idx
        gc.collect()

    R = pd.DataFrame(rows)
    D = pd.concat(dailies, ignore_index=True)
    G = pd.concat(regs, ignore_index=True)
    R = R.merge(G.groupby("fold")[["disp", "medvol", "volspread"]].mean().reset_index(),
                on="fold", how="left")
    # 窗口自身的 regime（用该窗口天数的逐日均值）
    win_reg = []
    for win, g in D[D.tag == "H01"].groupby("win"):
        gg = G[(G.fold == win[:-1]) & (G.trade_date.between(g.trade_date.min(), g.trade_date.max()))]
        win_reg.append(dict(win=win, w_disp=gg["disp"].mean(), w_medvol=gg["medvol"].mean(),
                            w_volspread=gg["volspread"].mean(), w_days=len(g)))
    R = R.merge(pd.DataFrame(win_reg), on="win", how="left")
    R.to_csv(OUT / "newfeat_halves.csv", index=False)
    D.to_csv(OUT / "newfeat_halves_daily.csv", index=False)
    return R, D


def report(R: pd.DataFrame) -> None:
    pd.set_option("display.width", 340)
    piv = R.pivot_table(index=["win", "fold", "half", "d0", "d1", "w_days",
                               "w_disp", "w_medvol", "w_volspread"],
                        columns="tag", values=["alpha", "ic", "profile", "final"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["da"] = piv["alpha_L2X"] - piv["alpha_L2"]
    piv["da_h01"] = piv["alpha_L2X"] - piv["alpha_H01"]
    piv["dic"] = piv["ic_L2X"] - piv["ic_L2"]
    piv["dprof"] = piv["profile_L2X"] - piv["profile_L2"]
    piv["dfin"] = piv["final_L2X"] - piv["final_L2"]
    cols = ["win", "d0", "d1", "w_days", "w_disp", "w_medvol", "alpha_H01", "alpha_L2",
            "alpha_L2X", "da", "dic", "dprof", "dfin"]
    print("\n=== 8 个样本外窗口（受控：L2X vs L2 只差 8 列）===")
    print(piv[cols].round(6).to_string(index=False))

    def line(lbl, sub):
        print(f"  {lbl:12s} n={len(sub)} Δα {sub['da'].mean():+.6f}（正 {int((sub['da'] > 0).sum())}/"
              f"{len(sub)}） ΔIC {sub['dic'].mean():+.6f}（正 {int((sub['dic'] > 0).sum())}/{len(sub)}） "
              f"Δprof {sub['dprof'].mean():+.6f}  regime 中位vol {sub['w_medvol'].mean():.4f}")

    print("\n=== 汇总 ===")
    line("全部 8 窗", piv)
    for fold in piv["fold"].unique():
        line(fold, piv[piv.fold == fold])
    for half in ("A", "B"):
        line(f"半窗 {half}", piv[piv.half == half])
    line("2023 两窗", piv[piv.fold == "wf2023"])
    line("非2023 6窗", piv[piv.fold != "wf2023"])

    print("\n=== 两个竞争解释：8 点相关（n=8，p<0.05 需 |r|>0.707）===")
    for name, x in (("H01 窗口内 α（越弱越该受益）", "alpha_H01"),
                    ("窗口 regime：截面离散度", "w_disp"),
                    ("窗口 regime：vol60 中位", "w_medvol"),
                    ("窗口 regime：低波-高波价差", "w_volspread")):
        print(f"  corr(Δα, {name:24s}) = {piv['da'].corr(piv[x], method='spearman'):+.3f}")
    for name, x in (("H01 窗口内 IC", "ic_H01"), ("H01 窗口内 final", "final_H01")):
        try:
            print(f"  corr(Δα, {name:24s}) = {piv['da'].corr(piv[x], method='spearman'):+.3f}")
        except KeyError:
            pass
    for name, x in (("H01 窗口内 α", "alpha_H01"),
                    ("窗口 vol60 中位", "w_medvol"),
                    ("窗口截面离散度", "w_disp")):
        print(f"  corr(ΔIC, {name:24s}) = {piv['dic'].corr(piv[x], method='spearman'):+.3f}")


if __name__ == "__main__":
    t0 = time.time()
    folds = [f.strip() for f in
             (sys.argv[1] if len(sys.argv) > 1 else ",".join(FOLDS)).split(",") if f.strip()]
    log("读因子面板...")
    fdf = load_factors(20210101, 20241231)
    log(f"因子面板 {fdf.shape}")
    R, D = run(folds, ROUNDS)
    report(R)
    log(f"完成，用时 {time.time()-t0:.0f}s")

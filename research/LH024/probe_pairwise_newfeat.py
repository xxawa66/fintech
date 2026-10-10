"""方向二判定探针：把 8 个新特征加进阶段2 重排器（受控对照）。

假说
----
阶段2（池内重排）目前只有 H01 一个信号。上一轮实测 H01 在候选池内的 rank IC
≈ −0.009（近乎正交于 y）⇒ 池内排序几乎不携带信息，这是 α 的上限来源。
方向二已筛出 8 个「与 vol60 正交 + 与 H01 正交 + 四折同号 + 独立 α 为正」的新特征，
本探针只回答一个问题：

    把它们加进阶段2 的输入特征，Δα 是多少？

受控设计（与 probe_pairwise_rerank.py 的 split 模式完全同构）
-----------------------------------------------------------
固定不动：候选池（TS_K40 阶段1，mix top 40%）、训练样本行（池内 & 有 y &
          折内前 60% 天）、超参（T030）、boosting 轮数、band(keep_q=q*)、
          评估窗口（折内后 40% 天）、评分口径。
唯一变量 = 阶段2 重排器的输入特征矩阵：
  H01 : 归档 TS_K40 的阶段2（不训练）——基线
  L2  : 59 特征点式回归（= 上一轮 split 的 L2，应逐位复现）
  L2X : 59 + 8 新特征

判定看的是 **L2X vs L2** 的配对 Δ，不是 L2X vs H01。L2→L2X 只改了 8 列，
是唯一干净的因果对照；H01 是归档产物、训练协议不同，只能当参照。

自检
----
① L2 臂必须复现上一轮 cv_design_audit/pairwise_rerank_split.csv 的 L2 行；
② 新特征在训练行上的 NaN 比例要打印（超窗口 / 覆盖不足的警报）。

产物（仓库外，未入库）：
  cv_design_audit/newfeat_stage2_split.csv
  cv_design_audit/newfeat_stage2_daily_split.csv
  cv_design_audit/newfeat_stage2_importance.csv
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

import argparse  # noqa: E402
import gc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402

HERE = Path(__file__).resolve()
REPO = _REPO
OUT = _LH024_DIR / "cv_design_audit"
OUT.mkdir(parents=True, exist_ok=True)
for _p in (str(REPO), str(REPO / "research" / "LH023"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_pairwise_rerank import (  # noqa: E402
    FEATS, FOLDS, KQ, T030_PARAMS, build_fold, diag, eval_arm, group_sizes,
    log, s2_of, stage2_encode, train_rows, _paired_fold,
)
from probe_new_features import build_features, load_raw  # noqa: E402
from _probe_dgtw import pct_rank_2d  # noqa: E402
from _probe_e_attr import load_factors, wide_of  # noqa: E402

# 方向二筛出的最小候选集（见 cv_design_audit/new_feature_screen.csv）
NEW8 = ["cumintra5", "cumintra20", "cumintra60", "gapspread20",
        "corrmkt60", "corrmkt120", "amihud20", "amihud60"]


# --------------------------------------------------------------------------
# 新特征：在完整面板上一次算完，再切片到折（滚动窗口需要折外历史）
# --------------------------------------------------------------------------
def build_new_features() -> dict[str, pd.DataFrame]:
    raw = load_raw()
    raw = raw.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    from _probe_root_cause import Panel
    panel_full = Panel(raw)
    log(f"特征面板 {panel_full.T} 天 × {panel_full.N} 只")
    allfeat = build_features(panel_full, raw)
    keep = {nm: allfeat[nm].astype("float32") for nm in NEW8 if nm in allfeat}
    missing = [nm for nm in NEW8 if nm not in keep]
    if missing:
        raise ValueError(f"build_features 缺列：{missing}")
    del allfeat, raw
    gc.collect()
    return keep


def slice_new(feats: dict[str, pd.DataFrame], panel) -> tuple[np.ndarray, dict[str, float]]:
    """把全样本宽表切片到折面板，返回长表矩阵 (n, 8) 与各列 NaN 比例。"""
    cols, nanfrac = [], {}
    for nm in NEW8:
        A = feats[nm].reindex(index=panel.dates, columns=panel.codes).to_numpy(dtype="float32")
        v = A[panel.ti, panel.si]
        cols.append(v)
        nanfrac[nm] = float(np.isnan(v).mean())
    return np.column_stack(cols), nanfrac


# --------------------------------------------------------------------------
# 训练 / 编码（L2 与 L2X 只差输入列）
# --------------------------------------------------------------------------
def fit_l2(Xtr, ytr, groups, names, rounds: int):
    p = dict(T030_PARAMS)
    p.update(objective="regression", metric="l2")
    ds = lgb.Dataset(Xtr, label=ytr, group=groups, feature_name=list(names),
                     params=p, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=rounds)


def s2_from_booster(b, X_full, booster) -> np.ndarray:
    panel, cand = b["panel"], b["cand"]
    sc = booster.predict(X_full, num_threads=8)
    M = np.full(panel.Y.shape, np.nan)
    M[panel.ti, panel.si] = sc
    M = np.where(cand & np.isfinite(M), M, np.nan)
    return pct_rank_2d(M)


def run(fdf, folds, rounds: int, h01_only: bool) -> pd.DataFrame:
    rows, dailies, dailym, imps, nans = [], [], [], [], []
    feats = None if h01_only else build_new_features()

    for fold in folds:
        b = build_fold(fold)
        W = wide_of(fdf, b["panel"])
        T = b["panel"].T
        n_tr = int(T * 0.6)
        days_tr = np.arange(0, n_tr)
        days_ev = np.arange(n_tr, T)

        s2h = s2_of(b, "H01")
        row, d, dd = eval_arm(stage2_encode(b["pr_mix"], s2h, b["cand"]), b, W, "H01", days_ev)
        rows.append(row); dailies.append(d); dailym.append(dd)
        log(f"  {fold} H01 : IC {row['ic']:.6f} E {row['excess']:.6f} T {row['turnover']:.6f} "
            f"a {row['alpha']:.6f} prof {row['profile']:.6f} F {row['final']:.6f}")
        if h01_only:
            continue

        X59 = b["X"]
        Xn, nanfrac = slice_new(feats, b["panel"])
        nans.append(dict(fold=fold, **nanfrac))
        X67 = np.hstack([X59, Xn]).astype("float32", copy=False)

        idx = train_rows(b, days_tr)
        ytr = b["long"]["y"][idx].astype("float64")
        grp = group_sizes(b["long"]["ti"][idx])
        assert grp.sum() == len(idx), (grp.sum(), len(idx))
        nan_tr = {k: float(np.isnan(v[idx]).mean()) for k, v in
                  zip(NEW8, Xn.T)}
        log(f"{fold}: split 训练 {len(idx):,} 行 / {len(grp)} 天；"
            f"评估窗口 {T - n_tr} 天；新特征训练行 NaN "
            f"{max(nan_tr.values()):.3f}(max) / {sum(nan_tr.values())/len(nan_tr):.3f}(mean)")

        for arm, Xtr, Xfull, names in (
            ("L2", X59[idx], X59, FEATS),
            ("L2X", X67[idx], X67, FEATS + NEW8),
        ):
            t0 = time.time()
            bo = fit_l2(Xtr, ytr, grp, names, rounds)
            log(f"    {arm} 拟合完成 {time.time()-t0:.0f}s  ({len(names)} 特征)")
            s2 = s2_from_booster(b, Xfull, bo)
            pic, ch1 = diag(b, s2, s2h, days_ev)
            row, d, dd = eval_arm(stage2_encode(b["pr_mix"], s2, b["cand"]), b, W, arm, days_ev)
            row.update(pool_ic=pic, corr_h01=ch1)
            rows.append(row); dailies.append(d); dailym.append(dd)
            log(f"    {fold} {arm:4s}: IC {row['ic']:.6f} E {row['excess']:.6f} "
                f"T {row['turnover']:.6f} a {row['alpha']:.6f} prof {row['profile']:.6f} "
                f"F {row['final']:.6f} | poolIC {pic:.4f} corrH01 {ch1:.4f}")

            g = bo.feature_importance(importance_type="gain")
            order = np.argsort(-g)
            for r_, j in enumerate(order):
                imps.append(dict(fold=fold, arm=arm, rank=r_ + 1,
                                 feature=names[j], gain=float(g[j]),
                                 is_new=int(names[j] in NEW8)))

        del X67, Xn, idx
        gc.collect()

    R = pd.DataFrame(rows)
    D = pd.concat(dailies, ignore_index=True)
    DM = pd.concat(dailym, ignore_index=True)
    R.to_csv(OUT / "newfeat_stage2_split.csv", index=False)
    D.to_csv(OUT / "newfeat_stage2_daily_split.csv", index=False)
    DM.to_csv(OUT / "newfeat_stage2_daymetrics_split.csv", index=False)
    if imps:
        pd.DataFrame(imps).to_csv(OUT / "newfeat_stage2_importance.csv", index=False)
    if nans:
        pd.DataFrame(nans).to_csv(OUT / "newfeat_stage2_nanfrac.csv", index=False)

    report(R, D, DM)
    if imps:
        report_importance(pd.DataFrame(imps))
    return R


def report(R: pd.DataFrame, D: pd.DataFrame, DM: pd.DataFrame) -> None:
    pd.set_option("display.width", 320)
    print("\n=== [split] 逐折 ===")
    cols = ["fold", "tag", "ic", "excess", "turnover", "alpha", "profile",
            "final", "noprofile", "pool_ic", "corr_h01", "n_slice"]
    print(R[[c for c in cols if c in R.columns]].round(6).to_string(index=False))

    print("\n=== [split] 均值 ===")
    g = R.groupby("tag")[["ic", "excess", "turnover", "alpha", "profile",
                          "final", "noprofile", "pool_ic", "corr_h01"]].mean()
    print(g.round(6).to_string())

    for base in ("H01", "L2"):
        if base not in set(R["tag"]):
            continue
        print(f"\n=== [split] 折级配对 Δ（vs {base}）===")
        out = []
        for metric in ("ic", "excess", "turnover", "alpha", "profile",
                       "final", "noprofile", "pool_ic"):
            out.extend(_paired_fold(R, metric, base=base))
        P = pd.DataFrame(out)[["tag", "metric", "delta", "t", "signs", "n"]]
        print(P.round(5).to_string(index=False))
        P.to_csv(OUT / f"newfeat_stage2_split_paired_vs_{base}.csv", index=False)

    print("\n=== [split] 逐日配对 Δ（不依赖折数）===")
    dd = []
    for src, metric in ((DM, "ic"), (DM, "excess"), (D, "alpha"), (D, "profile")):
        w = src.pivot_table(index="trade_date", columns="tag", values=metric)
        for base in ("H01", "L2"):
            if base not in w.columns:
                continue
            for tag in w.columns:
                if tag == base:
                    continue
                d = (w[tag] - w[base]).dropna()
                n = len(d)
                sd = float(np.std(d.to_numpy(), ddof=1))
                dd.append({"vs": base, "tag": tag, "metric": metric, "n": n,
                           "delta": float(d.mean()),
                           "t": float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan})
    Dp = pd.DataFrame(dd)
    print(Dp.round(6).to_string(index=False))
    Dp.to_csv(OUT / "newfeat_stage2_split_dailypaired.csv", index=False)


def report_importance(I: pd.DataFrame) -> None:
    print("\n=== feature importance（gain，四折均值；新特征已标记）===")
    for arm in I["arm"].unique():
        sub = I[I.arm == arm]
        g = sub.groupby(["feature", "is_new"])["gain"].mean().reset_index()
        g["rank"] = g["gain"].rank(ascending=False).astype(int)
        tot = g["gain"].sum()
        g["share"] = g["gain"] / tot
        new = g[g.is_new == 1].sort_values("rank")
        print(f"\n  --- {arm}（{len(g)} 特征）---")
        print(f"  新特征合计 gain 份额：{new['share'].sum():.4f}")
        print(new[["rank", "feature", "gain", "share"]].round(5).to_string(index=False))
        top = g.nsmallest(12, "rank")[["rank", "feature", "share"]]
        print("  前 12 名：")
        print(top.round(5).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="方向二：新特征进阶段2 重排器的受控对照")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--h01-only", action="store_true", help="只跑 H01 基线（不训练，用于自检）")
    a = ap.parse_args(argv)
    folds = [f.strip() for f in a.folds.split(",") if f.strip()]

    t0 = time.time()
    log("读因子面板（DGTW 分解用）...")
    fdf = load_factors(20210101, 20241231)
    log(f"因子面板 {fdf.shape}")
    run(fdf, folds, a.rounds, a.h01_only)
    log(f"完成，用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

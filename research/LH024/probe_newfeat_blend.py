"""blend 扫描：阶段2 分数 = w·s2(H01) + (1−w)·s2(L2X)。

问题
----
L2X（阶段2 输入 40→48 列）的 Δα 集中在 wf2023 两个半窗（占 8 窗总量 83.5%）；
而 H01 最强的 confirm2024 两窗，Δα 反而是负的（−0.0130 / −0.0167）。
⇒ L2X 不能替换 H01（折序被打乱），但可能是「H01 失效时的保险」。

本探针只回答一个问题：
    是否存在 w，使得同时「保住 confirm2024 的 H01 工时」与「拿到 2023 的增益」？

设计
----
固定不动：候选池（mix top 40%）、8 个样本外窗口（4 折 × 2 半窗，同
          probe_newfeat_halves.py）、训练协议（窗口内前段天数）、超参、band(keep_q=q*)、
          评分口径、H01 / L2X 两个信号本身（重训结果从缓存复用）。
唯一变量 = 混合权重 w：s2 = w·s2_H01 + (1−w)·s2_L2X，w ∈ {0.0,0.1,...,1.0}。
  w=1 → 纯 H01（= 基线，必须逐位复现 newfeat_halves.csv 的 H01 行）
  w=0 → 纯 L2X（必须逐位复现 L2X 行）

两者都是「池内百分位」（pct_rank_2d 后，池外 NaN），同尺度，可直接线性混合。
NaN 处理：只在两边都有限处加权；只有一边有限则取那一边（避免 0·NaN=NaN 的坑）。

判据（预注册）
--------------
① 若 w* → 1（内部无极值）⇒ blend 无用，关闭。
② 若 0<w*<1 且能同时满足 confirm2024 Δα ≥ −0.005、wf2023 Δα ≥ +0.02 ⇒ 保险成立。
③ 报告 w 与两个口径（严格 = 0.4·ΔIC+0.3·Δ(1−T)；含 Δα）的曲线，看是否有平台。

产物（仓库外）：cv_design_audit/newfeat_blend_windows.csv（逐窗×w）、
               newfeat_blend_scan.csv（汇总）、s2_cache/*.npz（信号缓存）
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

HERE = Path(__file__).resolve()
REPO = _REPO
OUT = _LH024_DIR / "cv_design_audit"
CACHE = OUT / "s2_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
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
from _probe_e_attr import load_factors, wide_of  # noqa: E402

ROUNDS = 500
WS = [round(i / 10.0, 1) for i in range(11)]
WINDOWS = (("A", 0.30, 0.60), ("B", 0.60, 1.00))
ANN = 252


def blend(s2h: np.ndarray, s2x: np.ndarray, w: float) -> np.ndarray:
    """两边都有限处加权；只有一边有限则取那一边。"""
    okh, okx = np.isfinite(s2h), np.isfinite(s2x)
    out = np.full(s2h.shape, np.nan, dtype="float64")
    both = okh & okx
    out[both] = w * s2h[both] + (1.0 - w) * s2x[both]
    out[okh & ~okx] = s2h[okh & ~okx]
    out[okx & ~okh] = s2x[okx & ~okh]
    return out


# --------------------------------------------------------------------------
# 阶段 1：算并缓存每窗口的两个信号
# --------------------------------------------------------------------------
def build_cache(folds, force: bool) -> None:
    todo = [(f, t) for f in folds for t, _, _ in WINDOWS
            if force or not (CACHE / f"{f}{t}.npz").exists()]
    if not todo:
        log("信号缓存齐备，跳过训练")
        return
    feats = build_new_features()
    for fold in folds:
        need = [t for f_, t in todo if f_ == fold]
        if not need:
            continue
        b = build_fold(fold)
        panel = b["panel"]
        T = panel.T
        s2h = s2_of(b, "H01")
        X59 = b["X"]
        Xn, _ = slice_new(feats, panel)
        X67 = np.hstack([X59, Xn]).astype("float32", copy=False)
        for tag_w, cut, ev_cut in WINDOWS:
            fp = CACHE / f"{fold}{tag_w}.npz"
            if not force and fp.exists():
                continue
            n_tr = int(T * cut)
            idx = train_rows(b, np.arange(0, n_tr))
            ytr = b["long"]["y"][idx].astype("float64")
            grp = group_sizes(b["long"]["ti"][idx])
            t0 = time.time()
            bo = fit_l2(X67[idx], ytr, grp, FEATS + NEW8, ROUNDS)
            s2x = s2_from_booster(b, X67, bo)
            np.savez_compressed(fp, s2h=s2h.astype("float32"), s2x=s2x.astype("float32"))
            log(f"  {fold}{tag_w}: 训练 {len(idx):,} 行/{len(grp)} 天，"
                f"缓存 {fp.name}（{time.time()-t0:.0f}s）")
        del X67, Xn
        gc.collect()
    del feats
    gc.collect()


# --------------------------------------------------------------------------
# 阶段 2：扫 w
# --------------------------------------------------------------------------
def scan(fdf, folds) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, dailies, dailym = [], [], []
    for fold in folds:
        b = build_fold(fold)
        W = wide_of(fdf, b["panel"])
        panel = b["panel"]
        T = panel.T
        for tag_w, cut, ev_cut in WINDOWS:
            fp = CACHE / f"{fold}{tag_w}.npz"
            z = np.load(fp)
            s2h = z["s2h"].astype("float64")
            s2x = z["s2x"].astype("float64")
            days_ev = np.arange(int(T * cut), int(T * ev_cut))
            win = f"{fold}{tag_w}"
            for w in WS:
                s2 = blend(s2h, s2x, w)
                P0 = stage2_encode(b["pr_mix"], s2, b["cand"])
                row, d, dd = eval_arm(P0, b, W, f"w{w:.1f}", days_ev)
                row.update(win=win, fold=fold, half=tag_w, w=w,
                           d0=int(panel.dates[days_ev[0]]),
                           d1=int(panel.dates[days_ev[-1]]), n_days=len(days_ev))
                rows.append(row)
                dailies.append(d.assign(win=win, fold=fold, half=tag_w, w=w))
                dailym.append(dd.assign(win=win, fold=fold, half=tag_w, w=w))
            log(f"  {win}: 扫完 {len(WS)} 个 w")
        del b
        gc.collect()
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "newfeat_blend_windows.csv", index=False)
    pd.concat(dailies, ignore_index=True).to_csv(
        OUT / "newfeat_blend_daily.csv", index=False)
    pd.concat(dailym, ignore_index=True).to_csv(
        OUT / "newfeat_blend_daymetrics.csv", index=False)
    return R


def _selfcheck(R: pd.DataFrame) -> None:
    print("\n=== 自检：w=1 应逐位复现 H01；w=0 应逐位复现 L2X ===")
    try:
        H = pd.read_csv(OUT / "newfeat_halves.csv")
        H["win"] = H["fold"] + H["half"]
    except FileNotFoundError:
        print("  未找到 newfeat_halves.csv，跳过")
        return
    piv = R.pivot_table(index=["win"], columns="w", values=["ic", "alpha", "final", "turnover"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    for w, tag in ((1.0, "H01"), (0.0, "L2X")):
        if f"final_{w}" not in piv.columns:
            continue
        j = H[H.tag == tag].set_index("win")
        dd = (piv[f"final_{w}"] - j["final"]).abs()
        print(f"  w={w} vs 归档 {tag}: max|Δfinal| = {dd.max():.2e}  "
              f"（IC max {np.abs(piv[f'ic_{w}']-j['ic']).max():.2e}）")
        assert dd.max() < 1e-6, f"w={w} 未复现 {tag}"


def report(R: pd.DataFrame) -> pd.DataFrame:
    pd.set_option("display.width", 340)
    GROUPS = {
        "全 8 窗": lambda d: d,
        "2023 两窗": lambda d: d[d.fold == "wf2023"],
        "confirm2024 两窗": lambda d: d[d.fold == "confirm2024"],
        "2021 两窗": lambda d: d[d.fold == "wf2021"],
        "2022 两窗": lambda d: d[d.fold == "wf2022"],
    }
    # 基线 w=1（纯 H01），用于算 Δ
    base = R[R.w == 1.0].set_index("win")

    out = []
    for gname, fn in GROUPS.items():
        sub = fn(R)
        for w in WS:
            s = sub[sub.w == w].set_index("win")
            bsel = base.reindex(s.index)
            dic = float((s["ic"] - bsel["ic"]).mean())
            da = float((s["alpha"] - bsel["alpha"]).mean())
            dprof = float((s["profile"] - bsel["profile"]).mean())
            dturn = float((s["turnover"] - bsel["turnover"]).mean())
            dex = float((s["excess"] - bsel["excess"]).mean())
            strict = 0.4 * dic - 0.3 * dturn          # 不含 Δα
            witha = strict + 0.3 * da                 # 含 Δα
            out.append(dict(group=gname, w=w,
                            ic=float(s["ic"].mean()), excess=float(s["excess"].mean()),
                            turnover=float(s["turnover"].mean()), alpha=float(s["alpha"].mean()),
                            profile=float(s["profile"].mean()), final=float(s["final"].mean()),
                            noprofile=float(s["noprofile"].mean()),
                            dic=dic, dex=dex, da=da, dprof=dprof, dturn=dturn,
                            strict_delta=strict, with_alpha_delta=witha, n_wins=len(s)))
    S = pd.DataFrame(out)
    S.to_csv(OUT / "newfeat_blend_scan.csv", index=False)

    print("\n=== 逐窗 × w（节选：Δ vs 纯 H01）===")
    piv = R.pivot_table(index="win", columns="w", values=["ic", "alpha", "final"])
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        if ("alpha", w) not in piv.columns:
            continue
        da = piv[("alpha", w)] - piv[("alpha", 1.0)]
        dic = piv[("ic", w)] - piv[("ic", 1.0)]
        print(f"\n  --- w={w} ---")
        t = pd.DataFrame({"Δα": da, "ΔIC": dic}).round(6)
        print(t.to_string())
        for gname in ("2023 两窗", "confirm2024 两窗", "全 8 窗"):
            sub = GROUPS[gname](R)
            ss = sub[sub.w == w].set_index("win")
            bs = base.reindex(ss.index)
            print(f"      {gname:16s} Δα {(ss['alpha']-bs['alpha']).mean():+.6f} "
                  f"ΔIC {(ss['ic']-bs['ic']).mean():+.6f}")

    print("\n=== 汇总：w 曲线（Δ vs 纯 H01）===")
    for gname, fn in GROUPS.items():
        sub = S[S.group == gname]
        print(f"\n  --- {gname} ---")
        print(sub[["w", "ic", "alpha", "profile", "turnover", "dic", "da",
                   "strict_delta", "with_alpha_delta"]].round(6).to_string(index=False))
    return S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阶段2 H01/L2X blend 扫描")
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--force", action="store_true", help="强制重训并覆盖信号缓存")
    ap.add_argument("--cache-only", action="store_true")
    a = ap.parse_args(argv)
    folds = [f.strip() for f in a.folds.split(",") if f.strip()]

    t0 = time.time()
    log("阶段1：准备信号缓存...")
    build_cache(folds, a.force)
    if a.cache_only:
        log(f"仅缓存，用时 {time.time()-t0:.0f}s")
        return 0
    log("读因子面板（DGTW 分解用）...")
    fdf = load_factors(20210101, 20241231)
    log(f"因子面板 {fdf.shape}")
    log("阶段2：扫 w...")
    R = scan(fdf, folds)
    log("阶段3：报告")
    _selfcheck(R)
    S = report(R)
    log(f"完成，用时 {time.time()-t0:.0f}s")

    print("\n=== 判据检查 ===")
    c24 = S[(S.group == "confirm2024 两窗")]
    y23 = S[(S.group == "wf2023 两窗")]
    allg = S[S.group == "全 8 窗"]
    best = allg.loc[allg["with_alpha_delta"].idxmax()]
    print(f"  含 Δα 口径最优 w = {best['w']:.1f}（增益 {best['with_alpha_delta']:+.5f}）")
    bs = allg.loc[allg["strict_delta"].idxmax()]
    print(f"  严格口径最优 w   = {bs['w']:.1f}（增益 {bs['strict_delta']:+.5f}）")
    for _, r in c24.iterrows():
        sel = y23[y23.w == r["w"]]["da"]
        yv = float(sel.iloc[0]) if len(sel) else np.nan
        print(f"  w={r['w']:.1f}: confirm2024 Δα {r['da']:+.5f} | 2023 Δα "
              f"{yv:+.5f} | 8窗含Δα {r['with_alpha_delta']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

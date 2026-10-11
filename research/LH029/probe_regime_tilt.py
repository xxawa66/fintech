"""LH029 —— 牛熊分析能不能改变预测结构？把「完美预知」的 oracle 上限直接跑出来。

背景
----
LH020/LH027 的结论：长周期模型在 CV 上多出的 E **全部来自 profile**（vol60 十层倾斜
带来的风格暴露），而 profile 是 regime 依赖的——2018-2024 的 L0−L9 是 +43.21pp/年，
2025-2026 只有 +12.65pp（且逐年看是断崖：…57.0 / 40.0 / 53.5 → 15.96 / 4.60）。

⇒ 自然会问：能不能**预测牛熊**，该要 profile 的时候倾斜、不该要的时候不倾斜？

本脚本不猜，直接把上界跑出来。三档 tilt 施加在同一份 TS_K40 两阶段打分之上：

  score_t = P0 + phi_t · ( prank(−vol60) − 0.5 )      （只在两阶段内部重排，不破坏候选池边界）

  const φ      静态倾斜 ⇒ 给出「profile 的价格」：买 1pp profile 要付多少换手 / 多少 α
  oracle φ     每月的符号 = **该月已实现的 L0−L9**（前视，故意的）⇒ 牛熊择时的**上界**
  pred   φ     每月的符号 = 只用历史的扩展窗口 OLS 预测 ⇒ 真实可达版，用来对照

判据
----
四折 CV（2021-2024）配对 Δ / t / 逐折符号，口径 0.4·IC + 0.3·α + 0.3·(1−T)（不含 profile，
因为 profile 正是被测对象，把它放进判据等于自证）。
**全程只用 2021-2024，不读 2025 以后数据。**

自检：φ=0 必须逐位复现归档 TS_K40 四折。
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P

_HERE = _P(__file__).resolve()
_REPO = next(_p for _p in _HERE.parents if (_p / "configs" / "project.yaml").exists())
_OUT = _REPO / "experiments" / "LH029"
_OUT.mkdir(parents=True, exist_ok=True)

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time

import numpy as np
import pandas as pd

REPO = _REPO
CACHE = REPO / "outputs" / "long_horizon"
OUT = _OUT
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/LH023"))

from _probe_dgtw import decompose, pct_rank_2d, stratify  # noqa: E402
from _probe_e_attr import load_factors, top_sets, wide_of  # noqa: E402
from _probe_root_cause import (  # noqa: E402
    Panel,
    fast_band,
    holding_stats,
    load_fold,
    score,
    to_wide,
)

FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
KQ = 0.0022778298112255263
K_STAGE1 = 40.0
ANN = 252
NBIN = 10
LAYER = "vol60"
WEIGHTS = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
COSTS = [0.0010, 0.0020, 0.0025, 0.0040]
# tilt 幅度：delta = phi*(prank-0.5) ∈ [-phi/2, +phi/2]；phi<1 才能保住候选池与非候选池的分离
PHI_STATIC = [-0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.4, 0.6, 0.8]
PHI_TIMED = [0.2, 0.4]

ARCHIVE = {
    "wf2021": dict(ic=0.081156, excess=0.257336, turnover=0.022824,
                   final=0.402816, alpha=0.256086, profile=-0.009986),
    "wf2022": dict(ic=0.097536, excess=0.207012, turnover=0.016528,
                   final=0.396160, alpha=0.150050, profile=0.051173),
    "wf2023": dict(ic=0.069012, excess=0.138517, turnover=0.012252,
                   final=0.365484, alpha=0.140308, profile=-0.006093),
    "confirm2024": dict(ic=0.097487, excess=0.230244, turnover=0.021913,
                        final=0.401494, alpha=0.209000, profile=0.018148),
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
# 月度 L0-L9 spread 的历史：给 honest forecast 用，也用来核对 F1 那条逐年表
# --------------------------------------------------------------------------
def monthly_history() -> pd.DataFrame:
    cols = ["ts_code", "trade_date", "y_ret_1d", "flag_limit_up", "vol60"]
    df = pd.read_parquet(CACHE / "_probe_factors_full.parquet", columns=cols,
                         filters=[("trade_date", ">=", 20180102), ("trade_date", "<=", 20241231)])
    df["ts_code"] = df["ts_code"].astype(str)
    df["trade_date"] = df["trade_date"].astype("int64")
    for c in ["y_ret_1d", "vol60"]:
        df[c] = df[c].astype("float64")
    rows = []
    for d, g in df.groupby("trade_date", sort=True):
        y = g["y_ret_1d"].to_numpy()
        lu = g["flag_limit_up"].to_numpy() == 1
        vol = g["vol60"].to_numpy()
        inv = (~lu) & np.isfinite(y) & np.isfinite(vol)
        if inv.sum() < 300:
            continue
        b = stratify(vol[inv], NBIN)
        yy = y[inv]
        ok = b >= 0
        if ok.sum() < 300:
            continue
        s0 = b == 0
        s9 = b == 9
        rows.append(dict(trade_date=int(d), ym=int(d) // 100,
                         sp=float(yy[s0 & ok].mean() - yy[s9 & ok].mean()),
                         mkt=float(yy[ok].mean())))
    Dt = pd.DataFrame(rows)
    Mh = Dt.groupby("ym").agg(sp=("sp", "mean"), mkt=("mkt", "mean"), days=("sp", "size"))
    for c in ["sp", "mkt"]:
        Mh[c] = Mh[c] * ANN * 100
    return Mh.reset_index()


def walkforward_sign(Mh: pd.DataFrame) -> dict[int, int]:
    """只用历史预测下个月的 spread 符号：扩展窗口 OLS。返回 {ym: +1/-1}。"""
    M = Mh.copy()
    M["sp_l1"] = M["sp"].shift(1)
    M["sp_l3"] = M["sp"].rolling(3).mean().shift(1)
    M["mk_l1"] = M["mkt"].shift(1)
    M["mk_l3"] = M["mkt"].rolling(3).mean().shift(1)
    feats = ["sp_l1", "sp_l3", "mk_l1", "mk_l3"]
    out = {}
    for i in range(len(M)):
        if i < 24:
            continue
        x = M[feats].iloc[:i].to_numpy(float)
        y = M["sp"].iloc[:i].to_numpy(float)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if ok.sum() < 12:
            continue
        Xa, ya = x[ok], y[ok]
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
        Z = np.column_stack([np.ones(ok.sum()), (Xa - mu) / sd])
        b, *_ = np.linalg.lstsq(Z, ya, rcond=None)
        xi = M[feats].iloc[i].to_numpy(float)
        if not np.isfinite(xi).all():
            continue
        yhat = float(np.r_[1.0, (xi - mu) / sd] @ b)
        out[int(M["ym"].iloc[i])] = 1 if yhat > 0 else -1
    return out


# --------------------------------------------------------------------------
def two_stage(pr_lh: np.ndarray, pr_h1: np.ndarray, K: float) -> np.ndarray:
    out = np.full(pr_lh.shape, np.nan)
    for t in range(pr_lh.shape[0]):
        a, b = pr_lh[t], pr_h1[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K / 100.0)
        cand = m & (a >= thr)
        if cand.sum() < 20:
            continue
        bb = np.where(cand & np.isfinite(b), b, np.nan)
        v = bb[cand & np.isfinite(b)]
        lo, hi = v.min(), v.max()
        if hi > lo:
            within = np.where(np.isfinite(bb), (bb - lo) / (hi - lo), 0.5)
        else:
            within = np.where(np.isfinite(bb), 0.5, 0.5)
        out[t] = np.where(cand, 1.0 + within, np.where(m, a, np.nan))
    return out


def load_weights() -> dict:
    dw = pd.read_csv(REPO / "experiments/LH019_deploy_weights.csv")
    row = dw[np.isclose(dw.lam, 0.5)].iloc[0]
    return {kv.split(":")[0]: float(kv.split(":")[1]) for kv in str(row["w"]).split(";")}


def tilt(P0: np.ndarray, pr_lowvol: np.ndarray, phi_t: np.ndarray) -> np.ndarray:
    """组内重排：delta=phi_t*(pr-0.5) 只对各自组内部起作用，候选池边界保持不变。"""
    D = (pr_lowvol - 0.5) * phi_t[:, None]
    P = P0 + D
    iscand = P0 >= 1.0
    P = np.where(iscand, P, P - 1.0)
    P[~np.isfinite(P0)] = np.nan
    return P


def fold_block(fold, fdf, sign_pred) -> list[dict]:
    df = load_fold(fold)
    pred = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    extra = [m for m in WEIGHTS if m in pred.columns and m not in df.columns]
    if extra:
        df = df.drop(columns=[c for c in extra if c in df.columns]).merge(
            pred[["ts_code", "trade_date"] + extra], on=["ts_code", "trade_date"], how="left")
        df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    W = wide_of(fdf, panel)
    vol = W[LAYER]
    Lbin = np.vstack([stratify(vol[t], NBIN) for t in range(panel.T)])
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in WEIGHTS}
    pct = {m: pct_rank_2d(P) for m, P in raw.items()}
    mix = np.nansum(np.stack([w * pct[m] for m, w in WEIGHTS.items()]), axis=0)
    P0 = two_stage(pct_rank_2d(mix), pct["H01"], K_STAGE1)

    # 低波倾斜打分：prank(-vol60)，缺 vol 的置中性 0.5
    plv = pct_rank_2d(np.where(np.isfinite(vol), -vol, np.nan))
    plv = np.where(np.isfinite(plv), plv, 0.5)

    ym = np.array([int(d) // 100 for d in panel.dates])

    def evaluate(tag, P):
        r = score(P, panel)
        S = top_sets(P, panel)
        d = decompose(W, panel, S, [LAYER], [NBIN], fold, tag)
        DAILY[tag] = d.set_index("trade_date")["excess"]
        alpha = float(d["alpha"].mean()) * ANN
        prof = float(d["profile"].mean()) * ANN
        hs = holding_stats(r["_S"])
        c = composition(r["_S"], Lbin, panel.valid_turn)
        T = r["mean_turnover"]
        ann_turn = 504.0 * T / (2.0 - T)
        rec = dict(fold=fold, tag=tag, mode=tag.split("|")[0], param=float(tag.split("|")[1]),
                   ic=r["ic_mean"], excess=r["annual_excess"], turnover=T,
                   final=r["final_score"], alpha=alpha, profile=prof,
                   noprofile=0.4 * r["ic_mean"] + 0.3 * alpha + 0.3 * (1 - T),
                   hold_days=hs["avg_hold_days"], list_size=hs["list_size"],
                   ann_turn=ann_turn, **{f"L{k}": c[k] for k in range(NBIN)})
        for cp in COSTS:
            rec[f"netE_{cp:.4f}"] = rec["excess"] - ann_turn * cp
            rec[f"netA_{cp:.4f}"] = alpha - ann_turn * cp
        return rec

    def composition(S, LB, valid_turn):
        out = {}
        for k in range(NBIN):
            num = 0.0
            den = 0.0
            for t in range(S.shape[0]):
                idx = S[t]
                idx = idx[idx >= 0]
                if idx.size == 0:
                    continue
                m = valid_turn[t][idx] & (LB[t][idx] >= 0)
                if m.sum() == 0:
                    continue
                num += (LB[t][idx][m] == k).sum()
                den += m.sum()
            out[k] = num / den if den > 0 else np.nan
        return out

    rows = []
    DAILY = {}
    # 基线
    P, _ = fast_band(P0, panel.LU, KQ)
    rows.append(evaluate(f"base|0", P))
    log(f"  {fold} base          IC {rows[-1]['ic']:.4f} E {rows[-1]['excess']:.4f} "
        f"T {rows[-1]['turnover']:.4f} F {rows[-1]['final']:.4f} "
        f"a {rows[-1]['alpha']:.4f} p {rows[-1]['profile']:+.4f} np {rows[-1]['noprofile']:.4f} "
        f"L0 {rows[-1]['L0']:.3f} L9 {rows[-1]['L9']:.3f} hold {rows[-1]['hold_days']:.0f}")

    # 静态倾斜
    for phi in PHI_STATIC:
        for sg in (+1, -1):
            phit = np.full(panel.T, sg * phi)
            Pt, _ = fast_band(tilt(P0, plv, phit), panel.LU, KQ)
            rows.append(evaluate(f"const|{sg*phi:g}", Pt))
            r = rows[-1]
            log(f"  {fold} const {sg*phi:+.1f}    IC {r['ic']:.4f} E {r['excess']:.4f} "
                f"T {r['turnover']:.4f} F {r['final']:.4f} a {r['alpha']:.4f} "
                f"p {r['profile']:+.4f} np {r['noprofile']:.4f} "
                f"L0 {r['L0']:.3f} L9 {r['L9']:.3f} hold {r['hold_days']:.0f}")

    # ---- 时变 tilt：oracle = 每月挑「当月实际表现更好」的那个方向（真·完美前视上界）----
    for g in PHI_TIMED:
        for sg in (+1, -1):
            tag = f"const|{sg*g:g}"
            if tag not in DAILY:
                phit = np.full(panel.T, sg * g)
                Pt, _ = fast_band(tilt(P0, plv, phit), panel.LU, KQ)
                rows.append(evaluate(tag, Pt))
                r = rows[-1]
                log(f"  {fold} const {sg*g:+.1f}    IC {r['ic']:.4f} E {r['excess']:.4f} "
                    f"T {r['turnover']:.4f} F {r['final']:.4f} a {r['alpha']:.4f} "
                    f"p {r['profile']:+.4f} np {r['noprofile']:.4f} "
                    f"L0 {r['L0']:.3f} L9 {r['L9']:.3f} hold {r['hold_days']:.0f}")
        up = DAILY[f"const|{g:g}"]
        dn = DAILY[f"const|{-g:g}"]
        common = up.index.intersection(dn.index)
        ym_map = pd.Series(common // 100, index=common)
        better = (up[common] > dn[common]).astype(float) * 2 - 1     # +1=低波更优, -1=高波更优
        bym = better.groupby(ym_map).mean()
        bym = (bym > 0).map({True: 1.0, False: -1.0}).to_dict()
        phit = np.array([g * bym.get(int(m), 0.0) for m in ym], dtype=float)
        Pt, _ = fast_band(tilt(P0, plv, phit), panel.LU, KQ)
        rows.append(evaluate(f"oracle|{g:g}", Pt))
        r = rows[-1]
        nflip = int(np.sum(np.abs(np.diff(phit)) > 0))
        log(f"  {fold} oracle {g:.1f}   IC {r['ic']:.4f} E {r['excess']:.4f} "
            f"T {r['turnover']:.4f} F {r['final']:.4f} a {r['alpha']:.4f} "
            f"p {r['profile']:+.4f} np {r['noprofile']:.4f} "
            f"L0 {r['L0']:.3f} L9 {r['L9']:.3f} hold {r['hold_days']:.0f} 翻转{nflip}次")
        # 诚实版：只用历史的 walk-forward 符号（低波=+1 / 高波=-1）
        phit2 = np.array([-g * float(sign_pred.get(int(m), 0.0)) for m in ym], dtype=float)
        Pt2, _ = fast_band(tilt(P0, plv, phit2), panel.LU, KQ)
        rows.append(evaluate(f"pred|{g:g}", Pt2))
        r = rows[-1]
        log(f"  {fold} pred {g:.1f}     IC {r['ic']:.4f} E {r['excess']:.4f} "
            f"T {r['turnover']:.4f} F {r['final']:.4f} a {r['alpha']:.4f} "
            f"p {r['profile']:+.4f} np {r['noprofile']:.4f} "
            f"L0 {r['L0']:.3f} L9 {r['L9']:.3f} hold {r['hold_days']:.0f}")

    return rows


def main() -> int:
    t0 = time.time()
    WEIGHTS.update(load_weights())
    log(f"部署权重 {WEIGHTS}")
    fdf = load_factors(20210101, 20241231)
    log("算月度 spread 历史")
    Mh = monthly_history()
    sign_pred = walkforward_sign(Mh)
    log(f"  {len(Mh)} 个月；walk-forward 符号预测覆盖 {len(sign_pred)} 个月")

    def hist_phi_fn(fold, ym, panel, vol):
        # 1) oracle：该月已实现的 L0-L9（前视——故意的，这是上界）
        realized = np.full(panel.T, np.nan)
        for t in range(panel.T):
            b = Lbin_of(vol[t], panel.valid_top[t])
            y = panel.Y[t]
            m = panel.valid_top[t] & np.isfinite(y) & (b >= 0)
            if m.sum() < 200:
                continue
            yy = y[m]
            bb = b[m]
            s0 = yy[bb == 0]
            s9 = yy[bb == 9]
            if s0.size and s9.size:
                realized[t] = s0.mean() - s9.mean()
        bym = {}
        for mth in np.unique(ym):
            mask = ym == mth
            v = np.nanmean(realized[mask]) if np.isfinite(realized[mask]).any() else 0.0
            bym[int(mth)] = 1.0 if v > 0 else -1.0
        # 2) honest：只用历史的 walk-forward 预测
        return {"oracle": bym, "pred": {k: float(v) for k, v in sign_pred.items()}}

    rows = []
    for fold in FOLDS:
        log(f"=== {fold} ===")
        rows.extend(fold_block(fold, fdf, sign_pred))
    M = pd.DataFrame(rows).drop_duplicates(subset=["fold", "tag"], keep="last")
    M.to_csv(OUT / "regime_tilt_grid.csv", index=False)

    pd.set_option("display.width", 300)
    print("\n=== 自检：base vs 归档 TS_K40 ===")
    bad = 0
    for fold in FOLDS:
        got = M[(M.fold == fold) & (M.tag == "base|0")].iloc[0]
        for k, v in ARCHIVE[fold].items():
            if abs(float(got[k]) - v) > 1e-5:
                bad += 1
                print(f"  FAIL {fold} {k}: {float(got[k]):.6f} vs {v:.6f}")
    print(f"  四折 × 6 项 = 24 项，不一致 {bad} 项 -> {'PASS' if bad == 0 else 'FAIL'}")

    cc = ["ic", "excess", "turnover", "final", "alpha", "profile", "noprofile",
          "hold_days", "ann_turn", "L0", "L9"]
    print("\n=== 四折均值 ===")
    print(M.groupby("tag")[cc].mean().round(4).to_string())

    print("\n=== 相对 base 的配对 Δ / t / 逐折符号 ===")
    base = M[M.tag == "base|0"].set_index("fold").reindex(FOLDS)
    out = []
    for tag in [t for t in M.tag.unique() if t != "base|0"]:
        o = M[M.tag == tag].set_index("fold").reindex(FOLDS)
        rec = dict(tag=tag)
        for k in ["ic", "alpha", "profile", "excess", "turnover", "final", "noprofile"]:
            d = (o[k].to_numpy() - base[k].to_numpy())
            sd = d.std(ddof=1)
            rec[f"d{k}"] = d.mean()
            rec[f"t_{k}"] = d.mean() / (sd / 2) if sd > 0 else np.nan
            rec[f"sgn_{k}"] = "".join("+" if v > 0 else "-" for v in d)
        out.append(rec)
    P = pd.DataFrame(out).set_index("tag")
    show = P[["dprofile", "dturnover", "dic", "dalpha", "dexcess", "dfinal", "dnoprofile"]]
    print(show.round(5).to_string())
    print()
    print(P[["t_ic", "t_alpha", "t_profile", "t_turnover", "t_final", "t_noprofile",
             "sgn_ic", "sgn_alpha", "sgn_profile", "sgn_final"]].round(2).to_string())

    print("\n=== profile 的价格：每买 1pp profile 要付多少 ===")
    G = M.groupby("tag")[["profile", "turnover", "alpha", "ic", "excess", "ann_turn", "final", "noprofile"]].mean()
    b0 = G.loc["base|0"]
    for tag in [t for t in G.index if t.startswith("const") or t.startswith("oracle")]:
        r = G.loc[tag]
        dprof = r["profile"] - b0["profile"]
        dturn = r["turnover"] - b0["turnover"]
        dfin = r["final"] - b0["final"]
        print(f"  {tag:12s} dprofile {dprof:+7.4f}  dT {dturn:+7.4f}  dFinal {dfin:+7.4f}  "
              f"| 每 pp profile 的换手代价 {dturn/(dprof*100) if abs(dprof)>1e-9 else float('nan'):+8.5f}  "
              f"| 官方视角 每 pp profile 净 {0.3*(dprof+0) / 1.0:+7.4f}分, 换手罚 {0.3*dturn:+7.4f}分")

    print("\n=== 真实成本下的净超额（c=0.10% / 0.20% / 0.25% / 0.40% 单边）===")
    key = [c for c in M.columns if c.startswith("netE_") or c.startswith("netA_")]
    print(M.groupby("tag")[key].mean().round(4).to_string())
    Mh.round(4).to_csv(OUT / "monthly_spread_history.csv", index=False)
    print(f"\n用时 {time.time()-t0:.0f}s")
    return 0


def Lbin_of(vol_t: np.ndarray, valid_t: np.ndarray) -> np.ndarray:
    b = np.full(vol_t.shape, -1, dtype=int)
    m = valid_t & np.isfinite(vol_t)
    if m.sum() >= 300:
        b[m] = stratify(vol_t[m], NBIN)
    return b


if __name__ == "__main__":
    raise SystemExit(main())

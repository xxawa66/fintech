"""方向二：新特征族筛选探针（不重训）。

问题：99 个现有特征（V1 40 + research 18 + 长周期 59）里**没有任何「相对市场」的量**
（beta / 市场因子 / IVOL / 协偏度 全库零命中），也没有 MAX、Amihud、累计隔夜拆分、AR1。
本探针先不重训，只回答三个问题：

  Q1  这些新特征本身有没有横截面信息（全池 IC / 候选池内 IC）？
  Q2  它们的信息是否已被现有信号（H01 / mix）占据（与 H01 的秩相关、给定 H01 的偏 IC）？
  Q3  它们的信号落在**可外推的 α** 上，还是落在**会塌的 profile（低波倾斜）**上？
      —— 把每个特征当作独立信号跑 band + DGTW 分解，直接看 α 与 profile。

只用 2021–2024 四折（特征构造回溯到 2020，但**评估窗口严格 2021–2024**）。

自检：本项目重算的 vol60 / rev1 必须逐日复现归档 `_probe_factors_full.parquet`，
否则整条链路不可信。

产物：cv_design_audit/new_feature_screen.csv、new_feature_basket.csv（仓库外，未入库）。
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve()
REPO = _REPO
CACHE = REPO / "outputs/long_horizon"
OUT = _LH024_DIR / "cv_design_audit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/LH023"))

from _probe_dgtw import decompose, pct_rank_2d, pct_rank_row, stratify  # noqa: E402
from _probe_e_attr import top_sets  # noqa: E402
from _probe_root_cause import (Panel, daily_excess, daily_topsets, fast_band,  # noqa: E402
                               turnover_from_sets)

KQ = 0.0022778298112255263
ANN = 252
FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
FOLD_WIN = {"wf2021": (20210101, 20211231), "wf2022": (20220101, 20221231),
            "wf2023": (20230101, 20231231), "confirm2024": (20240101, 20241231)}
DEPLOY = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
K_POOL = 40.0
LOOKBACK_START = 20200101
EVAL_START = 20210101
EVAL_END = 20241231


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
            "flag_limit_up", "flag_limit_down", "y_ret_1d"]
    dtypes = {"ts_code": "str", "trade_date": "int64",
              **{c: "float64" for c in ["open", "high", "low", "close", "vol", "amount", "y_ret_1d"]},
              "flag_limit_up": "int64", "flag_limit_down": "int64"}
    parts = []
    for chunk in pd.read_csv(REPO / "data/raw/训练集.csv", usecols=cols, dtype=dtypes,
                             chunksize=2_000_000):
        m = (chunk["trade_date"] >= LOOKBACK_START) & (chunk["trade_date"] <= EVAL_END)
        if m.any():
            parts.append(chunk.loc[m])
    df = pd.concat(parts, ignore_index=True)
    log(f"原始面板 {df.shape}  {df.trade_date.min()}~{df.trade_date.max()}")
    return df


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """折缓存 `_probe_rc_*` 缺 H05_F1T2，改从 LH003/<fold>/raw_predictions.parquet 取。"""
    frames = []
    for fold in FOLDS:
        p = pd.read_parquet(CACHE / "LH003" / fold / "raw_predictions.parquet",
                            columns=["ts_code", "trade_date", *DEPLOY.keys()])
        frames.append(p)
    pred = pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    df = df.merge(pred, on=["ts_code", "trade_date"], how="left")
    missing = [m for m in DEPLOY if m not in df.columns]
    if missing:
        raise ValueError(f"折预测缺列：{missing}")
    return df


def wide(panel: Panel, values: np.ndarray) -> pd.DataFrame:
    W = np.full((panel.T, panel.N), np.nan, dtype="float32")
    W[panel.ti, panel.si] = np.asarray(values, dtype="float32")
    return pd.DataFrame(W, index=panel.dates, columns=panel.codes)


# --------------------------------------------------------------------------
# 新特征
# --------------------------------------------------------------------------
def build_features(panel: Panel, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    C = wide(panel, df["close"].to_numpy())
    O = wide(panel, df["open"].to_numpy())
    A = wide(panel, df["amount"].to_numpy())
    prevC = C.shift(1)
    R = C / prevC - 1.0
    GAP = np.log(O / prevC)
    INTRA = np.log(C / O)
    mkt = R.mean(axis=1, skipna=True)
    out: dict[str, pd.DataFrame] = {}

    # 参考量（与 _probe_factors 同口径）
    out["vol60"] = R.rolling(60, min_periods=30).std()
    out["rev1"] = R

    def roll(df_, w, mp):
        return df_.rolling(w, min_periods=mp)

    for w, mp in [(60, 30), (120, 60)]:
        mean_r = roll(R, w, mp).mean()
        mean_m = mkt.rolling(w, min_periods=mp).mean()
        mean_rm = roll(R.mul(mkt, axis=0), w, mp).mean()
        cov = mean_rm.sub(mean_r.mul(mean_m, axis=0))
        var_m = (mkt ** 2).rolling(w, min_periods=mp).mean() - mean_m ** 2
        var_r = (R ** 2).rolling(w, min_periods=mp).mean() - mean_r ** 2
        beta = cov.div(var_m, axis=0)
        resid_var = var_r.sub(cov.pow(2).div(var_m, axis=0))
        out[f"beta{w}"] = beta
        out[f"ivol{w}"] = np.sqrt(resid_var.clip(lower=0))
        out[f"corrmkt{w}"] = cov.div(np.sqrt(var_r.mul(var_m, axis=0).clip(lower=0)))
        cum_r = roll(R, w, mp).sum()
        cum_m = mkt.rolling(w, min_periods=mp).sum()
        out[f"residmom{w}"] = cum_r.sub(beta.mul(cum_m, axis=0))

    for w, mp in [(20, 15), (60, 40)]:
        out[f"maxret{w}"] = roll(R, w, mp).max()
        out[f"amihud{w}"] = roll(R.abs() / A.where(A > 0), w, mp).mean() * 1e9

    for w, mp in [(5, 5), (20, 15), (60, 40)]:
        out[f"cumgap{w}"] = roll(GAP, w, mp).sum()
        out[f"cumintra{w}"] = roll(INTRA, w, mp).sum()
    out["gapspread20"] = out["cumgap20"] - out["cumintra20"]

    out["ar1_60"] = roll(R, 60, 40).corr(R.shift(1))
    return out


NEW = ["beta60", "beta120", "ivol60", "ivol120", "corrmkt60", "corrmkt120",
       "residmom60", "residmom120", "maxret20", "maxret60", "amihud20", "amihud60",
       "cumgap5", "cumgap20", "cumgap60", "cumintra5", "cumintra20", "cumintra60",
       "gapspread20", "ar1_60"]


# --------------------------------------------------------------------------
# 诊断
# --------------------------------------------------------------------------
def daily_rank_ic(fm: np.ndarray, Y: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    out = np.full(fm.shape[0], np.nan)
    for t in range(fm.shape[0]):
        m = base_mask[t] & np.isfinite(fm[t])
        if m.sum() < 100:
            continue
        a, b = fm[t][m], Y[t][m]
        ra, rb = rankdata(a), rankdata(b)
        if ra.std() == 0 or rb.std() == 0:
            continue
        out[t] = float(np.corrcoef(ra, rb)[0, 1])
    return out


def daily_rank_corr(fm: np.ndarray, gm: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    out = np.full(fm.shape[0], np.nan)
    for t in range(fm.shape[0]):
        m = base_mask[t] & np.isfinite(fm[t]) & np.isfinite(gm[t])
        if m.sum() < 100:
            continue
        ra, rb = rankdata(fm[t][m]), rankdata(gm[t][m])
        if ra.std() == 0 or rb.std() == 0:
            continue
        out[t] = float(np.corrcoef(ra, rb)[0, 1])
    return out


def daily_partial_ic(fm: np.ndarray, Y: np.ndarray, cond: np.ndarray,
                     base_mask: np.ndarray) -> np.ndarray:
    """给定 cond 的偏秩相关：把 rank(Y) 对 rank(cond) 回归取残差，再与 rank(f) 求相关。"""
    out = np.full(fm.shape[0], np.nan)
    for t in range(fm.shape[0]):
        m = base_mask[t] & np.isfinite(fm[t]) & np.isfinite(cond[t])
        if m.sum() < 100:
            continue
        ry, rc = rankdata(Y[t][m]), rankdata(cond[t][m])
        A = np.column_stack([np.ones(m.sum()), rc])
        if np.linalg.matrix_rank(A) < 2:
            continue
        coef, *_ = np.linalg.lstsq(A, ry, rcond=None)
        resid = ry - A @ coef
        rf = rankdata(fm[t][m])
        if rf.std() == 0 or resid.std() == 0:
            continue
        out[t] = float(np.corrcoef(rf, resid)[0, 1])
    return out


def daily_ic_robust(P: np.ndarray, panel: Panel) -> np.ndarray:
    """官方 daily_ic 的稳健版：显式剔除 P 非有限处。

    本项目 scipy 的 rankdata 遇 NaN 会把整行返回 NaN（实测），因此 band 输出
    若含 NaN（特征覆盖不全），官方 daily_ic 会整天作废。
    """
    out = np.full(panel.T, np.nan)
    for t in range(panel.T):
        m = panel.valid_ic[t] & np.isfinite(P[t])
        if m.sum() < 30:
            continue
        rp, ry = rankdata(P[t][m]), rankdata(panel.Y[t][m])
        if rp.std() == 0 or ry.std() == 0:
            continue
        out[t] = float(np.corrcoef(rp, ry)[0, 1])
    return out


def score_robust(P: np.ndarray, panel: Panel) -> dict:
    ic = daily_ic_robust(P, panel)
    top_ret, mkt, _ = daily_excess(P, panel)
    tu = turnover_from_sets(daily_topsets(P, panel))
    exc = top_ret - mkt
    ic_m = float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan
    ex_m = float(np.nanmean(exc)) * ANN if np.isfinite(exc).any() else np.nan
    tu_m = float(np.nanmean(tu)) if np.isfinite(tu).any() else np.nan
    return dict(ic_mean=ic_m, annual_excess=ex_m, mean_turnover=tu_m,
                final_score=0.4 * ic_m + 0.3 * ex_m + 0.3 * (1 - tu_m))


def fold_mean(series: np.ndarray, dates: np.ndarray) -> tuple[float, float, str]:
    """按四个年折求均值，返回（四折均值, 四折 sd, 符号串）。"""
    vals = []
    for f in FOLDS:
        s, e = FOLD_WIN[f]
        m = (dates >= s) & (dates <= e)
        v = np.nanmean(series[m]) if m.any() else np.nan
        vals.append(v)
    v = np.asarray(vals, dtype=float)
    sd = float(np.nanstd(v, ddof=1))
    signs = "".join("+" if x > 0 else "-" for x in v)
    return float(np.nanmean(v)), sd, signs


def main() -> int:
    t0 = time.time()
    df = add_predictions(load_raw())
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    panel = Panel(df)
    log(f"评估面板 {panel.T} 天 × {panel.N} 只")

    feats = build_features(panel, df)

    # ---- 自检：复现归档 vol60 / rev1 ----
    log("=== 自检：vol60 / rev1 对归档 ===")
    arch = pd.read_parquet(CACHE / "_probe_factors_full.parquet",
                           columns=["ts_code", "trade_date", "vol60", "rev1"],
                           filters=[("trade_date", ">=", EVAL_START), ("trade_date", "<=", EVAL_END)])
    arch["ts_code"] = arch["ts_code"].astype(str)
    check = df[["ts_code", "trade_date"]].copy()
    check["vol60"] = feats["vol60"].to_numpy()[panel.ti, panel.si]
    check["rev1"] = feats["rev1"].to_numpy()[panel.ti, panel.si]
    chk = check.merge(arch, on=["ts_code", "trade_date"], suffixes=("", "_a"))
    for c in ["vol60", "rev1"]:
        a, b = chk[c].to_numpy(), chk[f"{c}_a"].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        err = float(np.max(np.abs(a[m] - b[m]))) if m.any() else np.nan
        nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
        print(f"  {c}: n={m.sum():,}  NaN 不一致 {nan_mismatch}  max|diff|={err:.3e}")
    del arch, chk, check

    # ---- 评估窗口切片 ----
    eval_dates = [d for d in panel.dates if d >= EVAL_START]
    keep_rows = np.array([d in set(eval_dates) for d in panel.dates])
    row_idx = np.flatnonzero(keep_rows)
    df_ev = df[df.trade_date >= EVAL_START].sort_values(["trade_date", "ts_code"],
                                                        kind="stable").reset_index(drop=True)
    panel_ev = Panel(df_ev)
    log(f"评估窗口 {panel_ev.T} 天 × {panel_ev.N} 只")

    def slice_eval(F: pd.DataFrame) -> np.ndarray:
        return F.iloc[row_idx].reindex(columns=panel_ev.codes).to_numpy(dtype="float64")

    Y = panel_ev.Y
    valid_ic = panel_ev.valid_ic
    dates_ev = panel_ev.dates

    pct = {m: pct_rank_2d(wide(panel_ev, df_ev[m].to_numpy(dtype="float64")).to_numpy(dtype="float64"))
           for m in DEPLOY}
    mix = np.nansum(np.stack([w * pct[m] for m, w in DEPLOY.items()]), axis=0)
    H01 = wide(panel_ev, df_ev["H01"].to_numpy(dtype="float64")).to_numpy(dtype="float64")
    vol60 = slice_eval(feats["vol60"])

    cand = np.zeros_like(mix, dtype=bool)
    for t in range(panel_ev.T):
        a = mix[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K_POOL / 100.0)
        cand[t] = m & (a >= thr)
    log(f"候选池（mix top {K_POOL:.0f}%）日均 {cand.sum(axis=1).mean():.0f} 只")

    # ---- 逐特征诊断 ----
    rows, brows = [], []
    for name in NEW:
        F = slice_eval(feats[name])
        ic = daily_rank_ic(F, Y, valid_ic)
        icp = daily_rank_ic(F, Y, cand)
        cvol = daily_rank_corr(F, vol60, valid_ic)
        ch01 = daily_rank_corr(F, H01, valid_ic)
        pic = daily_partial_ic(F, Y, H01, valid_ic)
        ic_m, ic_sd, ic_sg = fold_mean(ic, dates_ev)
        pic_m, pic_sd, pic_sg = fold_mean(pic, dates_ev)
        rows.append(dict(feature=name,
                         ic_full=ic_m, ic_full_sd=ic_sd, ic_full_signs=ic_sg,
                         ic_pool=float(np.nanmean(icp)),
                         corr_vol60=float(np.nanmean(cvol)),
                         corr_h01=float(np.nanmean(ch01)),
                         partial_ic_h01=pic_m, partial_ic_h01_sd=pic_sd,
                         partial_ic_h01_signs=pic_sg))
        log(f"  {name:12s} IC {ic_m:+.4f} [{ic_sg}] 池内 {np.nanmean(icp):+.4f} "
            f"corrVol {np.nanmean(cvol):+.3f} corrH01 {np.nanmean(ch01):+.3f} "
            f"偏IC|H01 {pic_m:+.4f} [{pic_sg}]")

        # 独立信号：按 IC 符号翻转后跑 band + DGTW 分解
        sgn = 1.0 if ic_m >= 0 else -1.0
        P0 = sgn * F
        P, _ = fast_band(P0, panel_ev.LU, KQ)
        r = score_robust(P, panel_ev)
        d = decompose({"vol60": vol60}, panel_ev, top_sets(P, panel_ev),
                      ["vol60"], [10], "all", name)
        al = float(d["alpha"].mean()) * ANN
        pr = float(d["profile"].mean()) * ANN
        brows.append(dict(feature=name, sign=sgn,
                          ic=r["ic_mean"], excess=r["annual_excess"],
                          turnover=r["mean_turnover"], final=r["final_score"],
                          alpha=al, profile=pr,
                          noprofile=0.4 * r["ic_mean"] + 0.3 * al + 0.3 * (1 - r["mean_turnover"])))
        log(f"      band: IC {r['ic_mean']:+.4f} E {r['annual_excess']:+.4f} "
            f"T {r['mean_turnover']:.4f} α {al:+.4f} prof {pr:+.4f}")

    R = pd.DataFrame(rows)
    B = pd.DataFrame(brows)
    R.to_csv(OUT / "new_feature_screen.csv", index=False)
    B.to_csv(OUT / "new_feature_basket.csv", index=False)

    pd.set_option("display.width", 320)
    print("\n=== 新特征筛选（四折均值；signs 为四折符号）===")
    print(R[["feature", "ic_full", "ic_full_signs", "ic_pool", "corr_vol60",
             "corr_h01", "partial_ic_h01", "partial_ic_h01_signs"]].round(4).to_string(index=False))
    print("\n=== 独立信号 band 分解（四折合并 2021–2024）===")
    print(B.round(4).to_string(index=False))

    # 与 H01 基线对照
    P_h01, _ = fast_band(H01, panel_ev.LU, KQ)
    r0 = score_robust(P_h01, panel_ev)
    d0 = decompose({"vol60": vol60}, panel_ev, top_sets(P_h01, panel_ev),
                   ["vol60"], [10], "all", "H01_ref")
    print(f"\n[H01 参考] IC {r0['ic_mean']:+.4f} E {r0['annual_excess']:+.4f} "
          f"T {r0['mean_turnover']:.4f} α {float(d0['alpha'].mean())*ANN:+.4f} "
          f"prof {float(d0['profile'].mean())*ANN:+.4f}")

    log(f"完成，用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

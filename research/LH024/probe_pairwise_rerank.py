"""方向一探针：阶段2 从「点式回归」换成「排序损失重排器」（不重训主模型链）。

问题
----
官方分 = 0.4·IC + 0.3·E + 0.3·(1−T)：IC 是逐日**秩相关**、E 只看**最顶部 1/10 的集合**。
仓库里至今所有模型的目标都是「对某个 y 做点式回归」（T030 的 L2；S005 的 Y0–Y5 全是
逐元素 y 变换）。评分的形状（秩 + 顶部集合）从未进过损失函数。

受控对照设计（这是本探针的全部要点）
----------------------------------
固定不动：候选池（TS_K40 阶段1，mix top K=40%）、特征（V1 40）、训练样本集、
          band(keep_q=q*)、评分口径。
唯一的变量 = 阶段2 的目标函数：
  H01 : 现方案——池内按 T030 的预测（L2 回归）重排，**不重训**（= TS_K40 基线）
  L2  : 用同一池子、同一特征**重训**一个 L2 回归（隔离「换模型」的效应）
  LR  : 同一池子 + **lambdarank**（pairwise/listwise，label = y 的当日十分位 0..9）
        —— label 的十分位切法与官方 top 1/10 完全对齐

两个训练协议（都是无前视的）
-------------------------
  split : 在同一折内按时间切 60%/40%，只用前 60% 的日子训练，只在后 40% 评分（4 折）
  walk  : 只在**严格更早**的折上训练（eval wf2022 ← train wf2021；eval wf2023 ← wf2021-22；
          eval confirm2024 ← wf2021-23），最贴近部署（3 折）

自检（必须通过）
--------------
  base 模式四折跑 H01 臂，必须逐位复现 experiments/LH020_twostage_folds.csv 的 TS_K40：
  IC 0.086298 / E 0.208280 / T 0.018378 / α 0.188861 / profile 0.013311 / final 0.391489。

产物（仓库外）：cv_design_audit/pairwise_rerank_{base,split,walk}.csv、
               pairwise_rerank_daily_{split,walk}.csv
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
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = Path(__file__).resolve()
REPO = _REPO
OUT = _LH024_DIR / "cv_design_audit"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "LH023"))

from _probe_root_cause import Panel, fast_band, load_fold, score, to_wide  # noqa: E402
from _probe_dgtw import decompose, pct_rank_2d  # noqa: E402
from _probe_e_attr import load_factors, top_sets, wide_of  # noqa: E402

FEAT_PATH = REPO / "data" / "processed" / "E001_baseline_repeat" / "features.parquet"
FOLDS = ["wf2021", "wf2022", "wf2023", "confirm2024"]
DEPLOY = {"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
K_POOL = 40.0
KQ = 0.0022778298112255263
ANN = 252
S2_MODELS = ["H01", "H05_F1T2", "H05_F2T2"]

# T030（S003 Optuna winner）的模型参数——重训臂一律用它，保证与 H01 同族
T030_PARAMS = {
    "boosting_type": "gbdt", "learning_rate": 0.04065467822968424, "num_leaves": 28,
    "max_depth": 8, "min_data_in_leaf": 415, "feature_fraction": 0.660945618691071,
    "bagging_fraction": 0.9204918282790925, "bagging_freq": 1, "lambda_l1": 2.086527798462595,
    "lambda_l2": 0.011862211698474496, "max_bin": 255, "seed": 42, "num_threads": 8,
    "device_type": "cpu", "deterministic": True, "force_col_wise": True,
    "use_missing": True, "zero_as_missing": False, "verbosity": -1,
}
# 归档自检对照（experiments/LH020_twostage_folds.csv 的 TS_K40 行）
ARCHIVE_TS_K40 = {"ic": 0.086298, "excess": 0.208280, "turnover": 0.018378,
                  "final": 0.391489, "alpha": 0.188861, "profile": 0.013311}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def feat_cols() -> list[str]:
    import pyarrow.parquet as pq
    names = pq.ParquetFile(FEAT_PATH).schema_arrow.names
    return [c for c in names if c not in ("ts_code", "trade_date")]


FEATS_RAW = feat_cols()
# 特征表里有 flag_limit_up / flag_limit_down / zero_volume，与面板同名列会撞名 ⇒ 加前缀
FEATS = [f"f_{c}" for c in FEATS_RAW]
FEAT_RENAME = dict(zip(FEATS_RAW, FEATS))


# --------------------------------------------------------------------------
# 一折的数据装配
# --------------------------------------------------------------------------
def build_fold(fold: str) -> dict:
    """返回该折的面板 + 因子 + 池 + 特征矩阵（行序 = (trade_date, ts_code) 升序）。"""
    df = load_fold(fold)
    pred = pd.read_parquet(REPO / "outputs/long_horizon/LH003" / fold / "raw_predictions.parquet")
    pred["ts_code"] = pred["ts_code"].astype(str)
    pred["trade_date"] = pred["trade_date"].astype("int64")
    extra = [m for m in S2_MODELS if m in pred.columns and m not in df.columns]
    if extra:
        df = df.drop(columns=[c for c in extra if c in df.columns]).merge(
            pred[["ts_code", "trade_date"] + extra], on=["ts_code", "trade_date"],
            how="left", validate="one_to_one")
    start, end = int(df.trade_date.min()), int(df.trade_date.max())
    feat = pd.read_parquet(FEAT_PATH,
                           filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
                           columns=["ts_code", "trade_date"] + FEATS_RAW)
    feat["ts_code"] = feat["ts_code"].astype(str)
    feat["trade_date"] = feat["trade_date"].astype("int64")
    feat = feat.rename(columns=FEAT_RENAME)
    n0 = len(df)
    df = df.merge(feat, on=["ts_code", "trade_date"], how="left",
                  validate="one_to_one", sort=False)
    assert len(df) == n0, f"{fold}: 特征 merge 改变了行数 {n0} -> {len(df)}"
    df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)

    panel = Panel(df)
    raw = {m: to_wide(panel, df[m].to_numpy(dtype="float64")) for m in S2_MODELS}
    pct = {m: pct_rank_2d(P) for m, P in raw.items()}
    mix = np.nansum(np.stack([w * pct[m] for m, w in DEPLOY.items()]), axis=0)
    pr_mix = pct_rank_2d(mix)

    # 阶段1 候选池：与 _probe_twostage.two_stage 同一规则
    cand = np.zeros(pr_mix.shape, dtype=bool)
    for t in range(panel.T):
        a = pr_mix[t]
        m = np.isfinite(a)
        if m.sum() < 200:
            continue
        thr = np.nanquantile(a[m], 1.0 - K_POOL / 100.0)
        cand[t] = m & (a >= thr)

    # 训练标签：y 的当日十分位（0=最差 .. 9=最好），口径与官方 top 1/10 对齐
    grade = np.full(panel.Y.shape, -1, dtype=np.int32)
    ypr = np.full(panel.Y.shape, np.nan)
    for t in range(panel.T):
        m = panel.valid_top[t]
        if m.sum() < 100:
            continue
        r = rankdata(panel.Y[t][m], method="average") / m.sum()
        grade[t][m] = np.minimum((r * 10.0).astype(np.int32), 9)
        ypr[t][m] = r

    ti, si = panel.ti, panel.si
    long = {
        "ti": ti, "si": si,
        "cand": cand[ti, si],
        "has_y": np.isfinite(panel.Y[ti, si]),
        "valid_top": panel.valid_top[ti, si],
        "y": panel.Y[ti, si],
        "grade": grade[ti, si],
        "ypr": ypr[ti, si],
    }
    X = df[FEATS].to_numpy(dtype="float32", copy=False)
    return dict(fold=fold, df=df, panel=panel, raw=raw, pct=pct, pr_mix=pr_mix,
                cand=cand, grade=grade, ypr=ypr, X=X, long=long)


def train_rows(bundle: dict, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """给定要用的 day 下标集合，返回训练用行下标（池内、有 y）。"""
    lo = bundle["long"]
    sel = lo["cand"] & lo["valid_top"]
    if days is not None:
        sel = sel & np.isin(lo["ti"], days)
    return np.flatnonzero(sel)


def label_of(arm: str, ycont: np.ndarray, grade: np.ndarray,
             ypr: np.ndarray) -> np.ndarray:
    """各臂的训练标签。

    L2   : 连续 y（点式回归，与 H01 同族）
    L2R  : y 的当日百分位（连续，无量化；S005 的 Y1「daily_rank」在阶段2 的对应物）
    L2W  : 连续 y + top 十分位加权（把「只看头部」写进目标，但不量化标签）
    LR   : 十分位 grade（lambdarank，LightGBM 要求非负整数标签 ⇒ 被迫量化）
    LRTD : 「是否 top 十分位」0/1（最直接对准 E，但量化最狠）
    """
    if arm == "L2":
        return ycont.astype("float64")
    if arm == "L2R":
        return ypr.astype("float64")
    if arm == "L2W":
        return ycont.astype("float64")
    if arm == "LR":
        return grade.astype("int32")
    if arm == "LRTD":
        return (grade == 9).astype("int32")
    raise ValueError(arm)


def weight_of(arm: str, grade: np.ndarray, lam: float = 3.0):
    """L2W 的样本权重：top 十分位给 1+λ 倍，其余 1 倍。"""
    if arm != "L2W":
        return None
    return np.where(grade == 9, 1.0 + lam, 1.0).astype("float64")


def fit_arm(arm: str, Xtr, ytr, groups, rounds: int, wtr=None):
    p = dict(T030_PARAMS)
    if arm in ("L2", "L2R", "L2W"):
        p.update(objective="regression", metric="l2")
    elif arm in ("LR", "LRTD"):
        p.update(objective="lambdarank", metric="ndcg",
                 lambdarank_truncation_level=500, lambdarank_norm=True)
    else:
        raise ValueError(arm)
    ds = lgb.Dataset(Xtr, label=ytr, group=groups, weight=wtr,
                     feature_name=FEATS, params=p, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=rounds)


def diag(bundle: dict, s2: np.ndarray, ref: np.ndarray | None = None, days=None):
    """池内诊断：① 池内 rank IC（重排器 vs y）；② 与 H01 池内排序的逐日相关。

    这条诊断用来区分「排序目标本身不行」与「实现有问题」：若重排器在池内的
    rank IC 不低于 H01，却在头部更差，那是目标对准了错的地方；若池内 IC 直接
    掉下来，那是模型本身弱。
    """
    panel = bundle["panel"]
    ds = None if days is None else set(int(x) for x in days)
    ic, cr = [], []
    for t in range(panel.T):
        if ds is not None and t not in ds:
            continue
        m = np.isfinite(s2[t]) & panel.valid_top[t]
        if m.sum() < 100:
            continue
        rp = rankdata(s2[t][m]); ry = rankdata(panel.Y[t][m])
        if rp.std() > 0 and ry.std() > 0:
            ic.append(float(np.corrcoef(rp, ry)[0, 1]))
        if ref is not None:
            m2 = m & np.isfinite(ref[t])
            if m2.sum() >= 100:
                a = rankdata(s2[t][m2]); b2 = rankdata(ref[t][m2])
                if a.std() > 0 and b2.std() > 0:
                    cr.append(float(np.corrcoef(a, b2)[0, 1]))
    return (float(np.mean(ic)) if ic else np.nan,
            float(np.mean(cr)) if cr else np.nan)


def group_sizes(keys: np.ndarray) -> np.ndarray:
    """keys 必须已按升序排列（同一天的样本连续）。返回每天的样本数。"""
    uniq, cnt = np.unique(keys, return_counts=True)
    order = np.argsort(uniq, kind="stable")
    return cnt[order].astype(np.int32)


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def stage2_encode(pr_mix: np.ndarray, s2: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """池内落 [1,2]（按 s2 排序），池外保持 pr_mix 相对序、落 [0,1]。

    s2 为池内百分位（只有池内非 NaN）。与 _probe_twostage.two_stage 同构，
    但池 mask 由调用方给定，保证各臂池子逐位相同。
    """
    return np.where(cand, 1.0 + np.where(np.isfinite(s2), s2, 0.5),
                    np.where(np.isfinite(pr_mix), pr_mix, np.nan))


def eval_arm(P0: np.ndarray, bundle: dict, W, tag: str,
             days=None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    panel = bundle["panel"]
    P, _ = fast_band(P0, panel.LU, KQ)
    r = score(P, panel)
    S = top_sets(P, panel)
    d = decompose(W, panel, S, ["vol60"], [10], bundle["fold"], tag)
    if days is None:
        sl = np.arange(panel.T)
        ic_m, ex_m, tu_m = r["ic_mean"], r["annual_excess"], r["mean_turnover"]
        n_slice = panel.T
    else:
        sl = np.asarray(days)
        ic_m = float(np.nanmean(r["_ic"][sl]))
        ex_m = float(np.nanmean(r["_exc"][sl])) * ANN
        tu_m = float(np.nanmean(r["_tu"][sl]))
        n_slice = len(sl)
        d = d[np.isin(d.trade_date.to_numpy(), panel.dates[sl])]
    row = dict(fold=bundle["fold"], tag=tag, ic=ic_m, excess=ex_m, turnover=tu_m,
               final=0.4 * ic_m + 0.3 * ex_m + 0.3 * (1 - tu_m),
               alpha=float(d["alpha"].mean()) * ANN,
               profile=float(d["profile"].mean()) * ANN, n_slice=n_slice)
    row["noprofile"] = 0.4 * ic_m + 0.3 * row["alpha"] + 0.3 * (1 - tu_m)
    dd = pd.DataFrame({"trade_date": panel.dates[sl], "rank": sl})
    dd["ic"] = r["_ic"][sl]
    dd["excess"] = r["_exc"][sl]
    dd["turnover"] = r["_tu"][sl]
    dd["fold"] = bundle["fold"]
    dd["tag"] = tag
    return row, d, dd


def s2_of(bundle: dict, arm: str, booster=None) -> np.ndarray:
    """返回池内百分位 s2（池外 NaN）。"""
    panel, cand = bundle["panel"], bundle["cand"]
    if arm == "H01":
        score_long = bundle["raw"]["H01"][panel.ti, panel.si]
    else:
        score_long = booster.predict(bundle["X"], num_threads=8)
    M = np.full(panel.Y.shape, np.nan)
    M[panel.ti, panel.si] = score_long
    M = np.where(cand & np.isfinite(M), M, np.nan)
    return pct_rank_2d(M)


# --------------------------------------------------------------------------
# 模式
# --------------------------------------------------------------------------
def mode_base(fdf, folds) -> pd.DataFrame:
    rows, dailies = [], []
    for fold in folds:
        b = build_fold(fold)
        W = wide_of(fdf, b["panel"])
        s2 = s2_of(b, "H01")
        row, d, _ = eval_arm(stage2_encode(b["pr_mix"], s2, b["cand"]), b, W, "H01")
        rows.append(row); dailies.append(d)
        log(f"  {fold} H01: IC {row['ic']:.6f} E {row['excess']:.6f} T {row['turnover']:.6f} "
            f"α {row['alpha']:.6f} prof {row['profile']:.6f} F {row['final']:.6f}")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "pairwise_rerank_base.csv", index=False)
    g = R[["ic", "excess", "turnover", "final", "alpha", "profile", "noprofile"]].mean()
    print("\n=== base 自检（四折均值 vs LH020_twostage_folds.csv 的 TS_K40）===")
    print(f"{'指标':>10} {'本脚本':>12} {'归档':>12} {'差':>10}")
    for k, av in ARCHIVE_TS_K40.items():
        print(f"{k:>10} {g[k]:12.6f} {av:12.6f} {abs(g[k]-av):10.2e}")
    print(f"{'noprofile':>10} {g['noprofile']:12.6f} {'0.385664':>12} "
          f"{abs(g['noprofile']-0.385664):10.2e}")
    return R


def _paired_fold(R: pd.DataFrame, metric: str, base: str = "H01") -> list[dict]:
    """折级配对 Δ：t = mean / (sd/√n_folds)。"""
    w = R.pivot_table(index="fold", columns="tag", values=metric)
    if base not in w.columns:
        return []
    d = w.sub(w[base], axis=0).drop(columns=[base])
    out = []
    for tag in d.columns:
        col = d[tag].to_numpy(dtype="float64")
        n = len(col)
        sd = float(np.std(col, ddof=1)) if n > 1 else np.nan
        out.append({"tag": tag, "metric": metric, "delta": float(col.mean()),
                    "t": float(col.mean() / (sd / np.sqrt(n))) if sd and sd > 0 else np.nan,
                    "signs": "".join("+" if v > 0 else "-" for v in col), "n": n})
    return out


def prep_train(spec):
    """按 [(*bundle*, days_or_None, fold_offset), ...] 组装训练矩阵，行序按天升序。"""
    xs, ys, gs, ps, ds = [], [], [], [], []
    for pb, days, off in spec:
        idx = train_rows(pb, days)
        if len(idx) == 0:
            continue
        xs.append(pb["X"][idx])
        ys.append(pb["long"]["y"][idx].astype("float64"))
        gs.append(pb["long"]["grade"][idx].astype("int32"))
        ps.append(pb["long"]["ypr"][idx].astype("float64"))
        ds.append(pb["long"]["ti"][idx] + off)
    day = np.concatenate(ds)
    X = np.concatenate(xs, axis=0)
    order = np.argsort(day, kind="stable")
    y = np.concatenate(ys)[order]
    g = np.concatenate(gs)[order]
    pr = np.concatenate(ps)[order]
    return X[order], y, g, pr, group_sizes(day[order])


def run_arms(b, W, arms, rounds, Xtr, yc, yg, yp, grp, days_ev, s2_h01):
    rows, dailies, dailym = [], [], []
    for arm in arms:
        booster = None
        if arm != "H01":
            t0 = time.time()
            booster = fit_arm(arm, Xtr, label_of(arm, yc, yg, yp), grp, rounds,
                              wtr=weight_of(arm, yg))
            log(f"    {arm} 拟合完成 {time.time()-t0:.0f}s")
        s2 = s2_of(b, arm, booster)
        pic, ch1 = diag(b, s2, s2_h01, days_ev)
        row, d, dd = eval_arm(stage2_encode(b["pr_mix"], s2, b["cand"]), b, W, arm, days_ev)
        row.update(pool_ic=pic, corr_h01=ch1)
        rows.append(row); dailies.append(d); dailym.append(dd)
        log(f"    {b['fold']} {arm:5s}: IC {row['ic']:.6f} E {row['excess']:.6f} "
            f"T {row['turnover']:.6f} a {row['alpha']:.6f} prof {row['profile']:.6f} "
            f"F {row['final']:.6f} | poolIC {pic:.4f} corrH01 {ch1:.4f}")
    return rows, dailies, dailym


def mode_split(fdf, folds, arms, rounds) -> pd.DataFrame:
    rows, dailies, dailym = [], [], []
    for fold in folds:
        b = build_fold(fold)
        W = wide_of(fdf, b["panel"])
        T = b["panel"].T
        n_tr = int(T * 0.6)
        days_tr = np.arange(0, n_tr)
        days_ev = np.arange(n_tr, T)
        Xtr, yc, yg, yp, grp = prep_train([(b, days_tr, 0)])
        log(f"{fold}: split 训练 {len(Xtr):,} 行 / {len(grp)} 天；评估窗口 {T-n_tr} 天")
        r, d, dm = run_arms(b, W, arms, rounds, Xtr, yc, yg, yp, grp,
                            days_ev, s2_of(b, "H01"))
        rows += r; dailies += d; dailym += dm
    return _report(pd.DataFrame(rows), pd.concat(dailies, ignore_index=True),
                   pd.concat(dailym, ignore_index=True), "split")


def mode_walk(fdf, folds, arms, rounds) -> pd.DataFrame:
    rows, dailies, dailym = [], [], []
    bundles = {f: build_fold(f) for f in folds}
    for k, fold in enumerate(folds):
        if k == 0:
            log(f"{fold}: 无更早的折可训练，跳过（walk 只评估后 {len(folds) - 1} 折）")
            continue
        b = bundles[fold]
        W = wide_of(fdf, b["panel"])
        spec = [(bundles[p], None, j) for j, p in enumerate(folds[:k])]
        Xtr, yc, yg, yp, grp = prep_train(spec)
        assert grp.sum() == len(Xtr), (grp.sum(), len(Xtr))
        log(f"{fold}: walk 训练 {len(Xtr):,} 行 / {len(grp)} 天（来源 {folds[:k]}）")
        r, d, dm = run_arms(b, W, arms, rounds, Xtr, yc, yg, yp, grp,
                            None, s2_of(b, "H01"))
        rows += r; dailies += d; dailym += dm
    return _report(pd.DataFrame(rows), pd.concat(dailies, ignore_index=True),
                   pd.concat(dailym, ignore_index=True), "walk")


def _report(R: pd.DataFrame, D: pd.DataFrame, DM: pd.DataFrame, mode: str) -> pd.DataFrame:
    R.to_csv(OUT / f"pairwise_rerank_{mode}.csv", index=False)
    D.to_csv(OUT / f"pairwise_rerank_daily_{mode}.csv", index=False)
    DM.to_csv(OUT / f"pairwise_rerank_daymetrics_{mode}.csv", index=False)
    pd.set_option("display.width", 300)
    print(f"\n=== [{mode}] 逐折 ===")
    cols = ["fold", "tag", "ic", "excess", "turnover", "alpha", "profile",
            "final", "noprofile", "pool_ic", "corr_h01", "n_slice"]
    print(R[[c for c in cols if c in R.columns]].round(6).to_string(index=False))
    print(f"\n=== [{mode}] 均值 ===")
    g = R.groupby("tag")[["ic", "excess", "turnover", "alpha", "profile",
                          "final", "noprofile", "pool_ic", "corr_h01"]].mean()
    print(g.round(6).to_string())
    print(f"\n=== [{mode}] 折级配对 Δ（vs H01）===")
    out = []
    for metric in ("ic", "excess", "turnover", "alpha", "profile", "final",
                   "noprofile", "pool_ic"):
        out.extend(_paired_fold(R, metric))
    P = pd.DataFrame(out)[["tag", "metric", "delta", "t", "signs", "n"]]
    print(P.round(5).to_string(index=False))
    P.to_csv(OUT / f"pairwise_rerank_{mode}_paired.csv", index=False)
    print(f"\n=== [{mode}] 逐日配对 Δ（不依赖折数；α 逐日来自 DGTW 分解）===")
    dd = []
    for src, metric in ((DM, "ic"), (DM, "excess"), (D, "alpha"), (D, "profile")):
        w = src.pivot_table(index="trade_date", columns="tag", values=metric)
        if "H01" not in w.columns:
            continue
        for tag in w.columns:
            if tag == "H01":
                continue
            d = (w[tag] - w["H01"]).dropna()
            n = len(d)
            sd = float(np.std(d.to_numpy(), ddof=1))
            dd.append({"tag": tag, "metric": metric, "n": n, "delta": float(d.mean()),
                       "t": float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan})
    Dp = pd.DataFrame(dd)
    print(Dp.round(6).to_string(index=False))
    Dp.to_csv(OUT / f"pairwise_rerank_{mode}_dailypaired.csv", index=False)
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阶段2 排序损失重排器探针")
    ap.add_argument("--mode", default="base", choices=("base", "split", "walk"))
    ap.add_argument("--arms", default="H01,L2,LR")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--folds", default=",".join(FOLDS))
    a = ap.parse_args(argv)
    folds = [f.strip() for f in a.folds.split(",") if f.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    t0 = time.time()
    log("读因子面板（DGTW 分解用）...")
    fdf = load_factors(20210101, 20241231)
    log(f"因子面板 {fdf.shape}")
    if a.mode == "base":
        mode_base(fdf, folds)
    elif a.mode == "split":
        mode_split(fdf, folds, arms, a.rounds)
    else:
        mode_walk(fdf, folds, arms, a.rounds)
    log(f"完成，用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

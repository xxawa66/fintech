"""LH032 第二步：profile 到底能不能外推？—— 滚动半年窗口的严格样本外检验。

不看「溢价稳不稳定」这种模糊指标，直接回答可执行的问题：

    用截止到 t 之前所有窗口估计出的层溢价剖面，决定往哪些层倾斜，
    在下一个窗口里到底赚不赚钱？

若「跟着历史溢价倾斜」在每个未见窗口上都拿到正 profile ⇒ profile 可外推，
    值得为它设计选股规则。
若时正时负 ⇒ profile 不可外推，E 只能靠 α。

设计（纯暴露层，与选股无关，是 profile 这一项的**上界**）：
  - 分层变量 f，十层；层溢价 prem[k,b] = 窗口 k 内「层 b 平均收益 − 市场收益」（年化）
  - 窗口 k（半年）的倾斜权重：w_b ∝ exp(θ · z_b)，z_b = 历史均值溢价的标准化
  - 实现 profile_k = Σ_b (w_b − 1/10) · prem[k,b]
  - 对照组：① 反向倾斜（往历史低溢价层）；② 随机方向；③ 全样本前视（上界的上界）

只用 20180101–20241231（训练期），不碰测试期。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "long_horizon"
sys.path.insert(0, str(ROOT))

from scipy import stats as st  # noqa: E402

FACTORS = ["vol60", "log_amt", "mom60", "rev1", "log_price", "mom20"]
NB = 10
ANN = 252
START, END = 20180101, 20241231


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wide(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    p = df.pivot(index="trade_date", columns="ts_code", values=col)
    return p, p.index.to_numpy(), p.columns.to_numpy()


def main() -> int:
    log("读因子面板（2018–2024）")
    df = pd.read_parquet(CACHE / "_probe_factors_full.parquet",
                         filters=[("trade_date", ">=", START), ("trade_date", "<=", END)])
    df = df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    log(f"  {len(df)} 行")

    dates = np.sort(df["trade_date"].unique())
    Y, _, codes = wide(df, "y_ret_1d")
    Yv = Y.to_numpy(dtype="float64", copy=True)
    LU = wide(df, "flag_limit_up")[0].to_numpy(dtype="float64", copy=True)
    LU = np.nan_to_num(LU, nan=0.0) > 0.5
    W = {f: wide(df, f)[0].to_numpy(dtype="float64", copy=True) for f in FACTORS}
    log(f"  宽矩阵 {Yv.shape}，涨停占比 {np.nanmean(LU):.4f}")

    # 逐日层溢价
    rows = []
    for t in range(Yv.shape[0]):
        y = Yv[t]
        ok = np.isfinite(y) & ~LU[t]
        if ok.sum() < 200:
            continue
        mkt = float(np.mean(y[ok]))
        d = int(dates[t])
        win = d // 10000 * 10 + (1 if (d // 100) % 100 <= 6 else 2)  # e.g. 20211
        for f in FACTORS:
            x = W[f][t]
            m = ok & np.isfinite(x)
            if m.sum() < NB * 20:
                continue
            xm = x[m]
            ym = y[m]
            q = np.quantile(xm, np.linspace(0, 1, NB + 1)[1:-1])
            b = np.searchsorted(q, xm, side="right")
            for c in range(NB):
                s = b == c
                if s.sum() < 10:
                    continue
                rows.append(dict(win=win, factor=f, layer=c,
                                 prem=float(np.mean(ym[s])) - mkt, n=int(s.sum())))
    A = pd.DataFrame(rows)
    log(f"  逐日层表 {A.shape}")
    A.to_csv(ROOT / "experiments" / "LH032" / "prem_daily.csv", index=False)

    # 窗口 × 因子 × 层 的年化溢价
    G = A.groupby(["win", "factor", "layer"])["prem"].mean().mul(ANN).reset_index()
    G.to_csv(ROOT / "experiments" / "LH032" / "prem_window.csv", index=False)
    wins = sorted(G["win"].unique())
    log(f"  半年窗口 {len(wins)} 个：{wins[0]} ~ {wins[-1]}")

    # ---------------- 滚动外推 ----------------
    def softmax_tilt(z: np.ndarray, theta: float) -> np.ndarray:
        e = np.exp(theta * (z - z.mean()) / (z.std() + 1e-12))
        return e / e.sum()

    def realized(prem_row: np.ndarray, w: np.ndarray) -> float:
        return float(((w - 1.0 / NB) * prem_row).sum())

    out = []
    rng = np.random.default_rng(20261011)
    for f in FACTORS:
        M = G[G.factor == f].pivot_table(index="win", columns="layer", values="prem").reindex(wins)
        M = M.dropna()
        P = M.to_numpy()
        idx = list(M.index)
        for theta in [0.5, 1.0, 2.0]:
            for k in range(3, len(P)):          # 至少用 3 个窗口估计历史
                hist = P[:k].mean(axis=0)
                w_pos = softmax_tilt(hist, theta)
                w_neg = softmax_tilt(-hist, theta)
                w_rnd = softmax_tilt(rng.standard_normal(NB), theta)
                full = P.mean(axis=0)            # 前视上界
                w_full = softmax_tilt(full, theta)
                out.append(dict(factor=f, theta=theta, win=idx[k],
                                follow=realized(P[k], w_pos),
                                reverse=realized(P[k], w_neg),
                                random=realized(P[k], w_rnd),
                                oracle=realized(P[k], w_full)))
    R = pd.DataFrame(out)
    R.to_csv(ROOT / "experiments" / "LH032" / "prem_extrap.csv", index=False)

    print("\n" + "=" * 104)
    print("滚动外推：用历史层溢价决定倾斜方向，在**未见窗口**实现的 profile（年化，正=赚钱）")
    print("=" * 104)
    for theta in [0.5, 1.0, 2.0]:
        s = R[R.theta == theta]
        g = s.groupby("factor")[["follow", "reverse", "random", "oracle"]].agg(["mean", "std"])
        n = s.groupby("factor")["follow"].size()
        t = s.groupby("factor")["follow"].apply(
            lambda v: v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 1 and v.std() > 0 else np.nan)
        pos = s.groupby("factor")["follow"].apply(lambda v: "".join("+" if x > 0 else "-" for x in v))
        tab = pd.DataFrame({"n_win": n, "follow_mean": g[("follow", "mean")],
                            "sd": g[("follow", "std")], "t": t, "pos_rate": s.groupby("factor")["follow"].apply(lambda v: (v > 0).mean()),
                            "sign": pos,
                            "reverse": g[("reverse", "mean")], "random": g[("random", "mean")],
                            "oracle": g[("oracle", "mean")]})
        print(f"\n### 倾斜强度 θ={theta}（每层权重 ∝ exp(θ·z_b)）")
        print(tab.round(4).to_string())

    # ---------------- 断档检验（复刻 LH020 口径） ----------------
    print("\n" + "=" * 104)
    print("断档检验：最后 3 个半年 vs 之前（spread = L9 − L0，年化）")
    print("=" * 104)
    rows = []
    for f in FACTORS:
        M = G[G.factor == f].pivot_table(index="win", columns="layer", values="prem").reindex(wins).dropna()
        sp = (M[9] - M[0]).to_numpy()
        k = np.arange(len(sp), dtype=float)
        kk = k - k.mean()
        yy = sp - sp.mean()
        den = float((kk * kk).sum())
        slope = float((kk * yy).sum() / den)
        resid = yy - slope * kk
        se = float(np.sqrt((resid ** 2).sum() / (len(yy) - 2)) / np.sqrt(den))
        late, early = sp[-3:], sp[:-3]
        u, p = st.mannwhitneyu(late, early, alternative="two-sided")
        tt, pt = st.ttest_ind(late, early, equal_var=False)
        rows.append(dict(factor=f, n=len(sp), mean=sp.mean(), sd=sp.std(),
                         slope_t=slope / se, last3=late.mean(), prior=early.mean(),
                         mwu_p=p, welch_t=tt))
    B = pd.DataFrame(rows).set_index("factor")
    B.to_csv(ROOT / "experiments" / "LH032" / "prem_break.csv")
    print(B.round(4).to_string())
    print("\n层溢价 spread 全序列（年化）：")
    for f in FACTORS:
        M = G[G.factor == f].pivot_table(index="win", columns="layer", values="prem").reindex(wins).dropna()
        sp = (M[9] - M[0])
        print(f"  {f:10s} " + " ".join(f"{v:+.3f}" for v in sp.values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

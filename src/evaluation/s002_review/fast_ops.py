"""复核用加速层：把长表面板摊成 (D, S) 稠密数组，缓存换手变换的截面排名。

动机：``src.evaluation.turnover`` 的 ``band_scores`` / ``smooth_scores`` 每次调用都
在 110 万行上做两次 pivot（合计约 7.2s），而参数网格里同一个预测要被扫十几个
``keep_q``／``alpha``——排名宽表完全不变，重复计算被浪费掉。本模块把面板一次性
摊成 (交易日 × 股票) 稠密数组，截面排名只算一次，之后每次变换只跑那 242 天的循环。

**口径纪律**：变换逻辑逐行照抄 ``src.evaluation.turnover``（含 ``delta`` 的 1e-9
余量、``prev_top`` 递归、``TOP_MIN_VALID`` 重置分支），评分仍一律走
``src.evaluation.official_eval.evaluate_frame``。``selfcheck`` 会把本模块的
band / smooth 与 ``src`` 版本逐位比对（1e-12），不通过就抛错。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import KEYS
from src.evaluation.official_eval import TOP_MIN_VALID
from src.evaluation.turnover import TOP_FRACTION, band_scores, smooth_scores


class Panel:
    """完全平衡面板的稠密视图，行序固定为 ``(trade_date, ts_code)`` 字典序。"""

    def __init__(self, keys: pd.DataFrame, labels: pd.DataFrame):
        k = keys.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
        dates = np.sort(k["trade_date"].unique())
        codes = np.sort(k["ts_code"].unique())
        self.D, self.S = len(dates), len(codes)
        if len(k) != self.D * self.S:
            raise ValueError(f"非完全平衡面板：{len(k):,} 行 != {self.D}×{self.S}")
        if not np.array_equal(k["trade_date"].to_numpy(), np.repeat(dates, self.S)):
            raise ValueError("键序不是 (日期, 代码) 的行主序")
        if not np.array_equal(k["ts_code"].to_numpy(), np.tile(codes, self.D)):
            raise ValueError("各交易日的股票序列不一致")
        self.keys = k
        self.dates = dates
        self.codes = codes

        lab = labels.set_index(KEYS).reindex(pd.MultiIndex.from_frame(k[KEYS]))
        if lab["y_ret_1d"].isna().all():
            raise ValueError("标签对齐失败：y 全缺失")
        self.y = lab["y_ret_1d"].to_numpy(dtype=float).reshape(self.D, self.S)
        self.limit = lab["flag_limit_up"].to_numpy(dtype=float).reshape(self.D, self.S)
        self.eligible = self.limit == 0

    # ---------- 打包 / 解包 ----------
    def pack(self, values) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(self.D, self.S)

    def unpack(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(-1)

    def frame(self, pred: np.ndarray) -> pd.DataFrame:
        """构造成 evaluate_frame 可吃的长表（列序与官方适配器一致）。"""
        return pd.DataFrame({
            "ts_code": self.keys["ts_code"].to_numpy(),
            "trade_date": self.keys["trade_date"].to_numpy(),
            "pred": pred,
            "y_ret_1d": self.y.reshape(-1),
            "flag_limit_up": self.limit.reshape(-1),
        })

    # ---------- 截面排名（与 _wide_rank 等价） ----------
    def rank_full(self, P: np.ndarray) -> np.ndarray:
        """当日全部有预测股票内的百分位排名（NaN 保留）。"""
        return pd.DataFrame(P).rank(axis=1, method="average", pct=True).to_numpy()

    def rank_elig(self, P: np.ndarray) -> np.ndarray:
        """仅非涨停股票内排名，其余位置 NaN。"""
        masked = np.where(self.eligible, P, np.nan)
        return pd.DataFrame(masked).rank(axis=1, method="average", pct=True).to_numpy()

    # ---------- 留仓带（照抄 src.evaluation.turnover.band_scores） ----------
    def band(self, pred, keep_q: float, ranks: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if ranks is None:
            ranks = self.ranks(pred)
        rank_full, rank_elig = ranks
        out = rank_full.copy()
        delta = 1.0 - keep_q + 1e-9
        vals = rank_elig
        prev_top: list[int] = []
        for t in range(self.D):
            row = vals[t]
            ok = ~np.isnan(row)
            n_elig = int(ok.sum())
            if n_elig < TOP_MIN_VALID:
                prev_top = []
                continue
            n_top = max(n_elig // TOP_FRACTION, 1)
            keep = [i for i in prev_top if ok[i] and row[i] >= keep_q]
            if len(keep) > n_top:
                keep = [keep[i] for i in np.argsort(-row[keep])[:n_top]]
            if len(keep) < n_top:
                chosen = set(keep)
                for idx in np.argsort(-np.where(ok, row, -np.inf)):
                    if len(keep) >= n_top:
                        break
                    if idx not in chosen:
                        keep.append(int(idx))
                        chosen.add(int(idx))
            drop = np.array([i for i in np.flatnonzero(ok) if i not in set(keep)], dtype=int)
            out[t, drop] = rank_full[t, drop] - delta
            prev_top = keep
        return self.unpack(out)

    # ---------- 排名平滑（照抄 src.evaluation.turnover.smooth_scores） ----------
    def smooth(self, pred, alpha: float, ranks: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        rank_full = self.ranks(pred)[0] if ranks is None else ranks[0]
        out = np.empty_like(rank_full)
        out[0] = rank_full[0]
        for t in range(1, self.D):
            cur = rank_full[t]
            out[t] = np.where(np.isnan(cur), np.nan, alpha * cur + (1 - alpha) * out[t - 1])
        return self.unpack(out)

    def apply(self, pred, method: str, value: float,
              ranks: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if method == "band":
            return self.band(pred, value, ranks)
        if method == "smooth":
            return self.smooth(pred, value, ranks)
        raise ValueError(f"未知变换 {method!r}")

    def ranks(self, pred) -> tuple[np.ndarray, np.ndarray]:
        """一次算出 (rank_full, rank_elig)，供同一预测的多点扫描复用。"""
        P = self.pack(pred)
        return self.rank_full(P), self.rank_elig(P)


def selfcheck(panel: Panel, pred, values=((1.0, "band"), (0.5, "band"), (0.1, "band"), (0.05, "band"),
                                          (1.0, "smooth"), (0.8, "smooth"), (0.6, "smooth")),
              tol: float = 1e-12) -> list[dict]:
    """把本模块的变换与 ``src`` 版本逐位比对，返回逐项差异表。"""
    scored = panel.frame(np.asarray(pred, dtype=float))
    reports = []
    for value, method in values:
        mine = panel.apply(pred, method, value)
        theirs = (band_scores(scored, value) if method == "band"
                  else smooth_scores(scored, value)).to_numpy(dtype=float)
        worst = float(np.nanmax(np.abs(mine - theirs)))
        reports.append({"method": method, "value": value, "max_abs_diff": worst,
                        "ok": worst < tol})
        if worst >= tol:
            raise AssertionError(f"{method}={value} 与 src 实现不一致：{worst:.3e}")
    return reports

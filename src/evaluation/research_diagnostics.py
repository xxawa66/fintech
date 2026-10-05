"""Descriptive diagnostics; annual selection always uses the official evaluator."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.official_eval import ANNUALIZATION_DAYS, IC_MIN_VALID, TOP_MIN_VALID


def monthly_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = frame["trade_date"].astype(str).str[:6]
    rows = []
    for month, group in frame.groupby("month"):
        ic = group["ic"].dropna()
        std = ic.std(ddof=1)
        annual = group["top_excess"].mean() * ANNUALIZATION_DAYS
        turn = group["turnover"].mean()
        rows.append({"month": month, "days": len(group), "ic_mean": ic.mean(), "ic_std": std,
                     "icir": ic.mean() / std if std > 0 else 0.,
                     "ic_positive_ratio": (ic > 0).mean(),
                     "annual_excess": annual,
                     "top1_annual_ret": group["top1_ret"].mean() * ANNUALIZATION_DAYS,
                     "mean_turnover": turn,
                     "final_score": .4 * ic.mean() + .3 * annual + .3 * (1 - turn),
                     "n_days_ic": len(ic), "n_days_top": int(group.top_excess.notna().sum()),
                     "n_days_turnover": int(group.turnover.notna().sum())})
    return pd.DataFrame(rows)


def portfolio_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    """Top/Bottom use the same label/limit-up universe as the official Top metric."""
    rows = []
    for day, group in scored.groupby("trade_date"):
        valid = group[(group.flag_limit_up == 0) & group.y_ret_1d.notna()].sort_values(
            "pred", ascending=False).reset_index(drop=True)
        if len(valid) < TOP_MIN_VALID:
            continue
        n = max(len(valid) // 10, 1)
        top = valid.y_ret_1d.iloc[:n].mean()
        bottom = valid.y_ret_1d.iloc[-n:].mean()
        rows.append({"trade_date": day, "n_valid": len(valid), "top10_ret": top,
                     "top20_ret": valid.y_ret_1d.iloc[:max(len(valid)//5, 1)].mean(),
                     "bottom10_ret": bottom, "market_ret": valid.y_ret_1d.mean(),
                     "top10_excess": top - valid.y_ret_1d.mean(), "top_bottom": top - bottom})
    return pd.DataFrame(rows)


def factor_diagnostics(valid: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only screening-year labels are used; statistics do not auto-delete features."""
    rows = []
    for day, group in valid.groupby("trade_date"):
        for name in columns:
            subset = group[[name, "y_ret_1d"]].dropna()
            ic = np.nan
            if (len(subset) >= IC_MIN_VALID and subset[name].nunique() > 1
                    and subset.y_ret_1d.nunique() > 1):
                ic = float(spearmanr(subset[name], subset.y_ret_1d).statistic)
            rows.append({"trade_date": day, "feature": name, "n_valid": len(subset), "ic": ic})
    daily = pd.DataFrame(rows)
    summary = []
    for name in columns:
        series = daily.loc[daily.feature == name, "ic"].dropna()
        summary.append({"feature": name, "missing_ratio": float(valid[name].isna().mean()),
                        "ic_mean": series.mean(), "ic_std": series.std(ddof=1),
                        "ic_positive_ratio": (series > 0).mean(), "n_days_ic": len(series)})
    return daily, pd.DataFrame(summary)

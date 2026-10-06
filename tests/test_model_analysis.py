"""model_analysis 通用评分分析模板的开发检查。

合成数据（300 只 × 80 日，含涨停与标签缺失）覆盖：加权拆解恒等式、
IC/Top/换手三分项与官方指标的一致性、换手诊断的边界抖动识别、
留仓带 keep_q=1.0 自检、参照归因恒等式与端到端产物完整性。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.model_analysis import (
    _top_membership,
    analyze_model,
    ic_analysis,
    load_band_keep_q,
    load_weights,
    reference_attribution,
    score_decomposition,
    top_analysis,
    turnover_analysis,
)
from src.evaluation.official_eval import OFFICIAL_METRICS, evaluate_frame

N_STOCKS = 300
N_DAYS = 80
TOP_CUT = 0.9  # 入选分位切线（1 - 1/10）


def _synthetic(seed: int = 7) -> pd.DataFrame:
    """合成评分帧：pred 与 y 弱正相关，2% 涨停，5% 标签缺失。"""
    rng = np.random.default_rng(seed)
    dates = 20240102 + np.arange(N_DAYS)
    codes = [f"S{i:04d}" for i in range(N_STOCKS)]
    rows = []
    for d in dates:
        y = rng.normal(0, 0.02, N_STOCKS)
        pred = 0.6 * (y / 0.02) + rng.normal(0, 1, N_STOCKS)
        frame = pd.DataFrame({
            "ts_code": codes, "trade_date": int(d), "pred": pred,
            "y_ret_1d": y, "flag_limit_up": 0,
        })
        frame.loc[rng.choice(N_STOCKS, 6, replace=False), "flag_limit_up"] = 1
        frame.loc[rng.choice(N_STOCKS, 15, replace=False), "y_ret_1d"] = np.nan
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


class TestScoreDecomposition(unittest.TestCase):
    def test_identity(self):
        metrics = evaluate_frame(_synthetic())
        dec = score_decomposition(metrics)
        self.assertAlmostEqual(
            dec["final_recomputed"], metrics["final_score"], delta=1e-12)
        self.assertAlmostEqual(
            dec["ic_term"] + dec["excess_term"] + dec["stability_term"],
            metrics["final_score"], delta=1e-12)

    def test_config_weights_match_official(self):
        weights = load_weights()
        self.assertEqual(weights, {"rank_ic": 0.4, "annual_excess": 0.3,
                                   "stability": 0.3})

    def test_band_keep_q_from_config(self):
        self.assertAlmostEqual(load_band_keep_q(), 0.1, delta=1e-12)


class TestSingleMetricAnalyses(unittest.TestCase):
    def setUp(self):
        self.scored = _synthetic()
        self.metrics = evaluate_frame(self.scored)

    def test_ic_analysis_consistent_with_official(self):
        summary, monthly, quarterly = ic_analysis(self.scored, self.metrics)
        self.assertAlmostEqual(summary["ic_mean"], self.metrics["ic_mean"],
                               delta=1e-12)
        self.assertEqual(summary["n_months"], monthly["month"].nunique())
        self.assertEqual(len(quarterly), 1)  # 单季度
        self.assertTrue((monthly["ic_mean"] != 0).all())

    def test_top_analysis_consistent_with_official(self):
        summary, monthly, daily = top_analysis(self.scored)
        self.assertAlmostEqual(summary["annual_excess"],
                               self.metrics["annual_excess"], delta=1e-12)
        self.assertAlmostEqual(summary["top10_annual_ret"],
                               self.metrics["top1_annual_ret"], delta=1e-12)
        self.assertEqual(monthly["month"].nunique(), monthly.shape[0])

    def test_turnover_analysis_consistent_with_official(self):
        summary, daily = turnover_analysis(self.scored, self.metrics)
        self.assertAlmostEqual(summary["mean_turnover"],
                               self.metrics["mean_turnover"], delta=1e-12)
        self.assertEqual(summary["n_turnover_days"],
                         int(daily["turnover"].notna().sum()))
        # 合成 pred 日际重排几乎全新 → 换手接近 1，且进出分位贴近切线
        self.assertGreater(summary["mean_turnover"], 0.9)
        self.assertAlmostEqual(summary["exit_rank_quantile"], TOP_CUT,
                               delta=0.15)
        self.assertAlmostEqual(summary["entry_rank_quantile"], TOP_CUT,
                               delta=0.15)

    def test_membership_size_and_reset(self):
        scored = _synthetic()
        # 构造一个有效样本 < 100 的交易日触发官方重置分支
        bad_day = scored["trade_date"].iloc[-1]
        mask = (scored["trade_date"] == bad_day) & (scored["flag_limit_up"] == 0)
        scored.loc[mask, "flag_limit_up"] = 1
        n_elig_bad = int(((scored["trade_date"] == bad_day)
                          & (scored["flag_limit_up"] == 0)).sum())
        self.assertLess(n_elig_bad, 100)
        members, _ = _top_membership(scored)
        self.assertEqual(members[-1], set())  # 有效样本不足 → 重置为空集合


class TestBandAndReference(unittest.TestCase):
    def setUp(self):
        self.scored = _synthetic()

    def test_band_keep_q_1_reproduces_baseline(self):
        from src.evaluation.turnover import evaluate_band
        base = evaluate_frame(self.scored)
        band_metrics, _ = evaluate_band(self.scored, keep_q=1.0)
        for key in OFFICIAL_METRICS:
            self.assertAlmostEqual(band_metrics[key], base[key], delta=1e-12)

    def test_reference_attribution_identity(self):
        metrics = evaluate_frame(self.scored)
        ref = dict(metrics)
        ref["ic_mean"] += 0.01
        ref["annual_excess"] -= 0.05
        ref["mean_turnover"] += 0.1
        ref["final_score"] = (0.4 * ref["ic_mean"] + 0.3 * ref["annual_excess"]
                              + 0.3 * (1 - ref["mean_turnover"]))
        at = reference_attribution(metrics, ref)
        total = (at["attrib_ic_term"] + at["attrib_excess_term"]
                 + at["attrib_stability_term"])
        self.assertAlmostEqual(total, at["delta_final"], delta=1e-12)


class TestEndToEnd(unittest.TestCase):
    def test_analyze_model_outputs(self):
        scored = _synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pred_path = tmp / "pred.csv"
            labels_path = tmp / "labels.csv"
            scored[["ts_code", "trade_date", "pred"]].to_csv(
                pred_path, index=False)
            scored[["ts_code", "trade_date", "y_ret_1d",
                    "flag_limit_up"]].to_csv(labels_path, index=False)

            out_dir = tmp / "analysis"
            summary = analyze_model(pred_path, labels_path, out_dir,
                                    name="test_exp", band_keep_q=0.1)
            for fname in ("analysis_summary.json", "REPORT.md", "ic_daily.csv",
                          "ic_monthly.csv", "ic_quarterly.csv", "top_daily.csv",
                          "top_monthly.csv", "top_summary.json",
                          "turnover_daily.csv", "turnover_diagnosis.json",
                          "band_summary.json"):
                self.assertTrue((out_dir / fname).exists(), fname)

            loaded = json.loads((out_dir / "analysis_summary.json").read_text(
                encoding="utf-8"))
            self.assertEqual(loaded["name"], "test_exp")
            self.assertIn("band_layer", loaded)
            dec = loaded["decomposition"]
            self.assertAlmostEqual(dec["final_recomputed"],
                                   loaded["official_metrics"]["final_score"],
                                   delta=1e-12)
            report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("test_exp", report)
            self.assertIn("留仓带换手层", report)

    def test_analyze_model_no_band(self):
        scored = _synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pred_path = tmp / "pred.csv"
            labels_path = tmp / "labels.csv"
            scored[["ts_code", "trade_date", "pred"]].to_csv(
                pred_path, index=False)
            scored[["ts_code", "trade_date", "y_ret_1d",
                    "flag_limit_up"]].to_csv(labels_path, index=False)
            summary = analyze_model(pred_path, labels_path, tmp / "a2",
                                    band_keep_q=0.0)
            self.assertNotIn("band_layer", summary)


if __name__ == "__main__":
    unittest.main()

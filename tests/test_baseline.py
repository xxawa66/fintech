"""Behavior checks: historical causality, calendars, coverage, scoring and reproducibility."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import clean_history
from src.data.load_data import KEYS, X_COLUMNS
from src.data.make_dataset import attach_labels, training_arrays
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, evaluate_frame
from src.evaluation.validation import Fold, split_train_valid
from src.features.build_features import build_features
from src.features.cross_section_features import percentile_rank
from src.models.lightgbm_model import load_model, save_model, train_model
from src.utils.experiments import FIELDS, append_record, check_experiment_id, read_records
from src.utils.project import load_config


def history(stocks=3, days=100):
    records = []
    dates = pd.bdate_range("2018-01-02", periods=days).strftime("%Y%m%d").astype(int)
    for s in range(stocks):
        for i, day in enumerate(dates):
            close = (s + 1) * (10 + 0.03 * i + 0.1 * np.sin(i / 3))
            records.append({"ts_code": f"{s:06d}.SZ", "trade_date": day,
                            "open": close * 0.99, "high": close * 1.02, "low": close * 0.98,
                            "close": close, "vol": 100 + 2 * i + s,
                            "amount": (100 + 2 * i + s) * close,
                            "flag_limit_up": 0, "flag_limit_down": 0,
                            "y_ret_1d": np.sin(i / 4) * 0.01})
    return pd.DataFrame(records)


class HistoricalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _ = load_config()
        cls.raw = history()
        cls.features, cls.columns = build_features(cls.raw[KEYS + X_COLUMNS], cls.cfg["features"])

    def test_fixed_count_types_and_keys(self):
        self.assertEqual(len(self.columns), 40)
        pd.testing.assert_frame_equal(self.raw[KEYS], self.features[KEYS])
        self.assertTrue(all(self.features[c].dtype == np.float32 for c in self.columns))
        self.assertFalse(np.isinf(self.features[self.columns].to_numpy()).any())
        self.assertNotIn("y_ret_1d", self.columns)
        self.assertNotIn("close", self.columns)

    def test_formulas_by_hand(self):
        i, raw, actual = 70, self.raw, self.features.iloc[70]
        close, vol, amount = raw["close"], raw["vol"], raw["amount"]
        for w in self.cfg["features"]["return_windows"]:
            self.assertAlmostEqual(actual[f"ret_{w}"], close[i] / close[i-w] - 1, places=6)
        self.assertAlmostEqual(actual["intraday_ret"], 1 / 0.99 - 1, places=6)
        self.assertAlmostEqual(actual["high_low_range"], 1.02 / 0.98 - 1, places=6)
        self.assertAlmostEqual(actual["high_close"], 0.02, places=6)
        self.assertAlmostEqual(actual["close_low"], 1 / 0.98 - 1, places=6)
        self.assertAlmostEqual(actual["open_gap"], raw["open"][i] / close[i-1] - 1, places=6)
        self.assertAlmostEqual(actual["close_position"], 0.5, places=6)
        self.assertAlmostEqual(actual["ma_bias_20"], close[i] / close[i-19:i+1].mean() - 1, places=6)
        returns = close / close.shift(1) - 1
        self.assertAlmostEqual(actual["volatility_20"], returns[i-19:i+1].std(ddof=1), places=6)
        self.assertAlmostEqual(actual["vol_ratio_20"], vol[i] / vol[i-19:i+1].mean(), places=6)
        self.assertAlmostEqual(actual["amount_ratio_5"], amount[i] / amount[i-4:i+1].mean(), places=6)
        self.assertAlmostEqual(actual["vol_change_5"], vol[i] / vol[i-5] - 1, places=6)
        window = close[i-19:i+1]
        self.assertAlmostEqual(actual["price_position_20"],
                               (close[i] - window.min()) / (window.max() - window.min()), places=6)

    def test_future_records_cannot_change_past(self):
        cutoff = sorted(self.raw.trade_date.unique())[75]
        prefix = self.raw[self.raw.trade_date <= cutoff]
        short, _ = build_features(prefix[KEYS + X_COLUMNS], self.cfg["features"])
        altered = self.raw.copy()
        altered.loc[altered.trade_date > cutoff, ["open", "high", "low", "close", "vol", "amount"]] *= 10000
        long, _ = build_features(altered[KEYS + X_COLUMNS], self.cfg["features"])
        past = long[long.trade_date <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(short, past)

    def test_stock_isolation_and_input_order(self):
        base = [name for name in self.columns if not name.startswith("rank_")]
        alone, _ = build_features(self.raw.iloc[:100][KEYS + X_COLUMNS], self.cfg["features"])
        pd.testing.assert_frame_equal(alone[base], self.features.iloc[:100][base])
        shuffled, _ = build_features(self.raw.sample(frac=1, random_state=2)[KEYS + X_COLUMNS], self.cfg["features"])
        pd.testing.assert_frame_equal(self.features, shuffled)

    def test_missing_calendar_row_is_not_filled_or_skipped(self):
        raw = self.raw.copy()
        raw.loc[5, ["open", "high", "low", "close", "vol", "amount"]] = np.nan
        values, _ = build_features(raw[KEYS + X_COLUMNS], self.cfg["features"])
        self.assertTrue(np.isnan(values.loc[6, "ret_1"]))
        self.assertTrue(np.isnan(values.loc[10, "ret_5"]))
        self.assertTrue(values.loc[5:9, "ma_bias_5"].isna().all())
        self.assertTrue(np.isfinite(values.loc[10, "ma_bias_5"]))
        self.assertTrue(values.loc[6:10, "volatility_5"].isna().all())

    def test_zero_denominators_and_true_zero_values(self):
        raw = self.raw.copy()
        raw.loc[5, "vol"] = 0
        raw.loc[5, ["open", "high", "low"]] = raw.loc[5, "close"]
        values, _ = build_features(raw[KEYS + X_COLUMNS], self.cfg["features"])
        self.assertEqual(values.loc[5, "zero_volume"], 1)
        self.assertEqual(values.loc[5, "vol_ratio_5"], 0)
        self.assertEqual(values.loc[5, "vol_change_1"], -1)
        self.assertTrue(np.isnan(values.loc[6, "vol_change_1"]))
        self.assertTrue(np.isnan(values.loc[5, "close_position"]))

    def test_rank_ties_missing_values_and_day_isolation(self):
        values = pd.Series([1., 1., 3., np.nan, 100., 50.])
        dates = pd.Series([1, 1, 1, 1, 2, 2])
        actual = percentile_rank(values, dates, pd.Series([True] * 6))
        np.testing.assert_allclose(actual, [0.5, 0.5, 1., np.nan, 1., 0.5], equal_nan=True)

    def test_target_column_rejected(self):
        with self.assertRaisesRegex(ValueError, "no labels"):
            build_features(self.raw, self.cfg["features"])


class PipelineContractTests(unittest.TestCase):
    def test_cleaning_preserves_missing_rows_and_rejects_duplicates(self):
        raw = history()
        raw.loc[5, ["open", "high", "low", "close", "vol", "amount", "y_ret_1d"]] = np.nan
        cleaned, stats = clean_history(raw)
        self.assertEqual(len(cleaned), len(raw))
        self.assertEqual(stats["invalid_quote_rows"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            clean_history(pd.concat([raw, raw.iloc[:1]], ignore_index=True))

    def test_invalid_prices_and_dates_stop_the_pipeline(self):
        raw = history()
        raw.loc[0, "close"] = -1
        with self.assertRaisesRegex(ValueError, "invalid quote"):
            clean_history(raw)
        raw = history()
        raw.loc[0, "trade_date"] = 20181332
        with self.assertRaises(ValueError):
            clean_history(raw)

    def test_split_boundary_mask_does_not_modify_original_labels(self):
        cfg, _ = load_config()
        raw = history()
        features, columns = build_features(raw[KEYS + X_COLUMNS], cfg["features"])
        combined = attach_labels(raw, features)
        dates = sorted(raw.trade_date.unique())
        fold = Fold("unit", dates[0], dates[69], dates[70], dates[-1])
        train, valid, info = split_train_valid(combined, fold)
        self.assertEqual(info["n_boundary_labels_dropped"], 3)
        self.assertTrue(train.loc[train.trade_date == dates[69], "y_ret_1d"].isna().all())
        self.assertTrue(combined.y_ret_1d.notna().all())
        _, y, training_info = training_arrays(train, columns)
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(training_info["used_rows"], 207)
        self.assertEqual(len(valid), 90)

    def test_prediction_coverage_duplicates_and_nonfinite_values(self):
        truth = history()[KEYS]
        pred = truth.copy().assign(pred=0.01)
        check_predictions(pred, truth)
        for bad in [pred.iloc[:-1], pd.concat([pred.iloc[:-1], pred.iloc[:1]])]:
            with self.assertRaises(ValueError):
                check_predictions(bad, truth)
        pred.loc[0, "pred"] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            check_predictions(pred, truth)

    def test_official_score_agreement_with_missing_labels_and_reset_days(self):
        rng = np.random.default_rng(7)
        dates = [20240102, 20240103, 20240104, 20240105, 20240108, 20240109, 20240110, 20240111]
        rows = []
        for day, date in enumerate(dates):
            for stock in range(120):
                rows.append({"ts_code": f"{stock:06d}.SZ", "trade_date": date,
                             "pred": float(rng.normal()),
                             "y_ret_1d": np.nan if day == 2 and stock >= 20 else float(rng.normal(0, .02)),
                             "flag_limit_up": int(day == 4 and stock >= 30)})
        frame = pd.DataFrame(rows)
        metrics = evaluate_frame(frame)
        self.assertEqual(metrics["n_days_ic"], 7)
        self.assertEqual(metrics["n_days_turnover"], 5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.csv"
            frame[KEYS + ["pred"]].to_csv(path, index=False)
            differences = compare_official(path, frame, metrics, 1e-10)
        self.assertTrue(all(value <= 1e-10 for value in differences.values()))

    def test_model_reload_repeatability_and_unicode_paths(self):
        cfg, _ = load_config()
        raw = history()
        features, columns = build_features(raw[KEYS + X_COLUMNS], cfg["features"])
        x, y = features[columns], raw.y_ret_1d.to_numpy()
        params = {**cfg["baseline"]["model"]["params"], "min_data_in_leaf": 5, "num_threads": 2}
        model1, model2 = train_model(x, y, params, 10), train_model(x, y, params, 10)
        first = model1.predict(x, num_threads=2)
        np.testing.assert_allclose(first, model2.predict(x, num_threads=2), rtol=0, atol=1e-12)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "中文模型目录"
            directory.mkdir()
            path = directory / "基线模型.txt"
            save_model(model1, path)
            loaded = load_model(path)
            np.testing.assert_allclose(first, loaded.predict(x, num_threads=2), rtol=0, atol=1e-12)

    def test_log_header_preservation_duplicate_guard_and_path_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment_log.csv"
            record = {field: "" for field in FIELDS}
            record["exp_id"] = "UNIT_fixture"
            append_record(path, record)
            self.assertEqual(read_records(path)[0]["exp_id"], "UNIT_fixture")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                append_record(path, record)
            with self.assertRaises(ValueError):
                check_experiment_id("../outside", path, [])
            with self.assertRaises(ValueError):
                check_experiment_id("UNIT_fixture", path, [])
            with self.assertRaises(FileExistsError):
                check_experiment_id("new", path, [Path(temporary)])


if __name__ == "__main__":
    unittest.main()

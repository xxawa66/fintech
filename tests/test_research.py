"""Day 5–8 formula, causality, cache, selection and end-to-end contract checks."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from test_baseline import history
from src.data.clean_data import clean_history
from src.data.load_data import KEYS, X_COLUMNS
from src.data.make_dataset import attach_labels
from src.evaluation.official_eval import daily_metrics, evaluate_frame
from src.evaluation.research_diagnostics import factor_diagnostics, monthly_metrics, portfolio_diagnostics
from src.evaluation.validation import Fold
from src.features.build_features import build_features
from src.features.research_features import research_names
from src.models.baseline import PreparedHistory, run_one
from src.models.research import PRESET_NAMES, choose_preset, preset_columns, promotion, verify_lock
from src.utils.experiments import FIELDS, append_record, read_records
from src.utils.project import ROOT, load_config, sha256
from src.utils.research_cache import cache_identity, cached_features, code_hashes, digest, protocol_config, validate_cache


class ResearchFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _ = load_config()
        cls.raw = history()
        cls.features, cls.columns = build_features(cls.raw[KEYS + X_COLUMNS], cls.cfg["features"],
                                                   research=cls.cfg["research"])

    def build(self, raw):
        return build_features(raw[KEYS + X_COLUMNS], self.cfg["features"], research=self.cfg["research"])[0]

    def test_v1_exactly_unchanged_and_58_float32_columns(self):
        old, old_names = build_features(self.raw[KEYS + X_COLUMNS], self.cfg["features"])
        pd.testing.assert_frame_equal(old, self.features[KEYS + old_names])
        self.assertEqual(len(self.columns), 58)
        self.assertTrue(all(self.features[c].dtype == np.float32 for c in self.columns))
        self.assertFalse(np.isinf(self.features[self.columns].to_numpy()).any())

    def test_preset_counts_and_order(self):
        self.assertEqual([p["name"] for p in self.cfg["research"]["presets"]], PRESET_NAMES)
        self.assertEqual([len(preset_columns(self.cfg, p)) for p in self.cfg["research"]["presets"]],
                         [40, 46, 46, 46, 58, 52, 52, 52])
        self.assertEqual(preset_columns(self.cfg, self.cfg["research"]["presets"][4]), self.columns)
        bad = {"name": "bad", "groups": ["T", "T"]}
        with self.assertRaises(ValueError):
            preset_columns(self.cfg, bad)

    def test_trend_formulas_by_hand(self):
        i = 75
        c = self.raw.close.iloc[:100]
        r = c / c.shift(1) - 1
        w = r.iloc[i-19:i+1]
        actual = self.features.iloc[i]
        self.assertAlmostEqual(actual.trend_efficiency_20, (c[i]-c[i-20])/c.diff().abs().iloc[i-19:i+1].sum(), places=6)
        self.assertAlmostEqual(actual.up_ratio_20, (w > 0).mean(), places=6)
        self.assertAlmostEqual(actual.downside_rms_20, np.sqrt(np.mean(np.minimum(w, 0)**2)), places=6)
        self.assertAlmostEqual(actual.return_skew_20, w.skew(), places=5)
        self.assertAlmostEqual(actual.return_kurt_20, w.kurt(), places=5)

    def test_volume_formulas_by_hand(self):
        i = 75
        c, v = self.raw.close.iloc[:100], self.raw.vol.iloc[:100]
        r = c / c.shift(1) - 1
        actual = self.features.iloc[i]
        w = slice(i-19, i+1)
        self.assertAlmostEqual(actual.corr_ret_logvol_20, np.corrcoef(r.iloc[w], np.log1p(v.iloc[w]))[0, 1], places=5)
        self.assertAlmostEqual(actual.corr_price_vol_20, np.corrcoef(c.iloc[w], v.iloc[w])[0, 1], places=5)
        for size in [5, 20]:
            idx = slice(i-size+1, i+1)
            self.assertAlmostEqual(actual[f"signed_vol_ratio_{size}"],
                                   (np.sign(r.iloc[idx])*v.iloc[idx]).sum()/v.iloc[idx].sum(), places=6)
        self.assertAlmostEqual(actual.volume_cv_20, v.iloc[w].std(ddof=1)/v.iloc[w].mean(), places=6)

    def test_risk_formulas_by_hand(self):
        i = 75
        s = self.raw.iloc[:100]
        c = s.close
        r = c / c.shift(1) - 1
        actual = self.features.iloc[i]
        def sigma(n):
            return r.iloc[i-n+1:i+1].std(ddof=1)
        self.assertAlmostEqual(actual.volatility_ratio_5_20, sigma(5)/sigma(20), places=5)
        self.assertAlmostEqual(actual.volatility_ratio_20_60, sigma(20)/sigma(60), places=5)
        self.assertAlmostEqual(actual.risk_adj_ret_20, (c[i]/c[i-20]-1)/(sigma(20)*np.sqrt(20)), places=5)
        previous = c.shift(1)
        parts = pd.concat([s.high-s.low, (s.high-previous).abs(), (s.low-previous).abs()], axis=1)
        tr = parts.max(axis=1, skipna=False)/previous
        self.assertAlmostEqual(actual.true_range_relative, tr[i], places=6)
        self.assertAlmostEqual(actual.atr_ratio_5_20, tr.iloc[i-4:i+1].mean()/tr.iloc[i-19:i+1].mean(), places=5)

    def test_ranks_use_current_x_only(self):
        for source in ["trend_efficiency_20", "signed_vol_ratio_20", "risk_adj_ret_20"]:
            expected = self.features.groupby("trade_date")[source].rank(method="average", pct=True).astype("float32")
            pd.testing.assert_series_equal(expected, self.features[f"rank_{source}"], check_names=False)
        altered = self.raw.copy()
        altered.y_ret_1d *= -100
        pd.testing.assert_frame_equal(self.features, self.build(altered))

    def test_future_rows_cannot_change_past(self):
        cutoff = sorted(self.raw.trade_date.unique())[75]
        before = self.build(self.raw[self.raw.trade_date <= cutoff])
        altered = self.raw.copy()
        altered.loc[altered.trade_date > cutoff, ["open", "high", "low", "close", "vol", "amount"]] *= 10000
        after = self.build(altered)
        pd.testing.assert_frame_equal(before, after[after.trade_date <= cutoff].reset_index(drop=True))

    def test_stock_isolation_and_input_shuffle(self):
        names = [n for n in self.columns if not n.startswith("rank_")]
        alone = self.build(self.raw.iloc[:100])
        pd.testing.assert_frame_equal(alone[names], self.features.iloc[:100][names])
        pd.testing.assert_frame_equal(self.features, self.build(self.raw.sample(frac=1, random_state=9)))

    def test_missing_calendar_gap_propagates(self):
        raw = self.raw.copy()
        raw.loc[30, ["open", "high", "low", "close", "vol", "amount"]] = np.nan
        f = self.build(raw)
        for name in ["trend_efficiency_20", "up_ratio_20", "downside_rms_20", "return_skew_20",
                     "return_kurt_20", "corr_ret_logvol_20", "signed_vol_ratio_20", "atr_ratio_5_20"]:
            self.assertTrue(f.loc[30:50, name].isna().all(), name)
            self.assertTrue(np.isfinite(f.loc[51, name]), name)
        self.assertTrue(np.isnan(f.loc[31, "true_range_relative"]))
        self.assertTrue(np.isfinite(f.loc[32, "true_range_relative"]))

    def test_full_windows_at_start(self):
        for n in ["trend_efficiency_20", "up_ratio_20", "downside_rms_20", "signed_vol_ratio_20", "atr_ratio_5_20"]:
            self.assertTrue(self.features.loc[:19, n].isna().all(), n)
            self.assertTrue(np.isfinite(self.features.loc[20, n]), n)
        self.assertTrue(self.features.loc[:59, "volatility_ratio_20_60"].isna().all())

    def test_constant_zero_denominators_and_true_zero(self):
        raw = history(stocks=1)
        for c in ["open", "high", "low", "close"]:
            raw[c] = 10.
        raw["vol"] = 100.
        f = self.build(raw)
        for n in ["trend_efficiency_20", "return_skew_20", "return_kurt_20", "corr_ret_logvol_20",
                  "corr_price_vol_20", "volatility_ratio_5_20", "risk_adj_ret_20", "atr_ratio_5_20"]:
            self.assertTrue(np.isnan(f.loc[75, n]), n)
        for n in ["up_ratio_20", "downside_rms_20", "signed_vol_ratio_20", "volume_cv_20", "true_range_relative"]:
            self.assertEqual(f.loc[75, n], 0, n)
        raw["vol"] = 0.
        zero = self.build(raw)
        self.assertTrue(zero.signed_vol_ratio_20.isna().all())
        self.assertTrue(zero.volume_cv_20.isna().all())

    def test_label_and_duplicate_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "no labels"):
            build_features(self.raw, self.cfg["features"], research=self.cfg["research"])
        with self.assertRaisesRegex(ValueError, "not unique"):
            self.build(pd.concat([self.raw, self.raw.iloc[:1]]))

    def test_invalid_quote_excluded_from_new_ranks(self):
        raw = self.raw.copy()
        raw.loc[75, "open"] = np.nan
        f = self.build(raw)
        for name in research_names(self.cfg["research"]):
            if name.startswith("rank_"):
                self.assertTrue(np.isnan(f.loc[75, name]))


class CacheAndSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg, _ = load_config()
        self.raw, _ = clean_history(history(stocks=2))
        dates = sorted(map(int, self.raw.trade_date.unique()))
        self.fold = Fold("unit", dates[0], dates[69], dates[70], dates[-1])
        self.provenance = {"sha256": "synthetic_fixture"}

    def results(self):
        counts = [40, 46, 46, 46, 58, 52, 52, 52]
        return [{"preset": n, "status": "passed", "feature_count": c, "metrics": {"final_score": .2}}
                for n, c in zip(PRESET_NAMES, counts)]

    def test_cache_fingerprint_changes_for_history_stocks_configuration(self):
        a = cache_identity(self.raw, self.cfg, self.fold, self.provenance)
        self.assertNotEqual(digest(a), digest(cache_identity(self.raw.iloc[:100], self.cfg, self.fold, self.provenance)))
        another = Fold("other", self.fold.train_start, self.fold.train_end, self.fold.valid_start, 20241231)
        self.assertNotEqual(digest(a), digest(cache_identity(self.raw, self.cfg, another, self.provenance)))
        cfg = copy.deepcopy(self.cfg)
        cfg["research"]["windows"]["path"] += 1
        self.assertNotEqual(digest(a), digest(cache_identity(self.raw, cfg, self.fold, self.provenance)))
        a["feature_code"]["src/features/research_features.py"] = "altered"
        self.assertNotEqual(digest(a), digest(cache_identity(self.raw, self.cfg, self.fold, self.provenance)))

    def test_cache_roundtrip_hit_and_corruption_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs/metrics") as temporary:
            cfg = copy.deepcopy(self.cfg)
            cfg["paths"]["research_cache"] = Path(temporary).relative_to(ROOT).as_posix()
            first, meta1 = cached_features(self.raw, cfg, self.fold, self.provenance, lambda _: None)
            second, meta2 = cached_features(self.raw, cfg, self.fold, self.provenance, lambda _: None)
            pd.testing.assert_frame_equal(first, second)
            self.assertFalse(meta1["hit"])
            self.assertTrue(meta2["hit"])
            self.assertNotIn("y_ret_1d", first)
            directory = ROOT / meta1["path"]
            path = directory / "features.parquet"
            with path.open("ab") as stream:
                stream.write(b"altered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_cache(directory, meta1["identity"], self.raw, meta1["columns"])

    def test_selection_requires_all_eight(self):
        with self.assertRaises(ValueError):
            choose_preset(self.results()[:-1], 1e-6)
        rows = self.results()
        rows[2]["status"] = "failed"
        with self.assertRaises(ValueError):
            choose_preset(rows, 1e-6)

    def test_selection_ties_and_small_gain_keep_base(self):
        rows = self.results()
        self.assertEqual(choose_preset(rows, 1e-6), "base")
        rows[1]["metrics"]["final_score"] += 5e-7
        self.assertEqual(choose_preset(rows, 1e-6), "base")
        rows[1]["metrics"]["final_score"] += .01
        rows[4]["metrics"]["final_score"] = rows[1]["metrics"]["final_score"] + 5e-7
        self.assertEqual(choose_preset(rows, 1e-6), "trend")
        rows[2]["metrics"]["final_score"] = rows[1]["metrics"]["final_score"]
        self.assertEqual(choose_preset(rows, 1e-6), "trend")

    def test_lock_rejects_candidate_code_config_and_manifest_changes(self):
        source = code_hashes()
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs/metrics") as temporary:
            path = Path(temporary) / "run.json"
            path.write_text("{}", encoding="utf-8")
            rows = self.results()
            for r in rows:
                r.update(manifest_path=path.relative_to(ROOT).as_posix(), manifest_sha256=sha256(path))
            selection = {"selected_preset": "base", "results": rows,
                         "protocol_sha256": digest(protocol_config(self.cfg)), "source_sha256": digest(source)}
            locked = digest(selection)
            verify_lock(selection, locked, self.cfg, source)
            changed = {**selection, "selected_preset": "trend"}
            with self.assertRaisesRegex(ValueError, "lock changed"):
                verify_lock(changed, locked, self.cfg, source)
            with self.assertRaisesRegex(ValueError, "Code/configuration"):
                verify_lock(selection, locked, self.cfg, {})
            changed_cfg = copy.deepcopy(self.cfg)
            changed_cfg["baseline"]["model"]["params"]["seed"] = 999
            with self.assertRaisesRegex(ValueError, "Code/configuration"):
                verify_lock(selection, locked, changed_cfg, source)
            path.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                verify_lock(selection, locked, self.cfg, source)

    def test_project_progress_does_not_change_protocol(self):
        changed = copy.deepcopy(self.cfg)
        changed["project"]["stage"] = "finished"
        self.assertEqual(digest(protocol_config(changed)), digest(protocol_config(self.cfg)))

    def test_cross_year_failure_keeps_baseline(self):
        m = lambda v: {"final_score": v}
        self.assertEqual(promotion("trend", m(.21), m(.2), m(.19), m(.2), 1e-6), "base")
        self.assertEqual(promotion("trend", m(.21), m(.2), m(.22), m(.2), 1e-6), "trend")
        self.assertEqual(promotion("trend", m(.2), m(.2), m(.22), m(.2), 1e-6), "base")

    def test_log_lock_preserves_another_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment_log.csv"
            lock = path.with_name(path.name + ".lock")
            lock.write_text("another writer", encoding="utf-8")
            row = {k: "" for k in FIELDS}
            row["exp_id"] = "UNIT_fixture"
            with self.assertRaisesRegex(RuntimeError, "locked"):
                append_record(path, row)
            self.assertEqual(lock.read_text(encoding="utf-8"), "another writer")
            self.assertFalse(path.exists())


class DiagnosticAndPipelineTests(unittest.TestCase):
    def scored(self):
        frame = history(stocks=160, days=45)
        rng = np.random.default_rng(13)
        frame["pred"] = rng.normal(size=len(frame))
        frame["y_ret_1d"] += rng.normal(scale=.01, size=len(frame))
        return frame

    def test_monthly_weighted_metrics_reconstruct_annual(self):
        scored = self.scored()
        metrics = evaluate_frame(scored)
        monthly = monthly_metrics(daily_metrics(scored))
        for name, weight in [("ic_mean", "n_days_ic"), ("annual_excess", "n_days_top"),
                             ("top1_annual_ret", "n_days_top"), ("mean_turnover", "n_days_turnover")]:
            self.assertAlmostEqual(np.average(monthly[name], weights=monthly[weight]), metrics[name], places=12)

    def test_top_bottom_uses_official_top_universe(self):
        scored = self.scored()
        scored.loc[::7, "flag_limit_up"] = 1
        scored.loc[::11, "y_ret_1d"] = np.nan
        official = daily_metrics(scored)
        diagnostic = portfolio_diagnostics(scored)
        np.testing.assert_allclose(diagnostic.top10_excess, official.top_excess, rtol=0, atol=1e-12)

    def test_factor_constant_and_missing_values(self):
        scored = self.scored()
        scored["constant"] = 1.
        scored["missing"] = np.nan
        daily, summary = factor_diagnostics(scored, ["constant", "missing"])
        self.assertTrue(daily.ic.isna().all())
        self.assertTrue(summary.n_days_ic.eq(0).all())
        self.assertEqual(summary.loc[summary.feature == "missing", "missing_ratio"].item(), 1.)

    def test_prepared_history_end_to_end_smoke_and_no_formal_log(self):
        cfg, _ = load_config()
        raw, cleaning = clean_history(history(stocks=120, days=100))
        features, columns = build_features(raw[KEYS + X_COLUMNS], cfg["features"], research=cfg["research"])
        raw["y_ret_1d"] = .2 * (features["vol_ratio_20"].fillna(1) - 1) + np.random.default_rng(9).normal(scale=.0001, size=len(raw))
        dates = sorted(map(int, raw.trade_date.unique()))
        fold = Fold("unit", dates[0], dates[89], dates[90], dates[-1])
        truth = raw.loc[raw.trade_date >= dates[90], KEYS + ["y_ret_1d", "flag_limit_up"]].copy()
        prepared = PreparedHistory(attach_labels(raw, features), truth, cleaning, {"fixture": True}, fold.name)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs/metrics") as temporary:
            root = Path(temporary)
            for name in ["models", "metrics", "predictions", "processed"]:
                cfg["paths"][name] = (root / name).relative_to(ROOT).as_posix()
            cfg["paths"]["experiment_log"] = (root / "log.csv").relative_to(ROOT).as_posix()
            source = root / "synthetic.csv"
            raw.to_csv(source, index=False)
            cfg["paths"]["train"] = source.relative_to(ROOT).as_posix()
            config_path = root / "config.yaml"
            config_path.write_text("synthetic fixture", encoding="utf-8")
            provenance = {"sha256": sha256(source), "official_attachments": {p: sha256(ROOT / p) for p in
                           ["evaluate.py", "evaluate.R", "docs/赛题五-更新.pdf"]}}
            context = {"feature_version": "UNIT_research_all"}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = run_one(cfg, config_path, fold, "UNIT_research", "test", provenance,
                                 {"commit": "fixture", "branch": "main", "dirty": False}, True,
                                 prepared=prepared, columns=columns, context=context)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["prediction"]["rows"], 1200)
            self.assertEqual(result["model"]["num_boost_round"], 10)
            self.assertEqual(result["prediction"]["model_reload_max_abs_difference"], 0.)
            self.assertEqual(result["official_score_max_difference"], 0.)
            self.assertEqual(read_records(root / "log.csv"), [])
            self.assertTrue((root / "metrics/UNIT_research/monthly_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()

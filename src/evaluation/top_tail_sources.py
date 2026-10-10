"""Read-only S009 source adapter and causal context; no model fitting."""
from __future__ import annotations

import gc
import importlib.metadata

import numpy as np
import pandas as pd

from src.data.clean_data import clean_history, valid_quote
from src.data.load_data import KEYS, X_COLUMNS, load_training_data
from src.features.build_features import feature_names
from src.features.market_features import MARKET_ALL, market_daily_table
from src.models.alpha_research import is_canonical, relative, save_csv
from src.models.optuna_tuning import read_json, save_json
from src.utils.project import ROOT, sha256
from src.utils.research_cache import contained_path, digest


class FrozenInputs:
    def __init__(self, hashes=None):
        self.hashes = dict(hashes or {})
        self.verified = set()

    def check(self, path, expected=None):
        path = contained_path(path)
        name = relative(path)
        if name not in self.verified:
            actual = sha256(path)
            if name in self.hashes and self.hashes[name] != actual:
                raise ValueError(f"Frozen input changed: {name}")
            self.hashes[name] = actual
            self.verified.add(name)
        if expected is not None and self.hashes[name] != expected:
            raise ValueError(f"Input checksum differs: {name}")
        return self.hashes[name]


def prepare_sources(cfg, frozen):
    settings = cfg["top_tail_diagnostics"]
    indexes = {}
    for study, path in settings["source_artifacts"].items():
        frozen.check(path)
        index = read_json(contained_path(path))
        if index["status"] != "passed":
            raise ValueError("Source study did not pass its artifact audit.")
        indexes[study] = {r["exp_id"]: r for r in index["runs"]}
    receipt_path = contained_path(settings["input_inventory"])
    frozen.check(receipt_path)
    receipt = read_json(receipt_path)
    if receipt["package_path"] != cfg["paths"]["research_handoff_inputs"]:
        raise ValueError("Handoff package path changed.")
    for entry in receipt["files"]:
        frozen.check(entry["path"], entry["sha256"])
    package = contained_path(receipt["package_path"])
    summary7 = read_json(ROOT / "experiments/market_regime_S007_summary.json")
    summary8 = read_json(ROOT / "experiments/optuna_s008_summary.json")
    selection3 = read_json(contained_path(cfg["alpha_research"]["source_selection"]))
    for path in ["experiments/market_regime_S007_summary.json", "experiments/optuna_s008_summary.json",
                 cfg["alpha_research"]["source_selection"], "docs/alpha_research_S005_selection.json",
                 "docs/alpha_research_S006_selection.json"]:
        frozen.check(path)
    if read_json(package / "studies/S008/selection.json")["winner_spec"] != summary8["winner_spec"]:
        raise ValueError("T033 package selection differs from published selection.")

    def local(study, exp_id):
        entry = indexes[study][exp_id]
        frozen.check(entry["manifest"], entry["manifest_sha256"])
        run = read_json(contained_path(entry["manifest"]))
        if run["status"] != "passed" or run["exp_id"] != exp_id:
            raise ValueError("Source run identity/status changed.")
        for path, expected in run["artifact_hashes"].items():
            frozen.check(path, expected)
        return {"prediction": run["prediction"], "metrics": run["metrics"],
                "manifest": entry["manifest"], "run": run}

    cases = []
    for fold in [*cfg["optuna"]["folds"], cfg["optuna"]["confirm"]]:
        name, year = fold["name"], fold["valid"][0]//10000
        suffix = "confirm2024" if year == 2024 else f"T030_{name}"
        pair = {layer: local("S003", f"S003_{suffix}_{'raw' if layer == 'raw' else 'band'}")
                for layer in ["raw", "band"]}
        cases.append({"model": "T030", "fold": name, "year": year, **pair})
        for target in ["Y1", "Y4"]:
            if year == 2024 and target == "Y1":
                continue
            if year == 2024:
                raw = local("S006", "S006_confirm2024_Y4_raw")
                band = local("S006", "S006_confirm2024_new_signal_qstar")
            else:
                raw = local("S005", f"S005_{target}_{name}_raw")
                band = local("S006", f"S006_{'F006' if target == 'Y1' else 'F003'}_{name}")
            cases.append({"model": target, "fold": name, "year": year, "raw": raw, "band": band})
        for model, directory, expected in [
            ("Market", package / f"studies/S007/market/{name}", summary7["folds"][f"market/{name}"]),
            ("T033", package / f"studies/S008/winner/T033/{name}", summary8["winner_folds"][name])]:
            manifest = directory / "metrics.json"
            run = read_json(manifest)
            spec = selection3["winner"]["spec"] if model == "Market" else summary8["winner_spec"]
            if run["spec_digest"] != digest(spec) or run["fold_window"] != fold["train"]+fold["valid"]:
                raise ValueError("Handoff model spec or fold boundary differs.")
            if model == "Market" and run["columns"] != feature_names(cfg["features"])+MARKET_ALL:
                raise ValueError("Market feature column order changed.")
            pair = {}
            for layer, label in [("raw", "raw"), ("band", "band_qstar")]:
                if run["metrics"][label] != expected[label]:
                    raise ValueError("Handoff metrics differ from GitHub summary.")
                pair[layer] = {"prediction": relative(directory / f"prediction_{label}.csv"),
                    "metrics": run["metrics"][label], "manifest": relative(manifest), "run": run}
            cases.append({"model": model, "fold": name, "year": year, **pair})
    if len(cases) != 19 or sum(len([c for c in cases if c["year"] == y]) for y in range(2021, 2025)) != 19:
        raise ValueError("The predeclared 19 model-year coverage changed.")
    return cases


def prepare_context(cfg, root, frozen, data, log):
    index = read_json(contained_path(cfg["top_tail_diagnostics"]["source_artifacts"]["S003"]))
    cache = index["confirmation_cache"]
    directory = contained_path(cache["path"])
    frozen.check(directory / "cache.json")
    frozen.check(directory / "features.parquet", cache["parquet_sha256"])
    identity = cache["identity"]
    if (identity["raw_sha256"] != data["sha256"] or identity["features"] != cfg["features"]
            or identity["history"] != [20180102, 20241231]):
        raise ValueError("Frozen V1 context cache identity differs.")
    for path, expected in identity["code"].items():
        frozen.check(path, expected)
    if identity["packages"] != {p: importlib.metadata.version(p) for p in identity["packages"]}:
        raise ValueError("Frozen cache package versions changed.")
    log("read original history and verified causal V1 context columns")
    raw, cleaning = clean_history(load_training_data(contained_path(cfg["paths"]["train"])))
    if len(raw) != 7_900_350:
        raise ValueError("The complete original training history is required.")
    feature = pd.read_parquet(directory / "features.parquet", columns=KEYS+["ret_1", "volatility_20"])
    feature["ts_code"] = pd.Categorical(feature.ts_code, categories=raw.ts_code.cat.categories)
    feature["trade_date"] = feature.trade_date.astype(raw.trade_date.dtype)
    if not feature[KEYS].equals(raw[KEYS]):
        raise ValueError("Context X cache keys changed.")
    x_only = raw[KEYS+X_COLUMNS]
    market = market_daily_table(feature, x_only)
    stop = 20231231
    prefix_market = market_daily_table(feature[feature.trade_date <= stop], x_only[x_only.trade_date <= stop])
    if not market[market.trade_date <= stop].reset_index(drop=True).equals(prefix_market.reset_index(drop=True)):
        raise ValueError("Actual market context prefix is not causal.")
    save_csv(root / "market_daily.csv", market)
    thresholds, contexts = [], {}
    quote = valid_quote(raw)
    for fold in [*cfg["optuna"]["folds"], cfg["optuna"]["confirm"]]:
        year = fold["valid"][0]//10000
        past = market[market.trade_date <= fold["train"][1]]
        threshold = {"year": year, "training_end": fold["train"][1], "past_days": len(past),
                     "vol_median": float(past.mkt_ret_vol_20.median()),
                     "breadth_median": float(past.mkt_breadth_20.median())}
        thresholds.append(threshold)
        marked = market[market.trade_date.between(*fold["valid"])].copy()
        marked["state_vol"] = np.where(marked.mkt_ret_vol_20 > threshold["vol_median"], "high", "low")
        marked["state_breadth"] = np.where(marked.mkt_breadth_20 > threshold["breadth_median"], "high", "low")
        marked["state_trend"] = np.where(marked.mkt_mom_20 > 0, "positive", "nonpositive")
        take = raw.trade_date.between(*fold["valid"])
        context = raw.loc[take, KEYS+["y_ret_1d", "flag_limit_up", "amount"]].copy()
        context["quote_valid"] = quote[take].to_numpy()
        context["volatility_20"] = feature.loc[take, "volatility_20"].to_numpy()
        context["ts_code"] = context.ts_code.astype(str)
        context = context.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
        labels_path = index["confirmation_cache" if year == 2024 else "search_cache"]["folds"][fold["name"]]["labels"]
        expected_sha = index["confirmation_cache" if year == 2024 else "search_cache"]["folds"][fold["name"]]["labels_sha256"]
        frozen.check(labels_path, expected_sha)
        labels = pd.read_csv(contained_path(labels_path), dtype={"ts_code": "str", "trade_date": "int32"})
        if (not is_canonical(labels) or not labels[KEYS].equals(context[KEYS])
                or not np.array_equal(labels.flag_limit_up, context.flag_limit_up)
                or not np.array_equal(labels.y_ret_1d.isna(), context.y_ret_1d.isna())
                or np.nanmax(abs(labels.y_ret_1d.to_numpy()-context.y_ret_1d.to_numpy())) > 1e-12):
            raise ValueError("Frozen labels differ from original complete history.")
        # Keep the historical CSV representation used by the official score.
        context["y_ret_1d"] = labels.y_ret_1d.to_numpy()
        context = context.merge(marked, on="trade_date", how="left", validate="many_to_one")
        for column, destination in [("amount", "liquidity_group"), ("volatility_20", "stock_vol_group")]:
            known = context[column].where(context.quote_valid & np.isfinite(context[column]))
            ranks = known.groupby(context.trade_date).rank(method="average", pct=True)
            group = np.ceil(ranks.fillna(0).to_numpy()*4).astype("int8")
            context[destination] = group
        path = root / "context" / f"{fold['name']}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        context.to_parquet(path, index=False)
        contexts[fold["name"]] = {"path": relative(path), "sha256": sha256(path),
            "rows": len(context), "days": context.trade_date.nunique(), "labels": labels_path,
            "fallback_rows": int((~context.quote_valid).sum()), "year": year}
        log(f"context {fold['name']}: {len(context):,} rows; causal historical thresholds")
    save_csv(root / "state_thresholds.csv", pd.DataFrame(thresholds))
    del raw, feature, x_only
    gc.collect()
    return {"contexts": contexts, "feature_cache": cache,
            "market_prefix_replay": {"end": stop, "passed": True}, "cleaning": cleaning}

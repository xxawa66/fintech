"""Frozen S010 test inference and explicitly nonofficial, ex-post price scoring.

No fitting, selection, or parameter search occurs here. Proxy labels are created
only by the separate score phase, after all complete predictions have been saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import clean_history, valid_quote
from src.data.load_data import KEYS, VALUE_COLUMNS, X_COLUMNS
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import daily_metrics, evaluate_frame
from src.evaluation.top_tail_tables import top_matrix
from src.evaluation.turnover import band_scores
from src.evaluation.turnover_controllers import order_fingerprint
from src.features.build_features import build_features
from src.models.lightgbm_model import load_model
from src.models.top_tail_targets import ARMS, causal_candidates, complete_signal, panel_shape
from src.utils.project import ROOT, git_state, load_config, project_path, sha256, timestamp, write_json


def read_json(path):
    return json.loads(project_path(path).read_text(encoding='utf-8'))


def relative(path):
    return str(path.relative_to(ROOT)).replace('\\', '/')


def read_x(path, chunksize=None):
    return pd.read_csv(path, usecols=KEYS + X_COLUMNS, chunksize=chunksize,
        dtype={'ts_code': 'str', 'trade_date': 'int32',
               **{c: 'float64' for c in VALUE_COLUMNS},
               'flag_limit_up': 'int8', 'flag_limit_down': 'int8'})


def canonical(frame):
    return frame.sort_values(['trade_date', 'ts_code'], kind='stable').reset_index(drop=True)


def check_hash(path, expected):
    if sha256(project_path(path)) != expected:
        raise ValueError(f'Artifact changed: {path}')


def predict(cfg, settings, out):
    if (out / 'run.json').exists():
        raise ValueError('Existing inference run must not be overwritten.')
    started = timestamp()
    study = read_json(settings['source_study'])
    if study['status'] != 'complete':
        raise ValueError('S010 is not complete.')
    for path, expected in study['source'].items():
        check_hash(path, expected)
    data_hashes = {}
    for item in read_json('data/manifest.json')['files']:
        check_hash(item['path'], item['sha256'])
        data_hashes[item['path']] = item['sha256']
    for attachment in ['evaluate.py', 'evaluate.R']:
        data_hashes[attachment] = sha256(ROOT / attachment)
    print('Loading test X and historical warm-up X; no labels enter inference.', flush=True)
    test = read_x(project_path(cfg['paths']['test_x']))
    chunks = [chunk.loc[chunk.trade_date >= settings['history_start']]
              for chunk in read_x(project_path(cfg['paths']['train']), chunksize=500_000)]
    history = pd.concat(chunks, ignore_index=True)
    first = int(test.trade_date.min())
    if history.trade_date.max() >= first:
        raise ValueError('History overlaps test dates.')
    windows = [v for key, values in cfg['features'].items()
               if key.endswith('_windows') for v in values]
    if history.groupby('ts_code').size().min() <= max(windows):
        raise ValueError('Insufficient historical rows for trailing feature windows.')
    # The common cleaner requires a label column; a placeholder is discarded
    # immediately. Neither the original training Y nor any proxy Y is loaded.
    joined, cleaning = clean_history(pd.concat([history, test], ignore_index=True).assign(y_ret_1d=np.nan))
    features, names = build_features(joined[KEYS + X_COLUMNS], cfg['features'],
                                     progress=lambda msg: print(msg, flush=True))
    features = canonical(features.loc[features.trade_date >= first])
    known = canonical(test[KEYS + ['flag_limit_up']].assign(quote_valid=valid_quote(test)))
    if not features[KEYS].equals(known[KEYS]):
        raise ValueError('Features and test keys differ.')
    shape = panel_shape(known)
    quote = known.quote_valid.to_numpy()
    source = read_json(study['oof_sources'][str(settings['source_year'])]['source_manifest'])
    first_path = source['model']
    check_hash(first_path, source['artifact_hashes'][first_path])
    first_model = load_model(project_path(first_path))
    if first_model.feature_name() != names or source['split']['last_train_day'] >= first:
        raise ValueError('First-stage feature or training boundary mismatch.')
    known['oof_raw'] = 0.
    known.loc[quote, 'oof_raw'] = first_model.predict(features.loc[quote, names], num_threads=8)
    features['temporal_Y4_percentile'] = known.oof_raw.groupby(known.trade_date).rank(
        method='average', pct=True).to_numpy(dtype='float32')
    model_hashes = {first_path: sha256(project_path(first_path))}
    runs, diagnostics = [], []
    log_path = project_path(cfg['paths']['experiment_log'])
    log_sha = sha256(log_path)
    out.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out / 'test_features.parquet', index=False)
    known.to_parquet(out / 'known_X_predictions.parquet', index=False)
    for arm in ARMS:
        exp_id = f"S010_{arm['id']}_wf{settings['source_year']}_raw"
        fit_path = project_path(cfg['paths']['models']) / exp_id / 'fit.json'
        fit = read_json(fit_path)
        if fit['status'] != 'passed' or fit['max_training_date'] >= first:
            raise ValueError(f'Invalid frozen second stage: {exp_id}')
        model_path = fit['model']
        check_hash(model_path, fit['artifact_hashes'][model_path])
        model_hashes[model_path] = sha256(project_path(model_path))
        model = load_model(project_path(model_path))
        if model.feature_name() != list(features.columns[2:]) or model.feature_name() != fit['spec']['features']:
            raise ValueError('Second-stage feature order mismatch.')
        mask = quote if arm['pool'] == 1 else causal_candidates(known, arm['pool'])
        native = np.zeros(len(known), dtype='float64')
        native[mask] = model.predict(features.loc[mask, fit['spec']['features']], num_threads=8)
        np.save(out / f"{arm['id']}_native.npy", native)
        raw, candidates = complete_signal(known, native, arm)
        np.save(out / f"{arm['id']}_candidates.npy", candidates)
        for layer in ['raw', 'band']:
            if layer == 'raw':
                values = raw
            else:
                values = band_scores(known[KEYS + ['flag_limit_up']].assign(pred=raw),
                                     settings['keep_q']).to_numpy()
            pred = known[KEYS].assign(pred=values)
            check_predictions(pred, test[KEYS])
            path = project_path(cfg['paths']['predictions']) / settings['run_id'] / f"{arm['id']}_{layer}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
            restored = pd.read_csv(path, dtype={'ts_code': 'str', 'trade_date': 'int32'})
            check_predictions(restored, test[KEYS])
            if order_fingerprint(values.reshape(shape)) != order_fingerprint(restored.pred.to_numpy().reshape(shape)):
                raise ValueError('CSV changed the complete ordering or ties.')
            np.save(out / f"{arm['id']}_{layer}.npy", values)
            top = top_matrix(restored.pred.to_numpy().reshape(shape),
                             known.flag_limit_up.eq(0).to_numpy().reshape(shape))
            union = (top[1:] | top[:-1]).sum(axis=1)
            turn = 1 - (top[1:] & top[:-1]).sum(axis=1) / union
            diagnostics.append({'arm': arm['id'], 'layer': layer, 'rows': len(pred),
                                'days': shape[0], 'mean_turnover_unlabeled': float(turn.mean()),
                                'official_final_score': None})
            runs.append({'arm': arm['id'], 'layer': layer, 'prediction': relative(path),
                         'prediction_sha256': sha256(path), 'model': model_path,
                         'fit_manifest': relative(fit_path), 'fit_manifest_sha256': sha256(fit_path),
                         'training_label_end': fit['max_training_date'],
                         'csv_order_and_ties_preserved': True})
        print(f"{arm['id']} frozen inference complete", flush=True)
    if sha256(log_path) != log_sha:
        raise ValueError('The original scored experiment log changed.')
    manifest = {'run_id': settings['run_id'], 'status': 'predicted', 'started_at': started,
        'finished_at': timestamp(), 'git': git_state(), 'settings': settings,
        'inference_source_sha256': sha256(Path(__file__)), 'data_hashes': data_hashes,
        'model_hashes': model_hashes, 'source_study': settings['source_study'],
        'source_study_sha256': sha256(project_path(settings['source_study'])),
        'first_stage_label_end': study['oof_sources'][str(settings['source_year'])]['max_training_label_date'],
        'date_start': first, 'date_end': int(test.trade_date.max()), 'rows': len(test),
        'days': shape[0], 'stocks': shape[1], 'cleaning': cleaning,
        'history_rows': len(history), 'history_start': int(history.trade_date.min()),
        'history_end': int(history.trade_date.max()), 'new_model_fits': 0,
        'official_test_labels_available': False, 'official_final_score': None,
        'controller_state': 'cold start at test first day; continuous across 2025/2026',
        'experiment_log_sha256': log_sha, 'runs': runs, 'diagnostics': diagnostics}
    write_json(out / 'run.json', manifest)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)


def score_proxy(cfg, settings, out):
    run = read_json(out / 'run.json')
    if run['status'] != 'predicted':
        raise ValueError('Complete frozen predictions required; completed scores are not overwritten.')
    if run['inference_source_sha256'] != sha256(Path(__file__)) or run['settings'] != settings:
        raise ValueError('Inference source or settings changed.')
    for path, expected in {**run['data_hashes'], **run['model_hashes']}.items():
        check_hash(path, expected)
    test = canonical(read_x(project_path(cfg['paths']['test_x'])))
    shape = panel_shape(test)
    close = test.close.to_numpy().reshape(shape)
    # Shift along the complete market calendar, never skip suspended/missing rows.
    nxt = np.vstack([close[1:], np.full((1, shape[1]), np.nan)])
    allowed = np.isfinite(close) & np.isfinite(nxt) & (close > 0) & (nxt > 0)
    y = np.full(shape, np.nan)
    y[allowed] = nxt[allowed] / close[allowed] - 1
    truth = test[KEYS + ['flag_limit_up']].assign(y_ret_1d=y.ravel())
    truth.to_parquet(out / 'proxy_labels_NOT_OFFICIAL.parquet', index=False)
    summary, comparisons = [], []
    for item in run['runs']:
        path = project_path(item['prediction'])
        check_hash(path, item['prediction_sha256'])
        pred = pd.read_csv(path, dtype={'ts_code': 'str', 'trade_date': 'int32'})
        check_predictions(pred, truth)
        scored = pred.merge(truth, on=KEYS, validate='one_to_one', sort=False)
        for period in ['2025', '2026', '2025_2026']:
            selected = scored if period == '2025_2026' else scored.loc[scored.trade_date // 10000 == int(period)]
            metrics = evaluate_frame(selected)
            # Formula parity only: the labels here are proxies, not official Y.
            comparison_path = out / 'formula_input.csv'
            selected[KEYS + ['pred']].to_csv(comparison_path, index=False)
            diffs = compare_official(comparison_path, selected, metrics, 1e-10)
            row = {'arm': item['arm'], 'layer': item['layer'], 'period': period,
                   'label_kind': 'EX_POST_CLOSE_PROXY_NOT_OFFICIAL',
                   'date_start': int(selected.trade_date.min()), 'date_end': int(selected.trade_date.max()),
                   'rows': len(selected), 'finite_proxy_labels': int(np.isfinite(selected.y_ret_1d).sum()),
                   **metrics, 'formula_max_difference': max(diffs.values())}
            summary.append(row)
            comparisons.append({'arm': item['arm'], 'layer': item['layer'], 'period': period, 'differences': diffs})
            print(f"NONOFFICIAL {item['arm']} {item['layer']} {period}: {metrics['final_score']:.10f}", flush=True)
        daily_metrics(scored).to_csv(out / f"{item['arm']}_{item['layer']}_proxy_daily.csv", index=False)
    (out / 'formula_input.csv').unlink()
    report_path = project_path(settings['summary'])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(report_path, index=False, encoding='utf-8', lineterminator='\n')
    run.update(status='proxy_scored', proxy_scored_at=timestamp(),
        proxy_label_definition='close(next market trading date)/close(t)-1; both finite and positive',
        proxy_label_warning='Not official hidden Y; unknown official missingness/eligibility. Not for model selection.',
        finite_proxy_labels=int(allowed.sum()), missing_proxy_labels=int((~allowed).sum()),
        proxy_coverage_by_year=truth.assign(year=truth.trade_date // 10000).groupby('year').y_ret_1d.agg(
            ['size', 'count']).reset_index().to_dict('records'),
        proxy_last_day_missing=True, year_boundary_label_uses_next_2026_day=True,
        yearly_turnover_excludes_cross_year_transition=True,
        summary=relative(report_path), summary_sha256=sha256(report_path),
        formula_comparisons=comparisons,
        local_artifact_hashes={relative(p): sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'run.json'})
    if sha256(project_path(cfg['paths']['experiment_log'])) != run['experiment_log_sha256']:
        raise ValueError('Original scored experiment log changed.')
    write_json(out / 'run.json', run)
    write_json(project_path(settings['manifest']), run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['predict', 'score-proxy'], required=True)
    parser.add_argument('--allow-nonofficial-price-labels', action='store_true')
    args = parser.parse_args()
    cfg, _ = load_config()
    settings = cfg['s010_test_replay']
    out = project_path(cfg['paths']['metrics']) / settings['run_id']
    if args.phase == 'predict':
        predict(cfg, settings, out)
    elif args.allow_nonofficial_price_labels:
        score_proxy(cfg, settings, out)
    else:
        raise ValueError('Proxy scoring requires explicit --allow-nonofficial-price-labels.')


if __name__ == '__main__':
    main()

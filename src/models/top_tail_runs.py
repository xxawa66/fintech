"""Audited S010 fits and scores using the project's original official adapter."""
from __future__ import annotations

import gc
import hashlib
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.data.load_data import KEYS
from src.evaluation.baseline_checks import check_predictions, compare_official
from src.evaluation.official_eval import OFFICIAL_METRICS, daily_metrics, evaluate_frame, load_validation_frame
from src.evaluation.research_diagnostics import monthly_metrics
from src.evaluation.turnover import band_scores
from src.evaluation.turnover_controllers import order_fingerprint
from src.evaluation.top_tail_tables import top_matrix
from src.models.alpha_research import relative, save_csv
from src.models.lightgbm_model import load_model, save_model
from src.models.optuna_tuning import read_json, save_json
from src.models.top_tail_targets import complete_signal, panel_shape
from src.utils.experiments import append_record, read_records
from src.utils.project import sha256, timestamp
from src.utils.research_cache import contained_path


def record(cfg, info, manifest):
    row = {'exp_id': info['exp_id'], 'date': info['finished_at'], 'owner': info['owner'],
           'git_commit': info['git']['commit'], 'config_path': 'configs/project.yaml',
           'features': info['features'], 'model': info['family'], 'params': json.dumps(info['spec'], sort_keys=True),
           'train_period': '-'.join(map(str, info['split']['train_window'])),
           'valid_period': '-'.join(map(str, info['split']['valid_window'])),
           **{k: info['metrics'][k] for k in OFFICIAL_METRICS},
           'artifact_path': relative(manifest), 'notes': info['notes']}
    path = contained_path(cfg['paths']['experiment_log'])
    old = next((r for r in read_records(path) if r['exp_id'] == row['exp_id']), None)
    if old is not None:
        if old != {k: str(v) for k,v in row.items()}:
            raise ValueError('Existing shared record differs from passed S010 artifact.')
    else:
        append_record(path, row)


def check_run(info):
    if info['status'] != 'passed':
        raise ValueError('Incomplete run cannot be treated as passed.')
    for path, expected in info['artifact_hashes'].items():
        if sha256(contained_path(path)) != expected:
            raise ValueError(f'Saved run changed: {path}')


def score(cfg, root, meta, exp_id, frame, values, spec, split, extra, log, model_path=None):
    directories = {k: contained_path(cfg['paths'][k]) / exp_id for k in ['models','metrics','predictions']}
    for path in directories.values(): path.mkdir(parents=True, exist_ok=True)
    manifest = directories['models'] / 'run.json'
    if manifest.exists():
        previous = read_json(manifest)
        if previous.get('status') == 'passed':
            if previous['spec'] != spec: raise ValueError('Existing score spec changed.')
            check_run(previous); record(cfg, previous, manifest); return previous
    prediction = directories['predictions'] / f"valid_{int(frame.trade_date.iloc[0])//10000}.csv"
    labels = root / 'labels' / f"{split['fold']}.csv"
    labels.parent.mkdir(parents=True, exist_ok=True)
    truth = frame[KEYS + ['y_ret_1d','flag_limit_up']].copy()
    if labels.exists():
        parsed = pd.read_csv(labels, dtype={'ts_code':str,'trade_date':'int32','flag_limit_up':'int8'})
        if not parsed[KEYS].equals(truth[KEYS]) or not np.allclose(parsed.y_ret_1d,truth.y_ret_1d,atol=1e-12,rtol=0,equal_nan=True):
            raise ValueError('Fold label file differs from the audited history.')
    else: save_csv(labels, truth)
    pred = frame[KEYS].assign(pred=np.asarray(values,dtype='float64'))
    check_predictions(pred, truth); save_csv(prediction, pred)
    parsed = pd.read_csv(prediction, dtype={'ts_code':str,'trade_date':'int32','pred':'float64'})
    shape = panel_shape(frame)
    order_ok = order_fingerprint(values.reshape(shape)) == order_fingerprint(parsed.pred.to_numpy().reshape(shape))
    if not order_ok: raise ValueError('Native / CSV ordering or exact ties changed.')
    scored = load_validation_frame(prediction, labels)
    metrics = evaluate_frame(scored)
    differences = compare_official(prediction, pd.read_csv(labels), metrics, cfg['baseline']['score_tolerance'])
    daily = daily_metrics(scored); month = monthly_metrics(daily)
    save_csv(directories['metrics']/'daily_metrics.csv',daily)
    save_csv(directories['metrics']/'monthly_metrics.csv',month)
    save_json(directories['metrics']/'metrics.json',metrics)
    matrix=scored.pred.to_numpy().reshape(shape)
    flags=scored.flag_limit_up.to_numpy().reshape(shape)==0
    labels_good=np.isfinite(scored.y_ret_1d.to_numpy().reshape(shape))
    np.savez_compressed(directories['metrics']/'actual_top_sets.npz',
        trade_date=frame.trade_date.drop_duplicates().to_numpy(),
        ts_code=frame.ts_code.iloc[:shape[1]].to_numpy(dtype=str),
        turnover=top_matrix(matrix,flags),returns=top_matrix(matrix,flags & labels_good))
    config = directories['models']/'config.yaml'
    config.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    info = {'exp_id':exp_id,'status':'passed','owner':meta['owner'],'finished_at':timestamp(),'git':meta.get('active_git',meta['git']),
            'source_digest':meta['source_digest'],'protocol_digest':meta['protocol_digest'],'data':meta['data'],
            'family': 'LightGBM-S010' if spec['layer']=='raw' else 'S010+qstar-band',
            'features': 'baseline_v1 (40)' if spec.get('warmup') else 'baseline_v1 + temporal_Y4_percentile (41)',
            'spec':spec,'split':split,'metrics':metrics,'official_differences':differences,
            'official_max_difference':max(differences.values()),'prediction':relative(prediction),
            'labels':relative(labels),'model':relative(model_path) if model_path else None,
            'validation_rows':len(frame),'validation_days':frame.trade_date.nunique(),
            'csv_order_and_ties_preserved':order_ok,**extra,
            'notes':f"{meta['study_id']}; {spec.get('arm','Y4-warmup')}; {spec['layer']}; "
                    f"new_model_fits={int(model_path is not None)}; cold_start; fixed rounds; no early stopping; "
                    "strict temporal OOF; historical validation; pending_B_review"}
    paths = [prediction,labels,config,*directories['metrics'].glob('*')]
    if model_path: paths.append(model_path)
    for key in ['native_prediction','candidate_mask','fit_manifest']:
        if extra.get(key): paths.append(contained_path(extra[key]))
    info['artifact_hashes'] = {relative(p):sha256(p) for p in paths if p.is_file()}
    save_json(manifest,info); record(cfg,info,manifest)
    log(f"PASS {exp_id}: score={metrics['final_score']:.10f}; official delta={info['official_max_difference']:.2e}")
    return info


def fit(cfg, root, meta, exp_id, x, y, valid, predict_mask, spec, split, log, group=None):
    directory = contained_path(cfg['paths']['models'])/exp_id
    directory.mkdir(parents=True,exist_ok=True)
    fit_path = directory/'fit.json'; model_path=directory/'model.txt'; native_path=directory/'native.npy'
    old = read_json(fit_path) if fit_path.exists() else None
    if old is not None:
        if old['status']!='passed' or old['spec']!=spec: raise ValueError('An incomplete fit is preserved; automatic retry is refused.')
        for p,h in old['artifact_hashes'].items():
            if sha256(contained_path(p))!=h: raise ValueError('Saved native fit changed.')
        native=np.load(native_path); return native,old,model_path
    if exp_id in {f['exp_id'] for f in meta['fit_attempts']} or len(meta['fit_attempts'])>=18:
        raise ValueError('Fit attempt cannot be retried or the 18-fit budget is exhausted.')
    if len(x)!=len(y) or not np.isfinite(y).all() or list(x.columns)!=spec['features']:
        raise ValueError('Actual fit inputs do not match the fixed spec.')
    if group is not None and (sum(group)!=len(y) or np.any(np.asarray(group)<=0)):
        raise ValueError('Invalid contiguous date query groups.')
    input_identity={'rows':len(y),'features':list(x.columns),'x_dtype':'float32',
        'x_sha256':hashlib.sha256(np.ascontiguousarray(x.to_numpy(),dtype='<f4')).hexdigest(),
        'y_dtype':str(y.dtype),'y_sha256':hashlib.sha256(np.ascontiguousarray(y)).hexdigest(),
        'group_sha256':hashlib.sha256(np.ascontiguousarray(group)).hexdigest() if group is not None else None}
    attempt={'exp_id':exp_id,'started_at':timestamp(),'status':'running','spec':spec,'git':meta.get('active_git',meta['git'])}
    meta['fit_attempts'].append(attempt); save_json(root/'study.json',meta)
    log(f"FIT {len(meta['fit_attempts'])}/18 {exp_id}: rows={len(y):,}; rounds={spec['rounds']}")
    started=time.perf_counter()
    def progress(env):
        done=env.iteration+1
        if done%100==0 or done==spec['rounds']: log(f"{exp_id}: rounds {done}/{spec['rounds']}")
    try:
        dataset=lgb.Dataset(x,label=y,group=group,feature_name=list(x.columns),free_raw_data=True)
        model=lgb.train(spec['model_params'],dataset,num_boost_round=spec['rounds'],callbacks=[progress])
        save_model(model,model_path)
        native=np.zeros(len(valid),dtype='float64')
        native[predict_mask]=model.predict(valid.loc[predict_mask,spec['features']],num_threads=8)
        restored=load_model(model_path)
        again=restored.predict(valid.loc[predict_mask,spec['features']],num_threads=8)
        delta=float(np.max(np.abs(native[predict_mask]-again)))
        if delta>1e-12 or not np.isfinite(native).all(): raise ValueError('Model reload or finite predictions failed.')
        np.save(native_path,native,allow_pickle=False)
        save_csv(directory/'feature_importance.csv',pd.DataFrame({'feature':list(x),
            'gain':model.feature_importance(importance_type='gain'),'split':model.feature_importance(importance_type='split')}))
        info={'exp_id':exp_id,'status':'passed','spec':spec,'rows':len(y),'groups':len(group) if group is not None else None,
              'max_training_date':int(split['supervised_last_day']),'split':split,
              'reload_max_difference':delta,'actual_iterations':model.current_iteration(),
              'native_prediction':relative(native_path),'model':relative(model_path),
              'input_identity':input_identity,'seconds':time.perf_counter()-started,'finished_at':timestamp(),
              'artifact_hashes':{relative(p):sha256(p) for p in [model_path,native_path,directory/'feature_importance.csv']}}
        save_json(fit_path,info); attempt.update(status='passed',finished_at=info['finished_at'],fit_manifest=relative(fit_path))
        save_json(root/'study.json',meta)
        del model,restored,dataset;gc.collect()
        return native,info,model_path
    except Exception as error:
        attempt.update(status='failed',error=repr(error));save_json(root/'study.json',meta)
        raise


def paired_layers(cfg,root,meta,arm,fold,valid,native,fit_info,model_path,spec,split,log):
    known=valid[KEYS+['quote_valid','flag_limit_up','oof_raw']]
    values,candidate=complete_signal(known,native,arm)
    prefix=valid.trade_date<=valid.trade_date.drop_duplicates().iloc[len(valid.trade_date.unique())//2-1]
    replay,_=complete_signal(known.loc[prefix].reset_index(drop=True),native[prefix.to_numpy()],arm)
    if not np.array_equal(replay,values[prefix.to_numpy()]): raise ValueError('Complete native signal prefix differs.')
    native_path=contained_path(fit_info['native_prediction'])
    candidate_path=model_path.parent/'candidate_mask.npy';np.save(candidate_path,candidate,allow_pickle=False)
    extra={'new_model_fits':1,'reload_max_difference':fit_info['reload_max_difference'],
           'native_prediction':relative(native_path),'candidate_mask':relative(candidate_path),
           'fit_manifest':relative(model_path.parent/'fit.json'),'signal_prefix_passed':True,
           'native_encoding':'twice_average_rank_zero_anchored', 'candidate_rows':int(candidate.sum())}
    raw_id=f"{meta['study_id']}_{arm['id']}_{fold}_raw"
    raw=score(cfg,root,meta,raw_id,valid,values,spec,split,extra,log,model_path)
    # The official parsed raw CSV is the one fixed controller input.
    parsed=pd.read_csv(contained_path(raw['prediction']),dtype={'ts_code':str,'trade_date':'int32'})
    known_band=parsed.assign(flag_limit_up=valid.flag_limit_up.to_numpy())
    band=band_scores(known_band,cfg['top_tail_learning']['keep_q']).to_numpy()
    again=band_scores(known_band.loc[prefix].reset_index(drop=True),cfg['top_tail_learning']['keep_q']).to_numpy()
    if not np.array_equal(again,band[prefix.to_numpy()]): raise ValueError('Continuous q* prefix differs.')
    band_spec={**spec,'layer':'band','raw_source':raw_id,'keep_q':cfg['top_tail_learning']['keep_q']}
    transformed=score(cfg,root,meta,f"{meta['study_id']}_{arm['id']}_{fold}_band",valid,band,band_spec,split,
        {'new_model_fits':0,'raw_source':raw_id,'band_prefix_passed':True,'single_band_application':True},log)
    return raw,transformed

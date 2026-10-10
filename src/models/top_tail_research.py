"""S010 finite temporal-OOF head learning, CV selection and one locked confirmation."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import re
import subprocess
import traceback

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, log_loss

from src.data.load_data import KEYS
from src.evaluation.top_tail_learning_handoff import publish
from src.models.alpha_research import relative, save_csv
from src.models.baseline import RunLog
from src.models.optuna_tuning import read_json, save_json, study_lock
from src.models.temporal_oof import TemporalOOF
from src.models.top_tail_runs import fit, paired_layers, check_run
from src.models.top_tail_targets import ARMS, training_targets, causal_candidates, panel_shape
from src.utils.experiments import read_records
from src.utils.project import ROOT, git_state, load_config, sha256, timestamp
from src.utils.research_cache import code_hashes, contained_path, digest

PACKAGES=['numpy','pandas','scipy','lightgbm','pyarrow','PyYAML','matplotlib','scikit-learn']


def protocol(cfg):
    s=cfg['top_tail_learning']
    if (s['phase']!='S010' or s['arms']!=ARMS or s['model_fit_budget']!=18 or s['first_stage_rounds']!=800
            or s['second_stage_rounds']!=400 or s['keep_q']!=.0022778298112255263
            or s['folds']!=cfg['optuna']['folds'] or s['confirm']!=cfg['optuna']['confirm']
            or s['warmup_years']!=[2019,2020] or s['second_stage_start']!=20190101
            or s['first_stage_transform']!={'transform':'daily_winsor_zscore','quantiles':[.01,.99],
                'interpolation':'linear','ddof':0,'zero_std':0.}
            or s['row_order']!=['trade_date','ts_code'] or s['initial_state']!='cold_start'
            or s['features']!=41 or s['query_truncation']!=466 or s['label_gain']!=[0,1,2,3,4]
            or s['bootstrap']!={'block_length':20,'repetitions':1000,'seed':42}
            or s['gates']!={'mean_score_delta':.003,'mean_excess_delta':.01,'max_excess_loss':.02,
                'max_worst_score_loss':.005,'ic_retention':.95,'max_turnover_increase':.01}
            or cfg['features']['expected_count']!=40):
        raise ValueError('S010 config differs from the accepted finite plan.')
    return {'schema':1,'learning':s,'features':cfg['features'],
        'paths':{k:cfg['paths'][k] for k in ['train','test_x','models','metrics','predictions','figures',
            'experiment_log','research_studies','optuna_cache']},'score_tolerance':cfg['baseline']['score_tolerance'],
        'coding':'twice average native tuple rank; zero quote-fallback group anchored to zero; exact ties retained',
        'selection':'all complete-qstar gates; highest mean, 1e-6 tie, worst year then ID'}


def identity(cfg,meta):
    if (code_hashes()!=meta['source'] or digest(protocol(cfg))!=meta['protocol_digest']
            or {p:importlib.metadata.version(p) for p in PACKAGES}!=meta['packages']):
        raise ValueError('Source/protocol/packages changed; preserve S010 under its existing identity.')


def target_map(cfg,root,meta,train,split,year,log):
    mapped,daily,summary=training_targets(train,year*10000+101)
    allowed=mapped.allowed.to_numpy()
    if mapped.loc[allowed,'trade_date'].max()>=split['last_train_day']:
        raise ValueError('A cross-boundary label survived the outer purge.')
    stop=int(train.trade_date.drop_duplicates().iloc[train.trade_date.nunique()//2-1])
    prefix=train.trade_date<=stop
    replay,_,_=training_targets(train.loc[prefix].reset_index(drop=True),year*10000+101)
    if not replay.equals(mapped.loc[prefix].reset_index(drop=True)):
        raise ValueError('Target/candidate map failed the real history prefix.')
    directory=root/'targets'/split['fold'];directory.mkdir(parents=True,exist_ok=True)
    path=directory/'mapping.parquet';mapped.to_parquet(path,index=False)
    restored=pd.read_parquet(path)
    if not mapped.equals(restored):raise ValueError('Saved actual-key target mapping is not lossless.')
    info={'status':'passed','split':split,'summary':summary,'mapping':relative(path),'mapping_sha256':sha256(path),
          'prefix_passed':True,'prefix_end':stop,'oof_sources':{str(y):meta['oof_sources'][str(y)] for y in split['oof_years']}}
    save_json(directory/'target.json',info)
    meta.setdefault('target_artifacts',{}).update({relative(p):sha256(p) for p in directory.iterdir() if p.is_file()})
    save_json(root/'study.json',meta)
    daily['year']=year;save_csv(root/'tables/targets'/f'{year}.csv',daily)
    log(f"targets {year}: {summary['supervised_rows']:,} eligible labels; genuine prefix and key mapping passed")
    return mapped,info


def diagnostic(cfg,root,arm,year,valid,native,runs,train_stats):
    known=valid[KEYS+['quote_valid','flag_limit_up','oof_raw']]
    candidate=causal_candidates(known,arm['pool']) if arm['pool']<1 else valid.quote_valid.to_numpy()
    allowed=valid.quote_valid & valid.flag_limit_up.eq(0) & np.isfinite(valid.y_ret_1d)
    ranks=pd.Series(np.nan,index=valid.index)
    ranks.loc[allowed]=valid.loc[allowed,'y_ret_1d'].groupby(valid.loc[allowed,'trade_date']).rank(method='average',pct=True)
    mask=allowed.to_numpy() & candidate
    target=(ranks.to_numpy()>(arm.get('event',.90))).astype('int8')
    auc=float(roc_auc_score(target[mask],native[mask])) if arm['objective']=='binary' and np.unique(target[mask]).size==2 else None
    loss=float(log_loss(target[mask],native[mask],labels=[0,1])) if arm['objective']=='binary' else None
    record={'model':arm['id'],'year':year,'validation_supervised_rows':int(mask.sum()),
        'validation_positive_fraction':float(target[mask].mean()),'conditional_auc':auc,'conditional_logloss':loss,
        'train_supervised_rows':train_stats['train_rows'],'train_positive_fraction':train_stats['positive_fraction'],
        'train_single_class_days':train_stats['single_class_days'],'train_groups':train_stats['groups']}
    save_csv(root/'tables/auxiliary'/f"{arm['id']}_{year}.csv",pd.DataFrame([record]))
    full_candidate=causal_candidates(known,.20);wide_candidate=causal_candidates(known,.30)
    candidate_frame=valid[KEYS].copy();candidate_frame['model']=arm['id'];candidate_frame['year']=year
    candidate_frame['candidate20']=full_candidate;candidate_frame['candidate30']=wide_candidate
    candidate_frame['true_top10']=(ranks>.90).fillna(False).to_numpy()
    rows=[]
    for date,group in candidate_frame.groupby('trade_date'):
        total=int(group.true_top10.sum())
        rows.append({'model':arm['id'],'year':year,'trade_date':date,'true_top10':total,
            'candidate20':int(group.candidate20.sum()),'candidate30':int(group.candidate30.sum()),
            'recall20':float((group.true_top10&group.candidate20).sum()/total) if total else 0.,
            'recall30':float((group.true_top10&group.candidate30).sum()/total) if total else 0.})
    save_csv(root/'tables/candidates'/f"{arm['id']}_{year}.csv",pd.DataFrame(rows))
    # Output event precision/recall is descriptive and never used to select gates.
    shape=panel_shape(valid);labels=ranks.to_numpy().reshape(shape);good=allowed.to_numpy().reshape(shape)
    for info in runs:
        values=pd.read_csv(contained_path(info['prediction']),usecols=['pred']).pred.to_numpy().reshape(shape)
        day_rows=[]
        for t,date in enumerate(valid.trade_date.drop_duplicates()):
            order=np.flatnonzero(valid.quote_valid.to_numpy().reshape(shape)[t] & valid.flag_limit_up.to_numpy().reshape(shape)[t].__eq__(0))
            order=order[np.argsort(-values[t,order],kind='stable')];top=order[:len(order)//10]
            labelled=top[good[t,top]];true=labels[t]>.90
            day_rows.append({'trade_date':date,'model':arm['id'],'year':year,'layer':info['spec']['layer'],
                'precision10':float(true[labelled].mean()) if len(labelled) else 0.,
                'recall10':float(true[labelled].sum()/true.sum()) if true.sum() else 0.,
                'labelled_top_n':len(labelled)})
        save_csv(root/'tables/events'/f"{arm['id']}_{year}_{info['spec']['layer']}.csv",pd.DataFrame(day_rows))


def run_fold(cfg,root,meta,source,year,arms,log):
    train,valid,split=source.outer_training(year)
    mapped,target_info=target_map(cfg,root,meta,train,split,year,log)
    all_runs=[]
    columns=source.columns+['temporal_Y4_percentile']
    for arm in arms:
        pool=(mapped.candidate20.to_numpy() if arm['pool']==.20 else mapped.candidate30.to_numpy() if arm['pool']==.30 else np.ones(len(mapped),dtype=bool))
        mask=mapped.allowed.to_numpy() & pool
        label='L30' if arm['objective']=='lambdarank' else 'B20' if arm['event']==.80 else 'B10'
        selected=train.loc[mask].reset_index(drop=True);y=mapped.loc[mask,label].to_numpy()
        dates=selected.trade_date.to_numpy();group=selected.groupby('trade_date',sort=False).size().to_numpy(dtype='int32')
        if not np.all(np.diff(dates)>=0) or sum(group)!=len(y):raise ValueError('Query group continuity failed.')
        params={**source.base_spec['model_params'],'objective':arm['objective'],
                'metric':'ndcg' if arm['objective']=='lambdarank' else 'binary_logloss'}
        if arm['objective']=='lambdarank':params.update(label_gain=[0,1,2,3,4],lambdarank_truncation_level=466,eval_at=[466])
        spec={'arm':arm['id'],'arm_spec':arm,'layer':'raw','model_params':params,'rounds':400,'features':columns,
            'target_mapping':target_info['mapping'],'target_mapping_sha256':target_info['mapping_sha256'],
            'oof_years':split['oof_years'],'training_start':20190101,'pool_selection':'known X + past native Y4; stable code ties',
            'signal_encoding':'twice average tuple rank, zero anchored; exact ties retained'}
        actual_split={**split,'supervised_last_day':int(selected.trade_date.max()),'used_training_rows':len(y)}
        known=valid[KEYS+['quote_valid','flag_limit_up','oof_raw']]
        predict_mask=causal_candidates(known,arm['pool']) if arm['pool']<1 else valid.quote_valid.to_numpy()
        exp_id=f"{meta['study_id']}_{arm['id']}_{split['fold']}_raw"
        native,fit_info,model_path=fit(cfg,root,meta,exp_id,selected[columns],y,valid,predict_mask,
            spec,actual_split,log,group if arm['objective']=='lambdarank' else None)
        runs=paired_layers(cfg,root,meta,arm,split['fold'],valid,native,fit_info,model_path,spec,actual_split,log)
        stats={'train_rows':len(y),'positive_fraction':float((y>0).mean()),'groups':len(group),
               'single_class_days':int(pd.Series(y).groupby(dates).nunique().eq(1).sum())}
        diagnostic(cfg,root,arm,year,valid,native,runs,stats)
        importance=pd.read_csv(model_path.parent/'feature_importance.csv').assign(model=arm['id'],year=year)
        save_csv(root/'tables/importance'/f"{arm['id']}_{year}.csv",importance)
        all_runs.extend((arm['id'],year,info['spec']['layer'],info) for info in runs)
        meta.setdefault('completed_runs',{})[exp_id]=relative(model_path.parent/'run.json')
        save_json(root/'study.json',meta)
        del selected,y,native;gc.collect()
    del train,valid,mapped;gc.collect()
    return all_runs


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--study-id',default='S010');parser.add_argument('--owner',default='A')
    parser.add_argument('--phase',choices=['screen','confirm'],default='screen');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'S010(?:_[A-Za-z0-9_-]+)?',args.study_id):raise ValueError('Use S010 or a distinct S010_repeat identity.')
    cfg,_=load_config();spec=protocol(cfg);root=contained_path(cfg['paths']['research_studies'])/args.study_id
    state=git_state();remote=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    if state['branch']!='main' or state['commit']!=remote or (state['dirty'] and not args.resume):
        raise ValueError('First implementation/selection must be pushed to clean main before numerical execution.')
    if root.exists():
        meta=read_json(root/'study.json');identity(cfg,meta)
        if args.phase=='screen' and meta['status'] in ['screen_complete','complete']:raise ValueError('Completed S010 screening refuses overwrite.')
        if args.phase=='screen' and not args.resume:raise ValueError('Existing incomplete study needs explicit resume.')
    else:
        if args.phase!='screen':raise ValueError('CV screening is required before confirmation.')
        root.mkdir(parents=True)
        hashes=code_hashes()
        meta={'study_id':args.study_id,'owner':args.owner,'status':'running','started_at':timestamp(),'git':state,
            'source':hashes,'source_digest':digest(hashes),'protocol':spec,'protocol_digest':digest(spec),
            'packages':{p:importlib.metadata.version(p) for p in PACKAGES},
            'initial_records':read_records(contained_path(cfg['paths']['experiment_log'])),'fit_attempts':[],
            'completed_runs':{},'context_artifacts':{}}
        save_json(root/'study.json',meta)
        (root/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    meta['active_git']=state;save_json(root/'study.json',meta)
    with study_lock(root):
        log=RunLog(root/'run.log')
        try:
            source=TemporalOOF(cfg,root,meta,log);source.feature_prefix()
            for year in [2019,2020]:source.warmup(year)
            oof_rows=[{'year':y,'max_training_label_date':s['max_training_label_date'],
                'native_prediction':s['native_prediction'],'source_manifest':s['source_manifest'],
                'new_model_fits':s['new_model_fits']} for y,s in sorted(source.sources.items())]
            save_csv(root/'tables/oof/sources.csv',pd.DataFrame(oof_rows))
            if args.phase=='screen':
                runs=[]
                for year in [2021,2022,2023]:runs.extend(run_fold(cfg,root,meta,source,year,ARMS,log))
                refs=[('T030',year,layer,ref['run']) for (year,layer),ref in source.refs.items() if year<=2023]
                selection,_=publish(cfg,root,meta,runs,refs,'screen')
                meta['status']='screen_complete' if selection['winner'] else 'complete'
                meta['selection_digest']=selection['selection_digest'];meta['screen_finished_at']=timestamp()
                if not selection['winner']:meta['completed_at']=timestamp();meta['recommendation']='retain_S003'
                save_json(root/'study.json',meta)
                log(f"SCREEN COMPLETE: locked={selection['winner']}; fits={len(meta['fit_attempts'])}/18")
            else:
                if meta['status']!='screen_complete':raise ValueError('Confirmation is only permitted for one CV-qualified arm.')
                selection=read_json(root/'selection.json')
                public=ROOT/'docs'/f'top_tail_{args.study_id}_selection.json'
                if read_json(public)!=selection or digest({k:v for k,v in selection.items() if k!='selection_digest'})!=selection['selection_digest']:
                    raise ValueError('Frozen selection changed.')
                pushed=json.loads(subprocess.check_output(['git','show',f'origin/main:{relative(public)}'],cwd=ROOT,text=True,encoding='utf-8'))
                if pushed!=selection:raise ValueError('Selection must be pushed before confirmation.')
                arm=next(a for a in ARMS if a['id']==selection['winner'])
                runs=[]
                for exp_id,path in meta['completed_runs'].items():
                    info=read_json(contained_path(path));check_run(info)
                    year=info['split']['valid_window'][0]//10000
                    runs.append((info['spec']['arm'],year,'raw',info))
                    band=read_json(contained_path(cfg['paths']['models'])/exp_id.replace('_raw','_band')/'run.json')
                    check_run(band);runs.append((info['spec']['arm'],year,'band',band))
                runs.extend(run_fold(cfg,root,meta,source,2024,[arm],log))
                refs=[('T030',year,layer,ref['run']) for (year,layer),ref in source.refs.items()]
                _,confirmation=publish(cfg,root,meta,runs,refs,'confirm',selection)
                meta.update(status='complete',completed_at=timestamp(),recommendation=confirmation['recommendation'])
                save_json(root/'study.json',meta);log('CONFIRM COMPLETE: '+confirmation['recommendation'])
        except Exception as error:
            meta.update(status='failed',failure_time=timestamp(),error=repr(error),traceback=traceback.format_exc())
            save_json(root/'study.json',meta);log(meta['traceback']);raise


if __name__=='__main__':
    main()

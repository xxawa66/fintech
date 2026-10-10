"""Frozen V1 features, verified Y4 forecasts and two strictly past warmup fits."""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from src.data.clean_data import clean_history, valid_quote
from src.data.load_data import KEYS, X_COLUMNS, load_training_data
from src.evaluation.top_tail_sources import FrozenInputs
from src.evaluation.validation import split_train_valid
from src.features.build_features import build_features, feature_names
from src.models.alpha_research import relative, save_csv
from src.models.optuna_tuning import provenance, read_json, save_json
from src.models.target_transforms import array_digest, daily_target
from src.models.top_tail_runs import fit, score
from src.models.top_tail_targets import panel_shape
from src.utils.project import ROOT, sha256
from src.utils.research_cache import contained_path, digest
from src.utils.tuning_cache import canonical, make_fold


class TemporalOOF:
    def __init__(self,cfg,root,meta,log):
        self.cfg,self.root,self.meta,self.log=cfg,root,meta,log
        self.frozen=FrozenInputs(meta.get('frozen_inputs'))
        settings=cfg['top_tail_learning']
        self.frozen.check(settings['diagnostic_artifacts'])
        audit=read_json(contained_path(settings['diagnostic_artifacts']))
        decision=read_json(contained_path(settings['diagnostic_decision']))
        self.frozen.check(settings['diagnostic_decision'])
        if audit['status']!='passed' or not decision['advance_S010']:
            raise ValueError('S009 did not authorize the accepted S010 research route.')
        if provenance(cfg)!=audit['data_provenance']:
            raise ValueError('Original input/official attachment identity differs from S009.')
        meta['data']=audit['data_provenance']
        s3_path=contained_path(settings['source_selection'])
        self.frozen.check(s3_path,audit['frozen_inputs'][relative(s3_path)])
        self.base_spec=read_json(s3_path)['winner']['spec']
        if self.base_spec['rounds']!=800 or self.base_spec['keep_q']!=settings['keep_q']:
            raise ValueError('Fixed T030 source differs.')
        self.columns=feature_names(cfg['features']); self.sources={};self.refs={}
        for year,fold in [(2021,'wf2021'),(2022,'wf2022'),(2023,'wf2023'),(2024,'confirm2024')]:
            for model,layer in [('Y4','raw'),('T030','raw'),('T030','band')]:
                entry=next(r for r in audit['runs'] if r['model']==model and r['fold']==fold and r['layer']==layer)
                self.frozen.check(entry['manifest'],entry['sha256'])
                cell=read_json(contained_path(entry['manifest']))
                self.frozen.check(cell['source_manifest'],audit['frozen_inputs'][cell['source_manifest']])
                source=read_json(contained_path(cell['source_manifest']))
                for path,expected in source['artifact_hashes'].items():self.frozen.check(path,expected)
                if source['status']!='passed' or source['split']['train_window'][1]//10000!=year-1:
                    raise ValueError('Legacy forecast is not from the declared past training window.')
                if model=='Y4':
                    mapping=source['target_mapping']
                    if (source['spec']['model_params']['objective']!='huber' or source['spec']['rounds']!=800
                            or mapping['summary']['end']>=source['split']['last_train_day']):
                        raise ValueError('Legacy Y4 transform or purged label boundary changed.')
                    native=cell['native'];self.frozen.check(native['prediction'],native['sha256'])
                    # All diagnostic context paths come from the verified S009 audit.
                    prep_path=contained_path(settings['diagnostic_context'])
                    self.frozen.check(prep_path,audit['context_artifacts'][relative(prep_path)])
                    context=read_json(prep_path)['contexts'][fold]
                    self.frozen.check(context['path'],context['sha256'])
                    self.sources[year]={'native_prediction':native['prediction'],'source_manifest':cell['source_manifest'],
                        'context':context['path'],'split':source['split'],'new_model_fits':0,
                        'max_training_label_date':mapping['summary']['end']}
                else:self.refs[(year,layer)]={'run':source,'manifest':cell['source_manifest']}
        prep=read_json(contained_path(settings['diagnostic_context']))
        self.cache=prep['feature_cache']
        self.cache_path=contained_path(self.cache['path'])/'features.parquet'
        self.frozen.check(self.cache_path,audit['frozen_inputs'][relative(self.cache_path)])
        cache_meta=self.cache_path.with_name('cache.json')
        self.frozen.check(cache_meta,audit['frozen_inputs'][relative(cache_meta)])
        identity=read_json(cache_meta)['identity']
        if identity['features']!=cfg['features'] or identity['history']!=[20180102,20241231]:
            raise ValueError('Frozen complete 40-feature history changed.')
        for path,expected in identity['code'].items():self.frozen.check(path,expected)
        self.raw,cleaning=clean_history(load_training_data(contained_path(cfg['paths']['train'])))
        self.raw['quote_valid']=valid_quote(self.raw)
        if len(self.raw)!=7900350:raise ValueError('The complete original training panel is required.')
        meta['cleaning']=cleaning;meta['frozen_inputs']=self.frozen.hashes
        save_json(root/'study.json',meta)

    def frame(self,year,with_oof=False):
        path=self.root/'context'/f'{year}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            expected=self.meta.get('context_artifacts',{}).get(relative(path))
            if expected is None or sha256(path)!=expected:raise ValueError('Incomplete or changed year context.')
            frame=pd.read_parquet(path)
        else:
            raw=canonical(self.raw.loc[self.raw.trade_date.between(year*10000+101,year*10000+1231)])
            features=pd.read_parquet(self.cache_path,filters=[('trade_date','>=',year*10000+101),('trade_date','<=',year*10000+1231)])
            features=canonical(features)
            if not features[KEYS].equals(raw[KEYS]) or list(features)!=KEYS+self.columns:
                raise ValueError('Year feature keys or frozen column order changed.')
            if not np.array_equal(features.flag_limit_up.to_numpy(),raw.flag_limit_up.to_numpy()):
                raise ValueError('Cached known eligibility flag differs from raw X.')
            frame=pd.concat([raw[KEYS+['y_ret_1d','quote_valid']],features[self.columns]],axis=1)
            if any(frame[c].dtype!=np.float32 for c in self.columns):raise ValueError('V1 feature dtypes changed.')
            panel_shape(frame);frame.to_parquet(path,index=False)
            self.meta.setdefault('context_artifacts',{})[relative(path)]=sha256(path)
            save_json(self.root/'study.json',self.meta)
            self.log(f'context {year}: {len(frame):,} complete keys; X floats32; original y attached after X')
            del raw,features;gc.collect()
        if with_oof:
            source=self.sources[year]
            native_path=contained_path(source['native_prediction'])
            if source.get('new_model_fits'):
                run=read_json(contained_path(source['source_manifest']))
                if sha256(native_path)!=run['artifact_hashes'][relative(native_path)]:raise ValueError('Warmup native changed.')
            native=np.load(native_path,allow_pickle=False)
            if len(native)!=len(frame) or not np.isfinite(native).all():raise ValueError('OOF panel length/values differ.')
            if year>=2021:
                original=pd.read_parquet(contained_path(source['context']))
                if (not frame[KEYS].equals(original[KEYS]) or not np.array_equal(frame.quote_valid,original.quote_valid)
                        or not np.allclose(frame.y_ret_1d,original.y_ret_1d,atol=1e-12,rtol=0,equal_nan=True)):
                    raise ValueError('Reused native forecast and original X/y keys do not align.')
            frame['oof_raw']=native
            frame['temporal_Y4_percentile']=pd.Series(native).groupby(frame.trade_date).rank(method='average',pct=True).astype('float32')
            if source['max_training_label_date']>=int(frame.trade_date.min()):raise ValueError('OOF source used future labels.')
        return frame

    def feature_prefix(self):
        if self.meta.get('feature_prefix',{}).get('passed'):return
        stop=20181231
        prefix,_=build_features(self.raw.loc[self.raw.trade_date<=stop,KEYS+X_COLUMNS],self.cfg['features'],self.log)
        prefix=canonical(prefix);expected=self.frame(2018)
        if not prefix[KEYS].equals(expected[KEYS]) or not np.array_equal(prefix[self.columns].to_numpy(),expected[self.columns].to_numpy(),equal_nan=True):
            raise ValueError('Frozen features differ from real past-only full-market recomputation.')
        self.meta['feature_prefix']={'end':stop,'rows':len(prefix),'stocks':4650,'passed':True,'max_difference':0}
        save_json(self.root/'study.json',self.meta);del prefix,expected;gc.collect()
        self.log('V1 actual full-market 2018 prefix recomputation passed')

    def warmup(self,year):
        exp_id=f"{self.meta['study_id']}_Y4_warm{year}_raw"
        manifest=contained_path(self.cfg['paths']['models'])/exp_id/'run.json'
        valid=self.frame(year)
        histories=[self.frame(y) for y in range(2018,year)]
        history=pd.concat(histories+[valid],ignore_index=True);del histories
        fold=make_fold({'name':f'warm{year}','train':[20180102,(year-1)*10000+1231],'valid':[year*10000+101,year*10000+1231]})
        train,_,split=split_train_valid(history,fold);del history
        allowed=train.quote_valid & np.isfinite(train.y_ret_1d)
        train=train.loc[allowed].sort_values(KEYS,kind='stable').reset_index(drop=True)
        target_spec=self.cfg['top_tail_learning']['first_stage_transform']
        target,daily,summary=daily_target(train[KEYS+['y_ret_1d']],target_spec,fold.valid_start)
        stop=train.trade_date.drop_duplicates().sort_values().iloc[train.trade_date.nunique()//2-1]
        prefix=train.trade_date<=stop
        replay,_,_=daily_target(train.loc[prefix,KEYS+['y_ret_1d']].reset_index(drop=True),target_spec,fold.valid_start)
        if not np.array_equal(replay,target[prefix.to_numpy()]):raise ValueError('Warmup daily target prefix differs.')
        directory=self.root/'targets'/f'warm{year}';directory.mkdir(parents=True,exist_ok=True)
        mapping=train[KEYS+['y_ret_1d']].assign(transformed_y=target)
        mapping.to_parquet(directory/'mapping.parquet',index=False)
        save_csv(directory/'daily_statistics.csv',daily)
        save_json(directory/'target.json',{'summary':summary,'prefix_passed':True,'prefix_end':int(stop),
            'mapping_sha256':sha256(directory/'mapping.parquet')})
        self.meta.setdefault('target_artifacts',{}).update({relative(p):sha256(p) for p in directory.iterdir() if p.is_file()})
        params={**self.base_spec['model_params'],'objective':'huber','metric':'huber','alpha':.9}
        spec={'warmup':True,'arm':'Y4','layer':'raw','features':self.columns,'model_params':params,'rounds':800,
              'target_transform':target_spec,'training_order':KEYS,'mapping':relative(directory/'mapping.parquet'),
              'mapping_sha256':sha256(directory/'mapping.parquet')}
        split['supervised_last_day']=int(train.trade_date.max())
        native,fit_info,model_path=fit(self.cfg,self.root,self.meta,exp_id,train[self.columns],target,valid,
                                      valid.quote_valid.to_numpy(),spec,split,self.log)
        info=score(self.cfg,self.root,self.meta,exp_id,valid,native,spec,split,
            {'new_model_fits':1,'reload_max_difference':fit_info['reload_max_difference'],
             'native_prediction':fit_info['native_prediction'],'fit_manifest':relative(model_path.parent/'fit.json'),
             'warmup_target_mapping':relative(directory/'target.json'),'training_target_prefix_passed':True},self.log,model_path)
        self.sources[year]={'native_prediction':fit_info['native_prediction'],'source_manifest':relative(manifest),
            'split':split,'max_training_label_date':int(train.trade_date.max()),'new_model_fits':1}
        self.meta.setdefault('warmups',{})[str(year)]=relative(manifest)
        self.meta['oof_sources']={str(y):s for y,s in self.sources.items()}
        save_json(self.root/'study.json',self.meta)
        del train,valid,target,mapping,native;gc.collect()
        return info

    def outer_training(self,year):
        past=[self.frame(y,True) for y in range(2019,year)]
        valid=self.frame(year,True)
        history=pd.concat(past+[valid],ignore_index=True);del past
        definition={'name':f'wf{year}' if year!=2024 else 'confirm2024',
                    'train':[20190101,(year-1)*10000+1231],'valid':[year*10000+101,year*10000+1231]}
        train,_,split=split_train_valid(history,make_fold(definition));del history
        if split['n_boundary_rows']!=4650 or not train.trade_date.max()<valid.trade_date.min():
            raise ValueError('Outer training/purge dates differ.')
        used=sorted(y for y in self.sources if 2019<=y<year)
        if used!=list(range(2019,year)):raise ValueError('OOF history is incomplete or future OOF entered an outer fold.')
        split['oof_years']=used
        return train.reset_index(drop=True),valid.reset_index(drop=True),split

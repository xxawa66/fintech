"""Finite S010 gates, real date-block intervals, reports and source handoff."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evaluation.official_eval import OFFICIAL_METRICS
from src.evaluation.paired_block_bootstrap import comparisons
from src.models.alpha_research import relative, save_csv, write_text
from src.models.optuna_tuning import read_json, save_json
from src.models.target_research import table_markdown
from src.models.top_tail_runs import check_run
from src.utils.experiments import read_records
from src.utils.project import ROOT, sha256, timestamp
from src.utils.research_cache import contained_path, digest


def collect(cfg,runs,refs):
    official,daily,monthly=[],[],[]
    for model,year,layer,info in runs+refs:
        metrics_dir=contained_path(cfg['paths']['metrics'])/info['exp_id']
        official.append({'model':model,'year':year,'layer':layer,'exp_id':info['exp_id'],**info['metrics']})
        for target,name in [(daily,'daily_metrics.csv'),(monthly,'monthly_metrics.csv')]:
            frame=pd.read_csv(metrics_dir/name);frame['model']=model;frame['year']=year;frame['layer']=layer;target.append(frame)
    return pd.DataFrame(official),pd.concat(daily,ignore_index=True),pd.concat(monthly,ignore_index=True)


def gates(official,months,bootstrap,settings,years):
    rows=[];base=official[(official.model=='T030')&(official.layer=='band')].set_index('year').loc[years]
    bm=months[(months.model=='T030')&(months.layer=='band')&(months.year.isin(years))]
    worst_month_count=int(math.ceil(len(bm)*.10)); period='CV2021_2023' if len(years)==3 else 'Historical2021_2024'
    for arm in settings['arms']:
        model=arm['id'];f=official[(official.model==model)&(official.layer=='band')].set_index('year')
        if not all(y in f.index for y in years):continue
        f=f.loc[years];delta=f[['ic_mean','annual_excess','mean_turnover','final_score']]-base[['ic_mean','annual_excess','mean_turnover','final_score']]
        m=months[(months.model==model)&(months.layer=='band')&(months.year.isin(years))]
        if set(m.month)!=set(bm.month) or len(m)!=len(bm):raise ValueError('Monthly gate dates differ.')
        intervals=bootstrap[(bootstrap.model==model)&(bootstrap.layer=='band')&(bootstrap.period==period)].set_index('metric')
        flags={'mean_score':delta.final_score.mean()>=settings['gates']['mean_score_delta'],
            'mean_excess':delta.annual_excess.mean()>=settings['gates']['mean_excess_delta'],
            'positive_excess_years':int((delta.annual_excess>0).sum())>=(2 if len(years)==3 else 3),
            'each_excess':bool((delta.annual_excess>=-settings['gates']['max_excess_loss']).all()),
            'year2023_score':float(delta.loc[2023,'final_score'])>=-settings['gates']['max_worst_score_loss'],
            'worst_year_score':float(f.final_score.min()-base.final_score.min())>=-settings['gates']['max_worst_score_loss'],
            'each_ic':bool((f.ic_mean>=base.ic_mean*settings['gates']['ic_retention']).all()),
            'mean_turnover':float(delta.mean_turnover.mean())<=settings['gates']['max_turnover_increase'],
            'positive_months':int((m.annual_excess>0).sum())>=int((bm.annual_excess>0).sum()),
            'worst_months':float(m.nsmallest(worst_month_count,'final_score').final_score.mean()-
                                bm.nsmallest(worst_month_count,'final_score').final_score.mean())>=-settings['gates']['max_worst_score_loss'],
            'score_ci':float(intervals.loc['final_score','lower'])>0,
            'excess_ci':float(intervals.loc['annual_excess','lower'])>0}
        if len(years)==4:
            flags.update(confirm_score=delta.loc[2024,'final_score']>=-1e-6,
                confirm_excess=delta.loc[2024,'annual_excess']>=-settings['gates']['max_excess_loss'],
                confirm_ic=f.loc[2024,'ic_mean']>=base.loc[2024,'ic_mean']*settings['gates']['ic_retention'],
                confirm_turnover=delta.loc[2024,'mean_turnover']<=settings['gates']['max_turnover_increase'])
        rows.append({'model':model,'period':period,'mean_final':f.final_score.mean(),'worst_final':f.final_score.min(),
            'mean_delta_final':delta.final_score.mean(),'mean_delta_excess':delta.annual_excess.mean(),
            'excess_positive_years':int((delta.annual_excess>0).sum()),
            'positive_excess_months':int((m.annual_excess>0).sum()),
            'worst_month_count':worst_month_count,'final_ci_lower':intervals.loc['final_score','lower'],
            'excess_ci_lower':intervals.loc['annual_excess','lower'],
            'qualified':all(flags.values()),'failed_gates':','.join(k for k,v in flags.items() if not v),**flags})
    return pd.DataFrame(rows)


def publish(cfg,root,meta,runs,refs,phase,selection=None,confirmation=None):
    settings=cfg['top_tail_learning'];off,daily,monthly=collect(cfg,runs,refs)
    boot=comparisons(daily,settings['bootstrap'])
    gate=gates(off,monthly,boot,settings,[2021,2022,2023])
    if phase=='confirm':gate=pd.concat([gate,gates(off,monthly,boot,settings,[2021,2022,2023,2024])],ignore_index=True)
    tables={'official':off,'monthly':monthly,'bootstrap':boot,'gates':gate}
    for name in ['targets','candidates','auxiliary','events','oof','importance']:
        entries=[]
        for path in sorted((root/'tables'/name).glob('*.csv')) if (root/'tables'/name).exists() else []:
            entries.append(pd.read_csv(path))
        if entries:tables[name]=pd.concat(entries,ignore_index=True)
    for name,frame in tables.items():save_csv(ROOT/'experiments'/f"top_tail_{meta['study_id']}_{name}.csv",frame)
    if selection is None:
        qualified=gate.loc[gate.qualified].sort_values(['mean_final','worst_final','model'],ascending=[False,False,True])
        winner=None
        if len(qualified):
            best=float(qualified.mean_final.max())
            winner=qualified[qualified.mean_final>=best-1e-6].sort_values(['worst_final','model'],ascending=[False,True]).iloc[0].model
        selection={'study_id':meta['study_id'],'created_at':timestamp(),'source_digest':meta['source_digest'],
            'protocol_digest':meta['protocol_digest'],'winner':winner,'status':'selected' if winner else 'no_qualified_arm',
            'screen_git':meta['git'],'arms':settings['arms'],'keep_q':settings['keep_q'],
            'gates':gate.to_dict('records'),'oof_sources':meta['oof_sources'],
            'required_confirm_fits':int(winner is not None),'rule':'predeclared complete qstar gates; mean score, tolerance, worst fold, ID'}
        selection['selection_digest']=digest(selection)
        save_json(root/'selection.json',selection);save_json(ROOT/'docs'/f"top_tail_{meta['study_id']}_selection.json",selection)
    if phase=='confirm':
        winner=selection['winner'];candidate=gate[(gate.model==winner)&(gate.period=='Historical2021_2024')]
        promote=bool(len(candidate) and candidate.iloc[0].qualified)
        confirmation={'study_id':meta['study_id'],'selection_digest':selection['selection_digest'],
            'winner':winner,'status':'historical_check_complete','recommendation':'handoff_candidate_to_B' if promote else 'retain_S003',
            'historical_2024_used_previously':True,'criteria':candidate.to_dict('records')}
        save_json(root/'confirmation.json',confirmation)
        save_json(ROOT/'docs'/f"top_tail_{meta['study_id']}_confirmation.json",confirmation)
    draw_figures(cfg,meta,tables)
    report(cfg,meta,tables,selection,confirmation)
    current=read_records(contained_path(cfg['paths']['experiment_log']))
    if current[:len(meta['initial_records'])]!=meta['initial_records']:raise ValueError('Original shared records changed.')
    extra=current[len(meta['initial_records']):]
    all_manifests=[contained_path(cfg['paths']['models'])/r['exp_id']/'run.json' for r in extra]
    if any(not r['exp_id'].startswith(meta['study_id']+'_') for r in extra):raise ValueError('Unexpected external log changes.')
    audit_runs=[]
    for manifest in all_manifests:
        info=read_json(manifest);check_run(info)
        audit_runs.append({'exp_id':info['exp_id'],'manifest':relative(manifest),'sha256':sha256(manifest),
                          'artifact_hashes':info['artifact_hashes']})
    for path,expected in meta['frozen_inputs'].items():
        if sha256(contained_path(path))!=expected:raise ValueError('Frozen legacy input changed during S010.')
    for path,expected in meta['context_artifacts'].items():
        if sha256(contained_path(path))!=expected:raise ValueError('Prepared context changed.')
    for path,expected in meta['target_artifacts'].items():
        if sha256(contained_path(path))!=expected:raise ValueError('Saved target mapping or its actual audit changed.')
    public=[ROOT/'docs'/f"top_tail_{meta['study_id']}.md",ROOT/'docs'/f"top_tail_{meta['study_id']}_selection.json",
            *sorted((ROOT/'experiments').glob(f"top_tail_{meta['study_id']}_*.csv")),
            *sorted((ROOT/'docs/figures').glob(f"{meta['study_id']}_*.png"))]
    if confirmation:public.append(ROOT/'docs'/f"top_tail_{meta['study_id']}_confirmation.json")
    audit={'status':'passed','study_id':meta['study_id'],'time':timestamp(),'phase':phase,
        'implementation_commit':meta['git']['commit'],'source_digest':meta['source_digest'],'protocol_digest':meta['protocol_digest'],
        'data':meta['data'],'fit_attempts':meta['fit_attempts'],'new_model_fits':len(meta['fit_attempts']),
        'fit_budget':18,'shared_records_before':len(meta['initial_records']),'shared_records_after':len(current),
        'shared_records_unchanged':True,'new_shared_records':len(extra),'runs':audit_runs,
        'max_official_difference':max(read_json(p)['official_max_difference'] for p in all_manifests),
        'max_reload_difference':max(read_json(p)['reload_max_difference'] for p in all_manifests if read_json(p)['new_model_fits']),
        'frozen_inputs':meta['frozen_inputs'],'context_artifacts':meta['context_artifacts'],
        'target_artifacts':meta['target_artifacts'],'feature_prefix':meta['feature_prefix'],
        'selection_digest':selection['selection_digest'],'visual_inspection_complete':False,
        'published_files':{relative(p):{'sha256':sha256(p),'size_bytes':p.stat().st_size} for p in public}}
    save_json(root/'audit.json',audit);save_json(ROOT/'docs'/f"top_tail_{meta['study_id']}_artifacts.json",audit)
    return selection,confirmation


def draw_figures(cfg,meta,tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study=meta['study_id'];out=contained_path(cfg['paths']['figures'])/study;out.mkdir(parents=True,exist_ok=True)
    models=['T030','B10','B20','R20','R30','L30'];colors=dict(zip(models,['#333333','#2874a6','#5b9548','#b7823c','#985d8a','#b54a46']))
    off=tables['official'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for layer,ax in zip(['raw','band'],axes):
        for model in models:
            f=off[(off.model==model)&(off.layer==layer)].sort_values('year')
            if len(f):ax.plot(f.year,f.final_score,marker='o',label=model,color=colors[model])
        ax.axhline(.4,color='#888888',ls=':',lw=1);ax.set_title(f'Official final score: {layer}');ax.set_xticks([2021,2022,2023,2024]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(out/f'{study}_scores.png',dpi=150);plt.close(fig)
    f=off[(off.layer=='band')&(off.year<=2023)].groupby('model')[['ic_mean','annual_excess','mean_turnover']].mean().reindex(models)
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True);x=np.arange(len(f));width=.25
    for i,(name,values) in enumerate([('IC contribution',.4*f.ic_mean),('Excess contribution',.3*f.annual_excess),('Stability contribution',.3*(1-f.mean_turnover))]):
        ax.bar(x+(i-1)*width,values,width,label=name)
    ax.set_xticks(x,models);ax.set_title('CV equal-year official score contributions');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(out/f'{study}_components.png',dpi=150);plt.close(fig)
    month=tables['monthly'];band=month[month.layer=='band']
    matrix=band.pivot(index='model',columns='month',values='final_score').reindex(models)
    fig,ax=plt.subplots(figsize=(14,4),constrained_layout=True)
    im=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='RdYlGn',vmin=.1,vmax=.55)
    ax.set_yticks(range(len(matrix)),matrix.index);ticks=np.arange(0,len(matrix.columns),3)
    ax.set_xticks(ticks,[str(matrix.columns[i]) for i in ticks],rotation=45,ha='right');ax.set_title('Continuous annual controller: monthly final scores');fig.colorbar(im,ax=ax)
    fig.savefig(out/f'{study}_months.png',dpi=150);plt.close(fig)
    b=tables['bootstrap'];b=b[(b.layer=='band')&(b.period=='CV2021_2023')&(b.metric=='annual_excess')]
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for i,row in enumerate(b.itertuples()):
        ax.errorbar(row.delta,i,xerr=[[max(0,row.delta-row.lower)],[max(0,row.upper-row.delta)]],fmt='o',color=colors[row.model])
    ax.set_yticks(range(len(b)),b.model);ax.axvline(0,color='black',lw=.8);ax.set_title('Paired 20-day block CI: CV excess difference versus T030');ax.grid(axis='x',alpha=.2)
    fig.savefig(out/f'{study}_intervals.png',dpi=150);plt.close(fig)
    for p in out.glob('*.png'):(ROOT/'docs/figures'/p.name).write_bytes(p.read_bytes())


def report(cfg,meta,tables,selection,confirmation):
    study=meta['study_id'];decision=(confirmation or {}).get('recommendation','retain_S003' if selection['winner'] is None else 'CV_candidate_pending_historical_check')
    off=tables['official'];lines=[f'# {study}：严格时间 OOF 的头部分类与排序', '',f'发布时间：{timestamp()}','',
        f"执行提交 `{meta['git']['commit']}`；源码摘要 `{meta['source_digest']}`；协议摘要 `{meta['protocol_digest']}`。",'',
        f"**阶段判断：{decision}。** CV 锁定臂：{selection['winner'] or '无合格臂'}。新增拟合 {len(meta['fit_attempts'])}/18。",'',
        '## 协议与实际时间边界','',
        '第一阶段固定 Y4 / T030 / 800 轮：只新增 2018→2019、2018–2019→2020 两次预热，其余年度复用 S009 已验收的原生预测。'
        '第二阶段五臂各固定 400 轮与 T030 参数，40 个原特征加一个 Y4 时间外百分位；全部第二阶段从 2019 起训练。'
        '每个外层只用更早年度 OOF；第一阶段与第二阶段各自剔除最后交易日跨界标签。第一阶段的既有参数经过历史选择，嵌套日期正确并不能消除既往择模偏差。','',
        'B10/B20 在全允许监督池学习未来 Top10/20 事件；R20/R30 在由当日 X 与 OOF 形成的候选中学习全市场 Top10 事件；'
        'L30 使用全市场 0–4 序数、线性 label_gain、按交易日连续分组及固定截断 466。'
        '事件标签在候选筛选前定义，相同收益保持平均排名并列。所有候选先仅由 X / 预测形成，之后才移除无监督标签行。','',
        '完整输出采用精确整数平均排名编码，保留 native 的严格次序和全部 tuple ties；缺价格原始零组的编码锚定为 0。'
        'R 系列候选整体领先池外，池内按第二阶段 / 第一阶段排序，池外保持第一阶段顺序；不以代码打破第二阶段的同分同原分并列。'
        '完整信号最终只叠加一次原 q*，每年冷启动、全年连续；掉出候选池不额外清仓。原生值与候选 mask 单独无损保存。','',
        '## 官方全年结果','',table_markdown(off,['model','year','layer','ic_mean','annual_excess','mean_turnover','final_score']),'',
        '## 预声明晋级门槛','',table_markdown(tables['gates'],['model','period','mean_final','mean_delta_final','mean_delta_excess','final_ci_lower','excess_ci_lower','qualified','failed_gates']),'',
        '晋级只比较完整 q* 方案；逐年、月度尾部、IC 保留、超额和区间同时约束。没有合格臂则不生成新的 2024 第二阶段预测。'
        '只有唯一锁定清单提交 / 推送后才允许一次 2024 历史检查，不事后换臂、扩大预算或改变 keep_q。','',
        '## 核对与限制','',
        '实际 V1 全市场 2018 前缀重算、日标签 / 候选真实前缀、原生模型保存重载、完整键 / 有限值、'
        '编码 / CSV 严格排序与并列、完整信号 / q* 前缀及原官方八指标落盘核对随真实运行记录。'
        '每个拟合在调用前计入预算，失败不自动重试。已有 718 条共享记录保留，只追加真正通过的训练和派生完整方案。','',
        '配对区间抽取已在原连续路径计算的日 IC / 超额 / 换手：20 日循环块、1000 次、seed=42+year，年份等权。'
        '不在拼接月份或 bootstrap 边界重建控制器；区间是已使用历史样本的条件描述，不校正多轮历史选择。'
        '2021–2024 已用于既往研究；本研究不使用隐藏测试标签，也不推断正式测试成绩。事件命中改善不等于组合收益改善。','',
        '原 Y4 与 T030 从 2018 起训，第二阶段从 2019 起训；收益差也可能来自训练时期差异，不能全部归因于训练目标。'
        'S007/T033 原模型未保存与 Y1 2024 缺失不影响本轮选定的 Y4/T030 来源，仍保留为原阶段限制。成员 B 独立复核及团队采纳待接续。','',
        'LightGBM binary 标签、LambdaRank 整数标签与组数要求按[官方参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)'
        '及[Dataset 说明](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Dataset.html)落实。','']
    for name in ['targets','candidates','auxiliary','oof']:
        if name in tables:lines.extend([f'## {name} 实际统计','',f'完整小型表：`experiments/top_tail_{study}_{name}.csv`。',''])
    for p in sorted((ROOT/'docs/figures').glob(f'{study}_*.png')):lines.extend([f'![{p.stem}](figures/{p.name})',''])
    write_text(ROOT/'docs'/f'top_tail_{study}.md','\n'.join(lines))

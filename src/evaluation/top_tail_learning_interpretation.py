"""Post-completion S010 publication only; no fits, scores, or decision changes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.alpha_research import relative, save_csv, write_text
from src.models.optuna_tuning import read_json, save_json
from src.models.target_research import table_markdown
from src.utils.project import ROOT, load_config, sha256, timestamp
from src.utils.research_cache import contained_path


def publish():
    cfg,_=load_config();study=cfg['project']['current_head_study'];root=contained_path(cfg['paths']['research_studies'])/study
    meta,audit=read_json(root/'study.json'),read_json(root/'audit.json')
    if meta['status']!='complete' or audit['status']!='passed':raise ValueError('Only publish a completed S010 numerical run.')
    for path,expected in meta['source'].items():
        if sha256(contained_path(path))!=expected:raise ValueError('Executed numerical source changed.')
    for path,info in audit['published_files'].items():
        if sha256(contained_path(path))!=info['sha256']:raise ValueError('Completed public input changed.')
    selection=read_json(root/'selection.json');decision=meta['recommendation']
    exp=contained_path(cfg['paths']['experiment_log']).parent
    official=pd.read_csv(exp/f'top_tail_{study}_official.csv')
    auxiliary=pd.read_csv(exp/f'top_tail_{study}_auxiliary.csv')
    gates=pd.read_csv(exp/f'top_tail_{study}_gates.csv')
    s9index=read_json(contained_path(cfg['top_tail_learning']['diagnostic_artifacts']))
    s9path=ROOT/'experiments/top_tail_S009_official.csv'
    if sha256(s9path)!=s9index['published_files'][relative(s9path)]['sha256']:raise ValueError('S009 descriptive Y4 reference changed.')
    y4=pd.read_csv(s9path);y4=y4[(y4.model=='Y4')&y4.year.isin(official.year.unique())]
    combined=pd.concat([official[['model','year','layer','ic_mean','annual_excess','mean_turnover','final_score']],
        y4[['model','year','layer','ic_mean','annual_excess','mean_turnover','final_score']]],ignore_index=True)
    summary=combined[combined.year<=2023].groupby(['model','layer'],sort=True)[['ic_mean','annual_excess','mean_turnover','final_score']].mean().reset_index()
    save_csv(exp/f'top_tail_{study}_cv_summary.csv',summary)
    base=summary[(summary.model=='T030')&(summary.layer=='band')].iloc[0]
    attributions=[]
    for row in summary[(summary.layer=='band')&~summary.model.isin(['T030','Y4'])].itertuples():
        ci=.4*(row.ic_mean-base.ic_mean);ce=.3*(row.annual_excess-base.annual_excess)
        ct=-.3*(row.mean_turnover-base.mean_turnover);total=row.final_score-base.final_score
        if abs(ci+ce+ct-total)>1e-12:raise ValueError('CV score contributions do not recover the published difference.')
        attributions.append({'model':row.model,'ic_contribution':ci,'excess_contribution':ce,
            'turnover_contribution':ct,'final_delta':total})
    attribution=pd.DataFrame(attributions);save_csv(exp/f'top_tail_{study}_score_attribution.csv',attribution)
    auc=auxiliary[auxiliary.model!='L30'].groupby('model')[['conditional_auc','conditional_logloss']].mean().reset_index()
    classifier=auc.merge(summary[summary.layer=='raw'],on='model',validate='one_to_one')
    save_csv(exp/f'top_tail_{study}_events_vs_returns.csv',classifier)
    # Conditional official Top-return statistics, from already saved actual sets.
    details=[];ndcg_rows=[];extra_inputs={relative(s9path):sha256(s9path)}
    for year in sorted(official.year.unique()):
        context_path=root/'context'/f'{int(year)}.parquet'
        if sha256(context_path)!=meta['context_artifacts'][relative(context_path)]:raise ValueError('Context changed.')
        context=pd.read_parquet(context_path);n=context.ts_code.nunique();shape=(context.trade_date.nunique(),n)
        y=context.y_ret_1d.to_numpy().reshape(shape);vol=context.rank_volatility_20.to_numpy().reshape(shape)
        for row in official[official.year==year].itertuples():
            if row.model=='T030':
                fold='confirm2024' if year==2024 else f'wf{int(year)}'
                item=next(r for r in s9index['runs'] if r['model']=='T030' and r['fold']==fold and r['layer']==row.layer)
                path=contained_path(item['manifest']).parent/'actual_top_sets.npz'
                if sha256(path)!=item['artifact_hashes'][relative(path)]:raise ValueError('Legacy actual Top changed.')
                extra_inputs[relative(path)]=sha256(path)
            else:
                path=contained_path(cfg['paths']['metrics'])/row.exp_id/'actual_top_sets.npz'
                run=read_json(contained_path(cfg['paths']['models'])/row.exp_id/'run.json')
                if sha256(path)!=run['artifact_hashes'][relative(path)]:raise ValueError('Saved actual Top changed.')
            saved=np.load(path,allow_pickle=False)
            key='returns' if 'returns' in saved else 'return_top'
            if key not in saved:raise ValueError('Actual return-set key unavailable.')
            top=saved[key].astype(bool);counts=top.sum(axis=1)
            positive=252*np.mean(np.divide(np.where(top&(y>0),y,0).sum(axis=1),counts))
            negative=252*np.mean(np.divide(np.where(top&(y<0),y,0).sum(axis=1),counts))
            if abs(positive+negative-row.top1_annual_ret)>1e-10:raise ValueError('Weighted realised return components do not recover the official value.')
            vc=(top&np.isfinite(vol)).sum(axis=1)
            vr=np.divide(np.where(top&np.isfinite(vol),vol,0).sum(axis=1),vc,out=np.full(len(vc),np.nan),where=vc>0)
            details.append({'model':row.model,'year':int(year),'layer':row.layer,
                'top_volatility_percentile':np.nanmean(vr),'positive_return_component':positive,
                'negative_return_component':negative,'absolute_top_return':row.top1_annual_ret,
                'annual_excess':row.annual_excess,'final_score':row.final_score})
            if row.model=='L30' and row.layer=='raw':
                candidate_path=contained_path(run['candidate_mask'])
                if sha256(candidate_path)!=run['artifact_hashes'][relative(candidate_path)]:raise ValueError('Actual ranking candidate mask changed.')
                candidate=np.load(candidate_path,allow_pickle=False).reshape(shape)
                scores=pd.read_csv(contained_path(run['prediction']),usecols=['pred']).pred.to_numpy().reshape(shape)
                allowed=context.quote_valid & context.flag_limit_up.eq(0) & np.isfinite(context.y_ret_1d)
                p=pd.Series(np.nan,index=context.index)
                p.loc[allowed]=context.loc[allowed,'y_ret_1d'].groupby(context.loc[allowed,'trade_date']).rank(method='average',pct=True)
                levels=np.sum(p.to_numpy()[:,None]>[.60,.80,.90,.95],axis=1).reshape(shape)
                good=allowed.to_numpy().reshape(shape)
                for t,date in enumerate(context.trade_date.drop_duplicates()):
                    use=np.flatnonzero(candidate[t]&good[t]);gain=levels[t,use];s=scores[t,use];k=min(466,len(use))
                    discount=1/np.log2(np.arange(k)+2);ideal=float(np.dot(np.sort(gain)[::-1][:k],discount))
                    order=np.argsort(-s,kind='stable');ordered=s[order]
                    starts=np.flatnonzero(np.r_[True,ordered[1:]!=ordered[:-1]]);ends=np.r_[starts[1:],len(use)]
                    dcg=sum(float(gain[order[start:end]].mean())*float(discount[start:min(end,k)].sum())
                            for start,end in zip(starts,ends) if start<k)
                    ndcg_rows.append({'model':'L30','year':int(year),'trade_date':int(date),'query_rows':len(use),
                        'k':k,'ndcg466_linear_tie_average':dcg/ideal if ideal>0 else np.nan,
                        'single_class':np.unique(gain).size==1})
    exposure=pd.DataFrame(details);save_csv(exp/f'top_tail_{study}_return_components.csv',exposure)
    ndcg=pd.DataFrame(ndcg_rows);save_csv(exp/f'top_tail_{study}_ndcg.csv',ndcg)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    axes[0].bar(classifier.model,classifier.conditional_auc,color='#4c819e');axes[0].axhline(.5,color='black',ls=':')
    axes[0].set_ylim(.4,.8);axes[0].set_title('CV binary event AUC (each declared prediction pool)')
    axes[1].bar(classifier.model,classifier.annual_excess,color='#a27556');axes[1].axhline(0,color='black',lw=.8)
    axes[1].set_title('Same models: raw official annual excess')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    out=contained_path(cfg['paths']['figures'])/study/f'{study}_events_returns.png';fig.savefig(out,dpi=150);plt.close(fig)
    (ROOT/'docs/figures'/out.name).write_bytes(out.read_bytes())
    gate_cv=gates[gates.period=='CV2021_2023']
    top=gate_cv.sort_values('mean_final',ascending=False).iloc[0]
    r20=attribution[attribution.model=='R20'].iloc[0]
    r20_years=official[(official.model=='R20')&(official.layer=='band')].sort_values('year')
    global_b10=classifier[classifier.model=='B10'].iloc[0]
    global_exposure=exposure[(exposure.model=='B10')&(exposure.layer=='raw')].mean(numeric_only=True)
    findings=['## 完成结论与判读','',
        f"实际新增拟合 **{len(meta['fit_attempts'])}/18**，失败 {sum(f['status']!='passed' for f in meta['fit_attempts'])}；"
        f"原 {len(meta['initial_records'])} 条共享记录保留，新增 {audit['new_shared_records']} 条。"
        f"官方八指标最大差 {audit['max_official_difference']:.1e}，模型重载最大差 {audit['max_reload_difference']:.1e}。",'',
        f"三折最高完整均分为 **{top.model} / {top.mean_final:.10f}**。预声明选中方案：**{selection['winner'] or '无合格臂'}**；"
        f"阶段建议：**{decision}**。没有合格臂时保留 S003，2024 第二阶段拟合 / 新预测为 0；"
        '有合格臂时仅检查已先推送的唯一臂，检查结果不触发其他臂补跑。','',
        '### 为什么本轮没有晋级','',
        f"- R20 只在 2021 提高 Top 超额，2022 / 2023 的年化超额分别为 "
        f"{r20_years[r20_years.year==2022].annual_excess.iloc[0]:.6f} / {r20_years[r20_years.year==2023].annual_excess.iloc[0]:.6f}，"
        '低于 S003 的 0.196025 / 0.135785，也超过逐年最多下降 0.02 的容忍范围。',
        f"- R20 均分提升 {r20.final_delta:+.6f} 的来源：IC 项 {r20.ic_contribution:+.6f}、"
        f"超额项 {r20.excess_contribution:+.6f}、换手项 {r20.turnover_contribution:+.6f}。"
        'Top 收益未随排序相关性一起提高；ΔFinal 与 ΔExcess 的 95% 下界均不为正。',
        f"- B10 的 AUC 为 {global_b10.conditional_auc:.6f}，raw 年化超额仍为 {global_b10.annual_excess:.6f}。"
        f"其实际 Top 组平均波动百分位 {global_exposure.top_volatility_percentile:.6f}；"
        f"正收益贡献 {global_exposure.positive_return_component:.6f}、负收益贡献 {global_exposure.negative_return_component:.6f}，"
        '负收益抵消了正收益。结果支持本轮概率目标伴随较高波动暴露、未同时控制收益下侧的解释，未证明因果关系。',
        '- R20 的 raw 超额和完整 q* 超额均未超过已存在的纯 Y4 同层 CV 对照；'
        '本轮没有发现足够稳健的第二阶段增量。训练时期不同仍是比较限制。','',
        table_markdown(attribution,['model','ic_contribution','excess_contribution','turnover_contribution','final_delta']),'',
        '### 与原有信号的同层对照','',
        table_markdown(summary,['model','layer','ic_mean','annual_excess','mean_turnover','final_score']),'',
        'Y4 行为 S009 已评分的原有归因对照，未在本轮新增训练 / 官方记录。它与 T030 都从 2018 起训，'
        '本轮第二阶段从 2019 起训；样本时期、筛池、目标及完整信号重排可能共同改变结果。'
        '因此超过 S003 不自动等于第二阶段相对纯 Y4 的独立机制改善，也不意味着隐藏测试泛化。','',
        '### 事件判别与收益','',
        table_markdown(classifier,['model','conditional_auc','conditional_logloss','ic_mean','annual_excess','final_score']),'',
        'AUC 评价正例概率的判别，官方超额评价收益金额。B10/B20 的 AUC 来自全允许监督池；R20/R30 来自各自候选池，'
        '不同池的 AUC 不能直接当成同一测试集的模型排名。应在每个方案内部同时看事件判别与金额收益，'
        '正例更常出现仍可能伴随较大的负收益；不会因为 AUC 高或某一折超过 0.4 而改动晋级门槛。','',
        '### 收益两侧与已知波动暴露','',
        table_markdown(exposure.groupby(['model','layer'])[['top_volatility_percentile','positive_return_component','negative_return_component','absolute_top_return','annual_excess']].mean().reset_index(),
            ['model','layer','top_volatility_percentile','positive_return_component','negative_return_component','absolute_top_return','annual_excess']),'',
        '上述收益分解直接读取已保存的实际官方收益 Top 集合，逐日按人数平均后年化；正负两项严格恢复官方 Top 年化绝对收益。'
        '波动百分位只用当日 V1 的 rank_volatility_20。它是暴露描述，不是风险导致收益变化的因果证明；'
        '不能仅凭 feature gain 或 AUC 给失败方案归因。完整年度行见 return_components 表。','',
        '### L30 排序辅助量','',
        table_markdown(ndcg.groupby(['model','year']).agg(query_days=('trade_date','size'),
            mean_query_rows=('query_rows','mean'),mean_ndcg466=('ndcg466_linear_tie_average','mean'),
            single_class_days=('single_class','sum')).reset_index(),
            ['model','year','query_days','mean_query_rows','mean_ndcg466','single_class_days']),'',
        'NDCG 使用验证候选中有监督标签的日组、全允许监督池的 0–4 序数与线性 gain，'
        '以完整 native tuple 编码的排序计算，精确预测并列按组内平均 gain 计期望 DCG；'
        'k=min(466,当日有监督候选数)。这是事后辅助统计，不进入训练、早停或晋级；标签总 gain 为零的日组标缺失。','',
        f'![事件与收益](figures/{out.name})','',
        '成员 B 独立复核待完成；S011 的 S009 跨年条件仍不满足，S012 不自动扩参。正式提交阶段继续按团队固定方案接续。','']
    report=ROOT/'docs'/f'top_tail_{study}.md'
    body=report.read_text(encoding='utf-8');head,rest=body.split('\n## 协议与实际时间边界',1)
    head=head.split('\n## 完成结论与判读',1)[0].rstrip()
    write_text(report,head+'\n\n'+'\n'.join(findings)+'\n## 协议与实际时间边界'+rest)
    audit['descriptive_publication']={'time':timestamp(),'source':f'src/evaluation/top_tail_learning_interpretation.py',
        'sha256':sha256(ROOT/'src/evaluation/top_tail_learning_interpretation.py'),
        'new_model_fits':0,'new_official_scores':0,'decision_changed':False,
        'extra_frozen_inputs':extra_inputs}
    public=[report,ROOT/'docs'/f'top_tail_{study}_selection.json',
        *sorted(exp.glob(f'top_tail_{study}_*.csv')),*sorted((ROOT/'docs/figures').glob(f'{study}_*.png'))]
    confirm=ROOT/'docs'/f'top_tail_{study}_confirmation.json'
    if confirm.exists():public.append(confirm)
    audit['published_files']={relative(p):{'sha256':sha256(p),'size_bytes':p.stat().st_size} for p in public}
    audit['visual_inspection_complete']=False;audit.pop('visual_inspection',None);audit.pop('final_handoff',None)
    save_json(root/'audit.json',audit);save_json(ROOT/'docs'/f'top_tail_{study}_artifacts.json',audit)
    print('S010 descriptive publication complete; no fitting, official rescoring or selection changes')


if __name__=='__main__':publish()

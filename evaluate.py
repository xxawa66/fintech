'''
量化比赛测试集评分脚本
评价指标: Rank IC(40%) + Top组超额收益(30%) + 预测换手率(30%)
'''
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from loguru import logger


def evaluate(_submission_path: str, _data_dir: str = '/home/quant6/data/比赛') -> dict:
    '''
    计算综合评分

    参数:
        _submission_path: submission.csv 路径 (ts_code, trade_date, pred)
        _data_dir: 比赛数据目录

    返回:
        各指标及综合得分字典
    '''
    data_dir = Path(_data_dir)

    # 读取数据
    df_pred = pd.read_csv(_submission_path)
    df_y = pd.read_csv(data_dir / '测试集_Y.csv')
    df_x = pd.read_csv(data_dir / '测试集_X.csv', usecols=['ts_code', 'trade_date', 'flag_limit_up'])

    # 合并
    df = df_pred.merge(df_y, on=['ts_code', 'trade_date'], how='inner')
    df = df.merge(df_x, on=['ts_code', 'trade_date'], how='inner')
    logger.info(f'合并后样本数: {len(df)}, 交易日数: {df["trade_date"].nunique()}')

    # ========== 1. Rank IC (40%) ==========
    ic_list = []
    for m_date, m_group in df.groupby('trade_date'):
        valid = m_group.dropna(subset=['y_ret_1d'])
        if len(valid) < 30:
            continue
        ic, _ = spearmanr(valid['pred'], valid['y_ret_1d'])
        ic_list.append(ic)

    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list, ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_positive_ratio = np.mean(np.array(ic_list) > 0)

    logger.info(f'Rank IC: mean={ic_mean:.6f}, std={ic_std:.6f}, ICIR={icir:.4f}, IC>0占比={ic_positive_ratio:.2%}')

    # ========== 2. Top组超额收益 (30%) ==========
    excess_list = []
    for m_date, m_group in df.groupby('trade_date'):
        # 剔除涨停股和Y缺失
        valid = m_group[(m_group['flag_limit_up'] == 0) & m_group['y_ret_1d'].notna()].copy()
        if len(valid) < 100:
            continue
        # 按pred降序排列，等分10组
        valid = valid.sort_values('pred', ascending=False).reset_index(drop=True)
        n_top = max(len(valid) // 10, 1)
        top1_ret = valid['y_ret_1d'].iloc[:n_top].mean()
        market_ret = valid['y_ret_1d'].mean()
        excess_list.append(top1_ret - market_ret)

    annual_excess = np.mean(excess_list) * 252
    top1_annual_ret = np.mean([
        m_group[(m_group['flag_limit_up'] == 0) & m_group['y_ret_1d'].notna()]
        .sort_values('pred', ascending=False)['y_ret_1d']
        .iloc[:max(len(m_group[(m_group['flag_limit_up'] == 0) & m_group['y_ret_1d'].notna()]) // 10, 1)]
        .mean()
        for _, m_group in df.groupby('trade_date')
        if len(m_group[(m_group['flag_limit_up'] == 0) & m_group['y_ret_1d'].notna()]) >= 100
    ]) * 252

    logger.info(f'Top组超额收益: 年化={annual_excess:.4f}, Top1年化绝对收益={top1_annual_ret:.4f}')

    # ========== 3. 预测换手率 (30%) ==========
    dates = sorted(df['trade_date'].unique())
    turnover_list = []
    prev_set = None

    for m_date in dates:
        m_group = df[df['trade_date'] == m_date]
        valid = m_group[m_group['flag_limit_up'] == 0].copy()
        if len(valid) < 100:
            prev_set = None
            continue
        valid = valid.sort_values('pred', ascending=False)
        n_top = max(len(valid) // 10, 1)
        curr_set = set(valid['ts_code'].iloc[:n_top])

        if prev_set is not None and len(prev_set) > 0:
            intersection = len(curr_set & prev_set)
            union = len(curr_set | prev_set)
            turnover = 1.0 - intersection / union
            turnover_list.append(turnover)

        prev_set = curr_set

    mean_turnover = np.mean(turnover_list)
    logger.info(f'预测换手率: mean={mean_turnover:.4f}, (1-turnover)={1 - mean_turnover:.4f}')

    # ========== 4. 综合评分 ==========
    final_score = ic_mean * 0.4 + annual_excess * 0.3 + (1 - mean_turnover) * 0.3
    logger.info(f'综合得分 = {ic_mean:.6f}×0.4 + {annual_excess:.4f}×0.3 + {1 - mean_turnover:.4f}×0.3 = {final_score:.6f}')

    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': ic_positive_ratio,
        'annual_excess': annual_excess,
        'top1_annual_ret': top1_annual_ret,
        'mean_turnover': mean_turnover,
        'final_score': final_score,
    }


if __name__ == '__main__':
    submission_path = '/home/quant6/data/比赛/submission.csv'
    result = evaluate(submission_path)
    print('\n===== 评分结果 =====')
    for m_key, m_val in result.items():
        print(f'  {m_key}: {m_val:.6f}')

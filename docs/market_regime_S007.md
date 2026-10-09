# S007：市场状态特征研究（固定 T030 配置）

更新时间：2026-10-09T16:50:26+08:00

## 协议

固定 S003 锁定的 T030 LightGBM 配置（800 轮、seed 42，不重搜），仅替换特征集：
baseline = 原 40 特征 V1；market = 40 + 14 个市场状态特征
（11 个日度横截面聚合与 20 日滚动体制统计 + 3 个动量/反转交互）。
市场特征只用赛题允许字段、只使用 t 日及以前信息，在连续历史上一次算完再切分。
换手层沿用既有留仓带（不新增控制器层）：主口径 keep_q = T030 q*，另存 0.1 参考。
四折：wf2021 / wf2022 / wf2023 + confirm2024，冷启动，训练标签不跨验证边界。
baseline 臂逐折复现 S003 冻结 T030 结果（断言 1e-9 内一致）后，与 market 臂同层对比。

成功标准（《下一步》第二十节）：2023 折 0.35710 → 0.38+，且 2024 折保持 0.38+。

## 特征清单

基础（11）：mkt_ret_mean、mkt_ret_median、mkt_ret_std（横截面离散度）、mkt_adv_ratio、
mkt_limit_up_ratio、mkt_limit_down_ratio、mkt_amount_chg（全市场成交额对数变化）、
mkt_ret_mean_20、mkt_ret_vol_20、mkt_breadth_20、mkt_mom_20（20 日市场复利动量）。

交互（3）：mkt_x_mom_vol20 = ret_20 × mkt_ret_vol_20；
mkt_x_mom_breadth = ret_20 × mkt_breadth_20；mkt_x_rev_mktmom = ret_1 × mkt_mom_20。

## 结果（band keep_q=q*，官方口径）

|臂|2021|2022|2023|三折均值|2024|
|---|---:|---:|---:|---:|---:|
|baseline|0.3885388115|0.3861836281|0.3570976762|0.3873823479|0.3772733720|0.3873823479|
|market|0.3728577135|0.3803332272|0.3181858045|0.4024032191|0.3571255817|0.4024032191|

## 分项（band q*，逐折）

|折|臂|IC|年化超额|换手|综合分|
|---|---|---:|---:|---:|---:|
|wf2021|baseline|0.0664610415|0.2297282756|0.0232136259|0.3885388115|
|wf2021|market|0.0705194159|0.1658728146|0.0170396573|0.3728577135|
|wf2022|baseline|0.0810647459|0.1960246105|0.0168321779|0.3861836281|
|wf2022|market|0.0776629284|0.1780294316|0.0138025789|0.3803332272|
|wf2023|baseline|0.0502418651|0.1357850961|0.0124486621|0.3570976762|
|wf2023|market|0.0464767883|0.0092891062|0.0106388091|0.3181858045|
|confirm2024|baseline|0.0843562247|0.2011622338|0.0223627069|0.3873823479|
|confirm2024|market|0.0903343209|0.2377218945|0.0168235922|0.4024032191|

## market 相对 baseline 的归因（band q*）

|折|总分变化|IC 项|超额项|稳定项|ΔIC|Δ超额|Δ换手|
|---|---:|---:|---:|---:|---:|---:|---:|
|wf2021|-0.0156810980|+0.0016233498|-0.0191566383|+0.0018521906|+0.0040583744|-0.0638554610|-0.0061739687|
|wf2022|-0.0058504010|-0.0013607270|-0.0053985537|+0.0009088797|-0.0034018175|-0.0179951790|-0.0030295990|
|wf2023|-0.0389118718|-0.0015060307|-0.0379487970|+0.0005429559|-0.0037650767|-0.1264959899|-0.0018098529|
|confirm2024|+0.0150208711|+0.0023912385|+0.0109678982|+0.0016617344|+0.0059780962|+0.0365596607|-0.0055391148|

## 市场状态特征的重要性（market 臂，gain）

|折|特征|gain|split|
|---|---|---:|---:|
|wf2021|mkt_ret_vol_20|857.8|1627|
|wf2021|mkt_ret_std|850.6|1791|
|wf2021|mkt_breadth_20|842.3|1592|
|wf2021|mkt_amount_chg|740.0|1589|
|wf2021|mkt_ret_mean|738.8|1377|
|wf2021|mkt_limit_down_ratio|718.1|1168|
|wf2021|mkt_ret_mean_20|678.9|1280|
|wf2021|mkt_limit_up_ratio|639.5|1409|
|wf2021|mkt_ret_median|458.6|1128|
|wf2021|mkt_adv_ratio|429.7|1193|
|wf2021|mkt_mom_20|279.7|768|
|wf2021|mkt_x_rev_mktmom|14.0|144|
|wf2021|mkt_x_mom_breadth|12.1|93|
|wf2021|mkt_x_mom_vol20|10.6|96|
|wf2022|mkt_breadth_20|950.5|1487|
|wf2022|mkt_ret_std|946.8|1930|
|wf2022|mkt_ret_mean|917.5|1474|
|wf2022|mkt_amount_chg|909.9|1760|
|wf2022|mkt_ret_mean_20|772.3|1347|
|wf2022|mkt_ret_vol_20|755.4|1633|
|wf2022|mkt_limit_up_ratio|745.5|1592|
|wf2022|mkt_limit_down_ratio|687.5|1265|
|wf2022|mkt_ret_median|671.0|1221|
|wf2022|mkt_adv_ratio|558.8|1095|
|wf2022|mkt_mom_20|425.8|838|
|wf2022|mkt_x_mom_vol20|13.4|109|
|wf2022|mkt_x_rev_mktmom|13.2|94|
|wf2022|mkt_x_mom_breadth|10.4|68|
|wf2023|mkt_breadth_20|1200.4|1621|
|wf2023|mkt_amount_chg|1161.5|1950|
|wf2023|mkt_ret_std|1123.3|1878|
|wf2023|mkt_ret_vol_20|1109.0|1709|
|wf2023|mkt_ret_mean|1073.5|1537|
|wf2023|mkt_limit_down_ratio|1036.1|1331|
|wf2023|mkt_ret_mean_20|1025.0|1426|
|wf2023|mkt_limit_up_ratio|885.9|1509|
|wf2023|mkt_ret_median|842.4|1282|
|wf2023|mkt_adv_ratio|791.6|1257|
|wf2023|mkt_mom_20|503.3|816|
|wf2023|mkt_x_mom_vol20|19.3|97|
|wf2023|mkt_x_rev_mktmom|14.4|106|
|wf2023|mkt_x_mom_breadth|10.3|48|
|confirm2024|mkt_ret_std|1379.6|1997|
|confirm2024|mkt_breadth_20|1324.6|1720|
|confirm2024|mkt_amount_chg|1282.0|2035|
|confirm2024|mkt_ret_vol_20|1276.0|1739|
|confirm2024|mkt_ret_mean|1126.7|1532|
|confirm2024|mkt_limit_down_ratio|1101.2|1326|
|confirm2024|mkt_ret_mean_20|982.6|1378|
|confirm2024|mkt_limit_up_ratio|876.4|1510|
|confirm2024|mkt_ret_median|824.1|1342|
|confirm2024|mkt_adv_ratio|725.8|1199|
|confirm2024|mkt_mom_20|591.4|861|
|confirm2024|mkt_x_mom_vol20|19.3|87|
|confirm2024|mkt_x_rev_mktmom|12.5|83|
|confirm2024|mkt_x_mom_breadth|10.3|41|

## 产物与解释边界

研究产物：outputs/metrics/research_studies/S007/；汇总表：experiments/market_regime_S007_{folds,comparison,monthly,importance}.csv。
40 特征直接复用 S003 已校验 parquet 缓存；市场特征由本仓库 src/features/market_features.py 计算。
本轮结论只覆盖 2021–2024 历史验证；测试期（2025–2026）标签已不干净，未参与本轮。
固定单一超参配置，未与特征集做联合搜索；若 market 臂有效，是否需要按特征数重调 feature_fraction 属后续 S008 的问题。

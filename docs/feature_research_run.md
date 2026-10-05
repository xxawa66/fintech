# Day 5–8：固定模型的特征研究

本阶段承接已由 A、B 分别复现的 `E000_baseline_retry1` / `E001_baseline_repeat`。
对应 `workflow.md` 的阶段 6；这里的 Day 编号是两周实施时间表，不是阶段编号。
正式流程在 `src/`，路径、窗口、特征组与实验顺序在 `configs/project.yaml`。

## 四天的交付

|时间|成员 A 的交付|成员 B 的配合|
|---|---|---|
|Day 5|保持共享实验表 20 列；新增实验入口、缓存和完整诊断记录|保留已完成的原 V1 独立复现；复核新增记录|
|Day 6|实现三组共 18 个特征，检查公式与未来信息边界|独立检查分组、日期、缺失日和截面排名|
|Day 7|完成 2023 年八组固定参数对比，保存分项/月度/Top-Bottom/因子 IC|审阅评分口径、样本一致性与分项归因|
|Day 8|完成删组消融、锁定候选及一次 2024 年确认，给出保留/替换 V1 的结论|独立复核新特征与跨年度结论|

## 特征定义

设 `c,h,l,v` 为收盘、最高、最低、成交量，`r=c/c.shift(1)-1`。
`MA_n` 为包含当日在内的完整 n 行均值；`sd_n` 使用样本标准差 ddof=1。
所有公式按股票、日期升序计算；保留价格缺失日，不前填/后填；窗口必须完整。
公式先用 float64，最终转成 float32；分母不大于 0 时为 NaN。

|组|特征|公式|
|---|---|---|
|T|trend_efficiency_20|`(c-c.shift(20)) / sum_20(abs(c.diff()))`|
|T|up_ratio_20|`MA_20(r>0)`；r 缺失时不能当作 False|
|T|downside_rms_20|`sqrt(MA_20(min(r,0)^2))`|
|T|return_skew_20|pandas rolling skew；零方差为 NaN|
|T|return_kurt_20|pandas rolling kurt（超额峰度）；零方差为 NaN|
|T|rank_trend_efficiency_20|当日有效价格股票的平均秩百分位|
|V|corr_ret_logvol_20|`corr_20(r,log1p(v))`，双方需非零方差|
|V|corr_price_vol_20|`corr_20(c,v)`，双方需非零方差|
|V|signed_vol_ratio_5|`sum_5(sign(r)*v)/sum_5(v)`|
|V|signed_vol_ratio_20|`sum_20(sign(r)*v)/sum_20(v)`|
|V|volume_cv_20|`sd_20(v)/MA_20(v)`|
|V|rank_signed_vol_ratio_20|当日有效价格股票的平均秩百分位|
|R|volatility_ratio_5_20|`sd_5(r)/sd_20(r)`|
|R|volatility_ratio_20_60|`sd_20(r)/sd_60(r)`|
|R|risk_adj_ret_20|`ret_20 / (sd_20(r)*sqrt(20))`|
|R|true_range_relative|`max(h-l,abs(h-c_prev),abs(l-c_prev))/c_prev`；max 不跳过缺失分量|
|R|atr_ratio_5_20|`MA_5(true_range_relative)/MA_20(true_range_relative)`|
|R|rank_risk_adj_ret_20|当日有效价格股票的平均秩百分位|

三个新排名与原五个排名一样，只使用当日 X 和有效价格，绝不按 y 是否存在来决定股票池。
真实零成交量仍为 0；相关系数的数值舍入误差限于 [-1,1]，无穷值为 NaN。

## 预设实验与候选锁定

统一 CPU LightGBM，保持 V1 原参数、200 轮、seed=42、8 线程、deterministic。
本阶段只改变特征组；调参、早停、融合、平滑和正式测试预测安排在后续阶段。

|顺序|组合|加入组|特征数|
|---|---|---|---:|
|1|base|无|40|
|2|trend|T|46|
|3|volume|V|46|
|4|risk|R|46|
|5|all|T+V+R|58|
|6|without_trend|V+R|52|
|7|without_volume|T+R|52|
|8|without_risk|T+V|52|

筛选只用 fold1（训练 2018–2022、验证 2023）。全八组通过后，按全年官方综合分最大选择；
距最高分不超过 1e-6 视作近似并列，优先更少特征，再按上表顺序。
相对 base 未提高超过 1e-6 时保留 base。月度、单因子和重要性仅用于解释。

2023 结果的模型运行清单、候选、源代码与数值配置指纹保存为 `selection.json`。
必须先将选择报告和共享实验表提交到干净的 `main`，才能运行 confirm；修改代码、配置、候选或清单会被拒绝。
确认使用 fold2（训练 2018–2023、验证 2024）：先复现 V1，8 项原始指标最大差不得超过 1e-10；
随后最多运行锁定的一个新增候选。候选在两年都提高超过 1e-6 才推荐替换 V1。
若锁定 base，则无需额外的 2024 候选实验。本阶段全量训练最多 10 次。

## 执行方式

在仓库根目录使用已安装依赖的项目环境。先提交实现，再执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.models.research --phase screen --study-id S001 --owner A
```

screen 自动先执行独立的 256 股票 / 10 轮 / 58 特征检查；该记录不进入正式实验表。
完成后提交 `docs/feature_research_S001.md`、共享实验表及项目状态，再执行：

```powershell
.\.venv\Scripts\python.exe -m src.models.research --phase confirm --study-id S001 --owner A
```

已有实验或研究目录不覆盖；失败保留原文件并标记失败，不从不完整的实验中挑选候选。
重做研究使用新 study-id；原基线示例的实验编号也应换成未使用的新编号。

## 缓存、检查与产物

每个截止年度独立生成一份全 58 特征的 X 缓存，多组实验仅选择列，样本池保持一致。
指纹包含原始数据 SHA、历史截止日期、股票集合、特征配置、相关代码和数值依赖版本。
缓存不包含标签；读取时验证 SHA、列、float32、行数、键与顺序，完成后原子落位。
每个模型仍从原始数据关联标签，再由 B 的切分代码屏蔽训练边界标签：2023 折 4,280 个，2024 折 4,522 个。

- `data/processed/research_cache/<指纹>/`：X 缓存与来源；每个实验 processed 目录保存引用。
- `outputs/models/<实验>/`：模型文本、完整配置、特征列表、环境和代码提交、运行清单。
- `outputs/predictions/<实验>/valid_<年度>.csv`：全部验证键；无缺失或无穷预测。
- `outputs/metrics/<实验>/`：原始官方 8 指标、逐日、月度、Top/Bottom、重要性、标签和日志。
- `outputs/metrics/research_studies/<研究>/`：阶段状态、对比表、锁定记录、2023 单因子 IC、最终确认。
- `experiments/experiment_log.csv`：只有通过覆盖、模型重载、官方评分和原文件校验的全量实验才写入。
- `docs/feature_research_<研究>.md`：可在 GitHub 查看的小型真实结果报告。

模型重载预测差要求 ≤1e-12，8 项官方分差要求 ≤1e-10；每次运行都核验原始训练 CSV 与官方附件未改动。
共享 CSV 保留原 20 列，追加使用独占写锁及原子替换；重复 ID、已存在目录和越界路径会被拒绝。
大型缓存、预测、模型和分析产物继续由 `.gitignore` 排除。

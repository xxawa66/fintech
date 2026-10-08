# S005 运行协议：固定 T030 的训练目标与损失研究

用户于 2026-10-08 要求完成 [计划书第 5 节](alpha_retention_plan.md) 的 S005。S004 已完成；本入口只实施 S005 的五个新目标 / 损失与三折原始预测，不进行 S006 融合或 2024 确认。

## 固定协议

- 配置独立放在 `configs/project.yaml` 的 `target_research` 段，保持已完成 S004 的 `alpha_research`、S003 的 `optuna` 数值协议不变。
- 来源为 S003 冻结选择、T030 三折 raw / band 清单、原始数据及附件，以及原有 40 特征缓存；同时核对 S004 冻结控制器交接。
- 共用 T030 的结构、正则、采样与学习率参数、800 轮、seed=42、CPU 8 线程，每次建立独立 Dataset。验证标签不进入训练目标或早停。
- 时间窗口：2018–2020→2021、2018–2021→2022、2018–2022→2023；跨界剔除 3853 / 4170 / 4280 标签，相同监督样本数为 2594048 / 3568878 / 4597785。
- `target_cache.py` 复用未经改动的 `tuning_cache.prepare_folds`，再通过相同排序与监督掩码恢复真实日期 / 股票键；核对 X 行索引、原始 y 数组、完整切分与训练计数，禁止从 y 数组猜测日期。
- 目标只在屏蔽跨界标签、筛除无 y 或无有效价格样本之后构造；保存 `ts_code,trade_date,y_ret_1d,training_position,transformed_y` 的无损 Parquet 对应表，以及每日统计与摘要。

|目标|变换|objective / metric|
|---|---|---|
|Y0|原始 y，复用原 T030，不拟合|regression / l2|
|Y1|当日平均并列百分位|regression / l2|
|Y2|当日线性插值 1%/99% 缩尾|regression / l2|
|Y3|Y2 按日标准化，ddof=0、零 std→0|regression / l2|
|Y4|与 Y3 相同目标和映射|huber / huber，alpha=0.9|
|Y5|原始 y|regression_l1 / l1|

Y3 / Y4 共用相同映射，隔离损失变化；Y5 / Y0 为相同原始目标的损失对照。总计 **15 次新训练，12 份唯一目标映射**。目标尺度发生变化时，固定正则的相对作用也会变化，因此本轮是固定参数下的归因比较。

## 执行入口

先提交、推送实现与数值协议，从干净 main 运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.models.target_research --study-id S005 --owner A
```

相同源码、配置、依赖和来源可以恢复已通过的折；失败或中断的拟合不自动覆盖、重试或增加预算：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.models.target_research --study-id S005 --owner A --resume
```

研究持有进程锁，实验表采用已有追加锁；完成的 S005 拒绝重跑。独立复现使用 `S005_*` 新编号，并完整保留自己的模型、来源和结果。

## 评分、选择与核对

全部原始预测覆盖该年所有原始键，缺价行模型输出固定为 0；CSV / 官方评分行序固定 `trade_date,ts_code`。验证始终使用原始 y_ret_1d，不加控制器。

选择辅助量为 `PredictiveScore = 0.4*ic_mean + 0.3*annual_excess` 的三折等权均值，**不是官方综合分**。公开全部八项官方指标、IC/超额二维非支配点及弱年结果；按均值选两个新目标，1e-6 内同分依次比较最差折、编号。

每次训练复用已有 `run_prediction` 的完整预测、模型保存/重载、实际 CSV 原始官方八指标核对和真实记录；新增目标信息写入 spec 与运行清单。额外核对 CSV 是否保持原生模型的完整排序和精确并列组。目标对应表回读须无损；每份唯一目标从真实训练期前半段重算，须与全训练期相应前缀逐位一致。

与 T030 的相关性只使用完整市场预测的当日平均百分位。Top 重合只剔除涨停，使用官方排序，不读 y 或缺失掩码；它是互补性诊断，不冒充已实现的融合得分。月度和季度是已保存全年预测的描述切片，不另拟合。

## 产物与交接

- `outputs/metrics/research_studies/S005/`：配置、来源、状态、候选、12 份目标映射、统计与审计。
- `outputs/models/S005_Yx_wf20xx_raw/`：模型、配置快照和 run.json；`outputs/predictions/`：完整 raw CSV。
- `outputs/metrics/S005_Yx_wf20xx_raw/`：八指标、日 / 月 / 季度、重要性、相关性与实际 Top 重合。
- `experiments/alpha_S005_{comparison,folds,monthly,quarterly,agreement,target_statistics,importance}.csv`：小型交接表；原 20 列共享实验表仅追加 15 条真实记录，Y0 复用不追加。
- `docs/alpha_research_S005.md`、`_selection.json`、`_artifacts.json` 及图表：结果、两个冻结伙伴、完整 SHA 索引。

本阶段执行真实全量训练与产物核对，不新增或运行测试套件。S005 的两个目标伙伴不直接替换正式方案；成员 B 独立审核、S006 有限融合 / 控制器比较及其锁定后的 2024 历史确认接续。

损失支持和 alpha 参数已核对 [LightGBM 4.7.0 官方文档](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html#objective-parameters)（2026-10-08）。

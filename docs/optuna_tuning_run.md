# S003：Optuna/TPE、滚动验证与官方综合分

用户于 2026-10-07 批准实施 LightGBM 与 band 联合调参。当前文档是执行协议；真实成绩以运行报告为准。

## 固定协议

- 40 个 V1 特征，现有清洗、缺价原始预测 0、训练标签边界规则保持一致。
- 扩展窗口：2018–2020 / 验证 2021；2018–2021 / 验证 2022；2018–2022 / 验证 2023。
- 每个 trial 使用同一组模型参数、轮数、keep_q 完整运行三个年份，各折 band 独立冷启动。
- 目标是三折 **band 后官方 final_score 的等权均值**。IC、超额、换手、标准差与最差年份同步保存；没有 IC 额外门槛。
- LightGBM 使用回归损失训练。轮数在 200 / 400 / 800 中选，不用外层验证标签早停，不启用 pruning。
- 预算 50 个 trial，包含两个固定对照；失败 trial 占用预算，不编造分数、不自动补数。
- TPESampler(seed=42, n_startup_trials=10, multivariate=True)，单进程顺序搜索；每次 LightGBM 使用 8 个线程、CPU、deterministic=true。
- 完整搜索范围见 configs/project.yaml 的 optuna 段。band_keep_q 为 0–1；有限树深时实际叶数不超过 2^max_depth，保存建议值与执行值。
- 每次训练重新构造 LightGBM Dataset；bagging_fraction<1 时 bagging_freq=1。

前十组依次为：V1+q0.1、L1+q0.1、其余模型参数采样并固定 q=0.05 / 0.075 / 0.09 / 0.11 / 0.125 / 0.15 / 0 / 1；随后 TPE 联合搜索。先在 0.1 附近启动，不限制最终候选一定接近 0.1。

## 选择与 2024

50 个 trial 后，在已完成的最优 trial 与 V1 / L1 的缓存预测上，交叉比较 q=0.1 和最优 trial 的 q；最多新增三种派生组合，不增加模型训练。所有完整方案按同一三折均值选出唯一候选；完全同分时优先 q 距离 0.1 更近、源 trial 更早的方案。

先把选择清单、报告和实验记录提交并推送 main，再从干净 main 执行 2024。确认入口核对代码、协议、依赖、原始文件摘要，以及 origin/main 中的实际选择清单。

2024 只拟合一次锁定模型。q=0.1 及原 V1 / L1 的 q=0.1 是预先约定的诊断对照；不根据 2024 修改候选。2024 已在历史研究中使用过，不宣称为全新独立留出集。

## 运行

先将实现与协议提交、推送 main：

```powershell
.\.venv\Scripts\python.exe -m src.models.optuna_tuning --study-id S003 --phase search --owner A
```

中断后保持数值协议、代码、包版本和原始文件不变：

```powershell
.\.venv\Scripts\python.exe -m src.models.optuna_tuning --study-id S003 --phase search --resume --owner A
```

提交、推送搜索产生的 docs/optuna_tuning_S003_selection.json、报告和实验记录后：

```powershell
.\.venv\Scripts\python.exe -m src.models.optuna_tuning --study-id S003 --phase confirm --owner A
```

已有研究目录拒绝普通重跑。恢复已完成的折时核对所有产物摘要、共享记录的一致性，不重复追加。

## 产物与核对

- outputs/metrics/research_studies/S003：SQLite、采样器与待运行 trial 检查点、配置和来源快照、trials.csv、选择、确认、审计与运行日志。
- data/processed/optuna_cache：仅包含原始 X 派生的 40 特征缓存，标签在特征计算之后附加。
- outputs/models、outputs/predictions、outputs/metrics：按研究 / trial / 折 / raw 或 band 保存模型、完整 CSV、八指标、日度与月度分析。
- outputs/figures/S003：收敛曲线、band 参数与 CV 分数关系、fANOVA 参数重要性。重要性基于本次有限搜索观察，不代表因果关系；num_leaves 的建议值可能被深度上限限制。
- experiments/experiment_log.csv：沿用原 20 列，逐折原始和 band 结果分别追加。CV 汇总不冒充官方单期分数。
- docs/optuna_tuning_S003.md、选择清单和最终产物索引：用于两人交接，模型与大型数据不进入 Git。

模型文本使用现有的 Python UTF-8 保存/重载方式，兼容 Windows 中文路径。预测文件、标签和评分帧按 trade_date / ts_code 排序；评分从落盘 CSV 重读，再与原始官方 evaluate.py 核对八指标，容差 1e-10。预测键完整、唯一、有限；模型重载容差 1e-12。band 只接收键、pred 与 flag_limit_up。

数据库与采样器状态同时保存。已运行过的本地 sampler.pkl 只从本研究目录读取。数值协议或来源变化、产物摘要不一致、数据或官方评分口径错误会停止研究并保留原因。非有限模型/评分属于失败 trial，不进入候选池。

原始数据和官方附件保持原样。正式推荐方案的更新由最终比较与成员 B 复核决定；本阶段不生成比赛测试提交。执行真实训练与产物核对，不新增或运行测试套件。

## S003 执行完成（2026-10-08）

50 个有效 trial、三组派生对照与锁定后一次 2024 确认已完成。选择先在 `4ff3853` 推送 main，确认使用同一个 T030 配置：800 轮、28 叶、keep_q=0.0022778298112255263。三折均分 0.3772733720，2024 为 0.3873823479；V1 + band(0.1) 对照分别为 0.3608045815、0.3491780142。151 次训练、312 条新记录通过核对，共享表 363 条，原 51 条不变，官方评分和模型重载差异均为 0。

已完成的 S003 命令作为执行协议保存；新一轮搜索使用新的研究编号。成员 B 复核入口是 `docs/optuna_tuning_S003.md`、冻结选择 / 确认清单及产物 SHA 索引。`experiments/optuna_S003_trials.csv` 保留完整 trial 指标，`optuna_S003_monthly.csv` / `optuna_S003_quarterly.csv` 是全年留仓状态的描述性切片；小型图表发布在 `docs/figures/`。模型、预测、标签、缓存、数据库和采样器仍保存在约定本地路径，索引记录其大小与 SHA。

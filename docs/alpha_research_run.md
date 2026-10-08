# S004 运行协议：固定 T030 的换手控制研究

用户于 2026-10-08 要求先完成 [计划书第 4 节](alpha_retention_plan.md) 的 S004。本入口仅实施该阶段；新目标训练、融合与 2024 确认由后续 S005 / S006 接续。

## 协议与入口

- 配置：`configs/project.yaml` 中独立的 `alpha_research` 段；旧 S002 / S003 数值协议不变。
- 输入：S003 冻结的 T030，2021 / 2022 / 2023 三折 raw CSV、标签及 q* band 参照。
- 来源核对：冻结选择的 canonical digest、产物索引、模型 / 预测 / 标签 / manifest 的 SHA，训练 / 测试 X 与官方附件。
- 控制器只接收键、pred、涨停标志和历史状态；标签留在评分侧。
- 67 个预定配置，完整排序和并列组在三折全部等价时，真实评分与记录只保留首次配置，比较表公布映射；两种不同家族的配置交给 S006，S003 的 q* band 始终保留。
- 新增模型训练为 0；评估年份为 2021–2023，每折冷启动；输出 / 评分行序固定为 `trade_date, ts_code`。
- 每个实际落盘折经原始官方评分器核对八指标（容差 1e-10），从 CSV 解码的 Top 与控制器意图一致；落盘后排名与并列组必须不变。
- 选择按 CV 均分排序；差在 1e-6 内优先最差折、预定编号。band 两种编码同属一个家族，raw 不进入控制器交接；等价配置不重复占两个名额。

先提交、推送实现与固定协议，再从干净 main 运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.models.alpha_research --study-id S004 --phase controllers --owner A
```

源码、配置、依赖和冻结来源保持一致时，可恢复已完成的配置和折：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m src.models.alpha_research --study-id S004 --phase controllers --owner A --resume
```

已完成研究拒绝重跑。中断留下的不完整折不被覆盖，需检查后用新编号；失败不补造分数，也不自动扩大预算。研究进程持有锁，拒绝并发写入共享记录。

## 具体控制器

1. **raw**：原始预测；bonus=0 作为同一原始排序的退化点。
2. **band**：直接调用现有 `band_scores`，保持旧数值和并列规则。
3. **band_minimal**：先取得旧 band 从实际 pred 解码出的换手 Top 集合；对相同集合计算当日必要的落选股平移量 `max(0, max_rank_unselected−min_rank_selected+1e-9)`。集合相同，只比较编码后的全市场排序与官方收益结果。
4. **bonus**：先以全市场平均百分位加留仓奖励，可选旧股获得 beta；beta>0 时非并列大小关系保持不变，精确并列按股票代码优先，再编码成严格次序。涨停股不获奖励，其基础 adjusted score 仍为当日全市场 rank；严格次序编码的数值尺度属于排序信号，不是收益率。
5. **gap**：按排名差和主动替换上限选股；先退出不可选旧股、按当日排名裁减规模和补位，再将最强新股与最弱旧股逐对比较。差值严格大于 delta 才替换，主动替换最多 `floor(rho*K)`；强制退出、规模裁减与空缺补位单独记录。并列优先较小股票代码，最弱并列旧股优先退出较大代码者。最后用按日必要间隔编码集合。

官方收益池另删除 y 缺失，和换手池的 K 可能不同；控制器不使用 y 缺失信息。所有收益和 IC 都从完整 pred 重新计算，不把自选集合冒充官方收益组合。

## 产物与验收

- `outputs/metrics/research_studies/S004/`：配置、来源 / 环境 / 代码快照、状态、逐配置清单、选择、审计与日志。
- `outputs/models/S004_Cxxx_wf20xx/run.json`：派生实验清单，无新模型文件；注明源 T030 模型与摘要。
- `outputs/predictions/S004_Cxxx_wf20xx/valid_20xx.csv`：完整预测。
- `outputs/metrics/S004_Cxxx_wf20xx/`：八指标、日 / 月 / 季指标、实际 Top 集合、留仓诊断、股票与持仓段长度分布。
- `experiments/alpha_S004_{comparison,folds,monthly,quarterly,holdings}.csv`：GitHub 交接的小型表；旧 20 列实验表只追加真实通过的派生折。
- `docs/alpha_research_S004.md`、`_selection.json`、`_artifacts.json` 和 `docs/figures/alpha_S004_*`：报告、两种控制器、产物 SHA 与图表。

冻结控制器使用真实三折的前半段重新运行，核对前缀的排序和并列组与全年对应前缀一致；这是本次运行内的因果性核对。原始文件、S003 来源和原有共享记录再次核验。持仓段长度包含折末右删失段，图表明确按交易观察计算。

本阶段执行真实预测处理、官方评分与产物核对，不新增或运行测试套件。成员 B 独立复核仍由团队后续完成；S004 的控制器交接不直接改变正式推荐或启用 R1。

## S004 执行完成（2026-10-08）

实现提交 `4563f48`，67 个配置全部完成，失败 0；排序等价去重 23 个，实际 44 种排序 / 132 个全量折。官方八指标与原始参照最大差均为 0，6 次真实前缀重放通过；原 363 条记录保持不变，新增 132 条派生记录，共享表共 495 条，新增模型训练为 0。48 个同集合编码对照的 Top 集合、年化超额、Top 绝对收益和换手均完全相同。

最高均分为 C018 的 0.3772752567，只比原方案高约 0.0000018847。C042 均分 0.3741114504，2023 为 0.3403279087，未通过替换门槛。两种控制器冻结供 S006 比较，正式研究参照保持 S003。完整报告、图表与 SHA 索引见 `docs/alpha_research_S004*`；S005 / S006 尚未运行。

`src/evaluation/controller_handoff.py` 只为已完成、已审计的研究补充判读与非支配图，读取真实产物并刷新发布文件摘要；不拟合、评分或改变选择。可运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.evaluation.controller_handoff --study-id S004
```

该步骤保留数值实验的原源码摘要，单独索引完成后新增的报告模块。用于文档和图表，不重复 S004 的参数比较。

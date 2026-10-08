# A 股收益预测比赛项目

项目仓库：<https://github.com/xxawa66/fintech>

赛题：基于原始量价数据预测股票未来一天收益率。每条预测对应一组 `ts_code + trade_date`。

当前已完成基础流程、S001、S002、S003 及 **S004 换手控制研究**。S004 的 67 个配置去重为 44 种完整排序，完成 132 个三折全量评分，新增模型训练 0 次；最高均分 **0.3772752567**，相对 S003 的 **0.3772733720** 仅增加约 **0.0000018847**，来自同一 Top 集合编码后的微小 IC 变化，未获得实质性提分。冻结 C018（按日编码 band，q=0.0022778298112255263）及 C042（排名差替换，delta=0、rho=0.01）供 S006 比较；正式方案继续以 S003 T030 + 锁定 band 为参照，其 2024 历史确认仍为 **0.3873823479**。结果见 [S004 报告](docs/alpha_research_S004.md) 与 [S003 报告](docs/optuna_tuning_S003.md)，S005 目标研究、S006 融合及成员 B 的独立复核待接续。分阶段方案见 [计划书](docs/alpha_retention_plan.md)，进度见 [项目状态](docs/project_status.md)。

## 项目依据

- [AGENTS.md](AGENTS.md)：后续操作的工作规则。
- [官方赛题 PDF](docs/赛题五-更新.pdf)、[Python 官方评分器](evaluate.py)、[R 官方评分器](evaluate.R)：正式要求与计算依据。
- [项目流程](docs/workflow.md)：推进顺序、两人分工和阶段交付。
- [项目状态](docs/project_status.md)：已完成事项、待办和变更记录。
- [模型研究计划书](docs/model_research_plan.md)：Day 9–12 已按固定协议完成；Day 13–14 衔接独立审核、最终训练与提交。
- [S004–S006 计划书](docs/alpha_retention_plan.md)：固定三折研究，S004 已完成，S005 / S006 待启动。
- [S004 运行说明](docs/alpha_research_run.md)：固定 T030 的 67 组控制器、排序去重、官方核对和真实交接。
- [数据字典](docs/data_dictionary.md)与[数据清单](data/manifest.json)：字段含义、文件大小与校验信息。

后续每一步都从仓库当前文件和实际 Git 状态出发；聊天记录用于补充背景，具体实现、进度和配置写回仓库。

## 目录

```text
fintech/
├── AGENTS.md
├── README.md
├── requirements.txt
├── requirements.lock.txt        # 本次 Windows / Python 3.12 环境的固定版本
├── evaluate.py                 # 官方附件，保持原样
├── evaluate.R                  # 官方附件，保持原样
├── configs/
│   └── project.yaml            # 数据路径、标签及评分权重
├── data/
│   ├── manifest.json           # 原始文件的大小与 SHA-256
│   ├── raw/                    # 原始 CSV，仅保存在本地
│   └── processed/              # 清洗、特征和验证数据，仅本地
├── docs/
│   ├── 赛题五-更新.pdf
│   ├── data_dictionary.md
│   ├── workflow.md
│   └── project_status.md
├── src/
│   ├── data/                   # 读取、清洗、构造数据集
│   ├── features/               # 价格、成交量、技术与截面特征
│   ├── models/                 # 基线、LightGBM 和融合
│   ├── evaluation/             # 官方评分适配、时间验证和回测
│   ├── utils/                  # 路径、版本信息、实验记录
│   └── submission/             # 生成与检查提交文件
├── tests/                      # 特征、时间边界和流程检查
├── notebooks/                  # 4 个探索 Notebook
├── experiments/
│   └── experiment_log.csv      # 统一实验记录表
└── outputs/
    ├── models/
    ├── figures/
    ├── metrics/
    ├── predictions/
    └── submissions/
```

## 本地准备

从 GitHub 克隆后，按 [data/raw/README.md](data/raw/README.md) 放置比赛原始 CSV。原始 CSV、处理数据、模型和提交结果不进入普通 Git 提交。

在项目根目录创建环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

首次全量运行环境为 Windows、Python 3.12.14。相同平台复现可以用 `requirements.lock.txt` 替代上面安装命令中的 `requirements.txt`；其他平台需要处理其中 Windows 专用依赖。所有相对数据路径都以仓库根目录为基准，具体见 `configs/project.yaml`。

## 一条命令运行基线

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m src.models.baseline --fold fold2 --exp-id E002_baseline_repeat --owner A
```

入口会先执行独立保存的 256 只股票、10 轮训练检查，再执行全部股票、200 轮训练。流程为：读取训练 CSV → 清洗 → 40 个历史特征 → 2018–2023 训练 / 2024 验证 → LightGBM → 完整验证预测 → 官方评分核对 → 实验记录。轮数与参数固定，2024 标签只用于事后评分。

每次运行使用新的 `--exp-id`；已有目录或记录会被拒绝，避免覆盖结果。上述示例为重复运行，已有成功实验编号为 `E000_baseline_retry1`。成功的全量实验写入 `experiments/experiment_log.csv`；小规模检查不写入该表。模型、预测、配置快照、逐日指标、运行日志及来源信息保存在各自的实验目录，详见 [基线说明](docs/baseline_run.md)。

开发检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`notebooks/01_eda.ipynb` 与 `notebooks/04_result_visualization.ipynb` 已由成员 B 执行；回测分析和换手模块已实现。排名融合模块现已实现，正式提交模块仍为占位说明，其他探索 Notebook 继续按研究需要补充。

## Day 5–8 特征研究

原 V1 已由成员 B 以 `E001_baseline_repeat` 独立复现，8 项指标逐位相同。
新增三组共 18 特征（完整池 58 列），固定原模型参数，先用 2023 八组筛选和删组消融，再锁定一个候选做 2024 确认。
成员 A 的 Day 5–8 已完成，40 项检查通过，`S001` 共完成 10 次全量实验。52 特征的量价+风险组合在 2023 从 0.120662 提高到 0.125195，但 2024 得分 0.193103 低于原 V1 的 0.197517，因此保留 **40 特征 V1**。实际对比、消融和稳定性分析见 [研究结果](docs/feature_research_S001.md)；运行方法见 [特征研究说明](docs/feature_research_run.md)。本轮新增特征与跨年结论的成员 B 独立审核暂缓，仍列为待办。

## Day 9–12 模型研究

S002 的固定参数、Ridge、排名融合和单候选跨年确认入口见 [运行说明](docs/model_research_run.md)。2023 完成 8 次新训练和 5 个融合权重对照，在 `688bece` 提交、推送候选后，仅用锁定的 L1 / R2 在 2024 新训练 2 次。F025 + band 两年均超过同层 V1，按既定门槛成为推荐方案；备用为 V1 + band(0.1)。36 条训练或派生记录已追加，共享表共 51 条，原 15 条未改动。两年每个预测文件完整覆盖 1,125,300 个键，全部官方评分核对差异为 0，10 个新模型重载预测差异为 0。

以下为本次执行命令。S002 已完成，已有目录拒绝覆盖；重新研究使用新编号，仍须先提交筛选锁定结果再确认。

```powershell
.\.venv\Scripts\python.exe -m src.models.model_research --study-id S002 --phase screen --owner A
# 将 2023 选择清单、报告与实验记录提交 main 后，再运行：
.\.venv\Scripts\python.exe -m src.models.model_research --study-id S002 --phase confirm --owner A
```

## S003 联合自动调参

S003 已完成 Optuna/TPE + 2021–2023 Walk-forward CV + 官方综合分的 50 个有效联合 trial，以及三组预定交叉对照。锁定 **T030：800 轮、28 叶、keep_q=0.0022778298112255263**，CV 均分 **0.3772733720**（V1 + band(0.1) 为 **0.3608045815**）。选择清单在 `4ff3853` 先提交、推送后，仅新拟合一个锁定模型确认 2024：完整方案 **0.3873823479**，高于同层 V1 的 **0.3491780142**；同一模型固定 band(0.1) 为 **0.3454167676**。提分主要来自低换手，2024 的 IC 和超额项也提高。151 次训练、312 条新记录均通过核对，共享表现 363 条，原 51 条未改动；模型重载与官方评分最大差均为 0。结果、图表、月度 / 季度诊断和交接索引见 [S003 报告](docs/optuna_tuning_S003.md)。完整 trial 表在 `experiments/optuna_S003_trials.csv`；配置与运行约定见 [联合调参说明](docs/optuna_tuning_run.md)。S003 为本轮最高分候选，正式提交前需成员 B 独立复核；2024 是此前已使用的历史验证期。

## S004 换手控制研究

已按 [计划书](docs/alpha_retention_plan.md) 完成 67 个参数配置和三折比较，23 个排序等价配置去重，实际 44 种排序 / 132 个全量折；全部官方八指标差异为 0，源 T030 raw / q* band 参照差异为 0。原有 363 条实验记录保持不变，新增 132 条派生记录，共 495 条。2024 未参与本阶段的新评估，模型训练次数为 0。

C018 的均分 **0.3772752567** 仅微升，Top 收益和换手保持相同；C042 的均分 **0.3741114504**，2021 / 2022 改善而 2023 降至 **0.3403279087**，未通过跨年替换门槛。两个控制器冻结供 S006 有限组合比较，S003 继续作为正式研究参照。排名差网格内 delta=0–0.2 在同一上限下结果完全等价，不能据此宣称唯一最优 delta。源文件、实际 Top、6 次前缀重放与旧记录核对通过；未新增或运行测试套件。

结果、四张图和交接见 [S004 报告](docs/alpha_research_S004.md)、[选择](docs/alpha_research_S004_selection.json)、[产物索引](docs/alpha_research_S004_artifacts.json)；小型表在 `experiments/alpha_S004_*`，完整预测和留仓诊断保存在本地忽略目录。按 [运行说明](docs/alpha_research_run.md) 可用新研究编号独立复现；已完成 S004 拒绝重跑。下一步 S005 研究训练目标，随后 S006 融合；成员 B 独立复核另行接续。

## 评分与交付

`final_score = 0.4 * ic_mean + 0.3 * annual_excess + 0.3 * (1 - mean_turnover)`。

提交 `submission.csv`（`ts_code,trade_date,pred`，覆盖全部测试行）和正文不超过 8 页的报告。报告包括数据介绍、描述性分析、模型分析、应用验证、产品思路及总结结论。

当前没有官方测试标签 `测试集_Y.csv`，不能计算正式测试得分。官方评分器中的 Linux 示例路径保持原样；`src/evaluation/official_eval.py` 已适配本地验证路径。基线入口还会将实际验证预测交给原始 `evaluate.py` 核对全部 8 项指标，误差超过 `1e-10` 时停止。

## 两人协作

成员 A 主负责特征、模型、融合和报告模型部分；成员 B 主负责数据审计、时间验证、评估、提交检查和报告数据/应用部分。双方共同审核未来信息泄漏、官方评分口径及最终提交。

按用户最新约定，后续代码、配置、文档与实验记录直接在 `main` 提交并推送到 `origin/main`。开始前查看 `git status`，同步远端并读取项目依据；完成阶段后更新状态文档与实验记录。具体流程见 [docs/workflow.md](docs/workflow.md)。

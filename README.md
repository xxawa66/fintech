# A 股收益预测比赛项目

项目仓库：<https://github.com/xxawa66/fintech>

赛题：基于原始量价数据预测股票未来一天收益率。每条预测对应一组 `ts_code + trade_date`。

当前已完成成员 B 的数据审计、评分适配与时间切分，以及成员 A 的基础特征和 LightGBM 全量流程。15 项检查通过；实验 `E000_baseline_retry1` 已训练 5,659,954 条样本、预测完整 2024 验证集，官方综合分为 **0.197517**。运行方式见 [基线说明](docs/baseline_run.md)，实际结果见 [基线结果](docs/baseline_result.md)，进度见 [项目状态](docs/project_status.md)。

## 项目依据

- [AGENTS.md](AGENTS.md)：后续操作的工作规则。
- [官方赛题 PDF](docs/赛题五-更新.pdf)、[Python 官方评分器](evaluate.py)、[R 官方评分器](evaluate.R)：正式要求与计算依据。
- [项目流程](docs/workflow.md)：推进顺序、两人分工和阶段交付。
- [项目状态](docs/project_status.md)：已完成事项、待办和变更记录。
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

`notebooks/01_eda.ipynb` 已由成员 B 执行；其他 Notebook、模型融合、回测与正式提交模块仍为预留位置。

## Day 5–8 特征研究

原 V1 已由成员 B 以 `E001_baseline_repeat` 独立复现，8 项指标逐位相同。
新增三组共 18 特征（完整池 58 列），固定原模型参数，先用 2023 八组筛选和删组消融，再锁定一个候选做 2024 确认。
成员 A 的 Day 5–8 已完成，40 项检查通过，`S001` 共完成 10 次全量实验。52 特征的量价+风险组合在 2023 从 0.120662 提高到 0.125195，但 2024 得分 0.193103 低于原 V1 的 0.197517，因此保留 **40 特征 V1**。实际对比、消融和稳定性分析见 [研究结果](docs/feature_research_S001.md)；运行方法见 [特征研究说明](docs/feature_research_run.md)。本轮新增特征与跨年结论的成员 B 独立审核待完成。

## 评分与交付

`final_score = 0.4 * ic_mean + 0.3 * annual_excess + 0.3 * (1 - mean_turnover)`。

提交 `submission.csv`（`ts_code,trade_date,pred`，覆盖全部测试行）和正文不超过 8 页的报告。报告包括数据介绍、描述性分析、模型分析、应用验证、产品思路及总结结论。

当前没有官方测试标签 `测试集_Y.csv`，不能计算正式测试得分。官方评分器中的 Linux 示例路径保持原样；`src/evaluation/official_eval.py` 已适配本地验证路径。基线入口还会将实际验证预测交给原始 `evaluate.py` 核对全部 8 项指标，误差超过 `1e-10` 时停止。

## 两人协作

成员 A 主负责特征、模型、融合和报告模型部分；成员 B 主负责数据审计、时间验证、评估、提交检查和报告数据/应用部分。双方共同审核未来信息泄漏、官方评分口径及最终提交。

按用户最新约定，后续代码、配置、文档与实验记录直接在 `main` 提交并推送到 `origin/main`。开始前查看 `git status`，同步远端并读取项目依据；完成阶段后更新状态文档与实验记录。具体流程见 [docs/workflow.md](docs/workflow.md)。

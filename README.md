# A 股收益预测比赛项目

项目仓库：<https://github.com/xxawa66/fintech>

赛题：基于原始量价数据预测股票未来一天收益率。每条预测对应一组 `ts_code + trade_date`。

当前阶段为**项目结构初始化**。模块和 Notebook 已预留位置；数据审计、特征计算、评分适配和模型训练尚未实现，当前没有模型结果或比赛成绩。

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
│   └── submission/             # 生成与检查提交文件
├── notebooks/                  # 4 个探索 Notebook
├── experiments/
│   └── experiment_log.csv      # 统一实验记录表
└── outputs/
    ├── models/
    ├── figures/
    ├── metrics/
    └── submissions/
```

## 本地准备

从 GitHub 克隆后，按 [data/raw/README.md](data/raw/README.md) 放置比赛原始 CSV。原始 CSV、处理数据、模型和提交结果不进入普通 Git 提交。

在项目根目录创建环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖清单用于后续开发，尚未生成固定版本的环境锁文件。所有相对数据路径都以仓库根目录为基准，具体见 `configs/project.yaml`。

目前 Notebook 只有阶段说明，源代码模块只有职责说明；可运行的训练和预测入口将在后续阶段加入。

## 评分与交付

`final_score = 0.4 * ic_mean + 0.3 * annual_excess + 0.3 * (1 - mean_turnover)`。

提交 `submission.csv`（`ts_code,trade_date,pred`，覆盖全部测试行）和正文不超过 8 页的报告。报告包括数据介绍、描述性分析、模型分析、应用验证、产品思路及总结结论。

当前没有官方测试标签 `测试集_Y.csv`，不能计算正式测试得分。官方评分器中的 Linux 示例路径保持原样；后续在 `src/evaluation/official_eval.py` 中适配本地验证路径，不修改官方指标定义。

## 两人协作

成员 A 主负责特征、模型、融合和报告模型部分；成员 B 主负责数据审计、时间验证、评估、提交检查和报告数据/应用部分。双方共同审核未来信息泄漏、官方评分口径及最终提交。

开始前查看 `git status`，同步远端并读取项目依据；完成阶段后更新状态文档与实验记录。具体流程见 [docs/workflow.md](docs/workflow.md)。

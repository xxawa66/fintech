# Day 9–12 模型研究运行说明

执行协议见 [计划书](model_research_plan.md)。使用 `src.models.model_research`，不改写 S001 的特征研究入口。

## 两阶段命令

仓库 `main` 干净、原始数据和配置引用的本地基线产物齐备时，先运行 2023 筛选：

```powershell
.\.venv\Scripts\python.exe -m src.models.model_research --study-id S002 --phase screen --owner A
```

该阶段复用 L0 的已保存模型及预测，并重新加载模型核对；恢复本地 band 后核对 E003 记录。随后依次训练 L1–L5、R1–R3，选择一个 LightGBM 与一个 Ridge，在五个固定权重下做排名融合。每个完整预测均保存原始及 band 两层、逐日 / 月度 / Top-Bottom 表和官方 8 指标。

结果锁定到 `outputs/metrics/research_studies/S002/selection.json`，并将同一份小型清单保存为 `docs/model_research_S002_selection.json`。先把报告、清单和实验记录提交并推送 `main`，再执行：

```powershell
.\.venv\Scripts\python.exe -m src.models.model_research --study-id S002 --phase confirm --owner A
```

确认阶段核对代码指纹、数值协议和全部筛选文件 SHA，只训练已锁定方案的非零权重组件，最多两次新训练；不能借此追加搜索。已有研究目录拒绝覆盖，复跑使用新研究编号。

## Ridge 与融合

- Ridge 使用 LSQR、`tol=1e-6`、最多 200 次迭代；记录 `n_iter_`、警告和是否触及上限，触及上限或出现收敛警告时不登记成功。参数含义见 [scikit-learn Ridge 官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。
- 40 列特征的中位数填充与标准化只拟合训练样本。预处理与模型一起保存为本地 `model.joblib`，重载后完整核对预测；只加载本项目自己生成的可信模型文件。
- 融合严格对齐完整键，每个模型只按当日预测做百分位排名，权重固定为 LGB 的 0 / 0.25 / 0.5 / 0.75 / 1。排名相关性和 Top 重合为无标签诊断。
- band 的输入只包含预测、键与当天涨停标记，不读取 Y；两折均冷启动。原始与 band 文件区分保存，band 只执行一次。

## 记录与产物

实际训练、引用预测、融合和 band 都使用独立实验编号。派生记录写明“无新增模型训练”，不能将记录条数当作训练次数。共享 CSV 保留 20 列，原有记录保持不变。

- `outputs/models/<exp>/`：原生 LightGBM 模型或 Ridge 模型、列顺序、配置快照与运行清单；派生实验保存来源。
- `outputs/predictions/<exp>/valid_2023.csv` 或 `valid_2024.csv`：全 1,125,300 个键。
- `outputs/metrics/<exp>/`：标签来源、8 项指标、逐日 / 月度 / Top-Bottom 及日志。
- `outputs/metrics/research_studies/S002/`：对照表、缓存清单、无标签模型相关性、锁定文件与确认记录。
- 研究目录下 `screen_analysis_raw/band`、`confirm_analysis_raw/band`：候选相对同层 V1 的 IC / Top / 换手分析及加权归因。分析已编码 band 文件时关闭默认再次 band。
- `docs/model_research_S002.md`：真实研究摘要；`docs/model_research_S002_selection.json`：跨成员可读的锁定方案与产物 SHA。

本地基线引用目前为 `S001_screen_base` 和 `S001_confirm_base`。Git 克隆不会带入模型 / 预测目录；新机器应交接这些可信产物与校验清单，或以独立编号复现基线后更新引用，再开启新研究，不覆盖旧实验。

失败保留运行清单与部分产物，不从未完成的网格选择候选。原始数据与官方附件在阶段开始 / 结束校验 SHA；所有落盘预测均经完整键、有限值、官方 8 指标核对。

成员 B 的本轮独立复核与正式测试预测、提交脚本、最终报告属于后续交接，执行状态以 `project_status.md` 为准。

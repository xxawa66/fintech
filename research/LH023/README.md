# LH023 —— 复现手册

本目录是研究报告 `docs/long_horizon_LH023_alpha_capacity.md` 的全部可执行部分：
**四折参数确认**与**α 目标优化**这两个实验的脚本链。
产物已经随包提交在 `experiments/LH023/`，可以直接对照学习；
想重跑则按本文末的流程执行。

结论摘要见 `docs/model_card_TS_K40.md`（最优模型的定义与成绩）。

---

## 1. 目录内容

| 文件 | 作用 |
|---|---|
| `cv_lambda_k_opt.py` | 核心评测器（`Evaluator`）+ λ/K 联合优化。提供 `mix` / `two_stage` / `paired` 等公共函数，另两个脚本都依赖它 |
| `cv_three_numbers.py` | 三轴重审：keep_q 轴、F1 权重轴（`--k-only` 时只跑含 35/45 的 K 加密网格） |
| `cv_alpha_lofo.py` | 首次把 α 直接放进优化目标的留一折权重实验，并输出 α 容量上界与 IC-α 前沿 |
| `_probe_*.py` | 底层依赖（面板加载、DGTW 分解、band、 IC 最优权重等），原本位于 `outputs/long_horizon/` |

---

## 2. 评测器里的关键常数

全部定义集中在 `cv_lambda_k_opt.py` 顶部，可以直接改：

```python
FOLDS  = ["wf2021", "wf2022", "wf2023", "confirm2024"]   # 四个 walk-forward 折
KQ     = 0.0022778298112255263                            # band 的 keep_q
ANN    = 252                                              # 年化天数
LAM_GRID = [0.0, 0.1, ..., 1.0]                           # λ 粗网格，步长 0.1
K_GRID   = [10, 15, 20, 25, 30, 40, 50, 60, 70, 85, 100]  # K 粗网格
```

部署权重（`cv_three_numbers.py` 的 `DEPLOY`）：

```python
{"H01": 0.5, "H05_F1T2": 0.1204, "H05_F2T2": 0.3796}
```

---

## 3. 运行前的三个前提

脚本只依赖 Python 标准科学栈（`numpy` / `pandas` / `scipy` / `pyarrow`），
但**输入数据不在版本库里**，这是唯一无法直接从仓库复现的部分：

1. **面板与标签**：`data/raw/训练集.csv`（原始训练数据，`.gitignore` 排除）。
2. **模型预测**：`outputs/long_horizon/LH003/<fold>/raw_predictions.parquet`，
   提供九个候选信号 `H01`、`H05/H10/H20/H30_F{1,2}T2`。整个 `outputs/long_horizon/`
   约 2.4 GB，同样被 `.gitignore` 排除。
3. **因子面板**：`outputs/long_horizon/_probe_factors_full.parquet`（约 533 MB），
   由 `_probe_factors.py` 生成；α 与 profile 的 DGTW 分解要用它。

> 也就是说：**脚本、判定口径、逐折结果全部可复现；中间数据需要各自按
> `data/raw` 与 LH003 的流程先生成。** 三步之间只有文件路径耦合，没有隐式状态。

---

## 4. 路径约定

脚本里不写任何个人机器的绝对路径，一律由文件位置推导，并允许环境变量覆盖：

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `FINTECH_ROOT` | 本文件上两级（仓库根） | 仓库根目录 |
| `LH_CACHE` | `$FINTECH_ROOT/outputs/long_horizon` | 预测与因子缓存 |
| `LH_OUT` | `$FINTECH_ROOT/experiments/LH023` | 产物输出目录 |
| `LH_TESTDIR` | `$FINTECH_ROOT/../../test_y_2025_2026` | 折绑定的辅助目录（仅参与 `sys.path`） |

---

## 5. 标准命令

在项目根目录执行，输出落到 `experiments/LH023/`。

```bash
# ① 三个具体数字：keep_q 轴 + F1 权重轴      → three_numbers_grid.csv
python research/LH023/cv_three_numbers.py

# ② K 轴的粗/加密对照（含 35 / 45）         → k_grid35_45.csv
python research/LH023/cv_three_numbers.py --k-only

# ③ λ 与 K 的联合前沿                        → lambda_k_lamgrid.csv, kgrid_lam*.csv,
#                                              lambda_k_joint_grid.csv
python research/LH023/cv_lambda_k_opt.py --parts 0,A,B,C

# ④ 以 α 为目标的留一折权重优化               → alpha_objective_grid.csv,
#                                              ic_alpha_frontier.csv
python research/LH023/cv_alpha_lofo.py
```

单次完整评估（IC + E + 换手 + α + profile）约 **0.5 秒**，所以 ④ 不需要代理函数，
直接以真实 α 做搜索目标。

> 注意：脚本开头会强制把 `OMP_*` / `MKL_*` 等线程数设为 1。多线程 BLAS 的浮点累加顺序
> 不确定，会让 SLSQP 收敛到略有差异的权重，同一配置两次结果可差到 $3\times10^{-4}$
> ——已经超过我们在这里要分辨的效应量。**不要去掉这段。**

---

## 6. 脚本与产物的对应关系

| 产物（`experiments/LH023/`） | 由哪条命令生成 | 行数 | 报告的哪一节 |
|---|---|---|---|
| `three_numbers_grid.csv` | ① | 72 | §5 keep_q、§6 F1 权重 |
| `k_grid35_45.csv` | ② | 52 | §4 K 轴加密对照 |
| `kgrid_lam0.5.csv` | ③ | 44 | §4 K 轴粗网格（① 的对照输入） |
| `lambda_k_lamgrid.csv` | ③ | 44 | §3 λ 轴 |
| `lamgrid_fine.csv` | ③ | 44 | §3 λ 加密（有害的证据） |
| `lambda_k_joint_grid.csv` | ③ | 296 | §8 α 容量上界（74 个 λ×K 组合） |
| `alpha_objective_grid.csv` | ④ | 352 | §7 α 目标 LOFO |
| `ic_alpha_frontier.csv` | ④ | 74 | §9 分数与 α 负相关 |
| `deploy_vs_lofo.csv`、`joint_pairwise_vs_deploy.csv` | ③ | 12 / 54 | §3–§4 的留一折选参对照 |

所有产物**只含四个 walk-forward 折（2021-01-01 至 2024-12-31）**，不含 2025 年以后的任何数据。

---

## 7. 判定口径

三项标准，四个参数、四个优化目标一视同仁：

1. **配对差 + 符号**：$\Delta_f = \text{指标}_{\text{候选},f} - \text{指标}_{\text{部署},f}$，
   看均值、$t = \bar{\Delta}/(\mathrm{sd}(\Delta)/2)$、以及四折符号是否一致。
2. **1 倍标准误平台**：上式 sd 的一个标准误内，有哪些取值与最优不可分辨。
3. **留一折 argmax + 留一折选参**：在三个训练折上挑最优点，看它命中几次部署点、以及
   它在未见折上究竟赚还是亏。

> 这里的噪声必须用**配对差的跨折标准误**衡量，不能用「同一配置跨折的水平 sd」——
> 后者会把年份之间的水平漂移算进来，把 IC 和 α 的信噪比算反（详见报告 §2.2）。

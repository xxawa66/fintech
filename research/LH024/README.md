# LH024 —— 复现手册

本目录是研究报告 `docs/long_horizon_LH024_optimization_exhaustion.md` 的全部可执行部分：
**TS_K40 之后三个优化方向**（层内排序标签 / 排序损失 / 新特征族）的探针脚本链。

产物已随包提交在 `experiments/LH024/cv_design_audit/`，可直接对照学习；想重跑则按本文末的流程执行。

前置结论见 `docs/long_horizon_LH023_alpha_capacity.md`（四个参数的四折确认）与
`docs/model_card_TS_K40.md`（交付模型的定义与成绩）。

---

## 1. 目录内容

| 文件 | 作用 | 产物 |
|---|---|---|
| `probe_within_layer.py` | 方向一（A）主探针：层内化程度 $\beta$ 的插值前沿 + DGTW 分解 | `within_layer_{grid,paired,crossfold,composition}.csv` |
| `probe_within_ic.py` | 方向一（A）机理补充：全池 IC 与层内 IC 的谱 | `within_ic_spectrum.csv` |
| `probe_pairwise_rerank.py` | 方向一（B）主探针：阶段 2 目标函数的六臂受控对照（`--mode base/split/walk`） | `pairwise_rerank_*.csv` |
| `probe_new_features.py` | 方向二第 1 步：20 个新特征的独立筛选（单特征 IC / $\alpha$ / profile） | `new_feature_{screen,basket}.csv` |
| `probe_pairwise_newfeat.py` | 方向二第 2 步：`L2`（40 列）× `L2X`（48 列）受控对照 | `newfeat_stage2_*.csv` |
| `probe_newfeat_halves.py` | 方向二第 3 步：每折再切半，扩成 8 个样本外窗口 | `newfeat_halves{,_daily}.csv` |
| `probe_newfeat_halves_stats.py` | 8 窗稳健性统计（逐日配对 $t$、留一窗、剔 2023） | 打印 |
| `probe_newfeat_regime.py` | 追问：wf2023 的 $\Delta\alpha$ 是「补弱 regime」还是「敞口换方向」 | 打印 |
| `probe_horizon_break_2023.py` | 追问：wf2023 的弱是 H01 独有还是整个长周期族共有 | 打印 |
| `probe_newfeat_cumintra.py` | 收窄：`L2C` = `L2` + cumintra$\{5,20,60\}$（3 列） | `newfeat_cumintra_*.csv` |
| `probe_newfeat_nocum.py` | 收窄：`L2N` = `L2` + 其余五列（去掉 cumintra） | `newfeat_nocum_*.csv` |
| `probe_newfeat_cumintra_stats.py` | 三条对照线（`L2C`/`L2N`/`L2X`）的逐日配对检验 | `newfeat_cumintra_paired.csv` |
| `probe_newfeat_blend.py` | 方向二收尾：阶段 2 分数按 $w$ 混合（$w=1$ 纯 H01，$w=0$ 纯 L2X） | `newfeat_blend_{scan,windows}.csv`、`s2_cache/*.npz` |
| `probe_newfeat_blend_windows.py` | blend 的稳健性判定（预注册判据①②） | 打印 |
| `probe_newfeat_blend_verdict.py` | 三个判据量：折级聚合、留一窗、曲线平滑度 | 打印 |
| `probe_newfeat_blend_daily_test.py` | 逐日配对检验（$n=681$ 天，含剔除 2023） | 打印 |
| `probe_blend_recheck.py` | 对「$w$ 仍有可取之处」这条反驳的逐项核查 | 打印 |

> 这些脚本**原本位于仓库外的分析目录**（与测试集同处），本次入库时统一改了路径解析，
> 计算逻辑一字未改。

---

## 2. 路径约定

脚本里不写任何个人机器的绝对路径。每个脚本开头都有一段 shim，按脚本位置反查仓库根，
并把工作目录切到数据目录（脚本内部用的是 `cv_design_audit/` 相对路径）：

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `FINTECH_ROOT` | 本文件上两级（仓库根） | 仓库根目录 |
| `LH024_DIR` | `$FINTECH_ROOT/experiments/LH024` | 数据目录（脚本会 `chdir` 到这里） |

不需要改任何源码即可换机器运行。

---

## 3. 运行前的两个前提

脚本只依赖 Python 标准科学栈（`numpy` / `pandas` / `scipy` / `pyarrow`），
但**输入数据不在版本库里**：

1. **面板与标签**：`data/raw/训练集.csv`（`.gitignore` 排除）。
2. **模型预测**：`outputs/long_horizon/LH003/<fold>/raw_predictions.parquet`（提供九个候选信号），
   以及面板缓存 `outputs/long_horizon/_probe_rc_<fold>.parquet`。整个 `outputs/long_horizon/`
   约 2.4 GB，同样被 `.gitignore` 排除。

> **脚本、判定口径、逐折结果全部可复现；中间数据需要先按 `data/raw` 与 LH003 的流程生成。**

底层依赖 `_probe_root_cause.py` / `_probe_dgtw.py` / `_probe_e_attr.py` / `_probe_factors.py`
在 `research/LH023/`（脚本已把该目录加入 `sys.path`）。

---

## 4. 标准命令

在项目根目录执行（输出落回 `experiments/LH024/cv_design_audit/`）。

```bash
# 方向一（A）层内排序标签
python research/LH024/probe_within_layer.py
python research/LH024/probe_within_ic.py

# 方向一（B）排序损失六臂（--mode base 仅做自检）
python research/LH024/probe_pairwise_rerank.py --mode split

# 方向二 新特征族（三步链）
python research/LH024/probe_new_features.py
python research/LH024/probe_pairwise_newfeat.py
python research/LH024/probe_newfeat_halves.py
python research/LH024/probe_newfeat_halves_stats.py

# 方向二 收窄（cumintra 单独立项的反证）
python research/LH024/probe_newfeat_cumintra.py
python research/LH024/probe_newfeat_nocum.py
python research/LH024/probe_newfeat_cumintra_stats.py

# blend 收尾（先跑 blend 落盘 s2_cache，再跑三个判定脚本）
python research/LH024/probe_newfeat_blend.py
python research/LH024/probe_newfeat_blend_windows.py
python research/LH024/probe_newfeat_blend_verdict.py
python research/LH024/probe_newfeat_blend_daily_test.py
python research/LH024/probe_blend_recheck.py
```

第四类（`probe_blend_recheck.py`、`*_stats.py`、`*_verdict.py`、`*_daily_test.py`、`*_windows.py`）
**只读已落盘的表、不重训**，只要 `experiments/LH024/cv_design_audit/` 里有表就能秒跑。

> 脚本开头会把 `OMP_*` / `MKL_*` 等线程数设为 1。多线程 BLAS 的浮点累加顺序不确定，
> 会让优化器收敛到略有差异的解，同一配置两次可差到 $3\times10^{-4}$——已超过这里要分辨的效应量。
> **不要去掉这段。**（仅 $w_{F1}$ 优化路径涉及 SLSQP；本目录多为 LGBM + band。）

---

## 5. 脚本与产物的对应关系

产物全部在 `experiments/LH024/cv_design_audit/`，**只含 2021-01-01 至 2024-12-31 的四个
walk-forward 折（部分实验扩成 8 个半折窗口）**，不含 2025 年以后的任何数据。

| 产物 | 生成脚本 | 报告的哪一节 |
|---|---|---|
| `within_layer_{grid,paired,crossfold,composition}.csv` | `probe_within_layer.py` | §2 层内标签 |
| `within_ic_spectrum.csv` | `probe_within_ic.py` | §2 层内 IC 谱 |
| `pairwise_rerank_{base,split,split_paired,split_dailypaired}.csv` 等 | `probe_pairwise_rerank.py` | §3 排序损失 |
| `new_feature_{screen,basket}.csv` | `probe_new_features.py` | §4 特征筛选 |
| `newfeat_stage2_*.csv` | `probe_pairwise_newfeat.py` | §4 受控对照 |
| `newfeat_halves{,_daily}.csv` | `probe_newfeat_halves.py` | §4 8 窗 |
| `newfeat_cumintra_*.csv`、`newfeat_nocum_*`、`newfeat_cumintra_paired.csv` | `probe_newfeat_cumintra.py` / `probe_newfeat_nocum.py` / `probe_newfeat_cumintra_stats.py` | §5 cumintra |
| `newfeat_blend_{scan,windows,daily,daymetrics}.csv` | `probe_newfeat_blend.py` | §6 blend |

---

## 6. 判定口径

与 LH023 同一套：**配对差 + 四折符号**、**逐日配对 $t$**（$n=681$ 天，8 窗串联）、
**留一窗**，以及两个口径的分离——

$$\text{含 }\Delta\alpha\text{ 口径} = 0.4\cdot\mathrm{IC}+0.3\cdot\alpha+0.3\cdot(1-T)$$

（即项目权威定义里的「可外推口径」，$\alpha$ **计入**、profile 不进判据；见 LH023 §1.3）

$$\text{下界口径} = 0.4\cdot\mathrm{IC}+0.3\cdot(1-T)$$

（不含 $\alpha$，是本轮为分离 $\alpha$ 影响而额外引入的更严下界。**不要把两个名字混用。**）

> 本项目的一条硬约束：**分析只允许用训练期 + CV 折（$\le$ 2024-12-31）的数据。**
> 本目录所有脚本都不读 2025 年以后的任何数据。

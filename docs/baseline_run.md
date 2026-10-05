# 基础 LightGBM：运行与复现

目的：打通训练 CSV → 清洗 → 特征 → 时间切分 → LightGBM → 2024 验证预测 → 官方评分 → 实验记录。当前只采用固定基础方案；首次全量结果完成后将记录在项目状态与本文件中。

## 运行命令

在仓库根目录、已配置的 Python 环境中运行：

```powershell
.\.venv\Scripts\python.exe -m src.models.baseline --fold fold2 --exp-id E000_baseline --owner A
```

默认配置为 `configs/project.yaml`。原始 CSV 按 `data/raw/README.md` 放置；入口检查文件是否与 `data/manifest.json` 一致。入口先跑 `E000_baseline__smoke`，固定取代码排序最前的 256 只股票、完整历史、10 轮训练；通过后跑正式全量实验。小规模数据仍有 2024 验证期，但其成绩仅用于运行检查。

重跑使用新编号，例如 `E001_baseline_repeat`。已有实验目录或 CSV 记录会拒绝覆盖。`--smoke-only` 可用于单独检查，但同一编号的后续全量执行也会被已有检查目录阻止，应使用新编号。

## 数据与清洗规则

- 读取日期为整数、股票代码为类别、原始数值与标签为 float64；按 `ts_code,trade_date` 排序。
- 键缺失、重复、非法日期、非正价格、价格区间矛盾、负成交量或非法涨跌停标记会停止运行。
- 原始文件内容保持不变。X 中无穷值转为缺失，记录转换数量；标签缺失与无穷值不用于训练。
- 缺失日行保留，既不前向/后向填充，也不删掉后重新压缩历史。真实零成交量、零成交额保留。
- 当日四个价格不完整或无有效正价格时，不纳入训练；验证行仍保留并输出固定预测 `0`。标签只用于训练筛选和评分，预测适用性只由当日价格确定。

## 固定 40 个特征

`c,o,h,l,v,a` 分别表示当日收盘、开盘、最高、最低、成交量、成交额。`MA_w`、`MIN_w`、`MAX_w` 均为含当日在内的过去 `w` 行窗口。每只股票独立计算；窗口要求 `w` 个有效值；分母非正时结果为缺失。最终特征存为 float32，原始计算与标签保留 float64 精度。

| 特征组 | 名称 / 窗口 | 数量 | 定义 |
| --- | --- | --- | --- |
| 历史收益 | `ret_{1,2,3,5,10,20,40,60}` | 8 | `c_t / c_(t-w) - 1`；严格按原始交易日行偏移 |
| 当日价格形态 | `intraday_ret, high_low_range, high_close, close_low, open_gap, close_position` | 6 | 依次为 `c/o-1, h/l-1, h/c-1, c/l-1, o/c_(t-1)-1, (c-l)/(h-l)` |
| 均价偏离 | `ma_bias_{5,10,20,60}` | 4 | `c / MA_w(c) - 1` |
| 收益波动 | `volatility_{5,10,20,60}` | 4 | `ret_1` 的窗口样本标准差，`ddof=1` |
| 成交量比例 | `vol_ratio_{5,10,20,60}` | 4 | `v / MA_w(v)` |
| 成交额比例 | `amount_ratio_{5,20}` | 2 | `a / MA_w(a)` |
| 成交量变化 | `vol_change_{1,5}` | 2 | `v_t / v_(t-w) - 1` |
| 历史价格位置 | `price_position_{20,60}` | 2 | `(c - MIN_w(c)) / (MAX_w(c) - MIN_w(c))` |
| 状态 | `flag_limit_up, flag_limit_down, zero_volume` | 3 | 原始两种标记；当日成交量为零则为 1，否则为 0 |
| 截面排名 | 上述 `ret_1,ret_5,ret_20,volatility_20,vol_ratio_20` 各加 `rank_` 前缀 | 5 | 按当日有效价格股票的非缺失值计算平均名次 / 有效数量；不使用标签筛选 |

历史不足或缺失值保留给 LightGBM 处理；截面排名不合并未来日期。特征入口拒绝目标列，检查通过修改未来 X 验证过去特征不变。

## 时间切分与模型

- 使用成员 B 的 `split_train_valid`：2018-01-02 至 2023-12-31 训练，2024-01-01 至 2024-12-31 验证。实际交易日起止来自数据。
- 特征在连续的 2018–2024 历史上计算后切分，2024 第一日能够使用 2023 历史。
- 训练最后交易日的标签在训练副本中屏蔽。全量 fold2 已审计应屏蔽 4,522 个非缺失跨界标签；原始验证标签保持不变，包括 2024 最后交易日的官方标签。
- 验证应覆盖 4,650 只股票、242 个交易日，共 1,125,300 个键；标签缺失行仍输出预测。
- LightGBM 原生接口、CPU、GBDT 回归、学习率 0.05、31 叶、最小叶子 200 条、L2=1、max_bin=255、seed=42、8 线程、200 轮；全量特征与样本，无随机抽样。启用 `deterministic` 和 `force_col_wise`，零值按真实值处理，NaN 交给模型。
- 2024 标签不参与选特征、调参或停止训练；没有验证集早停。全部参数以配置快照为准。

## 检查与评分

1. 运行开发检查：`.venv\Scripts\python.exe -m unittest discover -s tests -v`；15 项检查包括公式、边界、股票/日期隔离、预测覆盖、模型重载与评分口径。
2. 预测严格三列 `ts_code,trade_date,pred`，全部键唯一，与原始验证键完全匹配；预测不能缺失或为无穷值。落盘后再次读取核对。
3. 保存模型后重新加载，对实际全部有效验证行比较预测，最大绝对差应不超过 `1e-12`。
4. 通过成员 B 的评分适配计算全部 8 项官方指标，再把实际预测文件与原始验证标签交给根目录原始 `evaluate.py` 计算；逐项差异应不超过 `1e-10`。涨停、标签缺失和有效数量门槛均沿用官方口径。
5. 运行后重新计算原始训练文件和三个官方附件的 SHA-256，核对内容未变。
6. 全部通过后追加正式实验记录；失败时在 `run.json` 保存错误，已生成文件保留，不写成功实验行。

综合分仍为 `0.4 * ic_mean + 0.3 * annual_excess + 0.3 * (1 - mean_turnover)`。年化指标为官方规定的日均值乘 252；验证分不等于未来表现或正式测试分。

## 产物与追溯

| 位置（相对仓库根目录） | 内容 |
| --- | --- |
| `data/processed/<exp_id>/features.parquet` | 按键排序的完整 40 列特征和键，不含标签 |
| `outputs/models/<exp_id>/model.txt` | 可重新加载的 LightGBM 模型 |
| `outputs/models/<exp_id>/features.json` | 训练列名及顺序 |
| `outputs/models/<exp_id>/config.yaml` | 本次配置快照 |
| `outputs/models/<exp_id>/run.json` | 代码提交/是否有改动、环境、数据哈希、行数、屏蔽数量、耗时、检查和指标 |
| `outputs/predictions/<exp_id>/valid_2024.csv` | 完整 2024 验证预测（fold1 时文件名为 `valid_2023.csv`） |
| `outputs/metrics/<exp_id>/validation_labels.csv` | 原始验证标签与涨停标记，供本地评分复现 |
| `outputs/metrics/<exp_id>/metrics.json` | 8 项指标及有效天数 |
| `outputs/metrics/<exp_id>/daily_metrics.csv` | B 的逐日 IC、Top 收益、基准、超额与 Jaccard 换手表 |
| `outputs/metrics/<exp_id>/run.log` | 执行进度 |
| `experiments/experiment_log.csv` | 通过核对的全量实验共享记录 |

大文件由 `.gitignore` 排除；代码、配置、环境锁、说明和正式实验记录进入 Git。成员 B 从相同代码提交、相同配置和相同原始数据复现。模型融合、优化、正式测试预测和报告仍属后续阶段。

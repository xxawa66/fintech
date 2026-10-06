# 通用模型评分分析模板（model_analysis）

更新时间：2026-10-06（Asia/Shanghai）

每次模型/策略优化产出固定验证预测后，用 `src/evaluation/model_analysis.py`
一条命令完成三项官方指标（IC / Top 组合 / 换手）的**单独分析**与
**加权分析**（final_score 拆解与参照归因），产物可直接作比赛报告素材。

## 用法

```bash
.\.venv\Scripts\python.exe -m src.evaluation.model_analysis \
    --pred outputs/predictions/<exp>/valid_2024.csv \
    --labels outputs/metrics/<exp>/validation_labels.csv \
    --out-dir outputs/metrics/<exp>/analysis \
    --name <exp> \
    --ref-pred outputs/predictions/E001_baseline_repeat/valid_2024.csv \
    --ref-labels outputs/metrics/E001_baseline_repeat/validation_labels.csv
```

- `--ref-pred/--ref-labels`：参照实验（基线或 Champion，**必须与验证期同折**），
  提供后加权分析额外输出逐指标 Δ 与三项归因
  （`Δfinal = 0.4·ΔIC + 0.3·ΔExcess − 0.3·Δ换手`，三项合计恒等于 Δfinal）。
- 留仓带层默认叠加（`configs/project.yaml` 的 `evaluation.band.keep_q`，
  2026-10-06 锁定 0.1）；`--band-keep-q 0` 可关闭。
- 评分权重同样取自 `evaluation.weights`，加载时与官方口径断言一致。

## 输出（--out-dir）

| 文件 | 内容 |
| --- | --- |
| `analysis_summary.json` | 官方 8 项 + 三分项单独分析 + 加权拆解 + 参照归因（+ band 层）总表 |
| `REPORT.md` | 人类可读报告，可直接进比赛报告素材 |
| `ic_daily/monthly/quarterly.csv` | IC 逐日/月/季汇总 |
| `top_daily/monthly.csv`、`top_summary.json` | Top 组合逐日/月明细与汇总（复用 backtest 口径） |
| `turnover_daily.csv`、`turnover_diagnosis.json` | 换手逐日与来源诊断 |
| `band_summary.json` | 留仓带层指标与相对原始预测的 Δ（仅叠加时） |

## 分析口径（全部复用既有模块，不重新实现评分）

- 官方 8 项与逐日指标：`official_eval.py`（阈值 30/100、ddof=1、×252 与官方一致）；
- Top 组合：`backtest.py`（Top10/20、Bottom10、月度超额、最大回撤）；
- 留仓带：`turnover.py`；换手诊断的 Top 集合重建用官方**换手组**口径
  （剔除涨停、不要求标签非缺失、有效样本 < 100 重置前日集合）；
- 拆解恒等式 `0.4·IC + 0.3·Excess + 0.3·(1−换手)` 在每次分析时重新校验，
  与官方 `final_score` 不一致会直接断言失败。

## 已验证

- 51 项开发检查通过（`tests/test_model_analysis.py` 新增 11 项：拆解恒等式、
  三分项与官方指标一致性、换手诊断边界抖动识别、keep_q=1.0 自检、
  归因恒等式、端到端产物完整性）。
- 2024 折真实冒烟（E001_baseline_repeat）：官方 8 项与实验记录逐位一致；
  换手诊断与 notebook 04 第 7 节发表数字一致（换出 0.9479 / 新进 0.9482 /
  平均连续在榜 3.14 天 / 最长 12 天 / 累计最多 72 天 / 日均 |分位变化| 0.2503）；
  留仓带层 final 0.3491780142、换手 0.102134，与锁定记录一致。

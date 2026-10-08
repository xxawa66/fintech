"""LH001：长周期模型播种首日 Top 组（model/long-horizon 分支，未并入 main）。

背景：S003 锁定的 T030 方案 keep_q≈0.0023，留仓带下首日进入 Top 1/10 的股票
几乎全年不变——初始集合的质量直接决定超额项。本包在**完全相同的因果口径**下
把监督目标的 horizon 从 1 个交易日拉长到 h∈{5,10,20,30} 个交易日：

- 特征：沿用冻结的 40 个 V1 特征，全部只使用 t 日及以前的信息；
- 标签：``y_ret_{h}d = close(t+h)/close(t) - 1``，仅作训练监督目标，
  训练窗口末尾 h 个交易日的标签因使用验证期价格而置 NaN（1 日版边界规则的
  自然推广，见 ``src/evaluation/validation.py``）；
- 用法：t 日特征 → 一次性预测未来 h 个交易日的收益率；只替换验证期**首日**的
  pred，其余交易日 pred 不变，首日 Top 组由长周期预测选出，之后交由原留仓带
  机制（``src/evaluation/turnover.band_scores``，keep_q 沿用 T030）维持。
  预测阶段不使用任何未来信息。

评分一律复用官方口径适配器 ``src/evaluation/official_eval.evaluate_frame``。
"""

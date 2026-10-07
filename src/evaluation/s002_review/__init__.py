"""成员 B 对 S002（F025 融合 + band）的独立复核与参数重扫工具链。

模块之间是一次性的研究流水线，直接以脚本方式运行，不提供对外 API：

1. ``train_components``：独立重建每一折的 L1（LightGBM）与 R2（Ridge）组件，
   落盘到 ``tmp/review_s002_b/artifacts/<fold>/``；
2. ``phase2_grid``：稠密面板加速的权重 / ``keep_q`` / 平滑 × 留仓带网格，两折各跑一遍；
3. ``phase3_altblend``：排名融合 vs 当日 z-score 融合对照（含并列结构统计）；
4. ``summarize``：把两折网格汇总成候选阶梯与相对 A 锁定方案的逐项提分归因。

``fast_ops`` 是加速层：把长表摊成 (交易日 × 股票) 稠密数组并缓存截面排名，把每次调用
约 7.2s 的两次 pivot 摊销掉；变换逻辑照抄 ``src.evaluation.turnover``，启动时逐位自检。

重型中间产物（每折约 130MB 的组件预测 CSV）只落本地 ``tmp/``，不入 Git。
复核结论见 ``docs/model_research_S002_review.md``。
"""

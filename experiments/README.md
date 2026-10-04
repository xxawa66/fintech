# 实验记录

每次真实实验追加一行 `experiment_log.csv`。初始化时只有表头。

`git_commit` 记录代码版本，`config_path`、`features`、`params` 记录复现所需设置，`train_period` 和 `valid_period` 记录时间范围。指标列与官方评分器返回字段一致。大型预测与模型保存到 `outputs/`，在 `artifact_path` 记录相对路径。

# 量化比赛测试集评分脚本 (R版)
# 评价指标: Rank IC(40%) + Top组超额收益(30%) + 预测换手率(30%)

library(data.table)

evaluate <- function(submission_path, data_dir = '/home/quant6/data/比赛') {
  # 读取数据
  df_pred <- fread(submission_path)
  df_y <- fread(file.path(data_dir, '测试集_Y.csv'))
  df_x <- fread(file.path(data_dir, '测试集_X.csv'), select = c('ts_code', 'trade_date', 'flag_limit_up'))

  # 合并
  df <- merge(df_pred, df_y, by = c('ts_code', 'trade_date'))
  df <- merge(df, df_x, by = c('ts_code', 'trade_date'))
  cat(sprintf('合并后样本数: %d, 交易日数: %d\n', nrow(df), uniqueN(df$trade_date)))

  dates <- sort(unique(df$trade_date))

  # ========== 1. Rank IC (40%) ==========
  ic_list <- c()
  for (m_date in dates) {
    sub <- df[trade_date == m_date]
    valid <- sub[!is.na(y_ret_1d)]
    if (nrow(valid) < 30) next
    ic <- cor(valid$pred, valid$y_ret_1d, method = 'spearman')
    ic_list <- c(ic_list, ic)
  }

  ic_mean <- mean(ic_list)
  ic_std <- sd(ic_list)
  icir <- ic_mean / ic_std
  ic_positive_ratio <- mean(ic_list > 0)
  cat(sprintf('Rank IC: mean=%.6f, std=%.6f, ICIR=%.4f, IC>0占比=%.2f%%\n',
              ic_mean, ic_std, icir, ic_positive_ratio * 100))

  # ========== 2. Top组超额收益 (30%) ==========
  excess_list <- c()
  top1_ret_list <- c()
  for (m_date in dates) {
    sub <- df[trade_date == m_date]
    valid <- sub[flag_limit_up == 0 & !is.na(y_ret_1d)]
    if (nrow(valid) < 100) next
    setorder(valid, -pred)
    n_top <- max(nrow(valid) %/% 10, 1)
    top1_ret <- mean(valid$y_ret_1d[1:n_top])
    market_ret <- mean(valid$y_ret_1d)
    excess_list <- c(excess_list, top1_ret - market_ret)
    top1_ret_list <- c(top1_ret_list, top1_ret)
  }

  annual_excess <- mean(excess_list) * 252
  top1_annual_ret <- mean(top1_ret_list) * 252
  cat(sprintf('Top组超额收益: 年化=%.4f, Top1年化绝对收益=%.4f\n', annual_excess, top1_annual_ret))

  # ========== 3. 预测换手率 (30%) ==========
  turnover_list <- c()
  prev_set <- NULL
  for (m_date in dates) {
    sub <- df[trade_date == m_date]
    valid <- sub[flag_limit_up == 0]
    if (nrow(valid) < 100) {
      prev_set <- NULL
      next
    }
    setorder(valid, -pred)
    n_top <- max(nrow(valid) %/% 10, 1)
    curr_set <- valid$ts_code[1:n_top]

    if (!is.null(prev_set) && length(prev_set) > 0) {
      intersection <- length(intersect(curr_set, prev_set))
      union_size <- length(union(curr_set, prev_set))
      turnover <- 1.0 - intersection / union_size
      turnover_list <- c(turnover_list, turnover)
    }
    prev_set <- curr_set
  }

  mean_turnover <- mean(turnover_list)
  cat(sprintf('预测换手率: mean=%.4f, (1-turnover)=%.4f\n', mean_turnover, 1 - mean_turnover))

  # ========== 4. 综合评分 ==========
  final_score <- ic_mean * 0.4 + annual_excess * 0.3 + (1 - mean_turnover) * 0.3
  cat(sprintf('综合得分 = %.6f×0.4 + %.4f×0.3 + %.4f×0.3 = %.6f\n',
              ic_mean, annual_excess, 1 - mean_turnover, final_score))

  return(list(
    ic_mean = ic_mean,
    ic_std = ic_std,
    icir = icir,
    ic_positive_ratio = ic_positive_ratio,
    annual_excess = annual_excess,
    top1_annual_ret = top1_annual_ret,
    mean_turnover = mean_turnover,
    final_score = final_score
  ))
}

# 主函数
submission_path <- '/home/quant6/data/比赛/submission.csv'
result <- evaluate(submission_path)
cat('\n===== 评分结果 =====\n')
for (name in names(result)) {
  cat(sprintf('  %s: %.6f\n', name, result[[name]]))
}

# Script to perform all pairwise t-tests on filtered data


library(dplyr)
library(readr)
library(tibble)

# Data files
# datafile = "aggregated_results.csv"  # step 1 -- fine-tuning results
# datafile = "prediction_results.csv" # step 2 -- prediction results
# datafile = "fine_tune_results.csv" # step 3 -- domain adaptation results
datafile = "fine_tune_prediction_results.csv" # step 4 -- prediction using domain adapted models

# total number of samples
N = 10  # number of reps for each model
# N = 488    # number of validation samples in step 1
# N = 1000   # number of prediction samples in step 2 and 4
# N = 200    # number of validation samples in step 3

# parameters
learning_rates = c(0.0001, 0.0005)
dropouts = c(0.15, 0.2, 0.25)
unfreeze = 2
# metric to use
# metric = "accuracy"
metric = "f1_score"
metric_std = paste0(metric, "_std")

df <- read_csv(datafile)  

# Filter and subset
filtered_df <- df %>%
  filter(folder_unf == unfreeze, folder_lr %in% learning_rates, folder_dropout %in% dropouts) %>%
  mutate(label = paste0(optimiser, ", lr=", folder_lr, ", dropout=", folder_dropout)) %>%
  select(label, all_of(metric), all_of(metric_std))

# Define a function for pairwise t-test
compare_two_rows <- function(idx) {
  row1 <- filtered_df[idx[1], ]
  row2 <- filtered_df[idx[2], ]

  sim1 <- rnorm(N, mean = row1[[metric]], sd = row1[[metric_std]])
  sim2 <- rnorm(N, mean = row2[[metric]], sd = row2[[metric_std]])

  # Perform t-test 10 times and then take the mean
  # as we don't have the actual data at this point, it's best to run multiple times
  ttest_results <- replicate(10, {
    sim1 <- rnorm(N, mean = row1[[metric]], sd = row1[[metric_std]])
    sim2 <- rnorm(N, mean = row2[[metric]], sd = row2[[metric_std]])
    t.test(sim1, sim2)$p.value
  })
  # Calculate the mean p-value
  mean_p_value <- mean(ttest_results)

  tibble(
    comparison = paste(row1$label, "vs", row2$label),
    metric1 = sprintf("%.3f +- %.3f", row1[[metric]], row1[[metric_std]]),
    metric2 = sprintf("%.3f +- %.3f", row2[[metric]], row2[[metric_std]]),
    p_value = mean_p_value,
    significant = mean_p_value < 0.05
  )
}

# All pairwise comparisons
combinations <- combn(1:nrow(filtered_df), 2)
comparison_table <- apply(combinations, 2, compare_two_rows) %>% bind_rows()


# Output results
# print(filtered_df)
# print(comparison_table)

# Save the comparison table to a CSV file
write.csv(comparison_table, "pairwise_comparisons.csv", row.names = FALSE)

# Output the number of non-significant comparisons  
num_non_significant <- sum(!comparison_table$significant)
cat("Number of non-significant comparisons:", num_non_significant, "\n")
# Output the number of p-values less than 0.001
num_less_than_001 <- sum(comparison_table$p_value < 0.001)
cat("Number of p-values less than 0.001:", num_less_than_001, "\n")
# output the total number of comparisons
num_comparisons <- nrow(comparison_table)
cat("Total number of comparisons:", num_comparisons, "\n")
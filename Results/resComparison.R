n = 10

mean1 = 0.945
std1 = 0.011

mean2 = 0.936
std2 = 0.009

t_results <- t.test(rnorm(n, mean1, std1), rnorm(n, mean2, std2))

print(t_results)

print(t_results$p.value)
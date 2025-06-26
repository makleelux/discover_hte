# Data Documentation for the manuscript:
# 'Heterogeneity in effects of automated results feedback after online depression screening: 
# A secondary machine-learning based analysis of the DISCOVER trial

# Analysis code is in part adapted from the tutorial paper: 
# Sverdrup E, Petukhova M, Wager S. Estimating Treatment Effect Heterogeneity in Psychiatry: 
# A Review and Tutorial with Causal Forests. arXiv preprint arXiv:240901578; 2024.

library(grf)
# restore legacy option (update 11/24)
options(grf.legacy.seed = TRUE) 

library(haven)
library(tidyverse)
library(caTools)
library(ggplot2)
library(broom)
library(caret)

#### -------------- primary analysis -------------- ####

# seed init
seed = 1234
set.seed(seed)

Y <- 't0_t3_change_phq9'
W <- 'W_no_vs_nt_feedback' # this may be changed to examine comparison of control with targeted or any feedback

#### train test ####
# split
df_temp <- discover_sub_FAS_prepped_no_na %>% drop_na(W) %>% select(-condition, -trial_id)

sample <- caTools::sample.split(df_temp[[W]], SplitRatio = 0.5)

train <- subset(df_temp, sample == T)
test <- subset(df_temp, sample == F)

X.train <- train %>% select(-Y, -contains(c('W_')))
W.train <- train %>% pull(W)
Y.train <- train %>% pull(Y)

X.test <- test %>% select(-Y, -contains(c('W_')))
W.test <- test %>% pull(W)
Y.test <- test %>% pull(Y)

#### CATE ####
# Tau-forest based on training data
tau.forest <- causal_forest(X.train,
                            Y.train, W.train,
                            W.hat = mean(W.train), 
                            num.threads = 12, 
                            seed = seed)

# CATE for training data using out-of-bag prediction
tau.hat.oob <- predict(tau.forest, estimate.variance = TRUE)

# CATE for test data
tau.hat.test = predict(tau.forest, newdata = X.test)$predictions

# ATE for training data
average_treatment_effect(tau.forest)

#### HTE in CATE quartiles (adapted from Sverdrup et al., 2024) ####
n_groups = 4 
quartile = cut(tau.hat.test,
               quantile(tau.hat.test, seq(0, 1, by = 1 / n_groups)),
               labels = 1:n_groups,
               include.lowest = TRUE)
samples.by.quartile = split(seq_along(quartile), quartile)

# Eval-forest based on test data
eval.forest = causal_forest(X.test,
                            Y.test, W.test,
                            # only valid for 50:50
                            W.hat = mean(W.test), 
                            num.threads = 12, 
                            seed = seed)

# ATE for test data
average_treatment_effect(eval.forest)

# Doubly robust ATE across quartiles
ate.by.quartile = lapply(samples.by.quartile, function(samples) {
  average_treatment_effect(eval.forest, subset = samples)
})

# Plot group ATEs along with 95% confidence bars.
df.plot.ate = data.frame(
  matrix(unlist(ate.by.quartile), n_groups, byrow = TRUE, dimnames = list(NULL, c("estimate","std.err"))),
  group = 1:n_groups
)

ggplot(df.plot.ate, aes(x = group, y = estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = estimate - 1.96 * std.err, ymax = estimate + 1.96 * std.err, width = 0.1)) +
  xlab("Estimated CATE quantile") +
  ylab("Average treatment effect") +
  scale_y_continuous(limits = c(-10, 10)) +
  theme_bw() +
  geom_hline(yintercept = 0, lty = 'dotted')

#### TOC/AUTOC (adapted from Sverdrup et al., 2024) ####
# TOC curve
rate.cate = rank_average_treatment_effect(
  eval.forest,
  tau.hat.test,
  q = seq(0.1, 1, length.out = 100)
)
plot(rate.cate, ylab = 'Average treatment effect')

# AUTOC
p.autoc = 2 * pnorm(-abs(rate.cate$estimate / rate.cate$std.err))

# AUTOC CI
paste("AUTOC (95% CI):", round(rate.cate$estimate, 3), 
      "(", round(rate.cate$estimate - 1.96 * rate.cate$std.err, 3), ",", 
      round(rate.cate$estimate + 1.96 * rate.cate$std.err, 3), "), p = ", round(p.autoc, 4))

#### Calibration ####
# differential forest prediction reflects CATE - ATE and suggests HTE
calib = grf::test_calibration(tau.forest)  
calib

#### Important Variables ####
varimp = variable_importance(tau.forest)
ranked.variables = order(varimp, decreasing = TRUE)
top.varnames = colnames(X.test)[ranked.variables[1:4]]
print(top.varnames)

#### Covariate Profiles (adapted from Sverdrup et al., 2024) ####
low = samples.by.quartile[[1]]
high = samples.by.quartile[[n_groups]]

df.lo = data.frame(
  covariate.value = unlist(as.vector(X.test[low, top.varnames])),
  covariate.name = rep(top.varnames, each = length(low)),
  cate.estimates = "Low"
)
df.hi = data.frame(
  covariate.value = unlist(as.vector(X.test[high, top.varnames])),
  covariate.name = rep(top.varnames, each = length(high)),
  cate.estimates = "High"
)

df.plot.hist = rbind(df.lo, df.hi)

ggplot(df.plot.hist, aes(x = covariate.value, fill = cate.estimates)) +
  geom_histogram(alpha = .4, position = 'identity') +
  facet_wrap(~ covariate.name, scales = "free", ncol = 2) +
  theme_bw()

#### Best linear projections ####
blp_dat = X.train %>% select(all_of(top.varnames))
blp = best_linear_projection(
  tau.forest, 
  A = blp_dat) %>% 
  tidy() %>% 
  mutate(p.value = round(p.value, 3)) %>% 
  mutate_at(c('estimate', 'std.error', 'statistic'), ~round(.x, 2))
blp


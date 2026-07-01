# ------------------------------------------------------------
# GLMM-style analysis using emmeans (adapted for sample data)
# ------------------------------------------------------------

rm(list = ls())
cat("\014")

# -----------------------------
# 0. Packages
# -----------------------------
library(emmeans)
library(dplyr)
library(car)
library(rstudioapi)

# NOTE: GLM (Gamma) instead of lme4::glmer for cross-platform consistency

# -----------------------------
# 1. Generate equivalent sample data
# -----------------------------

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
n <- 400
df <- read.csv("emmeans_tutorial_data.csv")

# Convert factors
df$treatment <- factor(df$treatment,
                       levels = c("control", "drugA", "drugB"))
df$load <- factor(df$load,
                  levels = c("high_load", "low_load"))

# -----------------------------
# 2. Fit GLM (Gamma log-link)
# -----------------------------
fit1 <- glm(
  y ~ treatment * load + age,
  data = df,
  family = Gamma(link = "log")
)

# -----------------------------
# 3. Model summary
# -----------------------------
summary(fit1)

# NOTE:
# GLM does NOT support Type II ANOVA properly like lmer models
# So we skip car::Anova here (or use Type I if needed)

# -----------------------------
# 4. Estimated marginal means
# -----------------------------

# Main effect: treatment
emm_treatment <- emmeans(
  fit1,
  pairwise ~ treatment,
  type = "response",
  adjust = "bonferroni"
)

# Interaction: treatment within load
emm_treatment_by_load <- emmeans(
  fit1,
  pairwise ~ treatment | load,
  type = "response",
  adjust = "bonferroni"
)

# Reverse slicing: load within treatment
emm_load_by_treatment <- emmeans(
  fit1,
  pairwise ~ load | treatment,
  type = "response",
  adjust = "bonferroni"
)

# -----------------------------
# 5. Inspect outputs
# -----------------------------
emm_treatment$emmeans
emm_treatment$contrasts

emm_treatment_by_load$emmeans
emm_treatment_by_load$contrasts

# -----------------------------
# 6. Save outputs (R equivalent of Python pickle)
# -----------------------------
save(
  fit1,
  emm_treatment,
  emm_treatment_by_load,
  emm_load_by_treatment,
  file = "posthoc_results.RData"
)

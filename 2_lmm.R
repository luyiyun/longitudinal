library(lme4)      # 用于混合模型（后续使用）和加载数据
library(ggplot2)   # 用于数据可视化
library(dplyr)     # 用于数据处理
library(broom)     # 用于整理模型输出

data(sleepstudy)

model_intercept <- lmer(Reaction ~ Days + (1 | Subject), data = sleepstudy)
summary(model_intercept)

model_intercept <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy)
summary(model_intercept)
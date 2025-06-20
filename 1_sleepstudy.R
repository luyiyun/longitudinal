# renv::install("languageserver")
# renv::install(c("tidyverse", "lme4"))

library(lme4)      # 用于混合模型（后续使用）和加载数据
library(ggplot2)   # 用于数据可视化
library(dplyr)     # 用于数据处理
library(broom)     # 用于整理模型输出

data(sleepstudy)
head(sleepstudy)

# 首先，通过“意大利面图”(Spaghetti Plot)直观感受数据
ggplot(sleepstudy, aes(x = Days, y = Reaction, group = Subject)) +
  geom_line(alpha = 0.5) +  # 绘制每个受试者的轨迹线
  geom_point() +
  labs(title = "18名受试者反应时随睡眠剥夺天数的变化",
       x = "睡眠剥夺天数",
       y = "平均反应时 (ms)") +
  theme_minimal()
ggsave("./results/sleepstudy_spaghetti_plot.png")


# 将所有数据点合并，进行一个普通最小二乘回归(OLS)
naive_model <- lm(Reaction ~ Days, data = sleepstudy)
summary(naive_model)


# 使用dplyr和broom为每个受试者拟合独立的线性模型
individual_models <- sleepstudy %>%
  group_by(Subject) %>%
  do(tidy(lm(Reaction ~ Days, data = .))) %>%
  ungroup()
head(individual_models)

# 可视化每个人的回归线
ggplot(sleepstudy, aes(x = Days, y = Reaction, group = Subject)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, size = 0.5) + # 为每组添加线性拟合线
  labs(title = "为每个受试者单独拟合的回归线", x = "睡眠剥夺天数", y = "平均反应时 (ms)") +
  theme_minimal()
ggsave("./results/sleepstudy_individual_models.png")

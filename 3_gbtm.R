# install.packages("lcmm", repos = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/")

# 加载包
library(lcmm)
library(ggplot2)
library(tidyr)

# 加载并查看数据
data(paquid)
head(paquid)

# 初步可视化：绘制每个人的MMSE随年龄变化的“意大利面图”
ggplot(paquid, aes(x = age, y = MMSE, group = ID)) +
  geom_line(alpha = 0.8, color = "steelblue") +
  labs(
    title = "Paquid研究中个体MMSE得分随年龄的变化轨迹",
    x = "年龄 (岁)",
    y = "MMSE 得分"
  ) +
  theme_minimal()
ggsave("./results/paquid_mmseline.png", width = 8, height = 6)


# 标准化一下年龄，防止拟合时数值过大导致收敛问题
paquid$age65 <- (paquid$age - 65) / 10

# 拟合K=1的模型 (即一个标准的线性混合模型，但没有随机效应)，需要其作为后续模型的参数估计的初始值
gbtm1 <- hlme(MMSE ~ age65 + I(age65^2), subject = "ID", ng = 1, data = paquid)
# 拟合K=2的GBTM模型, 默认random = ~ -1表示不包含随机效应，也就是GBTM模型
gbtm2 <- hlme(
  MMSE ~ age65 + I(age65^2),
  mixture = ~ age65 + I(age65^2),
  subject = "ID", ng = 2, data = paquid, B = gbtm1,
)
# 拟合K=3的GBTM模型
gbtm3 <- hlme(
  MMSE ~ age65 + I(age65^2),
  mixture = ~ age65 + I(age65^2),
  subject = "ID", ng = 3, data = paquid, B = gbtm1
)
# 拟合K=4的GBTM模型
gbtm4 <- hlme(
  MMSE ~ age65 + I(age65^2),
  mixture = ~ age65 + I(age65^2),
  subject = "ID", ng = 4, data = paquid, B = gbtm1
)

# 比较模型
summarytable(gbtm1, gbtm2, gbtm3, gbtm4)
# summarytable(gbtm1, gbtm2, gbtm3, gbtm4, which = c("G", "loglik", "conv", "npm", "AIC", "BIC", "SABIC", "entropy", "ICL", "%class"))

png("./results/paquid_summaryplot.png", width = 800, height = 200)
summaryplot(gbtm1, gbtm2, gbtm3, gbtm4)
dev.off()

# 查看K=4模型的详细摘要
summary(gbtm4)

postprob(gbtm4) # 后验概率


png("./results/paquid_gbtm4.png", width = 800, height = 600)
plot(gbtm4, which = "fit", var.time = "age", marg = FALSE, shades = TRUE)
dev.off()


data_pred <- data.frame(age = seq(65, 95, length.out = 50))
data_pred$age65 <- (data_pred$age - 65) / 10
pred_y <- predictY(gbtm4, newdata = data_pred, var.time = "age", draws = TRUE)
png("./results/paquid_gbtm4_pred.png", width = 800, height = 600)
plot(pred_y,
  col = c("deeppink", "deepskyblue", "darkgreen", "darkorange"),
  ylab = "normMMSE", main = "Predicted trajectories for normMMSE",
  shades = TRUE, lty = 1
)
dev.off()

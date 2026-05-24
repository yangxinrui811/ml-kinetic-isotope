import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import numpy as np
import matplotlib.pyplot as plt
import shap

# 数据集定义
df = pd.read_csv("G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/2.数据整理-新/data.csv", sep=',')

# 数据预处理
scaler = StandardScaler()
X = df[['input1', 'input2', 'input3', 'input4']]
X_scaled = scaler.fit_transform(X)
y = df['target']

# 留出法分离训练集和预测集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 初始化LightGBM模型（假设已经优化好超参数）
model = LGBMRegressor(
    n_estimators=100,  # 示例参数，根据实际优化结果替换
    max_depth=3,
    learning_rate=0.731482,
    num_leaves=10,
    min_child_samples=5,
    subsample=0.9,
    colsample_bytree=1.0,
    reg_alpha=1e-10,
    reg_lambda=1e-10,
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 使用SHAP进行解释
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 绘制beeswarm图
shap.summary_plot(shap_values, X_train, plot_type="dot", feature_names=['T', 'k', 'KIE', 'ΔE'])

# 如果需要显示图形（在某些环境中可能需要）
plt.show()
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
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

# 使用已经优化好的超参数初始化CatBoost模型
model = CatBoostRegressor(
    n_estimators=100,  # 示例最佳参数，需根据实际情况替换
    max_depth=3,
    learning_rate=0.751409,
    l2_leaf_reg=0.001,
    subsample=1.0,
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 使用SHAP进行解释
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 绘制beeswarm图
shap.summary_plot(shap_values, X_train, plot_type="dot", feature_names=['T', 'k', 'KIE', 'ΔE'])

# 如果需要显示图形界面（在某些环境如Jupyter Notebook中不需要显式调用show）
plt.show()
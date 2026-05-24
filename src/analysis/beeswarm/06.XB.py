import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
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

# 使用已经优化好的超参数初始化XGBoost模型
model = XGBRegressor(
    n_estimators=100,  # 示例参数，实际使用优化后的参数
    max_depth=3,      # 示例参数，实际使用优化后的参数
    learning_rate=0.4750865, # 示例参数，实际使用优化后的参数
    min_child_weight=0.4970553, # 示例参数，实际使用优化后的参数
    subsample=1.0,    # 示例参数，实际使用优化后的参数
    colsample_bytree=0.93367401, # 示例参数，实际使用优化后的参数
    gamma=1e-10,        # 示例参数，实际使用优化后的参数
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 使用SHAP进行解释
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 绘制beeswarm图
shap.summary_plot(shap_values, X_train, plot_type="dot", feature_names=['T', 'k', 'KIE', 'ΔE'])

# 显示图形（通常shap.summary_plot会直接显示图形，所以不需要额外的plt.show()）
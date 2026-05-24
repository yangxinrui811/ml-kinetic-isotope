import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import shap

# 数据集定义
df = pd.read_csv("G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/2.数据整理-新/data.csv", sep=',')  # 假设CSV文件使用逗号分隔

# 提取数据
data = {
    'input1': df['input1'].tolist(),
    'input2': df['input2'].tolist(),
    'input3': df['input3'].tolist(),
    'input4': df['input4'].tolist(),
    'target': df['target'].tolist()
}

df = pd.DataFrame(data)

# 数据预处理
scaler = StandardScaler()
X = df[['input1', 'input2', 'input3', 'input4']]
X_scaled = scaler.fit_transform(X)
y = df['target']

# 数据进行留出法，分类为训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=24)

# 使用已经优化好的超参数创建岭回归模型
best_alpha = 1.06915939e-06  # 假设这是你已经优化好的alpha值
regressor = Ridge(alpha=best_alpha)
regressor.fit(X_train, y_train)

# 使用模型进行预测
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# 评估（可选，如果需要可以保留）
# ...

# SHAP分析
explainer = shap.Explainer(regressor, X_train)
shap_values = explainer(X_train)

# 绘制beeswarm图
shap.summary_plot(shap_values, X_train, plot_type="dot", feature_names=['T', 'k', 'KIE', 'ΔE'])

# 如果需要保存图像，可以使用以下代码（需要安装pillow库）
# plt.savefig('shap_beeswarm_plot.png')

plt.show()
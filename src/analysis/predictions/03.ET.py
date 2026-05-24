import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import shap  # 导入shap库

# 数据集定义
df = pd.read_csv("G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/2.数据整理-新/data.csv", sep=',')

# 数据预处理
scaler = StandardScaler()
X = df[['input1', 'input2', 'input3', 'input4']]
X_scaled = scaler.fit_transform(X)
y = df['target']

# 留出法分离训练集和预测集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 初始化极端随机树模型（使用已优化的超参数）
model = ExtraTreesRegressor(
    n_estimators=10,  # 示例最佳参数，根据实际优化结果调整
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 使用SHAP进行解释
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)  # 计算训练集的SHAP值

# 绘制beeswarm图
shap.summary_plot(shap_values, X_train, plot_type="dot", feature_names=['T', 'k', 'KIE', 'ΔE'])

# 如果需要保存图像，可以使用以下代码（需要安装pillow库）
# plt.savefig('shap_beeswarm_plot.png')

plt.show()
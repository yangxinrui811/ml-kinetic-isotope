import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# 初始化随机森林模型（假设已经优化好超参数）
model = RandomForestRegressor(
    n_estimators=85,  # 示例最佳参数
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 使用SHAP进行解释
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# 绘制beeswarm图
shap.summary_plot(shap_values, X_train, plot_type="dot", feature_names=['T', 'k', 'KIE', 'ΔE'])

# 如果需要保存图像，可以使用以下代码（需要安装Pillow库）
# plt.savefig('shap_beeswarm_plot.png')

# 显示图形（通常summary_plot已经调用plt.show()，所以这里不需要再次调用）
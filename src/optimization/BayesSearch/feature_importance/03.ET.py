import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

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

# 特征重要性分析
feature_importances = model.feature_importances_
feature_names = ['T', 'k', 'KIE', 'ΔE']  # 假设这是你的特征名称，与数据集中的列名对应

# 打印特征重要性
for name, importance in zip(feature_names, feature_importances):
    print(f"Feature {name}: {importance:.4f}")

# 设置全局字体大小并绘制等高线图
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# 可视化特征重要性
plt.figure(figsize=(10, 7))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from ExtraTreesRegressor')
plt.gca().invert_yaxis()  # 反转y轴，使重要性较高的特征在上面
plt.show()
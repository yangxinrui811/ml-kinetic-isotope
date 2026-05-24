import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 数据集定义
df = pd.read_csv("C:/Users/FD/Desktop/测试/data.csv", sep=',')  # 假设CSV文件使用逗号分隔

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

# 使用偏最小二乘回归模型，并通过交叉验证找到最佳参数
param_grid = {
    'n_components': [4],  # 组件的数量，通常小于或等于X中的变量数
    'max_iter': [20]  # 最大迭代次数，可以根据需要调整
}

grid_search = GridSearchCV(PLSRegression(), param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 输出最佳参数和对应的性能
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {-grid_search.best_score_}")

# 使用最佳参数进行预测
best_regressor = grid_search.best_estimator_
y_train_pred = best_regressor.predict(X_train)  # 预测训练集
y_test_pred = best_regressor.predict(X_test)  # 预测测试集

# 评估训练集
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
 
# 评估测试集
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
 
# 打印训练集和测试集的误差
print("Training Set - Mean Squared Error (MSE):", mse_train)
print("Training Set - Root Mean Squared Error (RMSE):", rmse_train)
print("Training Set - Mean Absolute Error (MAE):", mae_train)
print("Training Set - R^2 Score:", r2_train)
 
print("\nTest Set - Mean Squared Error (MSE):", mse_test)
print("Test Set - Root Mean Squared Error (RMSE):", rmse_test)
print("Test Set - Mean Absolute Error (MAE):", mae_test)
print("Test Set - R^2 Score:", r2_test)

# 生成科研绘图
plt.figure(figsize=(10, 6))  # 设置图形大小

# 绘制训练集的散点图
plt.scatter(y_train, y_train_pred, color='white', marker='o', edgecolor='green', alpha=1, label='Training Set')

# 绘制测试集的散点图
plt.scatter(y_test, y_test_pred, color='white', marker='o', edgecolor='red', alpha=1, label='Test Set')

# 绘制y=x的参考线
plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
         [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 'k--', lw=2)

# 添加标题和标签
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()  # 添加图例

# 显示网格
plt.grid(True)

# 显示图形
plt.show()
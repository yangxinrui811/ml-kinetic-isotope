import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import numpy as np  
import matplotlib.pyplot as plt  

# 数据集定义  
df = pd.read_csv("C:/Users/FD/Desktop/测试/data.csv", sep=',')  # 假设CSV文件使用逗号分隔  

# 数据预处理  
scaler = StandardScaler()  
X = df[['input1', 'input2', 'input3', 'input4']]  
X_scaled = scaler.fit_transform(X)  
y = df['target']  

# 分割数据集为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)  

# 使用神经网络回归模型，并通过交叉验证找到最佳参数  
param_grid = {  
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  
    'activation': ['tanh', 'relu'],  
    'solver': ['sgd', 'adam'],  
    'alpha': [0.0001, 0.05],  
    'learning_rate': ['constant', 'adaptive'],  
}  
grid_search = GridSearchCV(MLPRegressor(random_state=42, max_iter=1000), param_grid, cv=10, scoring='neg_mean_squared_error')  
grid_search.fit(X_train, y_train)  

# 输出最佳参数和最佳得分
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", np.sqrt(-grid_search.best_score_))

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

# 输出训练集和预测集的平均评分
print(f"Training Set Scores: MSE={mse_train}, RMSE={rmse_train}, MAE={mae_train}, R²={r2_train}")
print(f"Test Set Scores: MSE={mse_test}, RMSE={rmse_test}, MAE={mae_test}, R²={r2_test}")

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
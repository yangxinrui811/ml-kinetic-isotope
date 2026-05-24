import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.svm import SVR  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import numpy as np  # 用于计算RMSE  
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot用于绘图
  
# 数据集定义（与原代码相同）  
df = pd.read_csv("C:/Users/FD/Desktop/测试/data.csv", sep=',')  # 假设CSV文件使用逗号分隔  
  
data = {  
    'input1': df['input1'].tolist(),  
    'input2': df['input2'].tolist(),  
    'input3': df['input3'].tolist(),  
    'input4': df['input4'].tolist(),  
    'target': df['target'].tolist()  
}  
  
df = pd.DataFrame(data)  # 定义df，转换为二维的、大小可变的、有标签的数据结构  
  
# 数据预处理  
scaler = StandardScaler()  
X = df[['input1', 'input2', 'input3', 'input4']]  
X_scaled = scaler.fit_transform(X)  
y = df['target']  
  
# 数据进行留出法，分类为训练集与测试集  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  
  
# 使用支持向量机回归模型，并通过交叉验证找到最佳参数  
param_grid = {  
    'C': [0.1, 1, 10, 100],  
    'gamma': [1, 0.1, 0.01, 0.001],  
    'kernel': ['rbf', 'linear']  
}  
  
grid_search = GridSearchCV(SVR(), param_grid, cv=10, scoring='neg_mean_squared_error')  
grid_search.fit(X_train, y_train)  
  
# 输出最佳参数和对应的性能  
print(f"Best parameters found: {grid_search.best_params_}")  
print(f"Best cross-validation score: {-grid_search.best_score_}")  
  
# 使用最佳参数进行预测  
best_regressor = grid_search.best_estimator_  
y_pred = best_regressor.predict(X_test)  
  
# 评估模型  
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)  # 计算RMSE  
mae = mean_absolute_error(y_test, y_pred)  # 计算MAE  
r2 = r2_score(y_test, y_pred)  # 计算R2  
  
print(f"Mean Squared Error (MSE): {mse}")  
print(f"Root Mean Squared Error (RMSE): {rmse}")  
print(f"Mean Absolute Error (MAE): {mae}")  
print(f"R^2 Score: {r2}")

# 生成科研绘图  
plt.figure(figsize=(10, 6))  # 设置图形大小  
plt.scatter(y_test, y_pred, color='blue', marker='o', edgecolor='w', alpha=0.7)  # 绘制散点图  
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 绘制y=x的参考线  
  
# 添加标题和标签  
plt.title('Predicted vs Actual Values')  
plt.xlabel('Actual Values')  
plt.ylabel('Predicted Values')  
  
# 显示网格  
plt.grid(True)  
  
# 显示图形  
plt.show()
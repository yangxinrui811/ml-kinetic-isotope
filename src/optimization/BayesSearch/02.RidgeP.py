import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real, Integer

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

# 定义贝叶斯优化搜索空间
param_space = {
    'alpha': Real(1e-6, 1e2, 'log-uniform'),  # 岭回归的正则化强度
}

# 使用贝叶斯优化进行超参数搜索
bayes_search = BayesSearchCV(
    Ridge(),
    param_space,
    n_iter=100,  # 迭代次数，可以根据需要调整
    cv=10,
    scoring='neg_mean_squared_error',
    random_state=42
)

bayes_search.fit(X_train, y_train)

# 输出最佳参数和对应的性能
print(f"Best parameters found: {bayes_search.best_params_}")
print(f"Best cross-validation score: {-bayes_search.best_score_}")

# 使用最佳参数进行预测
best_regressor = bayes_search.best_estimator_
y_train_pred = best_regressor.predict(X_train)  # 预测训练集
y_test_pred = best_regressor.predict(X_test)  # 预测测试集

# 评估训练集
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# 计算训练集的平均偏差（自定义）
avg_deviation_train = (10 ** rmse_train) - 1
avg_deviation_train_percent = avg_deviation_train * 100  # 转换为百分比

# 评估测试集
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# 计算测试集的平均偏差（自定义）
avg_deviation_test = (10 ** rmse_test) - 1
avg_deviation_test_percent = avg_deviation_test * 100  # 转换为百分比

# 打印训练集误差，保留小数点后四位
print("Training Set - Mean Squared Error (MSE): {:.4f}".format(mse_train))
print("Training Set - Root Mean Squared Error (RMSE): {:.4f}".format(rmse_train))
print("Training Set - Mean Absolute Error (MAE): {:.4f}".format(mae_train))
print("Training Set - R^2 Score: {:.4f}".format(r2_train))
print("Training Set - Deviation: {:.4f}%".format(avg_deviation_train_percent))
 
# 打印测试集误差，保留小数点后四位
print("\nTest Set - Mean Squared Error (MSE): {:.4f}".format(mse_test))
print("Test Set - Root Mean Squared Error (RMSE): {:.4f}".format(rmse_test))
print("Test Set - Mean Absolute Error (MAE): {:.4f}".format(mae_test))
print("Test Set - R^2 Score: {:.4f}".format(r2_test))
print("Test Set - Deviation: {:.4f}%".format(avg_deviation_test_percent))

# 生成科研绘图
plt.figure(figsize=(8, 6))  # 设置图形大小

# 绘制训练集的散点图
plt.scatter(y_train, y_train_pred, color='white', marker='o', edgecolor='green', alpha=1, label='Training Set')

# 绘制测试集的散点图
plt.scatter(y_test, y_test_pred, color='white', marker='o', edgecolor='red', alpha=1, label='Test Set')

# 绘制y=x的参考线
plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
         [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 'k--', lw=2)

# 添加标题和标签
plt.xlabel('Actual Values', fontsize=15)
plt.ylabel('Predicted Values', fontsize=15)
plt.legend(fontsize=15)  # 添加图例

# 显示网格
plt.grid(True)

# 显示图形
plt.show()
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
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

# 初始化XGBoost模型
model = XGBRegressor(random_state=42)

# 定义贝叶斯优化搜索空间
param_space = {
    'n_estimators': Integer(5, 100),
    'max_depth': Integer(1, 3),
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'min_child_weight': Real(1e-3, 10, 'log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(1e-10, 10, 'log-uniform')
}

# 使用贝叶斯搜索和十折交叉验证
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, cv=10, n_iter=100, scoring='neg_mean_squared_error', n_jobs=1)
bayes_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print("Best parameters found: ", bayes_search.best_params_)
print("Best cross-validation score: ", np.sqrt(-bayes_search.best_score_))

# 使用最佳参数进行预测
best_regressor = bayes_search.best_estimator_
y_train_pred = best_regressor.predict(X_train)
y_test_pred = best_regressor.predict(X_test)

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
print("Training Set - MSE: {:.4f}".format(mse_train))
print("Training Set - RMSE: {:.4f}".format(rmse_train))
print("Training Set - MAE: {:.4f}".format(mae_train))
print("Training Set - R^2: {:.4f}".format(r2_train))
print("Training Set - Deviation: {:.4f}%".format(avg_deviation_train_percent))

# 打印测试集误差，保留小数点后四位
print("\nTest Set - MSE: {:.4f}".format(mse_test))
print("Test Set - RMSE: {:.4f}".format(rmse_test))
print("Test Set - MAE: {:.4f}".format(mae_test))
print("Test Set - R^2: {:.4f}".format(r2_test))
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
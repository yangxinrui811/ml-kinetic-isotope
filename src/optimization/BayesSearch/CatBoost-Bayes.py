import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from importlib_metadata import metadata, version
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real
 
# 数据集定义
df = pd.read_csv("C:/Users/FD/Desktop/测试/data.csv", sep=',')  # 假设CSV文件使用逗号分隔
 
# 数据预处理
scaler = StandardScaler()
X = df[['input1', 'input2', 'input3', 'input4']]
X_scaled = scaler.fit_transform(X)
y = df['target']
 
# 交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
 
# 定义CatBoost模型，使用固定的参数除了学习率
catboost_reg = CatBoostRegressor(iterations=100, depth=3, random_state=42, verbose=False)
 
# 定义贝叶斯优化的参数空间
param_space = {
    'learning_rate': Real(0.01, 1.0, 'log-uniform')
}
 
# 使用BayesSearchCV进行优化
bayes_cv = BayesSearchCV(
    catboost_reg,
    param_space,
    n_iter=32,  # 优化迭代的次数
    cv=kf,      # 交叉验证策略
    scoring='neg_mean_squared_error',  # 评分标准
    n_jobs=-1,  # 使用所有CPU核心
    random_state=42
)
 
# 执行优化
result = bayes_cv.fit(X_scaled, y)
 
# 输出最佳参数和最佳得分
print(f"Best parameters: {result.best_params_}")
print(f"Best negative MSE score: {result.best_score_}")
 
# 使用最佳模型进行预测和评估
best_model = result.best_estimator_
fold_scores = []
 
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
 
    # 训练最佳模型
    best_model.fit(X_train, y_train)
 
    # 评估模型
    y_pred = best_model.predict(X_test)
 
    # 计算并记录当前折的评分
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    fold_scores.append({
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })
 
# 输出交叉验证的平均评分
average_scores = {
    'MSE': np.mean([s['MSE'] for s in fold_scores]),
    'RMSE': np.mean([s['RMSE'] for s in fold_scores]),
    'MAE': np.mean([s['MAE'] for s in fold_scores]),
    'R²': np.mean([s['R²'] for s in fold_scores])
}
print(f"Cross-Validation Average Scores: {average_scores}")
 
# 收集所有折的预测值和实际值进行绘图
all_y_test = np.concatenate([y[test_index] for train_index, test_index in kf.split(X_scaled)])
all_y_pred = np.concatenate([best_model.predict(X_scaled[test_index]) for train_index, test_index in kf.split(X_scaled)])
 
# 生成科研绘图
plt.figure(figsize=(10, 6))
plt.scatter(all_y_test, all_y_pred, color='blue', marker='o', edgecolor='w', alpha=0.7)
plt.plot([all_y_test.min(), all_y_test.max()], [all_y_test.min(), all_y_test.max()], 'k--', lw=2)
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()
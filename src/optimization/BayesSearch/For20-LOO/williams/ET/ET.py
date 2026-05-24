import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 数据集定义
df = pd.read_csv("D:/BaiduSyncdisk/0.我的工作/5.隧穿相图/2.数据整理-新/data.csv", sep=',')

# 数据预处理
scaler = StandardScaler()
X = df[['input1', 'input2', 'input3', 'input4']]
X_scaled = scaler.fit_transform(X)
y = df['target']

# 使用最优参数初始化极端随机树模型
optimal_params = {
    'n_estimators': 10,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42
}
model = ExtraTreesRegressor(**optimal_params)

# 分割数据为20折
k_folds = 20
fold_size = len(X_scaled) // k_folds

# 初始化DataFrame来存储结果
results_df = pd.DataFrame(columns=[
    'Fold', 'Set', 'MSE', 'RMSE', 'MAE', 'R2', 'Deviation (%)', 'Standardized Residual', 'Leverage'
])

# 进行20次训练和测试
for i in range(k_folds):
    # 确定测试集和训练集索引
    test_index = slice(i * fold_size, (i + 1) * fold_size)
    train_index = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, len(X_scaled)))
    
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算训练集的残差
    train_residuals = y_train - y_train_pred

    # 正确计算杠杆值
    XTX_inv = np.linalg.inv(X_train.T @ X_train)
    H = X_train @ (XTX_inv @ X_train.T)
    train_leverage = np.diag(H)

    train_standardized_residuals = train_residuals / np.std(train_residuals)
    
    # 计算测试集的残差（实际应用中可能不需要测试集的杠杆值）
    test_residuals = y_test - y_test_pred
    test_standardized_residuals = test_residuals / np.std(test_residuals)  # 假设用训练集的残差标准差标准化

    # 评估训练集（省略，与原始代码相同）
    # ...

    # 创建临时DataFrame来存储当前折的结果（仅展示训练集的应用域分析）
    train_results_df = pd.DataFrame({
        'Fold': [i + 1] * len(train_standardized_residuals),
        'Set': ['Training'] * len(train_standardized_residuals),
        'MSE': [np.mean(mean_squared_error(y_train, y_train_pred))] * len(train_standardized_residuals),  # 示例填充
        'RMSE': [np.sqrt(np.mean(mean_squared_error(y_train, y_train_pred)))] * len(train_standardized_residuals),
        'MAE': [np.mean(mean_absolute_error(y_train, y_train_pred))] * len(train_standardized_residuals),
        'R2': [r2_score(y_train, y_train_pred)] * len(train_standardized_residuals),
        'Deviation (%)': [((10 ** np.sqrt(np.mean(mean_squared_error(y_train, y_train_pred)))) - 1) * 100] * len(train_standardized_residuals),
        'Standardized Residual': train_standardized_residuals,
        'Leverage': train_leverage
    })
    
    # 使用pd.concat合并DataFrame
    results_df = pd.concat([results_df, train_results_df], ignore_index=True)

# 绘制Williams图
plt.figure(figsize=(10, 6))
plt.scatter(results_df[results_df['Set'] == 'Training']['Leverage'], results_df[results_df['Set'] == 'Training']['Standardized Residual'], alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=np.mean(results_df[results_df['Set'] == 'Training']['Leverage']) + 2 * np.std(results_df[results_df['Set'] == 'Training']['Leverage']), color='g', linestyle='--')
plt.xlabel('Leverage')
plt.ylabel('Standardized Residual')
plt.title('Williams Plot')
plt.show()
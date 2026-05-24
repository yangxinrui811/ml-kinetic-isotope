import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
 
# 数据集定义
df = pd.read_csv("G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/2.数据整理-新/data.csv", sep=',')
 
# 数据预处理
scaler = StandardScaler()
X = df[['input1', 'input2', 'input3', 'input4']]
X_scaled = scaler.fit_transform(X)
y = df['target']
 
# 使用最优参数初始化CatBoost模型
optimal_params = {
    'n_estimators': 100,
    'learning_rate': 0.751409,
    'l2_leaf_reg': 0.001,
    'depth': 3,
    'subsample': 1.0,
    'random_state': 42,
    'verbose': 0
}
model = CatBoostRegressor(**optimal_params)
 
# 分割数据为20折
k_folds = 20
fold_size = len(X_scaled) // k_folds
 
# 初始化DataFrame来存储结果
results_df = pd.DataFrame(columns=[
    'Fold', 'Set', 'MSE', 'RMSE', 'MAE', 'R2', 'Deviation (%)'
])
 
# 存储所有预测值和实际值用于Williams图
all_y_test = []
all_y_test_pred = []
 
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
    
    # 存储测试集的实际值和预测值
    all_y_test.extend(y_test)
    all_y_test_pred.extend(y_test_pred)
    
    # ... (评估代码保持不变)
 
# 转换为numpy数组以便计算
all_y_test = np.array(all_y_test)
all_y_test_pred = np.array(all_y_test_pred)
 
# 计算残差
residuals = all_y_test - all_y_test_pred
 
# 制作Williams图
plt.figure(figsize=(10, 6))
plt.scatter(all_y_test_pred, residuals, alpha=0.5)
plt.hlines(0, xmin=all_y_test_pred.min(), xmax=all_y_test_pred.max(), colors='r', linestyles='dashed')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Williams Plot')
plt.grid(True)
plt.show()
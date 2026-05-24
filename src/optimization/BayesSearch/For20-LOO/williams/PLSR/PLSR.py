import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings
from collections import OrderedDict

# 忽略未来警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 数据集定义
df = pd.read_csv("G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/2.数据整理-新/data.csv", sep=',')

# 数据预处理
scaler = StandardScaler()
X = df[['input1', 'input2', 'input3', 'input4']]
X_scaled = scaler.fit_transform(X)
y = df['target']

# 使用最优参数初始化PLSRegression模型
optimal_params = OrderedDict([
    ('max_iter', 338),
    ('n_components', 4)
])

plsr_model = PLSRegression(**optimal_params)

# 分割数据为20折
k_folds = 20
fold_size = len(X_scaled) // k_folds

# 初始化DataFrame来存储结果
results_df = pd.DataFrame(columns=[
    'Fold', 'Set', 'MSE', 'RMSE', 'MAE', 'R2', 'Deviation (%)'
])

# 进行20次训练和测试
for i in range(k_folds):
    # 确定测试集和训练集索引
    test_index = slice(i * fold_size, (i + 1) * fold_size)
    train_index = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, len(X_scaled)))
    
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 训练模型
    plsr_model.fit(X_train, y_train)
    
    # 进行预测
    y_train_pred = plsr_model.predict(X_train)
    y_test_pred = plsr_model.predict(X_test)
    
    # 评估训练集
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    avg_deviation_train_percent = ((10 ** rmse_train) - 1) * 100 if rmse_train > 0 else 0
    
    # 评估测试集
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    avg_deviation_test_percent = ((10 ** rmse_test) - 1) * 100 if rmse_test > 0 else 0
    
    # 创建临时DataFrame来存储当前折的结果
    train_results_df = pd.DataFrame({
        'Fold': [i + 1],
        'Set': ['Training'],
        'MSE': [mse_train],
        'RMSE': [rmse_train],
        'MAE': [mae_train],
        'R2': [r2_train],
        'Deviation (%)': [avg_deviation_train_percent]
    })
    
    test_results_df = pd.DataFrame({
        'Fold': [i + 1],
        'Set': ['Test'],
        'MSE': [mse_test],
        'RMSE': [rmse_test],
        'MAE': [mae_test],
        'R2': [r2_test],
        'Deviation (%)': [avg_deviation_test_percent]
    })
    
    # 使用pd.concat合并DataFrame
    results_df = pd.concat([results_df, train_results_df, test_results_df], ignore_index=True)

# 计算平均训练集和测试集误差，并添加到DataFrame中
train_metrics = results_df[results_df['Set'] == 'Training'][['MSE', 'RMSE', 'MAE', 'R2', 'Deviation (%)']].mean()
test_metrics = results_df[results_df['Set'] == 'Test'][['MSE', 'RMSE', 'MAE', 'R2', 'Deviation (%)']].mean()

# 将平均结果添加到新的DataFrame中
average_results_df = pd.DataFrame({
    'Set': ['Training', 'Test'],
    'MSE': [train_metrics['MSE'], test_metrics['MSE']],
    'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
    'MAE': [train_metrics['MAE'], test_metrics['MAE']],
    'R2': [train_metrics['R2'], test_metrics['R2']],
    'Deviation (%)': [train_metrics['Deviation (%)'], test_metrics['Deviation (%)']]
})

# 将所有结果保存到CSV文件
results_df.to_csv("G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/1.正式代码/2.代码/BayesSearch/For20-LOO/PLSR/results.csv", index=False)
average_results_df.to_csv("G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/1.正式代码/2.代码/BayesSearch/For20-LOO/PLSR/average_results.csv", index=False)
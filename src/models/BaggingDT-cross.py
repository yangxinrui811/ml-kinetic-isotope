import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import KFold  
from sklearn.ensemble import BaggingRegressor  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import numpy as np  
import matplotlib.pyplot as plt  
  
# 数据集定义  
df = pd.read_csv("C:/Users/FD/Desktop/测试/data.csv", sep=',')  # 假设CSV文件使用逗号分隔  
  
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
  
# 交叉验证  
kf = KFold(n_splits=10, shuffle=True, random_state=42)  
fold_scores = []  # 用于存储每折的评分  
  
for train_index, test_index in kf.split(X_scaled):  
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]  
    y_train, y_test = y[train_index], y[test_index]  
  
    # 初始化Bagging模型  
    # 使用DecisionTreeRegressor作为基学习器  
    base_estimator = DecisionTreeRegressor(max_depth=3, random_state=42)  
    model = BaggingRegressor(base_estimator=base_estimator, n_estimators=100, random_state=42)  
  
    # 训练模型  
    model.fit(X_train, y_train)  
  
    # 评估模型  
    y_pred = model.predict(X_test)  
  
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
  
# 注意：下面的绘图代码不能直接使用，因为y_test和y_pred是在循环中定义的，并且每次循环都会被覆盖。  
# 我们需要先收集所有折的y_test和y_pred，然后再进行绘图。  
  
# 收集所有折的预测值和实际值  
all_y_test = np.concatenate([y[test_index] for train_index, test_index in kf.split(X_scaled)])  
all_y_pred = np.concatenate([model.predict(X_scaled[test_index]) for train_index, test_index in kf.split(X_scaled)])  
  
# 生成科研绘图  
plt.figure(figsize=(10, 6))  # 设置图形大小  
plt.scatter(all_y_test, all_y_pred, color='blue', marker='o', edgecolor='w', alpha=0.7)  # 绘制散点图  
plt.plot([all_y_test.min(), all_y_test.max()], [all_y_test.min(), all_y_test.max()], 'k--', lw=2)  # 绘制y=x的参考线  
  
# 添加标题和标签  
plt.title('Predicted vs Actual Values')  
plt.xlabel('Actual Values')  
plt.ylabel('Predicted Values')  
  
# 显示网格  
plt.grid(True)  
  
# 显示图形  
plt.show()
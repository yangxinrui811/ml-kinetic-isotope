import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error  
import numpy as np  
import matplotlib.pyplot as plt  
  
# 加载数据  
df = pd.read_csv("C:/Users/FD/Desktop/测试/data.csv", sep=',')  
  
# 数据预处理  
scaler = StandardScaler()  
X = df[['input1', 'input2', 'input3', 'input4']]  
X_scaled = scaler.fit_transform(X)  
y = df['target']  
  
# 分割数据集为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  
  
# 使用决策树回归模型，并通过交叉验证找到最佳参数  
param_grid = {  
    'max_depth': [1000],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]  
}  
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=10, scoring='neg_mean_squared_error')  
grid_search.fit(X_train, y_train)  
  
# 输出最佳参数和对应的性能  
print(f"Best parameters found: {grid_search.best_params_}")  
  
# 使用最佳参数配置模型  
best_regressor = grid_search.best_estimator_  
  
# 生成学习曲线  
train_sizes, train_scores, test_scores = learning_curve(best_regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error',  
                                                        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)  
  
# 转换得分为正数（因为scoring='neg_mean_squared_error'返回的是负值）  
train_scores_mean = -train_scores.mean(axis=1)  
train_scores_std = train_scores.std(axis=1)  
test_scores_mean = -test_scores.mean(axis=1)  
test_scores_std = test_scores.std(axis=1)  
  
# 绘制学习曲线  
plt.figure(figsize=(10, 6))  
plt.title('Learning Curve')  
plt.xlabel('Training examples')  
plt.ylabel('Mean Squared Error')  
plt.grid()  
  
# 绘制训练误差和验证误差的均值及标准差范围  
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')  
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                 test_scores_mean + test_scores_std, alpha=0.1, color='g')  
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')  
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')  
  
plt.legend(loc='best')  
plt.show()  
  
# 使用最佳参数进行预测并评估模型（这部分与原始代码相同）  
y_pred = best_regressor.predict(X_test)  
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)  
print(f"Mean Squared Error (MSE) on test set: {mse}")  
print(f"Root Mean Squared Error (RMSE) on test set: {rmse}")
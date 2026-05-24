import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.tree import DecisionTreeRegressor, export_graphviz  
from sklearn.metrics import mean_squared_error  
import numpy as np  
import pydotplus  
  
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
    'max_depth': [5],  # 注意：这里设置了一个非常大的深度，可能会导致过拟合，实际应用中应谨慎  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]  
}  
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=10, scoring='neg_mean_squared_error')  
grid_search.fit(X_train, y_train)  
  
# 输出最佳参数和对应的性能  
print(f"Best parameters found: {grid_search.best_params_}")  
  
# 使用最佳参数配置模型  
best_regressor = grid_search.best_estimator_  
  
# 生成决策树的图像数据  
dot_data = export_graphviz(best_regressor, out_file=None,  
                           feature_names=['input1', 'input2', 'input3', 'input4'],  
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
  
# 将图像保存到一个具体的文件路径中  
graph.write_png('C:/Users/FD/Desktop/测试/decision_tree.png')  
  
# 输出提示信息，告知用户图像已保存  
print("Decision tree image saved to 'C:/Users/FD/Desktop/测试/decision_tree.png'")  
  
  
# 使用最佳参数进行预测并评估模型  
y_pred = best_regressor.predict(X_test)  
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)  
print(f"Mean Squared Error (MSE) on test set: {mse}")  
print(f"Root Mean Squared Error (RMSE) on test set: {rmse}")
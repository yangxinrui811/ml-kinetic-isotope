import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor, plot_tree  # 导入plot_tree  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import numpy as np  # 用于计算RMSE  
import matplotlib.pyplot as plt  
  
# 数据集定义  
df = pd.read_csv("C:/Users/FD/Desktop/测试/data.csv", sep=',')  # 假设CSV文件使用逗号分隔  
  
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
X_scaled = scaler.fit_transform(X)  # 归一化处理  
y = df['target']  # 目标数据不需要进行归一化处理  
  
# 数据分为训练集与测试集  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  
  
# 使用决策树回归模型  
regressor = DecisionTreeRegressor(random_state=42)  
regressor.fit(X_train, y_train)  
  
# 预测  
y_pred = regressor.predict(X_test)  
  
# 评估模型  
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)  # 计算RMSE  
mae = mean_absolute_error(y_test, y_pred)  # 计算MAE  
r2 = r2_score(y_test, y_pred)  # 计算R2  
  
print(f"Mean Squared Error (MSE): {mse}")  
print(f"Root Mean Squared Error (RMSE): {rmse}")  
print(f"Mean Absolute Error (MAE): {mae}")  
print(f"R^2 Score: {r2}")

# 展示决策树模型  
plt.figure(figsize=(20,10))  
plot_tree(regressor, filled=True, feature_names=X.columns, rounded=True)  # 移除了class_names参数  
plt.show()
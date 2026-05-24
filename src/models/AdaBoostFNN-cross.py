import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import KFold  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
from sklearn.base import BaseEstimator, RegressorMixin  
from sklearn.ensemble import AdaBoostRegressor  
import numpy as np  
  
# 数据集定义（与原代码相同）  
df = pd.read_csv("C:/Users/FD/Desktop\测试\data.csv", sep=',')  # 假设CSV文件使用制表符分隔  

data = {  
    'input1': df['input1'].tolist(),  
    'input2': df['input2'].tolist(),  
    'input3': df['input3'].tolist(),  
    'input4': df['input4'].tolist(),  
    'target': df['target'].tolist()  
}  

df = pd.DataFrame(data)             # 定义df，转换为二维的、大小可变的、有标签的数据结构
  
# 数据预处理  
scaler = StandardScaler()  
X = df[['input1', 'input2', 'input3', 'input4']]  
X_scaled = scaler.fit_transform(X)  
y = df['target']  
  
# 定义BP神经网络  
class Net(torch.nn.Module):  
    def __init__(self):  
        super(Net, self).__init__()  
        self.fc1 = nn.Linear(4, 100)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(100, 100)  
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(100, 100)  
        self.relu3 = nn.ReLU()  
        self.fc4 = nn.Linear(100, 1)  
  
    def forward(self, x):  
        x = self.relu1(self.fc1(x))  
        x = self.relu2(self.fc2(x))  
        x = self.relu3(self.fc3(x))  
        x = self.fc4(x)  
        return x  
  
# 定义一个与scikit-learn兼容的PyTorch回归器  
class PyTorchRegressor(BaseEstimator, RegressorMixin):  
    def __init__(self, net, criterion, optimizer, epochs=1000):  
        self.net = net  
        self.criterion = criterion  
        self.optimizer = optimizer  
        self.epochs = epochs  
  
    def fit(self, X, y):  
        self.net.train()  
        X_tensor = torch.FloatTensor(X)  
        y_tensor = torch.FloatTensor(y).view(-1, 1)  
        for epoch in range(self.epochs):  
            output = self.net(X_tensor)  
            loss = self.criterion(output, y_tensor)  
            self.optimizer.zero_grad()  
            loss.backward()  
            self.optimizer.step()  
        return self  
  
    def predict(self, X):  
        self.net.eval()  
        with torch.no_grad():  
            X_tensor = torch.FloatTensor(X)  
            output = self.net(X_tensor).numpy()  
        return output.flatten()  
  
# 创建PyTorch回归器实例  
net = Net()  
criterion = nn.MSELoss()  
optimizer = optim.SGD(net.parameters(), lr=0.2)  
pytorch_regressor = PyTorchRegressor(net, criterion, optimizer)  
  
# 使用AdaBoostRegressor集成  
adaboost_model = AdaBoostRegressor(base_estimator=pytorch_regressor, n_estimators=10, random_state=42)  
  
# 交叉验证  
kf = KFold(n_splits=3, shuffle=True, random_state=42)  
fold_scores = []  
  
for train_index, test_index in kf.split(X_scaled):  
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]  
    y_train, y_test = y[train_index], y[test_index]  
      
    adaboost_model.fit(X_train, y_train)  
    y_pred = adaboost_model.predict(X_test)  
      
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
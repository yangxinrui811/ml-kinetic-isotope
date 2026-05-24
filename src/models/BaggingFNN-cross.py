import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import KFold, train_test_split  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import time  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
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
  
# Bagging配置  
num_models = 10  # 集成中模型的数量  
models = [Net() for _ in range(num_models)]  
criterion = nn.MSELoss()  
optimizers = [optim.SGD(model.parameters(), lr=0.2) for model in models]  
  
# 训练多个模型  
for model, optimizer in zip(models, optimizers):  
    # 每个模型在不同的数据子集上训练  
    X_train_resample, _, y_train_resample, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=np.random.randint(0, 100))  
    X_train = torch.FloatTensor(X_train_resample)  
    y_train = torch.FloatTensor(y_train_resample.values)  
  
    for t in range(1000):  
        output = model(X_train)  
        loss = criterion(output, y_train.view(-1, 1))  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
# 评估模型（使用全部数据）  
X_test = torch.FloatTensor(X_scaled)  
y_test = torch.FloatTensor(y.values)  
  
# 聚合预测  
predictions = torch.stack([model(X_test).detach() for model in models], dim=0)  
average_predictions = predictions.mean(dim=0).numpy()  # 使用 PyTorch 的 mean 方法，并转换为 NumPy 数组  
  
# 计算评分  
mse = mean_squared_error(y.values, average_predictions)  
rmse = np.sqrt(mse)  
mae = mean_absolute_error(y.values, average_predictions)  
r2 = r2_score(y.values, average_predictions)  
  
# 输出评分  
print(f"MSE: {mse}")  
print(f"RMSE: {rmse}")  
print(f"MAE: {mae}")  
print(f"R²: {r2}")
import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import KFold  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import time  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import numpy as np
import matplotlib.pyplot as plt 

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
  
# 数据预处理（与原代码相同）  
scaler = StandardScaler()  
X = df[['input1', 'input2', 'input3', 'input4']]  
X_scaled = scaler.fit_transform(X)  
y = df['target']  
  
# 2. 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)    # 第一层隐藏层  
        self.relu1 = nn.ReLU()          # 第一层隐藏层的ReLU激活函数  
        self.fc2 = nn.Linear(100, 100)  # 第二层隐藏层  
        self.relu2 = nn.ReLU()          # 第二层隐藏层的ReLU激活函数  
        self.fc3 = nn.Linear(100, 100)  # 第三层隐藏层  
        self.relu3 = nn.ReLU()          # 第三层隐藏层的ReLU激活函数  
        self.fc4 = nn.Linear(100, 1)    # 定义输出层网络

    def forward(self, x):
        x = self.relu1(self.fc1(x))  # 第一层隐藏层的激活函数  
        x = self.relu2(self.fc2(x))  # 第二层隐藏层的激活函数  
        x = self.relu3(self.fc3(x))  # 第三层隐藏层的激活函数  
        x = self.fc4(x)              # 输出层不用激活函数  
        return x

net = Net()
criterion = nn.MSELoss()  #均方误差损失
optimizer = optim.SGD(net.parameters(), lr=0.2)  #随机梯度下降
  
# 交叉验证  
kf = KFold(n_splits=10, shuffle=True, random_state=42)  
fold_scores = []  # 用于存储每折的评分  
all_y_test = []  
all_y_pred = []  
  
for train_index, test_index in kf.split(X_scaled):  
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]  
    y_train, y_test = y[train_index], y[test_index]  
      
    # 将数据转换为Tensor  
    X_train = torch.FloatTensor(X_train)  
    X_test = torch.FloatTensor(X_test)  
    y_train = torch.FloatTensor(y_train.values)  
    y_test = torch.FloatTensor(y_test.values)  
      
    # 初始化网络、损失函数和优化器  
    net = Net()  
    criterion = nn.MSELoss()  
    optimizer = optim.SGD(net.parameters(), lr=0.2)  
      
    # 训练网络  
    for t in range(1000):  
        output = net(X_train)  
        loss = criterion(output, y_train.view(-1, 1))  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
      
    # 评估模型  
    net.eval()  
    with torch.no_grad():  
        y_pred = net(X_test).numpy()  
        y_true = y_test.numpy()  
        all_y_test.append(y_test.numpy())  
        all_y_pred.append(y_pred) 
      
    # 计算并记录当前折的评分  
    mse = mean_squared_error(y_true, y_pred)  
    rmse = np.sqrt(mse)  
    mae = mean_absolute_error(y_true, y_pred)  
    r2 = r2_score(y_true, y_pred)  
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
all_y_test = np.concatenate(all_y_test)  
all_y_pred = np.concatenate(all_y_pred)  
  
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
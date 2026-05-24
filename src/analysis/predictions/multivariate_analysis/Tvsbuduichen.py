import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 设置全局字体大小并绘制等高线图
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# 数据集定义
df = pd.read_csv("G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/2.数据整理-新/data.csv", sep=',')

# 数据预处理
scaler = StandardScaler()
X = df[['input1', 'input2', 'input3', 'input4']]
X_scaled = scaler.fit_transform(X)
y = df['target']

# 留出法分离训练集和预测集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 使用已经优化好的超参数初始化XGBoost模型
model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.4750865,
    min_child_weight=0.4970553,
    subsample=1.0,
    colsample_bytree=0.93367401,
    gamma=1e-10,
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 定义交互交互函数
def plot_interInteraction(input1_data, input3_data):
    # 确保输入数据维度一致
    if len(input1_data) != len(input3_data):
        raise ValueError("输入1和输入3的数据长度必须相等")

    # 生成输入组合网格
    input1_constant = 300  # 基本温度值
    input2_range = np.linspace(-20, 10, 20)
    input3_range = np.logspace(0, 2, 20)

    input_data = np.c_[[input1_constant]*len(input2_range), input2_range, input3_range, [-0.9]*len(input2_range)]
    input_data_scaled = scaler.transform(input_data)

    # 预测
    predictions = model.predict(input_data_scaled)
    
    # 将预测结果转换为高斯分布
    target = 10**(-np.log10(predictions))  # 将预测值转换为高斯分布

    # 创建网格点
    mesh_i, mesh_j = input2_range.shape[0], input3_range.shape[0]
    X1, X2 = np.meshgrid(input2_range, input3_range)
    target_z = np.zeros((X1.shape[0], X1.shape[1]))

    # 确保计算结果与网格一致
    if len(target) != mesh_i * mesh_j:
        raise ValueError("预测结果和输入数据长度不匹配")

    for i in range(len(X1)):
        for j in range(len(X2)):
            x1 = X1[i,j]
            x3 = X2[i,j]
            input_vector = np.c_[input1_constant, x1, x3, -0.9]
            input_scaled = scaler.transform(input_vector)
            target_z_ij = 10**(-np.log10(model.predict(input_scaled)))
            target_z[i,j] = target_z_ij

    # 确保计算结果和输入数据长度一致
    if len(target_z) != len(predictions):
        raise ValueError("预测结果与网格点数不一致")

    predictions_grid = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            x1 = X1[i,j]
            x3 = X2[i,j]
            input_vector = np.c_[input1_constant, x1, x3, -0.9]
            input_scaled = scaler.transform(input_vector)
            predictions_grid_ij = model.predict(input_scaled)
            target_z_ij = 10**(-np.log10(predictions_grid_ij))
            target_z[i,j] = target_z_ij

    # 将预测结果重塑为网格形状
    target_z_grid_reshaped = target_z.reshape(X1.shape)

    # 使用contourf绘制等高线图，显示交互影响对target的贡献
    plt.figure(figsize=(20, 12))
    ax = plt.axes()
    ax.contourf(X1, X2, target_z_grid_reshaped, levels=20, cmap='Blues')
    ax.scatter(input2_range, input3_range, c=target_z_grid, cmap='Blues', s=50, alpha=0.5)
    plt.colorbar()
    plt.xlabel('k (x1)', fontsize=14)
    plt.ylabel('KIE (x3)', fontsize=14)

    # 显示图表
    plt.show()
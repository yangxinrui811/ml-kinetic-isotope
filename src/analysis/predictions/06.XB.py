import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 数据集定义
df = pd.read_csv("D:/BaiduSyncdisk/0.我的工作/5.隧穿相图/2.数据整理-新/data.csv", sep=',')
# 确保所有特征包括交互项
def create_feature_matrix(df):
    features = df[['input1', 'input2', 'input3', 'input4']]
    # 添加交互项：这里加入温度与势垒不对称度的乘积
    features['temp_separation_interaction'] = features['input1'] * features['input3']
    # 加入热力学参数作为新特征（假设存储在df中）
    #features['entropy'] = df['entropy']
    #features['free_energy_curvature'] = df['free_energy_curvature']
    return features

# 数据预处理
scaler = StandardScaler()
X = create_feature_matrix(df)
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

# 生成预测用的输入数据网格
input1_constant = 300
input2_range = np.linspace(-20, 10, 20)
input3_range = np.logspace(0, 2, 20)
input2_mesh, input3_mesh = np.meshgrid(input2_range, input3_range)
input4_constant = -0.9

# 标准化输入数据
input_data_grid = np.c_[
    [input1_constant] * len(input2_mesh.ravel()),
    input2_mesh.ravel(),
    input3_mesh.ravel(),
    [input4_constant] * len(input2_mesh.ravel())
]
input_data_grid_scaled = scaler.transform(input_data_grid)

# 进行预测
predictions_grid = model.predict(input_data_grid_scaled)

# 将预测结果重塑为网格形状
predictions_grid_reshaped = predictions_grid.reshape(input2_mesh.shape)

# 将预测值从对数空间转换到线性空间
predictions_grid_exp = 10**predictions_grid_reshaped

# 创建矩阵形式的 DataFrame
output_df = pd.DataFrame(predictions_grid_exp, index=input2_range, columns=input3_range)

# 将 DataFrame 写入 CSV 文件
try:
    output_df.to_csv('G:/BaiduSyncdisk/0.我的工作/5.隧穿相图/1.正式代码/2.代码/BayesSearch/预测/300Kpredictions_matrix.csv', index=True)
    print("文件已成功保存！")
except Exception as e:
    print(f"保存文件时发生错误：{e}")

# 设置全局字体大小并绘制等高线图
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

plt.figure(figsize=(10, 6))
plt.contourf(input2_mesh, input3_mesh, predictions_grid_exp, levels=10, cmap='GnBu')
plt.colorbar()
plt.xlabel('k')
plt.ylabel('KIE')
plt.xscale('linear')
plt.yscale('log')

# 添加多变量耦合的交互项影响线条（例如）
plt.plot(input2_range[:5], input3_range[:5], 'r', linewidth=2, label='温度与势垒不对称度协同效应')
plt.legend()

plt.show()
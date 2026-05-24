import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Integer
import matplotlib.pyplot as plt

# 数据集定义
df = pd.read_csv("D:/BaiduSyncdisk/0.坚果云/2.隧穿相图/测试/data.csv", sep=',')
X = df[['input1', 'input2', 'input3', 'input4']]
y = df['target']

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义一个函数来创建模型并计算交叉验证的负MSE
def objective(params):
    model = DecisionTreeRegressor(
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        random_state=42
    )
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = cross_val_score(model, X_scaled, y, scoring='neg_mean_squared_error', cv=kf)
    return np.mean(mse_scores)

# 定义超参数的空间
search_spaces = {
    'max_depth': Integer(1, 30),
    'min_samples_split': Integer(1, 10),
    'min_samples_leaf': Integer(1, 10)
}

# 运行贝叶斯优化
opt = BayesSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    search_spaces=search_spaces,
    scoring='neg_mean_squared_error',
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    random_state=42,
    n_iter=1,
)

opt.fit(X_scaled, y)

# 输出最佳超参数和对应的评分
print(f"Best parameters found: {opt.best_params_}")
print(f"Best negative MSE: {opt.best_score_}")

# 使用最佳超参数训练最终模型
best_params = opt.best_params_
model = DecisionTreeRegressor(
    max_depth=int(best_params['max_depth']),
    min_samples_split=int(best_params['min_samples_split']),
    min_samples_leaf=int(best_params['min_samples_leaf']),
    random_state=42
)
model.fit(X_scaled, y)

# 计算并输出最终模型在训练集上的误差
y_pred = model.predict(X_scaled)
train_mse = mean_squared_error(y, y_pred)
print(f"Training MSE of the final model: {train_mse}")

# 绘制预测值与实际值的对比图
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted values')
plt.show()
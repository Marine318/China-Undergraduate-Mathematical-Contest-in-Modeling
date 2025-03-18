import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import  pandas as pd
excel_file1 = "第2步，每日销售定价用于处理回归.xlsx"
excel_file2 = "第4步，各品类均利率.xlsx"
df1 = pd.read_excel(excel_file1)
df2 = pd.read_excel(excel_file2)
X=df1['食用菌']
y=df2['食用菌']

# 合并数据
X_all = np.concatenate([X, X])
y_all = np.concatenate([y, y])

# 异常值处理：删除超过3倍标准差的异常值
mean_y = np.mean(y_all)
std_y = np.std(y_all)
threshold = 3 * std_y
X_clean = X_all[abs(y_all - mean_y) < threshold]
y_clean = y_all[abs(y_all - mean_y) < threshold]

# 定义XGBoost回归模型参数
params = {
    'objective': 'reg:squarederror',  # 回归任务的目标函数
    'n_estimators': 100,             # 迭代次数
    'max_depth': 3,                 # 最大树深度
    'learning_rate': 0.1            # 学习率
}

# 创建并拟合XGBoost模型
model = xgb.XGBRegressor(**params)
model.fit(X_clean.reshape(-1, 1), y_clean)

# 预测并计算R值
y_pred = model.predict(X_all.reshape(-1, 1))
r_squared = r2_score(y_all, y_pred)

# 绘制原始数据和拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(X_all, y_all, label='原始数据')
plt.plot(X_all, y_pred, color='red', linewidth=2, label='XGBoost拟合曲线')
plt.title(f'R^2 = {r_squared:.2f}')
plt.legend()
plt.show()





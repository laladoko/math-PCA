import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


file_path = 'coverage\data\Prostate Cancer full.csv'
data = pd.read_csv(file_path)
print(data.head())

# 数据预处理
X = data.drop(columns=['id', 'lpsa', 'train'])
print(X.columns)

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 创建一个包含主成分的DataFrame
pca_df = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(X_pca.shape[1])])
print(pca_df.head())

# 打印各主成分的解释方差比例
print(pca.explained_variance_ratio_)

# 累计解释方差比例
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance)
plt.title('Cumulative Explained Variance by PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

# PCA进行线性回归
Z = pca_df[['PC_1']].values
y = data['lpsa']
model = LinearRegression()
model.fit(Z, y)
y_pred = model.predict(Z)


# 计算R-squared值
r2 = r2_score(y, y_pred)
print(f'R-squared using PCA: {r2}')

# # 创建一个包含主成分的DataFrame
# pca_df = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(X_pca.shape[1])])

# # # 目标变量
# y = data['lpsa']

# # 循环从PC_1到PC_8，绘制每个主成分与目标变量的散点图
# for i in range(1, 9):
#     plt.figure(figsize=(8, 6))  # 设置图像大小
#     Z = pca_df[f'PC_{i}'].values.reshape(-1, 1)  # 提取第i个主成分
#     model = LinearRegression()  # 初始化线性回归模型
#     model.fit(Z, y)  # 训练模型
#     y_pred = model.predict(Z)  # 预测目标变量值

#     plt.scatter(Z, y, color='blue', label=f'Actual values for PC_{i}')
#     plt.plot(Z, y_pred, color='red', linewidth=2, label=f'Regression line for PC_{i}')

#     plt.title(f'Scatter plot of PC_{i} vs Target Variable (lpsa)')
#     plt.xlabel(f'PC_{i}')
#     plt.ylabel('lpsa')

#     plt.legend()

#     plt.show()

# 残差图
residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot using PCA')
plt.show()

# 不使用PCA进行线性回归
model.fit(X_scaled, y)
y_pred_no_pca = model.predict(X_scaled)
r2_no_pca = r2_score(y, y_pred_no_pca)
print(f'R-squared without PCA: {r2_no_pca}')

residuals_no_pca = y - y_pred_no_pca
plt.scatter(y_pred_no_pca, residuals_no_pca)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot without PCA')
plt.show()
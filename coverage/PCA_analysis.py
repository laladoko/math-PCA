import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.linalg import svd

# 读取数据
file_path = 'coverage\data\Prostate Cancer full.csv'
data = pd.read_csv(file_path)

# 数据预处理
X = data.drop(columns=['id', 'lpsa', 'train'])
y = data['lpsa']

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def svd_pca(X, n_components):
    # 计算SVD
    U, s, Vt = svd(X)
    
    # 选择前n_components个主成分
    components = Vt[:n_components, :].T  # 注意这里取Vt的转置
    explained_variance = s[:n_components]**2 / np.sum(s**2)
    
    # 转换数据到主成分空间
    X_pca = X.dot(components)  # 使用U矩阵的列向量作为主成分
    return X_pca, explained_variance

# 假设我们选择前2个主成分
n_components = 4
X_pca_svd, explained_variance_svd = svd_pca(X_scaled, n_components)

# 创建一个包含主成分的DataFrame
pca_df_svd = pd.DataFrame(X_pca_svd, columns=[f'PC_{i+1}' for i in range(1, n_components + 1)])
# 线性回归分析
Z = pca_df_svd.values
model = LinearRegression()
model.fit(Z, y)
y_pred_svd = model.predict(Z)

# 计算R-squared值
r2_svd = r2_score(y, y_pred_svd)
print(f'R-squared using SVD-based PCA: {r2_svd}')

# 绘制残差图
residuals_svd = y - y_pred_svd
plt.scatter(Z[:,0], residuals_svd, label='Residuals using SVD-based PCA')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('PC_1')
plt.ylabel('Residuals')
plt.title('Residual Plot using SVD-based PCA')
plt.legend()
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
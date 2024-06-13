import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


print("start")
file_path = 'coverage\data\Prostate Cancer full.csv'
data = pd.read_csv(file_path)
print(data.head())
X = data.drop(columns=['id', 'lpsa', 'train'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 打印各主成分解释的方差比例s
print(pca.explained_variance_ratio_)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance)+1), cumulative_explained_variance)
plt.ylabel('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.show()
Z = X_pca[:, 0].reshape(-1, 1)  # 假设我们只使用第一主成分
y = data['lpsa']

model = LinearRegression()
model.fit(Z, y)
y_pred = model.predict(Z)

print(f'R-squared: {r2_score(y, y_pred)}')
# 直接使用原始X而不是X_pca
model.fit(X_scaled, y)  # 确保使用标准化后的X
y_pred = model.predict(X_scaled)

print(f'R-squared without PCA: {r2_score(y, y_pred)}')
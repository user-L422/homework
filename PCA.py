import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="class")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['class'] = y

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='class', palette='Set1')
plt.title('PCA of Wine Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

print("各主成分解释的方差比例（信息保留度）:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

print(f"\n前两个主成分共保留信息量: {np.sum(pca.explained_variance_ratio_):.4f}\n")

pc1 = pd.Series(pca.components_[0], index=X.columns)
pc2 = pd.Series(pca.components_[1], index=X.columns)

print("第一个主成分（PC1）系数：\n", pc1.sort_values(ascending=False))
print("\n第二个主成分（PC2）系数：\n", pc2.sort_values(ascending=False))
print("\n主成分解释的方差比例：", pca.explained_variance_ratio_)
print("总保留信息量：", sum(pca.explained_variance_ratio_))
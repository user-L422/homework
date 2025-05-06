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

print("�����ɷֽ��͵ķ����������Ϣ�����ȣ�:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

print(f"\nǰ�������ɷֹ�������Ϣ��: {np.sum(pca.explained_variance_ratio_):.4f}\n")

pc1 = pd.Series(pca.components_[0], index=X.columns)
pc2 = pd.Series(pca.components_[1], index=X.columns)

print("��һ�����ɷ֣�PC1��ϵ����\n", pc1.sort_values(ascending=False))
print("\n�ڶ������ɷ֣�PC2��ϵ����\n", pc2.sort_values(ascending=False))
print("\n���ɷֽ��͵ķ��������", pca.explained_variance_ratio_)
print("�ܱ�����Ϣ����", sum(pca.explained_variance_ratio_))
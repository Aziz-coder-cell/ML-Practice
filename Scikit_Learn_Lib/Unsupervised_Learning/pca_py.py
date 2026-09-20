import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    'Age': [25, 30, 35, 40, 45, 50],
    'Income': [30000, 40000, 50000, 60000, 70000, 80000],
    'Spending': [70, 60, 50, 40, 30, 20],
    'Savings': [1000, 5000, 8000, 10000, 15000, 20000]
}

df = pd.DataFrame(data)

scalar = StandardScaler()
scaled_data = scalar.fit_transform(df)

pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_results,columns=['col_1','col_2'])
pca_explain_var = pca.explained_variance_ratio_

print(pca_df)
print("Variance captured by PCA Component:")
print(pca_explain_var)
print(np.round(pca_explain_var*100,2))

plt.figure(figsize=(8,6))
plt.scatter(pca_df['col_1'],pca_df['col_2'],color='black')
plt.title("PCA Projection(2D View)")
plt.xlabel("Main")
plt.ylabel("Minor")
plt.grid(True)
plt.show()
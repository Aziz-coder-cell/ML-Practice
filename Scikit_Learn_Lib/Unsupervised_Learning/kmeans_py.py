import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'Customer': ['Kaushal', 'Aziz', 'Faizan', 'Neha', 'Imran', 'Customer'],
    'Age': [20, 30, 40, 22, 38, 25],
    'Spending': [100, 200, 300, 110, 290, 130]
}

df = pd.DataFrame(data)

model = KMeans(n_clusters=3,random_state=42,n_init=10)
X= df[['Age','Spending']]

df['Group'] = model.fit_predict(X)

plt.figure(figsize=(8,6))
for group in df['Group'].unique():
    group_data = df[df['Group']==group]
    plt.scatter(group_data['Age'],group_data['Spending'],label=f"Group {group}")

plt.title("Grouped data")
plt.grid(True)
plt.show()
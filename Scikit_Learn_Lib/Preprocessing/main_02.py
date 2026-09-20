import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.model_selection import train_test_split

data = {
    'Hours' : [2, 4, 6, 8, 10],
    'Scores' : [20, 40, 60, 80, 100]
}

df = pd.DataFrame(data)

standard_scalar = StandardScaler()
minmax_scalar = MinMaxScaler()

standard_scaled = standard_scalar.fit_transform(df)
minmax_scaled = minmax_scalar.fit_transform(df)

print("Standard Scaled Data:\n", pd.DataFrame(standard_scaled, columns=['Hours', 'Scores']))
print("\nMin-Max Scaled Data:\n", pd.DataFrame(minmax_scaled, columns=['Hours', 'Scores']))

X = df[['Hours']]
y = df[['Scores']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print("Trainind data:\n")
print(X_train)
print(X_test)
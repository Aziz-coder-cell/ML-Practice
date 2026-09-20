import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

data = {
    "StudyHours": [4, 5, 6, 2, 8, 5, 6, 2, 9, 1],
    "Marks": [50, 60, 65, 40, 80, 55, 70, 35, 90, 30]
}

df = pd.DataFrame(data)

scalar = StandardScaler()
scaled = scalar.fit_transform(df)

print(pd.DataFrame(scaled,columns=["StudyHours","Marks"]))

mmscaled = MinMaxScaler().fit_transform(df)

print(pd.DataFrame(mmscaled,columns=["StudyHours","Marks"]))
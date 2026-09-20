import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    "StudyHours": [4, 5, 6, 2, 8, 5, 6, 2, 9, 1],
    "Marks": [50, 60, 65, 40, 80, 55, 70, 35, 90, 30]
}

df = pd.DataFrame(data)

x_train,x_test,y_train,y_test = train_test_split(df['StudyHours'],df['Marks'],test_size=0.2,random_state=42)

print('X Train:')
print(x_train)

print('X Test')
print(x_test)
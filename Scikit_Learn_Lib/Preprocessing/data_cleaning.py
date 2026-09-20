import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

le=LabelEncoder()

df = pd.read_csv("dirty_dataset.csv")

df_label = df.copy()
df_label.drop('ID',axis=1,inplace=True)

df_label['Age'] = df_label['Age'].fillna(df_label['Age'].mean())

x=df_label['Gender'].mode()[0]
df_label['Gender'] = df_label['Gender'].fillna(x)

y = df_label['Profession'].mode()[0]
df_label['Profession'] = df_label['Profession'].fillna(y)

z= df_label['Marital status'].mode()[0]
df_label['Marital status'] = df_label['Marital status'].fillna(z)

a=df_label['City'].mode()[0]
df_label['City'] = df_label['City'].fillna(a)

df_label['Gender_encoded'] = le.fit_transform(df_label['Gender'])
df_label['Marital status_encoded'] = le.fit_transform(df_label['Marital status'])

df_encoded = pd.get_dummies(df_label,columns=['Profession','City'])

profession_cols = ['Profession_AI_Engineer', 'Profession_Doctor', 'Profession_MBA', 'Profession_ML_Engineer']
df_encoded['Profession_combined'] = df_encoded[profession_cols].astype(int).values.tolist()

city_cols = ['City_Colorado','City_Florida','City_LA','City_New york']
df_encoded['City_combined'] = df_encoded[city_cols].astype(int).values.tolist()

df_label = df_encoded.copy()
df_label = df_label.drop(city_cols,axis=1)
df_label = df_label.drop(profession_cols,axis=1)

df_label['Age'] = df_label['Age'].astype(int)

scaled = StandardScaler().fit_transform(df_label[['Age']])
mmscalar = MinMaxScaler().fit_transform(df_label[['Age']])

print(pd.DataFrame(scaled,columns=['Age']))
print(pd.DataFrame(mmscalar,columns=['Age']))

x_train,x_test,y_train,y_test = train_test_split(df_label['Age'],df_label['Gender_encoded'],test_size=0.46,random_state=43)

print("X Train")
print(x_train)

print("X Test")
print(x_test)
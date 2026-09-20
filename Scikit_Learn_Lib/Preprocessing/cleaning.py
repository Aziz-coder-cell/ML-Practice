import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("dirty_dataset.csv")
df_copy = df.copy()

#Cleaning the dataset
df_copy['Age'] = df['Age'].fillna(df['Age'].mean())
df_copy['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df_copy['City'] = df['City'].fillna(df['City'].mode()[0])
df_copy['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])
df_copy['Marital status'] = df['Marital status'].fillna(df['Marital status'].mode()[0])

#Encoding and scaling the features
df_encoded = pd.DataFrame()
le = LabelEncoder()
oe = OneHotEncoder(sparse_output=False)

df_encoded['Gender_encoded'] = le.fit_transform(df_copy['Gender'])
df_encoded['Marital status_encoded'] = le.fit_transform(df_copy['Marital status'])

encoded_city = oe.fit_transform(df_copy[['City']])
city_df = pd.DataFrame(
    encoded_city,
    columns = oe.get_feature_names_out(['City'])
)
df_encoded = pd.concat([df_encoded,city_df],axis=1)

encoded_profession = oe.fit_transform(df_copy[['Profession']])
profession_df = pd.DataFrame(
    encoded_profession,
    columns = oe.get_feature_names_out(['Profession'])
)
df_encoded = pd.concat([df_encoded,profession_df],axis=1)

mmscalar = MinMaxScaler()

age = mmscalar.fit_transform(df_copy[['Age']])
age_df = pd.DataFrame(age,columns=mmscalar.get_feature_names_out(['Age']))
df_encoded = pd.concat([df_encoded,age_df],axis=1)

print(df_encoded.head())

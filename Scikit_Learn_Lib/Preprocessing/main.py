import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("Dataset.csv")
print(df.columns.to_list())

label_encoder = LabelEncoder()
encoder = OneHotEncoder(sparse_output=False)

df['Gender_Encoded'] = label_encoder.fit_transform(df['Gender'])
df['Marital_status_Encoded'] = label_encoder.fit_transform(df['Marital status'])

encoded = encoder.fit_transform(df[['City']])

encoded_df = pd.DataFrame(
    encoded,
    columns = encoder.get_feature_names_out(['City'])
)

df = pd.concat([df.drop('City',axis=1),encoded_df],axis=1)

print(df[['ID','Gender','Gender_Encoded','Marital status','Marital_status_Encoded',]].head())
print('\n',df)
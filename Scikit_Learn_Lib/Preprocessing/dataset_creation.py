import numpy as np
import pandas as pd

n = 500

genders = ['Male','Female']
states = ['LA','New york','Colorado','Florida']
profession = ['ML_Engineer','Doctor','MBA','AI_Engineer']
status = ['Married','Unmarried']

data={
    'ID' : range(1,n+1),
    'Age' : np.random.randint(18,66,n),
    'Gender' : np.random.choice(genders,n),
    'City': np.random.choice(states,n),
    'Profession' : np.random.choice(profession,n),
    'Marital status' : np.random.choice(status,n)    
}

df = pd.DataFrame(data)
df.to_csv("Dataset.csv",index=False)
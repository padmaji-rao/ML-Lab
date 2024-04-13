import pandas as pd

temperatures = ["hot", "hot", "hot","mild","cool", "cool", "cool","mild","cool"]
df = pd.DataFrame({"Temperature": temperatures})

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
df["Temperature"]=le.fit_transform(df["Temperature"])
print(df)

colors=["red","green","blue","red","red","green","blue","green","blue","red","blue"]
df=pd.DataFrame({"Colors":colors})
print(df)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
encoder_df = pd.DataFrame(ohe.fit_transform(df[['Colors']]).toarray())
final_df = df.join(encoder_df)
final_df.drop('Colors', axis=1, inplace=True)
final_df.columns = [ 'Red', 'Green', 'Blue']
print("After one hot encoding")
print(final_df)

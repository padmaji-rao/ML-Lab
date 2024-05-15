import pandas as pd
df=pd.read_csv("C:\\Users\\satis\\Desktop\\ML Exp\\OHE_colors.csv")
print(df)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
transformed_column=pd.DataFrame(ohe.fit_transform(df[["Colors"]]).toarray())
df=df.join(transformed_column)
df.drop("Colors",axis=1,inplace=True)
df.columns=["SNO","Blue","Red","Green"]
print(df)

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("C:\\Users\\satis\\Desktop\\ML Exp\\Linear_Regression_Dataset.csv")
print(df)

x=df.drop("CropYield",axis=1)
y=df["CropYield"]


from sklearn.model_selection import train_test_split

x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_te,y_pred)
print(mse)




plt.scatter(range(len(y_te)),y_te,color="blue",label="Actual Values")
plt.scatter(range(len(y_pred)),y_pred,color="red",label="Predicted Values")
plt.legend()
plt.xlabel('Index')
plt.ylabel('Crop Yield')
plt.show()

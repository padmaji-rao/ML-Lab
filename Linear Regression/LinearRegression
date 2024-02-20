import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:\\Users\\TEMP.SRKRIT\\Downloads\\Salary_Data.csv")

X=df["YearsExperience"]
Y=df["Salary"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train.values.reshape(-1, 1),Y_train)
Y_pred=model.predict(X_test.values.reshape(-1, 1))

from sklearn.metrics import mean_squared_error, r2_score
mse=mean_squared_error(Y_test,Y_pred)
r2=r2_score(Y_test,Y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train, label="Training Data")
plt.scatter(X_test, Y_test, label="Testing Data")
plt.plot(X_test, Y_pred, label="Predicted Salaries")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs. Years of Experience")
plt.legend()
plt.show()

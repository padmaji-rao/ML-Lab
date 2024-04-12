import warnings
warnings.filterwarnings('ignore')

#Loading Boston House dataset
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)

#Dividing Dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#Initialising LinearRegression
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_pred=model_lr.predict(X_test)

#Finding mse, bia, variance
from mlxtend.evaluate import bias_variance_decomp
mse, bias, var = bias_variance_decomp(model_lr, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200,random_seed=123)
print("MSE from bias_variance_decomp lib:", mse)
print('Bias :',  bias)
print('Variance :',var)

from sklearn.metrics import mean_squared_error 
print('MSE from Sckit-learn lib :',mean_squared_error(y_test,y_pred))

import pandas as pd 
data=pd.read_csv("C:\\Users\\satis\\Documents\\Logistic_Regression.csv")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Default"]=le.fit_transform(data["Default"])

x=data.drop("Default",axis=1)
y=data["Default"]

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_tr,y_tr)
y_pred=lr.predict(x_te)

from sklearn.metrics import confusion_matrix,accuracy_score
acc=accuracy_score(y_te,y_pred)
cm=confusion_matrix(y_te,y_pred)
print(acc)
print(cm)

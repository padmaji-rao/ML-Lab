import pandas as pd
df=pd.read_csv("Diabetes.csv")

x=df.drop("Outcome",axis=1)
y=df["Outcome"]

from sklearn.model_selection import train_test_split 
x_tr,x_te,y_tr,y_te=train_test_split(x,y,random_state=1,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=7)
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(confusion_matrix(y_te,y_pred))
print(accuracy_score(y_te,y_pred))
print(classification_report(y_te,y_pred))

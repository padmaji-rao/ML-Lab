import pandas as pd
df=pd.read_csv("Loan_Default.csv")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Default"]=le.fit_transform(df["Default"])
x=df.drop("Default",axis=1)
y=df["Default"]

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.svm import SVC
model=SVC(kernel="linear")  #rbf,poly,sigmoid
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print(confusion_matrix(y_te,y_pred))
print(accuracy_score(y_te,y_pred))
print(classification_report(y_te,y_pred))

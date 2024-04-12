#Before applying PCA

import pandas as pd
df=pd.read_csv("Diabetes.csv")

x=df.drop("Outcome",axis=1)
y=df["Outcome"]

from sklearn.model_selection  import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x,y,random_state=1,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)

from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_te,y_pred))
print(classification_report(y_te,y_pred))



#After applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
reduced_x_tr=pca.fit_transform(x_tr)
reduced_x_te=pca.fit_transform(x_te)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(reduced_x_tr,y_tr)
y_pred=model.predict(reduced_x_te)

from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_te,y_pred))
print(classification_report(y_te,y_pred))

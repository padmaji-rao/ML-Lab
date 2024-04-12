import pandas as pd 
df=pd.read_csv("sentiment.csv")

x=df["Comment"]
y=df["Sentiment"]


#Transforming x(text) into tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(x)


#Transforming tagrt variable
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)


from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(y_te,y_pred))
print(accuracy_score(y_te,y_pred))

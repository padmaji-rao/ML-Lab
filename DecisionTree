import pandas as pd                                         #import packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


df=pd.read_csv("C:\\Users\\TEMP\\Downloads\\tennis.csv")   #load the dataset

le=LabelEncoder()                                           #Data pre processing
df["Outlook"]=le.fit_transform(df["Outlook"])
df["Temperature"]=le.fit_transform(df["Temperature"])
df["Humidity"]=le.fit_transform(df["Humidity"])
df["Windy"]=le.fit_transform(df["Windy"])
df["Play"]=le.fit_transform(df["Play"])

print(df)
                                                            #Divide the dataset into independent and dependent features
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)     #Divide X and Y for training and testing

clf=DecisionTreeClassifier()

dt=clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)

cm=confusion_matrix(Y_test,Y_pred)
print(cm)

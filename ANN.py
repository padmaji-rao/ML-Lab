from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
df=pd.read_csv("C:\\Users\\satis\\Desktop\\ML Exp\\Diabetes.csv")


x=df.drop("Outcome",axis=1)
y=df["Outcome"]

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.3,random_state=1)

model=Sequential()

model.add(Dense(units=6,input_dim=8,activation="relu",kernel_initializer="uniform"))
model.add(Dense(units=6,activation="relu",kernel_initializer="uniform"))
model.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(x_tr,y_tr,epochs=50,batch_size=30)

loss,accuracy=model.evaluate(x_te,y_te,batch_size=10)
print(accuracy)

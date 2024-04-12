# use sklearn to import a dataset
from sklearn.datasets import load_wine
wine = load_wine()

df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df["target"] = wine.target

print("Shape :",df.shape) #gives (rows,columns)

print("Information about Dataset :")
df.info()

print("Number of Duplicate Rows",df.duplicated().sum())

print("Count of Target variables \n",df.target.value_counts())

#visualizing target variables
import matplotlib.pyplot as plt
df.target.value_counts().plot(kind="bar")
plt.title("Value counts of the target variable")
plt.xlabel("Wine type")
plt.ylabel("Count")
plt.show() 

print("Skewness:", df['magnesium'].skew())  #tells about symmetry
print("Kurtosis:",df['magnesium'].kurt())   #tells about tail heaviness

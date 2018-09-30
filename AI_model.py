from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import numpy as np

df = pd.read_csv("results.csv")
print(df.head())
df_Y = df[["layer1", "layer2", "layer3", "layer4", "layer5"]]
df = df.drop(columns=["layer1", "layer2", "layer3", "layer4", "layer5", "name"])

# get data
Y = df_Y.values
X = df.values

clf = LinearRegression()
clf.fit(X, Y)

X_predict = np.array([[0,0,1599, 11, 1.0, 0.0, 0.42399152687677927, 0.29958942885547163, 0, 0, 1]])

prediction = clf.predict(X_predict)
print(prediction)

with open("model/LinearRegressionModel", "wb") as f:
    pickle.dump(clf, f)

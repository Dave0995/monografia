import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("data_clean.csv", sep=";")

X = df.drop(["gasto_familiar"], axis = 1)
Y = df["gasto_familiar"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=7)
reg = LazyRegressor(ignore_warnings=True, random_state=7, verbose=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

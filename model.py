import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv("clean_data.csv", sep = ",")
X = df.drop(["gasto_familiar"], axis = 1)
sc = StandardScaler().fit(X)
x_scaled = sc.transform(X)
Y = df["gasto_familiar"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=7)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'verbose':[2]}
svr = SVR()
model = GridSearchCV(svr, parameters, n_jobs = -1, verbose = 2)
model.fit(X_train, y_train)

final = model.best_estimator_
joblib.dump(final, "best.joblib")
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

df = pd.read_csv("dataF.csv", sep = ",")

X = df.drop(["gasto_familiar"], axis = 1)
sc = StandardScaler().fit(X)
x_scaled = sc.transform(X)
Y = df["gasto_familiar"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=7)

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'verbose':[2]}
#svr = SVR()
#model = GridSearchCV(svr, parameters, n_jobs = -1, verbose = 2)
#model.fit(X_train, y_train)

#final = model.best_estimator_

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2, verbose=True))
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"El MAPE es :{mape}")

joblib.dump(regr, "regr.joblib")

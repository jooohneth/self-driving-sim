import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# DATA
data = pd.read_csv("Session10/iris.data", header=None)
data.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

X = data.drop(columns=['species'])
y = data['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# MODEL
model = KNeighborsClassifier()
model.fit(X_train, y_train)


# EVALUATE
predictions = model.predict(X_test)
print(f'accuracy:{accuracy_score(y_test, predictions)*100}')

joblib.dump(model, 'Session10/model.z')
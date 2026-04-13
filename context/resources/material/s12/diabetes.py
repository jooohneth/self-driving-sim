import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# DATA
data = pd.read_csv("Session11\diabetes.csv")
# print(data.info())
# print(data.describe())

print(data.columns)
not_null = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']

for i in not_null:
    data[i] = data[i].replace(0, np.nan)
    mean = data[i].mean(skipna=True)
    data[i] = data[i].replace(np.nan, mean)

X = data.drop(columns=['Outcome'])
y = data['Outcome']



# print(data['Glucose'].min())

X_train, X_test, y_train, y_test = train_test_split(X, y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# MODEL
model = KNeighborsClassifier()
model.fit(X_train, y_train)


# EVALUATE
predictions = model.predict(X_test)
print(f'accuracy={accuracy_score(y_test, predictions)*100}')
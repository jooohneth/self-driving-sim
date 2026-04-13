import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



# DATA
data = pd.read_csv("Session12\diabetes.csv")
# print(data)

# print(data.info())
# print(data.describe())
# print(data.columns)
not_zero = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']


for column in not_zero:
    data[column] = data[column].replace(0, np.nan)
    mean = data[column].mean(skipna=True)
    data[column] = data[column].replace(np.nan, mean)



X = data.drop(columns=['Outcome'])
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# MODEL
model = LogisticRegression()
model.fit(X_train, y_train)



# EVALUATE
predictions = model.predict(X_test)
print(f'accuracy={accuracy_score(y_test, predictions)*100}')
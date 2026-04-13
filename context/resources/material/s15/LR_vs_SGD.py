import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# DATA
data = pd.read_csv("Session14\student_scores.csv")
# print(data.describe())
# print(data.info())

X = data.drop(columns=['Scores'])
y = data['Scores']


# plt.scatter(X, y)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# MODEL
model = LinearRegression()
model.fit(X_train, y_train)

model_opt = SGDRegressor()
model_opt.fit(X_train, y_train)


# EVALUATE
print(f'y = {model.coef_[0]} * X + {model.intercept_}')
predictions = model.predict(X_test)
predictions_opt = model_opt.predict(X_test)

print(f'MAE = {mean_absolute_error(y_test, predictions)}')
print(f'MSE = {mean_squared_error(y_test, predictions)}')

print(f'MAE opt = {mean_absolute_error(y_test, predictions_opt)}')
print(f'MSE opt = {mean_squared_error(y_test, predictions_opt)}')
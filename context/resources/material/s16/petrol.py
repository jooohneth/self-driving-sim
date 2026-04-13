import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# DATA
data = pd.read_csv("Session16\petrol_consumption.csv")

X = data.drop(columns=['Petrol_Consumption'])
y = data['Petrol_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# MODEL
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

opt_model = SGDRegressor()
opt_model.fit(X_train, y_train)

# EVALUATE
lr_predictions = lr_model.predict(X_test)
print(f"LR MAE = {mean_absolute_error(y_test, lr_predictions)}")


opt_predictions = opt_model.predict(X_test)
print(f"OPT MAE = {mean_absolute_error(y_test, opt_predictions)}")


print(f'y = {lr_model.coef_[0]}*X1 + {lr_model.coef_[1]}*X2 + {lr_model.coef_[2]}*X3 + {lr_model.coef_[3]}*X4 +{lr_model.intercept_}')
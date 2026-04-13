import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))

# handle 0 values in population, two steps: replace 0s with NaN so we don't account them in the mean, 
# replace NaNs with the population's mean value 
dataset["population"] = dataset["population"].replace(0, np.nan)
dataset["population"] = dataset["population"].fillna(dataset["population"].mean())

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# print(regressor.coef_)
# print(regressor.intercept_)

# x_line = np.linspace(X.min(), X.max(), 100)
# y_line = regressor.coef_[0] * x_line + regressor.intercept_

# plt.scatter(X, y)
# plt.plot(x_line, y_line, color="red")
# plt.title("Population vs Profit")
# plt.xlabel("Population")
# plt.ylabel("Profit")
# plt.show()

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, "prediction": y_pred})
print(df, "\n")
print(f"MAE:  {mean_absolute_error(y_test, y_pred)}")
print(f"MSE:  {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

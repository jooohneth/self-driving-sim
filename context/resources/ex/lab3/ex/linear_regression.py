import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


dataset = pd.read_csv("S12_student_scores.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

plt.scatter(X, y)
plt.title("Hours vs Marks")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.coef_)
print(regressor.intercept_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual':  y_test, "prediction": y_pred})

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MAE: {mean_squared_error(y_test, y_pred)}")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

train = pd.read_csv("/Users/jxhn/Desktop/asn-2/data/Q2/mnist_train.csv")
test  = pd.read_csv("/Users/jxhn/Desktop/asn-2/data/Q2/mnist_test.csv")

X_train = train.iloc[:, 1:].values / 255  # shape: (60000, 784)
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values / 255    # shape: (10000, 784)
y_test = test.iloc[:, 0].values

# max_iter increased because 784-dim, 10-class problem needs more iterations
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, preds):.4f}")

joblib.dump(model, 'models/logreg.pkl')
print("Saved to models/logreg.pkl")

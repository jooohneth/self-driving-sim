import cv2
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

X_test, y_test = [], []
for address in glob("/Users/jxhn/Desktop/asn-2/data/Q1/test/*/*"):
    image = cv2.imread(address)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image / 255
    X_test.append(image.flatten())
    y_test.append(address.split("/")[8])

X_train, y_train = [], []
for address in glob("/Users/jxhn/Desktop/asn-2/data/Q1/train/*/*"):
    image = cv2.imread(address)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image / 255
    X_train.append(image.flatten())
    y_train.append(address.split("/")[8])

X_train = np.array(X_train)  # shape: (2000, 12288)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc  = encoder.transform(y_test)

# max_iter increased because 12288-dim input needs more iterations to converge
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train_enc)

preds = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test_enc, preds):.4f}")

joblib.dump({'model': model, 'encoder': encoder}, 'models/logreg.pkl')
print("Saved to models/logreg.pkl")

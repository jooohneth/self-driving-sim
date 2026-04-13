import joblib


model = joblib.load('Session10/model.z')

print(model.predict([[2,1.1,4,2.1]]))
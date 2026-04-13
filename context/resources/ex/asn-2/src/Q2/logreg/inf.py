import cv2
import numpy as np
import argparse
import joblib

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-m', '--model', default='models/logreg.pkl', help='Path to saved model')
args = ap.parse_args()

# Preprocess image (must match training pipeline)
image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image / 255
image = image.flatten().reshape(1, -1)  # shape: (1, 784)

model      = joblib.load(args.model)
pred       = model.predict(image)[0]
proba      = model.predict_proba(image)[0]
confidence = proba[pred]

print(f'Prediction : {pred}')
print(f'Confidence : {confidence:.4f}')

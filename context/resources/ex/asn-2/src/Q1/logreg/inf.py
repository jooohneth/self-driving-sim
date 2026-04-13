import cv2
import numpy as np
import argparse
import joblib

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-m', '--model', default='models/logreg.pkl', help='Path to saved model')
args = ap.parse_args()

# Preprocess image (must match training pipeline)
image = cv2.imread(args.image)
image = cv2.resize(image, (64, 64))
image = image / 255
image = image.flatten().reshape(1, -1)  # shape: (1, 12288)

data    = joblib.load(args.model)
model   = data['model']
encoder = data['encoder']

pred       = model.predict(image)[0]
proba      = model.predict_proba(image)[0]
label      = encoder.inverse_transform([pred])[0]
confidence = proba[pred]

print(f'Prediction : {label}')
print(f'Confidence : {confidence:.4f}')

import numpy as np
import cv2
import argparse
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image (grayscale digit)')
ap.add_argument('-m', '--model', default='models/mnist.keras', help='Path to trained model')
args = ap.parse_args()

# Preprocess image (must match training pipeline)
image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image.astype('float32') / 255.0
image = image.reshape(1, 28, 28, 1)  # shape: (1, 28, 28, 1)

# Load model and predict
model = load_model(args.model)
pred = model.predict(image)[0]

label = np.argmax(pred)
confidence = pred[label]

print(f'Prediction : {label}')
print(f'Confidence : {confidence:.4f}')

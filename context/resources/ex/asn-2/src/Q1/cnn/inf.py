import numpy as np
import cv2
import argparse
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-m', '--model', default='leaky-softmax-cross.h5', help='Path to trained model')
args = ap.parse_args()

# Preprocess image (must match training pipeline)
image = cv2.imread(args.image)
image = cv2.resize(image, (64, 64))
image = image / 255
image = np.expand_dims(image, axis=0)  # shape: (1, 64, 64, 3)

# Load model and predict
model = load_model(args.model)
pred = model.predict(image)[0]

classes = ['Cat', 'Dog']
label = classes[np.argmax(pred)]
confidence = pred[np.argmax(pred)]

print(f'Prediction : {label}')
print(f'Confidence : {confidence:.4f}')

import numpy as np
import cv2
from keras.models import load_model


MODEL_PATH = '/Users/jxhn/Desktop/asn-2/src/Q3/captcha_model.keras'
ASSET_PATH = '/Users/jxhn/Desktop/asn-2/src/Q3/assets/digits.png'
CLASSES    = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

model = load_model(MODEL_PATH)

# LOAD CAPTCHA IMAGE
captcha_img = cv2.imread(ASSET_PATH)

# SEGMENT DIGITS WITH findContours
gray = cv2.cvtColor(captcha_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digit_boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w * h >= 50:
        digit_boxes.append((x, y, w, h))

digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

# INFERENCE
predicted_digits = []
for (x, y, w, h) in digit_boxes:
    roi = captcha_img[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.copyMakeBorder(roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    roi = cv2.resize(roi, (16, 16))
    roi = roi / 255.0
    roi = roi.flatten().reshape(1, -1)

    pred = model.predict(roi, verbose=0)[0]
    predicted_digits.append(CLASSES[np.argmax(pred)])

print(f"Predicted CAPTCHA: {''.join(predicted_digits)}")

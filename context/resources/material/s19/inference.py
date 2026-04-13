import numpy as np
import cv2
from keras.models import load_model

image = cv2.imread("Session17/test.jpg")
image = cv2.resize(image, (64,64))
image = image/255
image = image.flatten()

model = load_model("Session19/fire_nn.h5")
prediction = model.predict(np.array([image]))
label = np.argmax(prediction[0])
if label==0:
    print('fire')
else:
    print("no fire")
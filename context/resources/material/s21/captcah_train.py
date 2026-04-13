import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import layers, Sequential
import matplotlib.pyplot as plt


# DATA
data_list = []
label_list = []

for i, address in enumerate(glob.glob("Session21/kapcha/kapcha/*/*")):
    image = cv2.imread(address)
    image = cv2.resize(image, (16,16))
    image = image/255
    image = image.flatten()

    data_list.append(image)
    label_list.append(address.split("\\")[1])


X = np.array(data_list)
y = np.array(label_list)

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_test)



# MODEL
model = Sequential([
    layers.Dense(8, activation='relu'),
    layers.Dense(15, activation='relu'),
    layers.Dense(9, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)



# EVALUATE
plt.plot(np.arange(15), H.history['accuracy'], label='train accuracy')
plt.plot(np.arange(15), H.history['val_accuracy'], label='test accuracy')
plt.plot(np.arange(15), H.history['loss'], label='train loss')
plt.plot(np.arange(15), H.history['val_loss'], label='test loss')
plt.legend()
plt.show()

import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import layers, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


# DATA
data_list = []
label_list = []

for address in glob.glob("/Users/jxhn/Desktop/asn-2/data/Q3/*/*"):
    image = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (16, 16))
    image = image / 255
    image = image.flatten()
    data_list.append(image)
    label_list.append(address.split("/")[7])

X = np.array(data_list)
y = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(X, y)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# MODEL
model = Sequential([
    layers.Input(shape=(16 * 16,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(9, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('/Users/jxhn/Desktop/asn-2/src/Q3/captcha_model.keras', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, mode='max')
]

H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, callbacks=callbacks)

print('Model saved to /Users/jxhn/Desktop/asn-2/src/Q3/captcha_model.keras')


# EVALUATE
plt.plot(H.history['accuracy'], label='train accuracy')
plt.plot(H.history['val_accuracy'], label='test accuracy')
plt.plot(H.history['loss'], label='train loss')
plt.plot(H.history['val_loss'], label='test loss')
plt.legend()
plt.show()

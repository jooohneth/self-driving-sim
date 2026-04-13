import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense
from keras import Sequential, layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import save_model
import matplotlib.pyplot as plt
from keras.optimizers import SGD

# DATA
data_list = []
label_list = []


for i, address in enumerate(glob.glob("Session17/fire_dataset/*/*")):
    image = cv2.imread(address)
    image = cv2.resize(image, (64,64))
    image = image/255
    image = image.flatten()


    data_list.append(image)

    label_list.append(address.split('\\')[1])

    if i%200==0:
        print(f'[INFO] {i} images processed!')

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
    layers.Dense(20, activation='leaky_relu'),
    layers.Dense(8, activation='leaky_relu'),
    layers.Dense(2, activation='softmax')
])

# opt = SGD(learning_rate=0.0001)
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)


# EVALUATE

plt.plot(np.arange(0, 10), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, 10), H.history['val_accuracy'], label='val accuracy')
plt.plot(np.arange(0, 10), H.history['loss'], label='loss')
plt.plot(np.arange(0, 10), H.history['val_loss'], label='val loss')
plt.legend()
plt.show()

save_model(model, 'Session19/fire_nn.h5')
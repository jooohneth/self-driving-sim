import cv2
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


X_test, y_test = [], []
for i, address in enumerate(glob("/Users/jxhn/Downloads/Assignment2/Q1/test/*/*")):
    image = cv2.imread(address)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image / 255
    X_test.append(image)
    y_test.append(address.split("/")[8])

X_train, y_train = [], []
for i, address in enumerate(glob("/Users/jxhn/Downloads/Assignment2/Q1/train/*/*")):
    image = cv2.imread(address)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image / 255
    X_train.append(image)
    y_train.append(address.split("/")[8])

X_train = np.array(X_train)  # shape: (2000, 64, 64, 3)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Split 20% of training data for validation
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
X_val, y_val = X_test, y_test

y_train = to_categorical(y_train)
y_val   = to_categorical(y_val)
y_test  = to_categorical(y_test)

# MODEL
model = Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

callbacks = [
    ModelCheckpoint('models/fifth.keras', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, mode='max')
]

H = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=callbacks)

# EVALUATE
plt.plot(H.history['accuracy'], label='accuracy')
plt.plot(H.history['val_accuracy'], label='val accuracy')
plt.plot(H.history['loss'], label='loss')
plt.plot(H.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# best model already saved by ModelCheckpoint

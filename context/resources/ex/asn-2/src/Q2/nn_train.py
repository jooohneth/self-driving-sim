import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


# Load CSV: column 0 = label, columns 1-784 = pixels (28x28 flattened)
train_df = pd.read_csv('../../data/Q2/mnist_train.csv', header=None)
test_df  = pd.read_csv('../../data/Q2/mnist_test.csv',  header=None)

X_train = train_df.iloc[:, 1:].values.astype('float32') / 255.0  # (60000, 784)
y_train = train_df.iloc[:, 0].values

X_test  = test_df.iloc[:, 1:].values.astype('float32') / 255.0   # (10000, 784)
y_test  = test_df.iloc[:, 0].values

# Reshape to (N, 28, 28, 1) for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test,  num_classes=10)

# MODEL
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

callbacks = [
    ModelCheckpoint('models/mnist.keras', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
]

H = model.fit(X_train, y_train, validation_data=(X_test, y_test),
              batch_size=128, epochs=50, callbacks=callbacks)

plt.plot(H.history['accuracy'],     label='accuracy')
plt.plot(H.history['val_accuracy'], label='val accuracy')
plt.plot(H.history['loss'],         label='loss')
plt.plot(H.history['val_loss'],     label='val loss')
plt.legend()
plt.show()

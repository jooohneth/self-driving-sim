from keras import layers, Sequential
from keras.models import save_model
import numpy as np
from keras.applications import VGG16

def train(X_train: np.array, X_test, y_train, y_test, EPOCHS, BATCH_SIZE=1):
    """
    X_train:



    Return
    H: history of the model training
    """
    print("[INFO] TRAINING MODEL...")


    vgg = VGG16(include_top=False,
          weights='imagenet',
          input_shape=(244, 244, 3))
    
    for layer in vgg.layers:
        layer.trainable=False

    net = Sequential([
        vgg,
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

        
    net.compile(loss='categorical_crossentropy',
                optimizer='adam', 
                metrics=['accuracy'])
    
    print(net.summary())

    H = net.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    save_model(net, "covid_detector.h5")

    return H



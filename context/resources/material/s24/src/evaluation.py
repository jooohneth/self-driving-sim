import matplotlib.pyplot as plt
import numpy as np


def result_vis(H, EPOCHS):

    plt.subplot(121)
    plt.plot(np.arange(EPOCHS), H.history['accuracy'], label='train accuracy')
    plt.plot(np.arange(EPOCHS), H.history['val_accuracy'], label='test accuracy')
    plt.title("MODEL ACCURACY")

    plt.subplot(122)
    plt.plot(np.arange(EPOCHS), H.history['loss'], label='train loss')
    plt.plot(np.arange(EPOCHS), H.history['val_loss'], label='test loss')
    plt.title("MODEL LOSS")

    plt.legend()
    plt.show()
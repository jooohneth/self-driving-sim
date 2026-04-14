#!/usr/bin/env python3
"""
nvidia end-to-end cnn for predicting steering angle from a camera image.

5 conv layers, flatten, dropout, 3 dense layers, single linear output.
based on bojarski et al. 2016 "end to end learning for self-driving cars".
"""

from keras import Sequential, layers
from keras.optimizers.legacy import Adam  # legacy adam required on apple silicon

INPUT_SHAPE = (66, 200, 3)   # height × width × channels (YUV)


def build_model(input_shape: tuple = INPUT_SHAPE) -> Sequential:
    """
    builds and compiles the nvidia cnn. input should already be normalized
    to [0, 1] in yuv (dataset.preprocess handles that). elu activations
    throughout, dropout(0.5) after flatten, single linear output.
    """
    model = Sequential([
        # strided 5x5 layers shrink spatial dims fast
        layers.Conv2D(24, (5, 5), strides=(2, 2), padding='valid',
                      activation='elu', input_shape=input_shape),   # 24@31x98
        layers.Conv2D(36, (5, 5), strides=(2, 2), padding='valid',
                      activation='elu'),                            # 36@14x47
        layers.Conv2D(48, (5, 5), strides=(2, 2), padding='valid',
                      activation='elu'),                            # 48@5x22

        # 3x3 layers refine
        layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
                      activation='elu'),                            # 64@3x20
        layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
                      activation='elu'),                            # 64@1x18

        layers.Flatten(),                                           # 1152
        layers.Dropout(0.5),

        layers.Dense(100, activation='elu'),
        layers.Dense(50,  activation='elu'),
        layers.Dense(10,  activation='elu'),

        # single linear output — steering angle
        layers.Dense(1),
    ], name="nvidia_cnn")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mse',      # mean squared error for regression
        metrics=['mae'], # mean absolute error — interpretable in steering-angle units
    )

    model.summary()
    return model


if __name__ == "__main__":
    import numpy as np
    m = build_model()
    dummy = np.zeros((1, *INPUT_SHAPE), dtype="float32")
    out = m.predict(dummy, verbose=0)
    print(f"Output shape: {out.shape}  value: {out[0, 0]:.4f}")

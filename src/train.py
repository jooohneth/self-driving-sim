#!/usr/bin/env python3
"""
training pipeline for the nvidia cnn.

loads data, trains the model, plots loss curves, saves the best checkpoint.
run with: uv run src/train.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_data
from model import build_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BATCH_SIZE = 32
EPOCHS     = 30
MODEL_DIR  = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.h5"  # .keras format doesn't work with modelcheckpoint on tf-macos 2.14


def plot_history(H) -> None:
    """
    two subplots: training vs validation loss (mse) on the left,
    training vs validation mae on the right.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(H.history['loss'],     label='Training Loss')
    axes[0].plot(H.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()

    axes[1].plot(H.history['mae'],     label='Training MAE')
    axes[1].plot(H.history['val_mae'], label='Validation MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (steering angle units)')
    axes[1].legend()

    plt.suptitle('Training Performance', fontsize=13)
    plt.tight_layout()
    plt.show()


def main() -> None:
    train_gen, val_gen, n_train, n_val = get_data(batch_size=BATCH_SIZE)

    steps_per_epoch  = n_train // BATCH_SIZE
    validation_steps = n_val   // BATCH_SIZE

    print(f"\nsteps per epoch : {steps_per_epoch}")
    print(f"validation steps: {validation_steps}\n")

    model = build_model(input_shape=(66, 200, 3))

    MODEL_DIR.mkdir(exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            str(MODEL_PATH),
            save_best_only=True,
            monitor='val_loss',
            verbose=1,
        ),
        EarlyStopping(
            patience=10,
            monitor='val_loss',
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    H = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(H)

    best_val_loss = min(H.history['val_loss'])
    best_val_mae  = min(H.history['val_mae'])
    print(f"\nBest val_loss : {best_val_loss:.6f}")
    print(f"Best val_mae  : {best_val_mae:.4f}  (steering angle units)")
    print(f"Model saved   : {MODEL_PATH}")


if __name__ == "__main__":
    main()

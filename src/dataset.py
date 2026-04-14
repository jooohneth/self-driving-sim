#!/usr/bin/env python3
"""
loads, balances, preprocesses, and batches the driving dataset.
"""

import csv
import sys
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Augment is applied inside the training generator only
sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import augment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH     = PROJECT_ROOT / "data" / "driving_log.csv"

# crop rows and output size
CROP_TOP    = 60
CROP_BOTTOM = 135
IMG_WIDTH   = 200
IMG_HEIGHT  = 66

# balancing
SAMPLES_PER_BIN = 400
NUM_BINS        = 25



CORRECTION = 0.2  # steering correction for left/right cameras


def load_samples(csv_path: Path = CSV_PATH) -> list:
    """
    reads driving_log.csv and returns a list of (image_path, steering_angle) pairs.

    uses all three cameras. left and right get a fixed correction so the model
    learns to recover when it drifts off center:
      left  -> angle + 0.2 (needs to steer right to get back)
      right -> angle - 0.2 (needs to steer left to get back)

    triples the dataset and is the main reason the model handles curves better.
    """
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                angle = float(row[3])
            except ValueError:
                continue  # skip header if present
            center_path = row[0].strip()
            samples.append((center_path, angle))

            # left and right cameras with correction
            if len(row) >= 3:
                left_path  = row[1].strip()
                right_path = row[2].strip()
                if left_path:
                    samples.append((left_path,  angle + CORRECTION))
                if right_path:
                    samples.append((right_path, angle - CORRECTION))

    return samples


def balance_samples(
    samples: list,
    samples_per_bin: int = SAMPLES_PER_BIN,
    num_bins: int = NUM_BINS,
) -> list:
    """
    trims bins with too many samples so no bin exceeds samples_per_bin.
    the raw data has a big spike at 0.0 (most driving is straight) that
    would bias the model to ignore turns.
    """
    angles = np.array([s[1] for s in samples])
    bin_edges = np.linspace(-1.0, 1.0, num_bins + 1)

    keep_indices = []
    for i in range(num_bins):
        in_bin = np.where((angles >= bin_edges[i]) & (angles < bin_edges[i + 1]))[0]
        if len(in_bin) > samples_per_bin:
            in_bin = np.random.choice(in_bin, samples_per_bin, replace=False)
        keep_indices.extend(in_bin.tolist())

    np.random.shuffle(keep_indices)
    balanced = [samples[i] for i in keep_indices]
    print(f"Balancing: {len(samples):,} → {len(balanced):,} samples "
          f"(threshold: {samples_per_bin}/bin, {num_bins} bins)")
    return balanced


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    crops to the road area, converts bgr to yuv, resizes to 200x66,
    applies gaussian blur, and normalizes to [0, 1].
    """
    image = image[CROP_TOP:CROP_BOTTOM, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0
    return image.astype(np.float32)


def batch_generator(samples: list, batch_size: int = 32, training: bool = True):
    """
    infinite generator that yields (images, angles) batches.
    training=true applies augmentation. validation skips it.
    shuffles at the start of each epoch.
    """
    samples = list(samples)  # local copy so shuffle doesn't affect caller

    while True:
        np.random.shuffle(samples)

        for offset in range(0, len(samples), batch_size):
            batch = samples[offset:offset + batch_size]
            images, angles = [], []

            for path, angle in batch:
                img = cv2.imread(str(PROJECT_ROOT / path))
                if img is None:
                    continue  # skip missing frames

                if training:
                    img, angle = augment(img, angle)

                img = preprocess(img)
                images.append(img)
                angles.append(float(angle))

            if images:
                yield np.array(images, dtype=np.float32), np.array(angles, dtype=np.float32)


def get_data(
    csv_path: Path = CSV_PATH,
    balance: bool = True,
    val_split: float = 0.2,
    batch_size: int = 32,
):
    """
    full pipeline: load, balance, split, return generators.
    """
    samples = load_samples(csv_path)
    print(f"Loaded: {len(samples):,} samples")

    if balance:
        samples = balance_samples(samples)

    train_samples, val_samples = train_test_split(samples, test_size=val_split)
    print(f"Split:  {len(train_samples):,} train / {len(val_samples):,} val")

    train_gen = batch_generator(train_samples, batch_size=batch_size, training=True)
    val_gen   = batch_generator(val_samples,   batch_size=batch_size, training=False)

    return train_gen, val_gen, len(train_samples), len(val_samples)


# sanity check

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_gen, val_gen, n_train, n_val = get_data(batch_size=8)

    images, angles = next(train_gen)
    print(f"Batch shape : {images.shape}")   # (8, 66, 200, 3)
    print(f"Angle range : [{angles.min():.3f}, {angles.max():.3f}]")
    print(f"dtype       : {images.dtype}")

    # show the first 8 augmented frames
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    for ax, img, ang in zip(axes.flat, images, angles):
        # YUV → BGR → RGB for display
        display = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_YUV2RGB)
        ax.imshow(display)
        ax.set_title(f"angle={ang:.3f}", fontsize=8)
        ax.axis("off")
    plt.suptitle("Training batch (augmented + preprocessed, YUV→RGB for display)")
    plt.tight_layout()
    plt.show()

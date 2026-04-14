#!/usr/bin/env python3
"""
data augmentation for training images.

each function takes (image, angle) in bgr and returns the same pair with a
transform applied. augmentation runs before preprocessing so zoom and pan
work on the full-res image before the crop step cuts it down.
"""

import cv2
import numpy as np


def random_flip(image, angle):
    """
    flips horizontally with 50% chance. when the image flips the steering
    angle flips too — otherwise a right turn becomes a left turn.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle


def random_brightness(image, angle):
    """
    randomly scales brightness via the hsv v-channel. done in hsv because
    augmentation runs before the yuv conversion. range 0.4-1.2 keeps things
    from going fully dark or blown out.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = np.random.uniform(0.4, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), angle


def random_zoom(image, angle):
    """
    zooms in by a random factor between 1.0 and 1.3 then center-crops back to
    the original size. steering angle stays the same.
    """
    h, w = image.shape[:2]
    factor = np.random.uniform(1.0, 1.3)
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(image, (new_w, new_h))
    y0 = (new_h - h) // 2
    x0 = (new_w - w) // 2
    return resized[y0:y0 + h, x0:x0 + w], angle


def random_pan(image, angle):
    """
    shifts the image horizontally (up to 15% of width) and vertically (up to 10%
    of height). horizontal shift moves where the car appears to sit on the road,
    so a small steering correction gets added (0.002 per pixel).
    """
    h, w = image.shape[:2]
    tx = np.random.uniform(-0.15 * w, 0.15 * w)
    ty = np.random.uniform(-0.10 * h, 0.10 * h)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h))
    angle += tx * 0.002
    return image, angle


def random_rotation(image, angle):
    """
    rotates up to +/-10 degrees. this simulates camera tilt, not road curvature,
    so the steering angle is left alone.
    """
    h, w = image.shape[:2]
    deg = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    return image, angle


def augment(image, angle):
    """
    runs the full augmentation pipeline on one sample. flip is always
    attempted (50% chance). everything else has a 50% chance of applying.
    """
    image, angle = random_flip(image, angle)
    if np.random.rand() < 0.5:
        image, angle = random_brightness(image, angle)
    if np.random.rand() < 0.5:
        image, angle = random_zoom(image, angle)
    if np.random.rand() < 0.5:
        image, angle = random_pan(image, angle)
    if np.random.rand() < 0.5:
        image, angle = random_rotation(image, angle)
    return image, angle


# quick visual sanity check
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CSV_PATH = PROJECT_ROOT / "data" / "driving_log.csv"

    import csv as _csv
    with open(CSV_PATH, newline="") as f:
        first_row = next(_csv.reader(f))
    img_path = PROJECT_ROOT / first_row[0]
    angle = float(first_row[3])

    original = cv2.imread(str(img_path))
    if original is None:
        print(f"Could not load {img_path}", file=sys.stderr)
        sys.exit(1)

    transforms = [
        ("Original",          original,                      angle),
        ("Flip",              *random_flip(original.copy(),  angle)),
        ("Brightness",        *random_brightness(original.copy(), angle)),
        ("Zoom",              *random_zoom(original.copy(),  angle)),
        ("Pan",               *random_pan(original.copy(),   angle)),
        ("Rotation",          *random_rotation(original.copy(), angle)),
        ("All (augment())",   *augment(original.copy(),      angle)),
    ]

    fig, axes = plt.subplots(1, len(transforms), figsize=(20, 3))
    for ax, (name, img, ang) in zip(axes, transforms):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}\nangle={ang:.3f}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

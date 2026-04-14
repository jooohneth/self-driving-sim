#!/usr/bin/env python3
"""
plots the steering angle distribution from driving_log.csv.

a big spike at 0.0 means too much straight driving data. the red line
shows where the balance threshold cuts things off.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "driving_log.csv"

# max samples per bin, shown as a red line on the histogram
BALANCE_THRESHOLD = 200


def load_steering_angles(csv_path: Path) -> np.ndarray:
    angles = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                angles.append(float(row[3]))
            except ValueError:
                continue  # skip header if there is one
    return np.array(angles)


def plot_histogram(angles: np.ndarray, threshold: int = BALANCE_THRESHOLD) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    n, bins, patches = ax.hist(
        angles,
        bins=100,
        range=(-1.0, 1.0),
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )

    # balance threshold line
    ax.axhline(
        threshold,
        color="red",
        linewidth=1.5,
        linestyle="--",
        label=f"Balance threshold ({threshold} samples/bin)",
    )

    ax.set_xlabel("Steering Angle")
    ax.set_ylabel("Frame Count")
    ax.set_title("Steering Angle Distribution")
    ax.set_xlim(-1.05, 1.05)
    ax.legend()

    # stats annotation
    over = int(np.sum(n > threshold))
    pct_straight = float(np.sum(np.abs(angles) < 0.05) / len(angles) * 100)
    stats = (
        f"Total frames: {len(angles):,}\n"
        f"Near-zero (|angle| < 0.05): {pct_straight:.1f}%\n"
        f"Bins exceeding threshold: {over}"
    )
    ax.text(
        0.98, 0.97, stats,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import load_samples, balance_samples

    raw_angles = load_steering_angles(CSV_PATH)
    balanced_angles = np.array([a for _, a in balance_samples(load_samples(CSV_PATH))])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, angles, title in zip(
        axes,
        [raw_angles, balanced_angles],
        ["Raw Distribution", "After Balancing"],
    ):
        n, _, _ = ax.hist(angles, bins=100, range=(-1.0, 1.0),
                          color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axhline(BALANCE_THRESHOLD, color="red", linewidth=1.5,
                   linestyle="--", label=f"Threshold ({BALANCE_THRESHOLD})")
        over = int(np.sum(n > BALANCE_THRESHOLD))
        pct_zero = float(np.sum(np.abs(angles) < 0.05) / len(angles) * 100)
        ax.text(0.98, 0.97,
                f"Total: {len(angles):,}\nNear-zero: {pct_zero:.1f}%\nBins over threshold: {over}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
        ax.set_title(title)
        ax.set_xlabel("Steering Angle")
        ax.set_ylabel("Frame Count")
        ax.set_xlim(-1.05, 1.05)
        ax.legend()

    plt.tight_layout()
    plt.show()

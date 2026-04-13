# Session 24 — Transfer Learning & COVID-19 Detection (In-Class Project)

**Course:** DPS920  
**Instructor:** Ellie Azizi  
**Format:** Live online lecture (remote)  
**Topics:** Transfer learning, VGG16 as base model, modular project structure, COVID-19 chest X-ray classification, false positive/negative trade-offs, segmentation as preprocessing

---

## 1. Administrative Announcements

### Quiz 3

- Now available on Blackboard — two research-based questions, no time limit, due by tomorrow night
- One of the two questions is covered in today's session (transfer learning)
- The other was discussed previously but requires deeper independent study
- Both concepts are included in the final exam — familiarity is required, not just look-up

### Final Exam

- Date: **April 13th** (not next week — the week after)
- Covers everything from beginning to end
- ~80% from the second half of the course; ~20% from before midterm

### Remaining Sessions

- 2 sessions remain (April 6th and 7th)
- Session plan:
  - One session: go over sample final exam questions together
  - Other session: Q&A, or any topics students want to revisit
- **MediaPipe** likely skipped — very similar to Ultralytics; instructor sees limited additional value in covering it separately. If time permits, it may be introduced briefly.

### Final Project — Mac Dependencies

**Student (Jayant):** The `package_list.txt` dependencies are Windows-specific, not Mac-compatible. I had to create a cross-platform configuration — is it okay to use my modified versions?

**Instructor:** Definitely — the package list is just a guide for those having dependency issues; those versions worked on my Windows setup. You can change versions freely. If you need help, I can ask previous students what worked on Mac.

**Jayant:** If any Mac users need the working dependency versions, I can share my `.txt` files.

**Instructor:** Thank you — please do. For Mac users: the original `package_list.txt` is Windows-only. Refer to Jayant for Mac-compatible versions.

> **Note:** One issue also reported: the assignment 2 zip file couldn't be unzipped by some students. A separately uploaded file works — download that one instead.

---

## 2. Brief Review — Tracking & Sequence Architectures

**Instructor:** From last session, we covered classification, segmentation, object detection, and briefly mentioned tracking and pose estimation.

**Pose estimation** in Ultralytics is just a function call — very similar to detection. Tracking is harder conceptually:

### Why Tracking Is Harder Than Detection

Detection processes one frame independently. Tracking must:
1. Detect an object in frame N
2. Identify that same object in frame N+1, N+2, etc.
3. Assign a persistent ID that doesn't change across frames

This requires processing **sequences** of data, not independent frames.

### Architectures for Sequence Processing

| Architecture | Notes |
|---|---|
| **RNN** (Recurrent Neural Network) | Has a dedicated weight for memorizing sequence context; updates both feature weights and sequence-memory weights simultaneously |
| **LSTM** (Long Short-Term Memory) | Proposed by Jürgen Schmidhuber. Has explicit "forget" and "memorize" gates that learn what to keep and what to discard. More complex than Transformers in some ways. |
| **GRU** (Gated Recurrent Unit) | Simplified extension of LSTM |
| **Transformer** | Also processes sequences; now the dominant architecture in NLP and multimodal tasks |

All of these learn — through parameter updates — what to memorize and what to forget. The difference from standard CNNs is the mechanism for tracking sequential context across time steps. These architectures are commonly applied to text (which is a sequence) and video (also a sequence of frames).

---

## 3. In-Class Project — COVID-19 Detection from Chest X-Rays

### Dataset

Downloaded from Blackboard: `COVID-19 dataset.zip`

Structure after unzipping:
```
COVID-19 dataset/
├── COVID/        ← chest X-rays of COVID-positive patients
└── Normal/       ← chest X-rays of healthy patients
```

### Problem Framing

**Is this solvable by machine learning?** Yes.

**Supervised, unsupervised, or reinforcement learning?** Supervised — we have labeled examples (COVID vs. Normal).

**Classification or regression?** Classification — binary output: COVID or Normal. Despite being called "detection," the task is classifying whether a condition is present.

**Which algorithm?**

The instinctive answer is CNNs. But the right process is:

> Start simple → build up from there.

- Try logistic regression first — if the class boundary is obvious in feature space, it may suffice and would be faster at inference
- In this dataset, the visual difference between COVID and Normal X-rays is **not** obvious to the human eye — the features are subtle, fine-grained, and similar in pixel distribution across both classes (all black-and-white, similar structural patterns)
- Contrast with fire detection: colored images, obvious red/green distribution difference → logistic regression worked there
- For fine-grained medical imaging: CNNs are the right choice

### Key Constraint — Small Dataset

**Student:** Small amount of data.

**Instructor:** Exactly — the biggest constraint in medical imaging. Reasons:
- Patient data is confidential; approval to use/publish takes months or years
- Same issue in any user-data domain (HCI, games with player behavior data, etc.)

**Consequences of small data:**
- The more complex the model, the more data it needs
- CNNs with sparse data suffer from **data sparsity** — the data is so scattered the model can't distinguish between class distributions reliably
- Even if you achieve 95–98% accuracy, you must question: does this small sample represent real-world variability?
- After train-test split, training data is even smaller

---

## 4. Project Structure

Following the s24 modular layout (mirrors VGG / research-grade projects):

```
src/
├── project.py          ← main entry point; imports and calls all modules
├── preprocessing.py    ← reads data, resizes, normalizes, splits, encodes labels
├── model.py            ← defines and trains the CNN; returns history
└── evaluation.py       ← plots accuracy and loss curves
data/
└── COVID-19 dataset/
    ├── COVID/
    └── Normal/
```

---

## 5. `preprocessing.py`

```python
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def data_preprocessing(path: str):
    """
    Reads images from subdirectories of `path`, resizes, normalizes,
    splits into train/test, and one-hot encodes labels.

    Args:
        path (str): Root directory containing class subdirectories.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    data_list = []
    label_list = []

    for i, address in enumerate(glob.glob(path + '/**/*')):
        image = cv2.imread(address)
        if image is None:
            continue
        image = cv2.resize(image, (224, 224))   # must match VGG16 input; use 64 if not using transfer learning
        image = image / 255.0                   # normalize to [0, 1]
        data_list.append(image)
        label_list.append(address.split('\\')[1])  # extracts 'COVID' or 'Normal' from path

    X = np.array(data_list)
    y = np.array(label_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Encode string labels → integers → one-hot vectors
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test
```

> **Why `split('\\')[1]`?** The path looks like `data\COVID\image.jpg` — splitting on backslash and taking index 1 gives `'COVID'` or `'Normal'`.

> **Image size note:** Use `(64, 64)` if training your own CNN from scratch. Must use `(224, 224)` if using VGG16 as the base model — VGG16 was trained on 224×224 ImageNet images and expects that input shape.

---

## 6. `model.py`

### Option A — Custom CNN (from scratch)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def train(X_train: np.ndarray, X_test: np.ndarray,
          y_train: np.ndarray, y_test: np.ndarray,
          epochs: int = 10, batch_size: int = 1):
    """
    Builds and trains a CNN for binary image classification.

    Args:
        X_train: Training images, shape (N, H, W, C)
        X_test:  Test images
        y_train: One-hot encoded training labels
        y_test:  One-hot encoded test labels
        epochs:  Number of training epochs (default 10)
        batch_size: Batch size (default 1 = full gradient descent)

    Returns:
        history: Keras training history object
    """
    net = Sequential([
        layers.Conv2D(6, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax'),   # 2 classes: COVID, Normal
    ])

    net.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    history = net.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    return history
```

**Instructor:** A `model` is the result of training. The architecture you define (`net`) is the network/architecture — not yet a model until it's been trained.

**On batch size:**
- Default here is `batch_size=1` (full gradient descent — one sample per update)
- Preferred convention: use powers of 2 (2, 4, 8, 16, 32, 64, 128) — more memory-efficient when loading data
- Given the small dataset, batch size 1 is acceptable here
- If `batch_size` is not passed, the keyword argument default of `1` applies

### Option B — Transfer Learning with VGG16 (preferred for small datasets)

```python
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def train(X_train: np.ndarray, X_test: np.ndarray,
          y_train: np.ndarray, y_test: np.ndarray,
          epochs: int = 10, batch_size: int = 1):
    """
    Trains a COVID classifier using VGG16 as a frozen feature extractor
    with a custom classification head.

    Args:
        X_train: Training images, shape (N, 224, 224, 3)  ← must be 224x224 for VGG16
        X_test:  Test images
        y_train: One-hot encoded training labels
        y_test:  One-hot encoded test labels
        epochs:  Training epochs
        batch_size: Batch size

    Returns:
        history: Keras training history object
    """
    # Load VGG16 pre-trained on ImageNet, without its original top (classifier) layer
    base_model = VGG16(
        weights='imagenet',
        include_top=False,       # exclude VGG's 1000-class softmax head
        input_shape=(224, 224, 3)
    )

    # Freeze all VGG16 layers — do not update their weights during training
    for layer in base_model.layers:
        layer.trainable = False

    # Build the full model: frozen VGG16 base + custom classification head
    net = Sequential([
        base_model,
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax'),   # 2 classes: COVID, Normal
    ])

    net.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    print(net.summary())

    history = net.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    return history
```

**On `include_top=False`:** VGG16 was originally trained on ImageNet with 1,000 output classes. We have 2 classes (COVID, Normal). Setting `include_top=False` removes VGG16's final Dense + Softmax layers so we can attach our own.

**On `weights='imagenet'`:** Loads parameters that were trained on ImageNet (14M images). Other weight options exist; ImageNet is the most common and reliable starting point.

**On saving the model:**

```python
from tensorflow.keras.models import save_model

save_model(net, 'COVID_detector.h5')
# Save to a dedicated weights/ directory in practice
```

---

## 7. `evaluation.py`

```python
import numpy as np
import matplotlib.pyplot as plt


def results(history, epochs: int):
    """
    Plots training and validation accuracy and loss curves side-by-side.

    Args:
        history: Keras History object returned by model.fit()
        epochs:  Number of training epochs (used for x-axis range)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy subplot
    axes[0].plot(np.arange(0, epochs), history.history['accuracy'],    label='Train Accuracy')
    axes[0].plot(np.arange(0, epochs), history.history['val_accuracy'], label='Test Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()

    # Loss subplot
    axes[1].plot(np.arange(0, epochs), history.history['loss'],     label='Train Loss')
    axes[1].plot(np.arange(0, epochs), history.history['val_loss'], label='Test Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results.png')   # save figure; plt.show() optional
    plt.show()
```

> **Why subplots instead of one plot?** Loss and accuracy can have very different scales. Showing them on the same axes can make one curve unreadable. Subplots keep each metric on its own scale.

---

## 8. `project.py` (Main Entry Point)

```python
from preprocessing import data_preprocessing
from model import train
from evaluation import results

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 1

# Step 1 — Load and preprocess data
DATA_PATH = 'data/COVID-19 dataset/**/*'
X_train, X_test, y_train, y_test = data_preprocessing(DATA_PATH)

print("Data preprocessed.")

# Step 2 — Train model
h = train(X_train, X_test, y_train, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

print("Training complete.")

# Step 3 — Evaluate and plot
results(h, EPOCHS)
```

---

## 9. Transfer Learning — Concept

**Instructor:** *(with a drawn diagram)*

The idea: use a model that was already trained on a large dataset as a feature extractor, then attach a small custom head for your specific problem.

### Why It Works

As a CNN is trained, its layers learn to detect progressively more detailed features:
- **Early layers:** edges, lines, basic shapes
- **Middle layers:** textures, curves, partial shapes
- **Later layers:** domain-specific features (e.g., fur, wheels, tumors)

A model trained on ImageNet (14M images, 1,000 classes) has already learned to extract general visual features. Even though fire detection and COVID detection seem unrelated, the general concept of "what is a shape," "what is a texture," "what is a boundary" transfers across domains.

```
Pre-trained model (e.g., VGG16 trained on ImageNet)
┌────────────────────────────────────┐
│  CNN Block 1                       │   ← frozen (trainable=False)
│  CNN Block 2                       │   ← frozen
│  CNN Block 3                       │   ← frozen
│  ...                               │   ← frozen
└────────────────────────────────────┘
           ↓
┌────────────────────────────────────┐
│  MaxPool → Flatten → Dense → Dense │   ← trainable (your custom head)
│  Softmax(2)  ← COVID, Normal       │
└────────────────────────────────────┘
```

### Two Approaches to Transfer Learning

| Approach | Description | When to use |
|---|---|---|
| **Freeze base** (`trainable=False`) | Only update the custom head's parameters | Small dataset; faster training; prevents overwriting strong pre-trained features |
| **Fine-tune all** (`trainable=True`) | Update all parameters — base model + custom head | Larger dataset; want domain-specific feature refinement; requires GPU |

In this project: base frozen, only the custom Dense layers are updated.

### Parameter Count (from `model.summary()`)

```
VGG16 base:          ~14,700,000 parameters  (frozen — not updated)
Custom head added:   ~294,000 parameters     (trainable)
─────────────────────────────────────────────
Total:               ~50,000,000 parameters
Trainable:           ~294,000 parameters
```

This is also an implementation of a published 2021 paper on COVID-19 detection from chest X-rays using transfer learning.

### Pre-trained Models Available in Keras

```python
from tensorflow.keras.applications import VGG16, VGG19   # Oxford
from tensorflow.keras.applications import AlexNet         # via community ports
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.applications import InceptionV3     # GoogLeNet family
from tensorflow.keras.applications import MobileNet       # edge deployment
# ... and many more
```

All of these are trained on ImageNet by default. All can be used as frozen feature extractors via `include_top=False` + `weights='imagenet'`.

---

## 10. Results & Discussion

### Custom CNN Result (~64×64 input)

- Accuracy: ~100% on training + test
- No visible overfitting
- Loss decreasing steadily

### VGG16 Transfer Learning Result (~224×224 input)

- Accuracy: ~100% (comparable to custom CNN)
- Slower per epoch — VGG16 has more layers and more calculations even when frozen

**Which model is more reliable?**

**Instructor:** The VGG16 transfer learning model, because:
- It has been exposed to vastly more data (14M ImageNet images)
- Its feature extractor generalizes better even with our small dataset
- Our custom CNN trained only on a few dozen images — high accuracy may be due to memorization, not generalization

**Critical caution:** Even 95–98% accuracy on this small dataset is suspect. The dataset may not represent real-world variability. You must question your model's robustness before deploying in any medical context.

---

## 11. False Positives vs. False Negatives in Medical Contexts

**Instructor:** Accuracy alone is insufficient. You must also look at:
- **Precision** — of all predicted COVID cases, how many were actually COVID?
- **Recall** — of all actual COVID cases, how many did the model catch?
- **False positive rate** — model says COVID, patient is healthy
- **False negative rate** — model says Normal, patient actually has COVID

**For COVID-19 specifically:**

| Error type | Consequence |
|---|---|
| False positive | Patient takes medication unnecessarily — generally not severely harmful |
| False negative | Patient has COVID but is told they're fine — potentially dangerous |

**Conclusion:** You should optimize for **low false negative rate** (high recall). Missing a COVID case is worse than over-diagnosing. This is a general principle in medical ML — the cost of different error types is asymmetric.

> This is also a quiz/exam concept to be familiar with.

---

## 12. Improving the Pipeline — Segmentation as Preprocessing

**Instructor:** What else could improve accuracy or model robustness?

**Student:** Data augmentation.

**Instructor:** Excellent. Since we have so few images, augmentation (zoom, brightness shift — not rotation if orientation matters) creates more variation and makes the model more robust (potentially slightly lower accuracy, but much more reliable).

**Student:** Wouldn't it be better to segment the chest region first?

**Instructor:** 100% correct — this is a standard technique:

1. Run a segmentation model (Ultralytics YOLO-seg, or a custom U-Net) on the chest X-ray
2. The segmentation mask isolates the lung/chest region — everything else becomes black
3. Feed the masked image (only the ROI) into the classifier

**Why this helps:** In the raw X-ray images, the border regions, medical labels, and surrounding body parts are noise relative to the diagnostic task. Giving the CNN only the relevant region reduces noise and focuses feature extraction.

**Challenge:** Training a segmentation model for chest X-rays requires a labeled segmentation dataset — you must manually annotate which pixels belong to the lung region. Medical image segmentation datasets also have confidentiality constraints. This is a supervised learning problem too — labeled data required.

---

## 13. Code Quality Practices Demonstrated

The instructor explicitly highlighted these in the project as professional standards:

| Practice | Purpose |
|---|---|
| **Type hints** (`X_train: np.ndarray`) | Makes function signatures self-documenting; enables IDE tooling |
| **Docstrings** (`"""..."""` on functions) | Documents args, return values, and purpose for each module |
| **Modular files** (preprocessing, model, evaluation, project) | Separation of concerns; mirrors real ML team workflows |
| **Keyword arguments with defaults** (`epochs: int = 10`) | Allows overriding at call site; sensible defaults for quick runs |
| **Hyperparameters at top of main file** (`EPOCHS = 10`) | Single place to change experiment configuration |
| **Unit tests** (mentioned, not implemented today) | Verify each module independently before wiring together |
| **Weights saved separately** (`weights/` directory) | Keeps model artifacts organized and versionable |

> **Instructor:** Even senior developers working with Keras and PyTorch use this exact format — different problems, same structural approach: `train`, `test`, `display`/`results` modules.

---

## 14. Closing

**Instructor:** With everything covered — numerical data (SQL, pandas), images (OpenCV, CNNs, transfer learning), and pre-trained models (Ultralytics, Keras applications) — you're equipped to tackle a wide range of real ML and computer vision projects.

Remaining sessions:
- Final exam sample question walkthrough
- Q&A / anything not covered today

The quiz is now open — go ahead and complete it; today's session covers the transfer learning question directly.

---

*End of Session 24 transcript.*

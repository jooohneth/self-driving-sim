# Computer Vision — DPS920 — Session 23

> 1:1 markdown transcription of `Session23.pdf`.

---

## Slide 1 — Title

Computer Vision
DPS920
Session 23

---

## Slide 2 — Overview

- Deep Learning
- Hyperparameter Tuning
- Learning Rate
- Overfitting
- Batch Normalization
- Data Augmentation
- Adam, ReLu

---

## Slide 3 — Agenda

- YOLO
- Ultralytics
- Object Detection
- Segmentation
- Pose Estimation
- Tracking
- Medical Use cases

---

## Slide 4 — What is Left?

1. Optimization and Loss Function
2. Code + Logistic Regression
3. ML and Images
4. Perceptron and Neural Networks
5. Neural Networks
6. Convolution Neural Networks (CNN)
7. Advanced CNNs
8. **Introduction to object detection, Segmentation and image generation methods with AI**

(items 1–7 are greyed out; item 8 is highlighted)

---

## Slide 5 — Some Vision Tasks

(row of five example images)

- **Classify** — worker with tripod (labels: PERSON, TRIPOD, SAFETY VEST)
- **Detect** — street scene (Person 92%, Traffic cone 86%)
- **Segment** — surgical instruments on a tray
- **Track** — highway with cars (Car 98%, Car 96%, Car 97%)
- **Pose** — athlete with skeleton keypoints

---

## Slide 6 — Object Detection Methods

**Traditional:**
- Viola-Jones (Haar Cascades),
- HOG + SVM (Histogram of Oriented Gradients + Support Vector Machine)
- Selective Search + SVM
- EdgeBoxes

**Deep Learning 2 stage:**
- R-CNN
- Fast R-CNN
- Faster R-CNN
- Mask R-CNN (adds segmentation)
- Cascade R-CNN
- Libra R-CNN
- DCN (Deformable Convolutional Networks)

**Deep Learning 1 stage:**
- YOLO (You Only Look Once)
- SSD (Single Shot MultiBox Detector)
- RetinaNet
- EfficientDet
- CenterNet
- FCOS (Fully Convolutional One-Stage Object Detection)
- CornerNet
- YOLO-NAS (Neural Architecture Search)

---

## Slide 7 — YOLO

Joseph Redmon

(timeline diagram of YOLO versions)

- 2015 — YOLOv1
- 2016 — YOLOv2 / YOLOv9000
- 2018 — YOLOv3
- 2020 — PP-YOLO / YOLOv5 / YOLOv4
- 2021 — YOLOS / PP-YOLOv2
- 2022 — YOLOv7 / YOLOv6
- 2023 — YOLOv8

---

## Slide 8 — Ultralytics

- pip install ultralytics

(Ultralytics logo)

---

## Slide 9 — Object Detection

object, name, confidence score

(photo of a European city square with detected Building, Human, Pushcart, Bag, Bicycle bounding boxes)

---

## Slide 10 — YOLOv8

| YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | YOLOv8x |
|---|---|---|---|---|
| 3.2M | 11.2M | 25.9M | 43.7M | 68.2 |

---

## Slide 11 — Example

```python
import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Read an image using OpenCV
source = cv2.imread('path/to/image.jpg')

# Run inference on the source
results = model(source)  # list of Results objects
```

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define remote image or video URL
source = 'https://ultralytics.com/images/bus.jpg'

# Run inference on the source
results = model(source)  # list of Results objects
```

```python
import numpy as np
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Create a random numpy array of HWC shape (640, 640, 3) with values in range
source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

# Run inference on the source
results = model(source)  # list of Results objects
```

---

## Slide 12 — Example URL

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define source as YouTube video URL
source = 'https://youtu.be/LNwODJXcvt4'

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects
```

---

## Slide 13 — Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `conf` | float | 0.25 | object confidence threshold for detection |
| `classes` | list[int] | None | filter results by class, i.e. classes=0, or classes=[0,2,3] |
| `show` | bool | False | show predicted images and videos if environment allows |
| `save` | bool | False | save predicted images and videos |
| `save_txt` | bool | False | save results as `.txt` file |
| `save_conf` | bool | False | save results with confidence scores |

---

## Slide 14 — Segmentation

(photo of surgical instruments on a blue cloth with colored contours labelled MEDICAL HOOK, SCALPEL, SCISSORS)

---

## Slide 15 — Segmentation

(three panels: a chest X-ray, a binary lung mask, and the mask applied back to the X-ray showing only the lung regions)

---

## Slide 16 — UNET

(U-Net architecture diagram showing the encoder-decoder structure with skip connections)

- Input: 128×128×1
- Down path: 128×128×16 → 64×64×16 → 64×64×32 → 32×32×32 → 32×32×64 → 16×16×64 → 16×16×128 → 8×8×128 → 8×8×256
- Up path: 8×8×256 → 16×16×(128+128) → 16×16×128 → 32×32×(64+64) → 32×32×64 → 64×64×(32+32) → 64×64×32 → 128×128×(16+16) → 128×128×16
- Output: 128×128×2

Legend:
- Conv 3×3, Relu (purple arrow)
- maxpool, 2×2 (dark yellow arrow)
- upsample, 2×2 (green arrow)
- copy and connect (grey arrow)
- Conv 1×1 (yellow arrow)

---

## Slide 17 — UNET

(3D U-Net architecture diagram)

- Input: 1×24×200×200
- Down path: 32×24×200×200 → 64×12×100×100 → 128×6×50×50 → 256×3×25×25
- Up path: 256×3×25×25 → 128×6×50×50 → 64×12×100×100 → 32×24×200×200
- Output: 32×24×200×200

Legend:
- Input: 1×24×200×200 (yellow)
- 3×3×3 Conv – GN8 – ELU (blue)
- 1×1×1 Conv + softmax (green)
- 2×2×2 Max-pooling (blue arrow)
- 3×3×3 Transposed Conv – GN8 – ELU (orange arrow)
- Skip connection (dashed arrow)

---

## Slide 18 — Classification

(two photos: a tabby cat lying on a ledge, and a blue heeler dog standing in a grass field)

```python
model = YOLO('yolov8n-seg.pt')
results = model('italy.jpg', show=True)
```

---

## Slide 19 — Pose Estimation

(photo of a male runner on the left; table of keypoint indices in the middle; stick-figure skeleton visualization on the right)

| Index | Key point |
|---|---|
| 0 | Nose |
| 1 | Left-eye |
| 2 | Right-eye |
| 3 | Left-ear |
| 4 | Right-ear |
| 5 | Left-shoulder |
| 6 | Right-shoulder |
| 7 | Left-elbow |
| 8 | Right-elbow |
| 9 | Left-wrist |
| 10 | Right-wrist |
| 11 | Left-hip |
| 12 | Right-hip |
| 13 | Left-knee |
| 14 | Right-knee |
| 15 | Left-ankle |
| 16 | Right-ankle |

---

## Slide 20 — Tracking

(photo of a highway scene at dusk with two cars bounded as "Car 93%" and yellow trajectory lines trailing behind each car)

---

## Slide 21 — Covid Diagnosis

(two chest X-ray images side by side)

- Normal
- Covid

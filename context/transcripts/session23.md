# Session 23 — YOLO, Ultralytics & Segmentation Architecture

**Course:** DPS920  
**Instructor:** Ellie Azizi  
**Format:** Live online lecture (remote)  
**Topics:** Final project overview, YOLO history, Ultralytics library, object detection, segmentation (U-Net encoder-decoder), SAM, pose estimation, tracking

---

## 1. Administrative Announcements

### Final Project

**Instructor:** In the Assignments and Labs folder on Blackboard, there's a final project and group submission. The PDF explains that you need to install a simulator to collect your own data and then run the simulation after you've trained your model. The simulation has two components — one for training data collection, one for the actual simulation run.

`TestSimulation.py` is your inference code. Once your model is developed, you generate the model file and point this file at it. You don't write the inference code from scratch — you just provide the path inside the file, and it runs the simulation automatically. The goal is a self-driving car that drives automatically in the simulation.

> **Package warning:** There is a package list (`requirements.txt` or similar). Install everything in a virtual environment. There are many incompatibilities with existing installations. Some packages require exact versions — do not deviate.

**Student:** Are the dependencies in the package list compatible with Mac?

**Instructor:** I think so — I haven't heard otherwise.

### Final Exam

- Covers everything from beginning to end
- ~80% of questions from the second half of the course; ~20% from before midterm
- No part 1 / part 2 split — one unified exam
- Online, research-based questions (Lockdown Browser had issues)
- Two research questions answered on Blackboard
- Concepts from labs and quizzes are also included in the final — familiarity is required, not just copy-paste

Sample questions (10 questions) uploaded to Blackboard under Quizzes & Exams.

### Quiz (Next Week)

- Online, research-based
- Two questions — search and answer on Blackboard

### Project Presentation

- Short, online, approximately 5–10 minutes
- Instructor will ask a few questions about the code to verify ownership

### Assignment 2

- Extended; attempts increased to 3
- If already submitted, you may resubmit

---

## 2. Review of Session 22

**Instructor:** In the previous session we covered:

- **CNN architecture:** convolution → subsampling → more convolution → subsampling → flatten → multi-layer perceptron → output (number of neurons = number of classes for classification; 1 neuron for regression)
- **ILSVRC Challenge:** from 2010 to 2017, tracking the evolution from shallow networks to deep learning. In 2012, AlexNet (Krizhevsky, Sutzkever, Hinton at U of Toronto) introduced the idea of going deep — that is the dividing line between classical ML and deep learning. Up to 152 layers (ResNet, 2015), surpassing human accuracy.
- **Architectures covered:** AlexNet, VGG, ResNet — as general concepts
- **Learning rate:** must be just right. Too high → diverge. Too low → very slow convergence. Import the optimizer from Keras, assign the `learning_rate` value in SGD or Adam.
- **Overfitting:** gap between train accuracy/loss and test accuracy/loss. Model memorizes training data instead of generalizing. Prevented by: data augmentation, batch normalization, dropout, regularization (L1/L2), simpler model, decreasing number of kernels, more epochs (to increase generalization), collecting more diverse data.
- **Batch normalization:** normalize outputs after each convolution layer to keep training stable; applied via `layers.BatchNormalization()`.
- **Dropout:** randomly deactivates neurons during training — not covered in depth in this course but worth knowing.
- **Bagging and boosting:** techniques applied in Random Forest / decision trees to prevent overfitting.

---

## 3. Where the Course Goes from Here

**Instructor:** After the midterm, starting from classical ML, we covered:

1. Optimization + loss functions (the most important algorithm in ML)
2. Logistic regression (classification)
3. ML on images — flatten, normalize, resize, then apply linear/logistic regression / KNN
4. Perceptron and neural networks — activation functions, components
5. Advanced neural networks — fully connected multi-layer
6. CNNs — convolution, pooling, feature extraction, AlexNet, VGG, ResNet

**Exciting tasks we didn't build from scratch (but will use):** segmentation, object detection, pose estimation, object tracking, counting objects, image generation.

**Why not from scratch?**  
YOLO's loss function alone would take 3–4 sessions to cover. The field is moving toward **AI engineers / inference engineers** who use pre-trained models rather than training from scratch. Larger companies (Meta, OpenAI, etc.) train the foundation models; smaller companies fine-tune and deploy them. That's the paradigm we'll work within.

---

## 4. Computer Vision Task Overview

| Task | Typical Use Cases |
|------|------------------|
| Classification | Image-level label (cat vs. dog) |
| Object detection | Bounding boxes around all detected objects |
| Single object detection | Bounding box for the most probable single object |
| Segmentation | Pixel-level mask (medical imaging: brain tumors, chest X-rays) |
| Tracking | Assigning IDs to objects across video frames (sports analytics) |
| Pose estimation | Key point localization (body joints for movement analysis) |
| Image/video generation | Text → image; image → text (captioning); video → text (summarization) |

**Segmentation in practice:** used primarily in medical imaging (brain tumor segmentation, lung chest X-ray analysis), combined with robotics (e.g., robotic surgical arms). Also used to remove irrelevant background before classification — apply segmentation mask first, then feed masked region to classifier.

**Tracking in practice:** soccer / sports analytics — tracking each player's position and performance per frame. Companies building this are very recent startups.

**Pose estimation in practice:** analyzing body movement for fitness/training correctness. Company example: **Move AI** — instead of expensive body sensors, they use cameras to track motion and reconstruct 3D skeletons in Unreal Engine.

---

## 5. YOLO History

**YOLO = "You Only Look Once"**

| Year | Version | Notes |
|------|---------|-------|
| 2015 | YOLOv1 | Published by Joseph Redmon (PhD research). First one-stage detector — processes entire image in one pass rather than two stages. |
| 2016 | YOLOv2 | Published by Redmon — faster, more accurate. |
| 2018 | YOLOv3 | Published by Redmon — further improvements. **Redmon then stopped working on YOLO entirely** because he discovered it was being used in military applications and refused to continue. |
| 2020 | YOLOv4 | Continued by other researchers/companies after Redmon's departure. |
| 2021 | YOLOv5 | Published by **Ultralytics**. |
| 2021–2026 | v6–v11+ | Continued development; Ultralytics remains the primary maintainer. As of 2026, the latest is YOLOv11. |

**Why YOLO is dominant:** The core architecture introduced one-stage detection, which is significantly faster than two-stage methods (RCNN, Fast RCNN, Faster RCNN). Other architectures exist but YOLO has the largest gap in adoption.

**Traditional (pre-deep learning) detection methods:** HOG + SVM, EdgeBoxes (convolutional-adjacent math). Less accurate, less fast than deep-learning-based methods.

**Deep learning detection family before YOLO:** RCNN → Fast RCNN → Faster RCNN (all two-stage detectors).

---

## 6. Ultralytics Library

**Instructor:** Ultralytics is the company that maintains YOLO. They package the model weights, training pipelines, and inference API into a Python library. Many companies use Ultralytics directly rather than training their own detection models — the cost and compute required to train from scratch is prohibitive.

### Installation

```bash
# Activate your virtual environment first
source /path/to/venv/bin/activate   # e.g., source scripts/activate

pip install ultralytics
```

### Import

```python
from ultralytics import YOLO
```

### Model Naming Convention

```
yolov8s.pt
│      │└─ extension: .pt = PyTorch model file
│      └── size: n (nano), s (small), m (medium), l (large), x (extra large)
└────────── architecture version: v8
```

Ultralytics uses **PyTorch** (not Keras) to develop their models — that's why the file extension is `.pt` rather than `.h5`.

### Model Size Trade-offs

| Size | Parameters | Speed | Accuracy |
|------|-----------|-------|----------|
| Nano (n) | ~11.2M | Fastest | Lower |
| Small (s) | ~22M | Fast | Moderate |
| … | … | … | … |
| Extra Large (x) | ~68.2M | Slowest | Highest |

More parameters = more convolution layers = better feature extraction = higher accuracy, but more computation. Choose based on your constraint: if speed matters (edge computing), go smaller. If accuracy matters more, go larger.

---

## 7. Object Detection — Image

### Basic Usage

```python
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('yolov8s.pt')  # downloads from GitHub on first use

# Load image
image = cv2.imread('italy.jpg')

# Run detection
results = model(image, show=True)

# Required to prevent kernel crash (Ultralytics uses OpenCV internally)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Result:** Detects cars, persons, traffic lights, handbags, etc. with bounding boxes and labels.

### Available Classes

```python
model.names
# {0: 'person', 1: 'bicycle', 2: 'car', ..., 79: 'toothbrush'}
# 80 total classes
```

The final layer of the YOLO architecture has 80 neurons — one per class. Each neuron outputs a probability (0–1) for that class. The output vector has length 80; coordinates identify where in the image.

### Saving Detection Results

```python
# Save bounding box coordinates to a .txt file
results = model(image, save_txt=True)
# Creates a folder with a text file; each line: class_id x_center y_center width height
```

**Why float coordinates?**  
Coordinates are normalized (0.0–1.0 relative to image dimensions). This way, if you resize the image, the coordinates remain valid regardless of image size.

```python
# Save with confidence scores appended to text file
results = model(image, save_txt=True, save_conf=True)
```

### Filtering by Class

```python
# Only detect specific classes (e.g., person=0, car=2, bus=5)
results = model(image, classes=[0, 2, 5])
```

### Confidence Threshold

```python
# Default: 0.5 (show objects model is ≥50% confident about)
results = model(image, conf=0.9)   # strict: only ≥90% confident detections
results = model(image, conf=0.25)  # loose: show everything ≥25% confident
```

Higher confidence → fewer detections, fewer false positives. Lower confidence → more detections, may include false positives. In medical applications, tuning this threshold is critical.

---

## 8. Object Detection — Video

```python
# Give the video path directly as source; no need for cv2.VideoCapture
results = model.predict(source='nature.mp4', show=True, stream=True)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Note:** Using `model.track()` instead of `model.predict()` enables object tracking (assigns persistent IDs across frames).

**Extra large model on video:**

```python
model_xl = YOLO('yolov8x.pt')
results = model_xl.predict(source='nature.mp4', show=True, stream=True)
```

Extra large is noticeably slower — fewer frames per second — but more accurate per frame. The model is a larger file and runs more calculations per frame.

---

## 9. YOLO Segmentation

Segmentation produces pixel-level masks (exact region boundaries) rather than just bounding boxes.

```python
# Segmentation model variant
model_seg = YOLO('yolov8s-seg.pt')  # downloads segment model

image = cv2.imread('italy.jpg')
results = model_seg(image, show=True)
```

**Result:** Bounding boxes + exact region-of-interest outlines drawn over detected objects.

---

## 10. Segmentation Architecture (Encoder-Decoder / U-Net)

**Instructor:** I want you to understand the idea, not every detail.

### Encoder (left side — downsampling)

Standard CNN pipeline — same as what we've built:

```
Input image (128x128x3)
    ↓ Conv2D × 2 (padding='same' to preserve size)
128x128x16
    ↓ MaxPool2D
64x64x16
    ↓ Conv2D × 2
64x64x32
    ↓ MaxPool2D
32x32x32
    ↓ Conv2D × 2
32x32x64
    ↓ MaxPool2D
    ↓ ...
    ↓ Flatten
Feature vector  ← "rich feature vector" / bottleneck
```

> **Key:** In segmentation, `padding='same'` is used in convolutions so spatial dimensions are preserved (only MaxPool reduces size). This differs from plain classification CNNs where `padding='valid'` naturally shrinks dimensions.

### Decoder (right side — upsampling)

Instead of going to a Dense output, the network reverses the spatial compression:

```
Feature vector
    ↓ Upsample (×2)
    ↓ TransposedConv2D × 2
    ↓ Upsample (×2)
    ↓ TransposedConv2D × 2
    ↓ Upsample (×2)
    ↓ TransposedConv2D × 2
    ↓ Upsample (×2)
Output: 128x128x1   ← black-and-white mask (same spatial size as input)
```

**TransposedConv2D** is the reverse of Conv2D — it upscales feature maps rather than shrinking them. Also called "deconvolution" informally.

**Output mask:** White pixels = object present; black pixels = background. You overlay this mask on the original image to see the segmented region.

### Why it's called U-Net

The architecture looks like a U shape — downsampling on the left half, upsampling on the right half, with the bottleneck feature vector at the bottom.

### Connection to Image Generation

This encoder-decoder architecture is the **baseline for all image generation models**:

- **Input:** image → encode features → decode to mask *(segmentation)*
- **Input:** text → encode features → decode to image *(text-to-image generation)*
- **Input:** image → encode features → decode to text *(image captioning)*
- **Input:** video frames → encode each → decode to text *(video summarization)*

The encoder extracts features; the decoder regenerates something from those features. The input and output types vary; the underlying idea does not.

---

## 11. SAM — Segment Anything Model (Meta)

**Instructor:** After YOLO segmentation, Meta introduced **SAM (Segment Anything Model)**.

- Trained on more data, more GPUs, more classes than YOLO-seg
- Can be imported and used in code (model file available for download)
- Also has a web demo interface
- Architecture is also encoder-decoder (same U-Net-style design)

```python
# Conceptual usage (exact API varies by SAM version)
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
```

> The core architecture is still the same encoder-decoder pattern we discussed — SAM just scales it further.

---

## 12. Other Ultralytics / YOLO Tasks

### Classification

```python
model_cls = YOLO('yolov8s-cls.pt')
results = model_cls(image)
# Returns class name only, no bounding box
```

### Pose Estimation

Uses 17 key points (body joints). Each key point is a localization problem — closer to regression than classification (predicting x, y coordinate values rather than a class label).

```python
model_pose = YOLO('yolov8s-pose.pt')
results = model_pose(image, show=True)
```

### Tracking

Assigns a persistent ID to each detected object across video frames.

```python
results = model.track(source='video.mp4', show=True, stream=True)
# Each detected object gets an ID that persists across frames
```

Useful for: sports analytics (tracking individual players), traffic monitoring, security surveillance.

---

## 13. Fine-Tuning / Transfer Learning (Preview for Next Session)

**Instructor:** If you want to use YOLO for a domain it wasn't trained on (e.g., medical imaging — brain tumor detection), you don't retrain from scratch. Instead:

1. Take the YOLO baseline (pre-trained weights = good feature extractor)
2. Freeze or partially freeze the base layers
3. Add your own classification/detection head on top
4. Train only the new head on your domain-specific data

This is called **transfer learning**. We will cover this in the next session, with an in-class project: **COVID diagnosis from chest X-rays** (possibly with a segmentation step before classification).

---

## 14. Additional Tools Mentioned

- **MediaPipe** (Google) — another library for pose estimation, hand tracking, face mesh, etc. Similar use cases to Ultralytics, different ecosystem.
- **Ultralytics documentation:** thorough experiment reports included — epoch count, batch size, learning rate, optimizer (SGD variant), full hyperparameter logs for each model version. Use these as references when configuring your own training runs.

---

## 15. Closing

**Instructor:** Thank you all for joining. Play around with the Ultralytics library — go up and down the documentation, try things that work and things that don't. Next session we'll cover transfer learning and do the COVID chest X-ray project together. See you online Tuesday.

---

*End of Session 23 transcript.*

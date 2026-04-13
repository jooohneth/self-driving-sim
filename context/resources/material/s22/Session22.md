# Computer Vision — DPS920 — Session 22

> 1:1 markdown transcription of `Session22.pdf`.

---

## Slide 1 — Title

Computer Vision
DPS920
Session 22

---

## Slide 2 — ILSVRC Challenge (2010-2017)

- ImageNet dataset (14 M)
- 1.2M images with 1000 classes

- Classification
- Single object detection
- Object detection

(collage of ImageNet sample photos on the right)

---

## Slide 3 — Results

(bar chart of top-5 error rates on ILSVRC over the years)

| Year | Team | Error | Depth |
|---|---|---|---|
| 2010 | Lin et al | 28.2 | shallow |
| 2011 | Sanchez & Perronnin | 25.8 | shallow |
| 2012 | Krizhevsky et al (AlexNet) | 16.4 | 8 layers |
| 2013 | Zeiler & Fergus | 11.7 | 8 layers |
| 2014 | Simonyan & Zisserman (VGG) | 7.3 | 19 layers |
| 2014 | Szegedy et al (GoogLeNet) | 6.7 | 22 layers |
| 2015 | He et al (ResNet) | 3.6 | 152 layers |
| 2016 | Shao et al | 3 | 152 layers |
| 2017 | Hu et al (SENet) | 2.3 | 152 layers |
| Human | Russakovsky et al | 5.1 | — |

---

## Slide 4 — AlexNet

(AlexNet architecture diagram)

- Input: 227 × 227 × 3
- CONV 11×11, stride=4, 96 kernels → (227−11)/4 + 1 = 55 → 55×55×96
- Overlapping Max POOL 3×3, stride=2 → (55−3)/2 + 1 = 27 → 27×27×96
- CONV 5×5, pad=2, 256 kernels → (27+2·2−5)/1 + 1 = 27 → 27×27×256
- Overlapping Max POOL 3×3, stride=2 → (27−3)/2 + 1 = 13 → 13×13×256
- CONV 3×3, pad=1, 384 kernels → (13+2·1−3)/1 + 1 = 13 → 13×13×384
- CONV 3×3, pad=1, 384 kernels → 13×13×384
- CONV 3×3, pad=1, 256 kernels → 13×13×256
- Overlapping Max POOL 3×3, stride=2 → (13−3)/2 + 1 = 6 → 6×6×256
- FC → 9216
- FC → 4096
- FC → 4096
- 1000 Softmax

---

## Slide 5 — VGG

(VGG architecture diagram with legend: blue = convolution+ReLU, red = max pooling, green = fully connected+ReLU)

- Input: 224 × 224 × 64
- conv1 block → 112 × 112 × 128
- conv2 block → 56 × 56 × 256
- conv3 block → 28 × 28 × 512
- conv4 block → 14 × 14 × 512
- conv5 block → 7 × 7 × 512
- fc6 → 1 × 1 × 4096
- fc7 → 1 × 1 × 4096
- fc8 → 1 × 1 × 1000

---

## Slide 6 — Learning Rate

A hyperparameter that controls how much the model updates its weights in response to the loss gradient.

Too high → Model diverges
Too low → Slow convergence
Just right → Fast & stable training

```python
from keras.optimizers import SGD
...
opt=SGD(lr=..., decay=...)
net.compile(..., optimizer=opt)
```

---

## Slide 7 — Overfitting

(loss curve diagram: blue curve decreasing monotonically (training loss), red curve decreasing then rising again (validation loss), with a warning icon at the inflection point)

- Model works good on train data but not on test data
- It memorizes and not generalize

---

## Slide 8 — Batch Normalization

- We normalized the input data.
- Why not normalizing values after each layer?

- A technique to normalize activations in a neural network, layer by layer, during training.

- Stabilizes learning
- Speeds up convergence
- Reduces internal covariate shift
- Allows higher learning rates

---

## Slide 9 — Data Augmentation

- A technique to increase the size and diversity of training data by applying random transformations.

- More robust models with better performance on unseen data.

---

## Slide 10 — Rotation

(original photo of a house at the top; four rotated versions of the same house below)

---

## Slide 11 — Width Shift

(original house photo at the top; four horizontally shifted versions below)

---

## Slide 12 — Brightness

(original house photo at the top; four versions below with progressively increasing brightness, from very dark to fully bright)

---

## Slide 13 — Shear

(original house photo at the top; four sheared versions below)

---

## Slide 14 — Zoom

(original house photo at the top; four versions below at different zoom levels)

---

## Slide 15 — ImageDataGenerator

A tool to apply data augmentation

```python
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")
```

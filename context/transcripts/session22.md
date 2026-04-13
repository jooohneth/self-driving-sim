# Session 22 — Convolutional Neural Networks: Implementation, History & Hyperparameters

**Course:** DPS920  
**Instructor:** Ellie Azizi  
**Format:** Live online lecture (remote)  
**Topics:** CNN implementation in Keras, ILSVRC challenge history, learning rate, overfitting, batch normalization, data augmentation

---

## Opening & Context

**Instructor:** Hi everyone. I hope you guys are doing well. Good afternoon. How's it going? So again, for your 5-minute meeting, this is the third. Close to the end of the semester.

So in the previous session, if you guys remember, we went over one of our most important topics that we were waiting for the whole semester, and that's convolutional neural networks. As you saw, it was nothing really crazy about it, because we knew every single piece of it — it's just like a glue that we have to stick them together.

---

## 1. CNN Architecture Review

**Instructor:** Convolutional neural networks are created based on these sections:
1. Convolution layer
2. Pooling layer
3. Fully connected layer
4. Activation layer

Once you combine them together, if you have an input image — it can be a batch of images, your whole dataset, or just one sample — all of them go through the network one by one. At the end, if it is a batch, the loss function takes the average. If it is one sample, it calculates the loss for that one sample. If it is the whole data, it calculates an average across the whole data.

It goes through a convolution, and the number of output matrices depends on how many kernels you have. In our example we had 6 kernels, so we get 6 output matrices. Then you apply subsampling (pooling layer). Then you apply 16 different convolutions and get 16 different matrices, then another subsampling. Subsampling can halve the size, or shrink to whatever size you want — it doesn't necessarily need to be divided by 2.

After that, once you have these matrices, you flatten them, attach them to each other, and make something like input data. Each one can be flattened to feed into a neural network / multi-layer perceptron. At the end:
- **Classification:** number of neurons = number of classes
- **Regression:** 1 neuron at the end

As you go from left to right in a CNN, the depth of the image increases and the size of the image decreases. Increasing the number of channels and decreasing the size makes it more compatible with the fully connected layers at the end, and you are extracting more important features. Features extracted in later layers are more important / intense than those in earlier layers — as you go left to right, you're extracting more and more important features and discarding the ones that don't make an impact.

**Key note:** You can have 2 convolution layers back-to-back and then one subsampling. Or 1 convolution + 1 subsampling. Or 5 convolutions + 1 subsampling. The composition is a design choice.

All of these kernel values are learned during weight updates via the optimization algorithm. That's how the network learns what values the kernels should be, so that in the next round, it extracts more important features rather than random ones.

---

## 2. Keras Conv2D — Implementation

> Notebook: CAPTCHA code example (from previous session, extended here)

**Instructor:** When it comes to coding, it's very easy, nothing really complicated. In Keras `layers`, we have `Dense` for fully connected layers and `Conv2D` for convolutions.

### Conv2D Parameters

```python
layers.Conv2D(
    filters,       # number of kernels
    kernel_size,   # e.g. (3,3), (5,5), (9,9)
    strides,       # step size
    padding,       # 'valid' (shrinks size, default) or 'same' (preserves size)
    activation     # e.g. 'relu', 'sigmoid'
)
```

- **`filters`** — how many kernels you want in this layer
- **`kernel_size`** — e.g. `3x3`, `5x5`, `9x9`; hyperparameter
- **`strides`** — step size; default is 1
- **`padding`:**
  - `'valid'` — no padding added; output matrix shrinks after convolution (default)
  - `'same'` — padding added so size is preserved after convolution
- **`activation`** — activation function (e.g. ReLU)

### Example Architecture (CAPTCHA)

```python
from keras import layers

# Convolution layer 1: 8 kernels, 3x3
layers.Conv2D(8, (3,3), strides=1, padding='valid', activation='relu')

# Convolution layer 2: 16 kernels, 5x5
# (increase kernel count as we go deeper)
layers.Conv2D(16, (5,5), padding='valid', activation='relu')

# MaxPooling: halves the size
layers.MaxPool2D(pool_size=(2,2))

# Convolution layer 3: 32 kernels, 3x3
layers.Conv2D(32, (3,3))

# MaxPooling again
layers.MaxPool2D((2,2))

# Flatten and Dense layers
layers.Flatten()
layers.Dense(15, activation='relu')
layers.Dense(num_classes, activation='softmax')  # classification
# for regression: Dense(1)
```

**Instructor:** Usually as they go from left to right, they increase the number of kernels in each layer. The reason is to gradually get closer to flattening so that the neighborhood doesn't become disordered all at once. When you flatten gradually, pixels that were beside each other are still near each other.

**Important:** When using Conv2D, you no longer flatten the images before feeding them in. The images are given as 2D arrays to the CNN, and flattening happens in the middle of the network. Flattening at the beginning (as in plain MLP) would break the CNN.

### Results

After fixing the flattening issue and running:
- Validation accuracy: ~99.8%, close to 100%
- Curves are very clean — train accuracy and test accuracy both increase together; train and test loss decrease together

**On epochs:**

**Student:** It looks like epoch = 5 would be enough since it reaches near 100% at epoch 3–4.

**Instructor:** Exactly. You don't need to spend resources overtraining your model. It's already almost flat after epoch 4. For models like GPT or Sora that take days to train, wasting epochs wastes enormous money, energy, and time. There's a technique called **early stopping**: you set a threshold — if accuracy hasn't significantly increased for, say, 2 steps over 3 epochs, you stop. For very large models, they often use `epoch=1` with large batch sizes, because the sheer volume of data is enough for one pass.

> GPT-3: 175 billion parameters, took ~34 days to train the largest version.

---

## 3. Model Summary & Parameter Counting

```python
# Add input_shape to enable model.summary()
model.build(input_shape=(None, 16, 16, 3))
print(model.summary())
```

**Instructor:** From `model.summary()`:
- After Conv2D(8, 3x3): output shape is `14x14x8` — 8 channels (8 kernels), size shrank from 16x16 to 14x14
- After Conv2D(16, 5x5): output shape is `10x10x16` — 16 channels
- After MaxPool2D: size halves
- Total trainable parameters: ~8,000

**Parameter count explanation:**

A 3x3 kernel on a 3-channel image:
- Each kernel is actually 3x3x3 (3 spatial × 3 channels) = 27 weights per kernel
- With 8 kernels: 8 × 27 = 216 weights
- Plus 1 bias per kernel: 8 × 1 = 8
- Total: 216 + 8 = 224 parameters ✓

**Student:** So it's 3x3x3 times 8?

**Instructor:** Exactly. Your input image is 3-channel, so the kernels are also 3-channel. The 3 channel slices are convolved and added together to produce one output matrix per kernel. Then add bias — one bias per kernel. So the question of where bias applies: one bias per kernel (not per channel, not per pixel).

**On the number 495 (Dense layer):** When you flatten the last Conv2D output (e.g. 32 channels, certain spatial size), that flattened vector connects to 15 neurons: `flattened_size × 15 = 495` parameters (plus biases).

For reference — GPT-3 has 175 billion parameters. Not comparable to what we're working with.

---

## 4. Why 3×3 and 5×5 Kernels? Why Odd Numbers?

**Student:** Can you explain the 3×3, 5×5 kernel sizes?

**Instructor:** Good questions — three parts:

**What are they?**  
These are the spatial dimensions of the kernel (filter) applied over the image. The kernel slides across the image. If you have an 8-kernel layer with 3×3 kernels, you have 8 filters each of size 3×3 (×input channels). Each is a set of learnable weights, randomly initialized, then updated by the optimizer.

**Why odd numbers only?**

**Student:** Because we apply the value to the middle pixel.

**Instructor:** Exactly. An odd-size kernel has a well-defined center pixel. With an even-size kernel, there is no single center, which causes alignment ambiguity.

**Why specifically 3×3 and 5×5?**  
These were established as best practice by VGG research (see §5 below). They showed mathematically that small kernels (3×3, at most 5×5) produce better feature extractors compared to large kernels like 11×11 or 13×13. Using 11×11 is valid (AlexNet used it), but 3×3 and 5×5 were shown to be mathematically more reliable for feature extraction. You can use 11×11, 13×13, etc. — it's a hyperparameter — but 3×3 and 5×5 are the best-practice defaults.

---

## 5. ILSVRC Challenge & CNN History

**Instructor:** From 2010 to 2017, there was a competition called the **ILSVRC Challenge** (ImageNet Large Scale Visual Recognition Challenge), published on Kaggle. Different companies and research groups competed. The challenge had three tracks:
1. Classification
2. Single object detection (detect the one most probable object)
3. Object detection (detect all objects regardless of count)

**ImageNet Dataset:**  
A research group collected 14 million data samples and called it ImageNet. A subset of 1.2 million images across 1,000 classes was used for the ILSVRC challenge.

### ILSVRC Results Timeline (top-5 error rate)

| Year | Error | Winner / Architecture | Notes |
|------|-------|----------------------|-------|
| 2010 | 28.2% | Lin et al. | Shallow networks, few layers |
| 2011 | 25.8% | Sanchez & Peronin | Still shallow |
| 2012 | 16.4% | **AlexNet** (Krizhevsky, Sutzkever, Hinton — U of Toronto) | First deep CNN; marked the beginning of deep learning |
| 2013 | 11.7% | — | Continued improvements |
| 2014 | 6.7% / 7.3% | GoogLeNet (winner) / **VGG** (2nd place) | VGG's contributions more influential |
| 2015 | 3.57% | **ResNet** (Microsoft) — 152 layers | Surpassed human accuracy (5.1%) |
| 2017 | — | — | Challenge ended |

### AlexNet (2012)

- Input: 227×227 images
- 96 kernels of 11×11 → output 55×55×96
- MaxPool → 27×27
- More convolutions, more max pooling
- Flatten
- 2× Dense(4096)
- Softmax(1000) — 1,000 classes

Authors include Ilya Sutzkever and Geoffrey Hinton. The idea of **scaling** (going deep into layers instead of keeping shallow) stems from their work. Instead of 1–2 conv layers, they used many. If you control hyperparameters and ensure stable convergence, going deeper gives better feature extraction — that's why it's called **deep learning**.

**Instructor:** If you search AlexNet online, you can now read the entire paper — you know what ILSVRC is, you know CNNs, you know training error curves, you know ReLU. The paper is accessible to you now.

### VGG (2014)

Architecture: `Conv64×2 → MaxPool → Conv128×2 → MaxPool → Conv256×3 → MaxPool → Conv512×3 → MaxPool → Conv512×3 → MaxPool → FC×3 → Softmax`

VGG's key contribution was not just their accuracy, but **studying what happens in each layer mathematically and optimization-wise**. Key findings:
- Small kernels (3×3, at most 5×5) are better than large kernels
- As you go deeper with small kernels, you get stronger feature extractors
- These became best practices that persist today

Paper: *"Very Deep Convolutional Networks for Large-Scale Image Recognition"*

> *"In this work, we investigate the effect of convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3×3) convolution filters, which shows that a significant improvement on prior-art configurations can be achieved by pushing the depth to 16–19 weight layers."*

"Weight layers" = layers with parameters (Conv2D, Dense). Subsampling/pooling layers are not counted as weight layers.

VGG16 = 16 weight layers. VGG19 = 19 weight layers. Both are publicly available.

**Instructor:** I highly recommend reading through VGG and AlexNet papers. Look at what perspective they took, what ideas they had, what paths they chose. Step by step, you'll also be able to read about Transformers — "Attention Is All You Need" — the most advanced architecture right now.

### ResNet (2015)

- Microsoft
- 152 weighted layers
- Error rate: 3.57% — better than human accuracy (5.1%)
- Uses residual connections (skip connections) — that's how 152 layers is trainable without vanishing gradients

**Instructor:** Someone mentioned ResNet in the last session. By 2015, bigger companies (Google, Microsoft) came into play because they had the compute resources to scale to 152 layers.

---

## 6. Hyperparameter: Learning Rate

**Instructor:** You're all familiar with learning rate. It controls how much the model updates its weights in response to the loss gradient.

- **Too high:** Model diverges — it jumps around the loss surface and never converges to the minimum
- **Too low:** Takes a very long time to reach the minimum
- **Just right:** Fast and stable convergence

Modern strategy: go fast at the beginning, slow down toward the end (e.g., at epoch 5 or 10, update very slowly to avoid overshooting the minimum).

### Setting Learning Rate in Keras

```python
from keras.optimizers import SGD, Adam

optimizer = Adam(learning_rate=0.001)  # default is 0.001

model.compile(optimizer=optimizer, loss='...', metrics=['accuracy'])
# or
model.compile(optimizer=SGD(learning_rate=0.001), ...)
```

**Why Adam?**  
Adam automatically goes faster at the beginning and slower toward the end — that's why it's generally a better optimizer than plain SGD. It also adapts based on gradient magnitude: where the slope is steep, updates are larger; where the slope is flat (near the minimum), updates are smaller.

**Student:** Must the optimizer be linear regression or random?

**Instructor:** The optimizer (SGD, Adam, RMSprop) uses the same update rule as linear regression: `new_W = old_W - alpha * dL/dW`. The difference is in how the architecture is defined — AX+B, AX+B with sigmoid, convolution + dense layers — but the underlying parameter update mechanism is the same gradient descent formula. Parameters are initialized randomly at the start, then updated.

---

## 7. Overfitting

**Instructor:** There are some patterns you'll see when plotting accuracy and loss:

**Normal (healthy) case:** Both train loss and test loss decrease. Both train accuracy and test accuracy increase.

**Overfitting:** Train loss keeps decreasing every epoch. Test loss decreases at the beginning but then plateaus or increases. A gap opens between train loss and test loss.

**Why does it happen?**  
You update parameters only based on training data. The model memorizes patterns from the training data and cannot generalize to data it hasn't used for weight updates (test data). Example: the model learns that all cats are gray with green eyes. When it sees a white cat, it fails — because it memorized Lucy's features rather than learning general cat features.

Same idea in regression: instead of fitting a clean line, you overfit to every data point — great training accuracy but huge error on new points.

**Instructor:** Even if you get 99% training accuracy, we don't call that a good model. You must look at test/evaluation results.

### Methods to Prevent Overfitting

1. **Dropout** — randomly zeros out neurons during training, adds noise so the model can't memorize
2. **L1 / L2 regularization** — adds a penalty term to the loss to constrain weights
3. **Data augmentation** — exposes the model to more variations of the training data
4. **Simpler model** — fewer neurons, fewer layers = less capacity to memorize
5. **Collect more data** — more variety means less risk of memorization
6. **Refine data** — ensure training data is sufficiently diverse and generic
7. **Change algorithm/architecture** — sometimes linear regression generalizes better than a deep network for a given problem
8. **Bagging / boosting** — e.g., with Random Forest, these ensemble methods reduce overfitting

**Student:** Adding more variety of cats would help?

**Instructor:** 100% correct. Data augmentation partially does this — it adds variations of the same data. But explicitly collecting genuinely different examples (different colored cats, different lighting, etc.) is the stronger version of that.

---

## 8. Batch Normalization

**Instructor:** There are two places where normalization matters:
1. **Distance-based models** (e.g., KNN) — for fair comparison
2. **Optimization-based models** — helps convergence stability; limits the search space so the model isn't searching all of ℝⁿ but a bounded region

We normalize inputs (images, tabular data) before feeding the network. However, after the input passes through convolution layers, the intermediate outputs can be any number — kernels are randomly initialized, and multiplications/additions can produce arbitrary values. **Batch normalization** applies normalization after each convolution layer (or after some layers — it's an architecture choice).

**Why it helps:** Mathematically similar to input normalization — keeps the mean at 0 and standard deviation at 1, making each layer's inputs stable. This improves convergence stability throughout the network, not just at the input.

### Applying Batch Normalization in Keras

```python
layers.BatchNormalization()
```

Example — after a convolution:

```python
layers.Conv2D(32, (3,3), activation='relu')
layers.BatchNormalization()
layers.MaxPool2D((2,2))
```

You don't have to apply it after every single weight layer — some apply it every 2 layers. Architecture choice.

---

## 9. Data Augmentation

**Instructor:** Data augmentation adds different variations of the same data to the training set. Changes are applied randomly. For images, this can include:

- Rotation (random angle)
- Horizontal / vertical shift
- Horizontal / vertical flip
- Zoom in / out
- Brightness change
- Shear

The goal: expose the model to more variety so it doesn't only understand one brightness, one orientation, one scale.

> **Important:** Augmentation transformations should make semantic sense. For example, flipping a digit horizontally may produce a nonsensical digit. For a steering angle prediction, flipping the image horizontally + negating the steering angle does make sense. Always validate that your augmentation preserves the label's meaning.

### Option 1: Manual Augmentation

```python
import cv2
# read, transform, write augmented images to dataset
cv2.imwrite('path/augmented_image.jpg', transformed_image)
```

### Option 2: Keras ImageDataGenerator

```python
from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
    brightness_range=[0.5, 1.5],  # random brightness multiplier in this range
    horizontal_flip=True,
    zoom_range=0.5,
    shear_range=0.2,
    # width_shift_range, height_shift_range, rotation_range, etc.
)

# Use aug.flow() to apply augmentation during training
aug.flow(X_train, y_train, batch_size=32)
```

**Instructor demo result:** After adding `horizontal_flip=True` and a wide brightness range, accuracy dropped significantly (~20%). Reason: horizontal flip doesn't make sense for digit classification (flipped digits are meaningless). After removing horizontal flip and adjusting brightness range, results improved. This confirms: augmentation transformations must be semantically valid for your problem.

---

## 10. Q&A: Kernel Values & ReLU

**Student:** What are the real kernel values? How do calculations pass to the next layer?

**Instructor:** Great question — it means you're thinking deeply.

Kernel values are parameters. For a 3×3 kernel on an 8×8 image:

```
Initial kernel (randomly initialized):
 0.1  |  5   |  8
 7.2  | -1   |  2.3
 0.5  |  0.3 |  1
```

At the start, they're random. Via the optimization algorithm (gradient descent), they're updated after each forward + backward pass. Over many epochs, the values converge so that the kernel extracts useful features that maximize accuracy.

**How accuracy connects to kernel values:**
- Better kernel values → better feature extraction → higher accuracy
- You update them until you have a good feature extractor

**ReLU after convolution:**

After applying the kernel, you get an output matrix, e.g.:
```
0.8  |  5   | -100
...
```

With ReLU:
```
0.8  |  5   |  0    (negative → 0)
```

With Sigmoid: all values mapped to (0, 1):
```
≈0.69 | ≈1.0 | ≈0.0
```

---

## 11. What's Next

**Instructor:** In the next session, we'll be introduced to **Ultralytics** — a tool that integrates object detection, segmentation, and tracking models in one place. Many companies use Ultralytics instead of training their own models because training at that scale is not affordable for smaller companies.

We'll also get an introduction to YOLO architecture and image generation/segmentation models.

If you're interested in object detection architecture internals: the loss function alone (compared to MSE and cross-entropy we know) is significantly more complex. We didn't have time to go through object detection, segmentation, etc. in full detail — but Ultralytics lets us use these models at a higher level of abstraction.

---

## 12. Closing

**Instructor:** With all of this in hand, you're ready to solve many problems in image processing — mostly classification and regression supervised learning problems.

Your understanding of machine learning is now advanced. You understand optimization, and everything in ML really depends on optimization.

**Quiz announcement:** Next week, online quiz on CNN architecture — conceptual, no math, no coding. Will be on Lockdown Browser or research-based (TBD).

**Final project:** Published today or tomorrow. It is all about convolutional neural networks.

---

*End of Session 22 transcript.*

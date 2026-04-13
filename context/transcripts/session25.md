# Session 25 — Final Exam Review & Sample Questions Walkthrough

**Course:** DPS920  
**Instructor:** Ellie Azizi  
**Format:** Live online lecture (remote)  
**Topics:** Final exam sample questions, concept review across the full course, student Q&A

---

## 1. Administrative Announcements

### Session Plan

- No new material today — short review session
- **MediaPipe** is not being covered; it is too similar to Ultralytics. A sample code snippet can be shared on request, but it requires no instruction.

### Next Session (Optional)

- Topic: **Unsupervised learning** — clustering on image data
- Participation is **not mandatory** and will **not** appear on the exam
- Intended for students interested in extending their ML knowledge beyond the course scope

### Final Exam

- **Date:** April 13th
- **Format:** Diagrams and inference questions, true/false, conceptual questions
- No coding (except possibly CV2 operations)
- ~20% from OpenCV topics, ~80% from ML / deep learning topics
- True/false answers tend to be **false** — a pattern consistent with the midterm
- Expect: overfitting diagrams, feature extraction diagrams, questions about architecture components

### Final Project Rubric (Preview)

A detailed PDF rubric will be shared. Grading criteria include:
- **Code quality** — modular structure (functions, separate files, components that work together)
- **Code cleanliness** — readability, organization
- **PEP 8 compliance**
- **Accuracy** — whether the car drives correctly in simulation
- **Ability to run the project** — end-to-end execution

### Final Project — Key Difference from Session 24 Project

The session 24 COVID project template can be reused for the final project. The only significant change is in the model section:

| Session 24 (COVID) | Final Project (Self-Driving Car) |
|---|---|
| Multi-class classification | Regression |
| Last layer: `Dense(2, activation='softmax')` | Last layer: `Dense(1)` — **no activation function** |
| Loss: `categorical_crossentropy` | Loss: `mse` |
| Output: class label (COVID / Normal) | Output: steering angle (continuous value) |

This mirrors how real self-driving cars (e.g., Tesla) work — they receive images from the environment and output a continuous steering value based on those images, combining computer vision with control systems.

---

## 2. Review of Session 24

**Instructor:** In the previous session we implemented a full modular COVID-19 detection project using:
- `preprocessing.py` — read images, resize, normalize, train-test split, label encode, one-hot encode
- `model.py` — define and train the CNN or VGG16 transfer learning model
- `evaluation.py` — plot accuracy and loss curves
- `project.py` — main entry point calling all modules

**Transfer learning:** Used VGG16 pre-trained on ImageNet. Set `trainable=False` on all VGG16 layers — only the custom dense layers at the top are trained. You can alternatively set `trainable=True` to fine-tune all weights, but that requires more compute and data.

---

## 3. Final Exam Sample Questions — Walkthrough

### Q1 — Gaussian Blur Before Edge Detection

**Question:** You apply `cv2.GaussianBlur` before edge detection with `cv2.Canny`. The main reason is:
- A. To sharpen edges before detection
- B. To reduce noise that may cause false edges ✓
- C. To increase contrast
- D. To make processing faster

**Answer: B**

**Explanation:** Gaussian blur denoises the image. In image processing, "denoising" and "blurring" refer to the same operation. Before applying an edge detection algorithm (Canny), you remove noise so that the edge detector does not pick up spurious edges caused by noise. This same principle applies in your final project pipeline — Gaussian blur is applied before the image is fed to the model to remove noise that would otherwise be treated as a feature.

---

### Q2 — One-Hot Encoding vs. Label Encoding

**Question:** When should you prefer one-hot encoding over label encoding?

**Key distinction:** This question applies to both **labels** (outputs) and **features** (inputs). Do not assume it refers only to labels.

**Rule for labels (in neural networks / CNNs):**  
Always use one-hot encoding. Neural networks and CNNs always require one-hot encoded labels.

**Rule for features:**

| Situation | Preferred encoding |
|---|---|
| Feature values have an **ordinal (ordered) relationship** | **Label encoding** — preserves the ordering. Example: `Diploma < Bachelor's < Master's < PhD` |
| Feature values have **no ordinal relationship** | **One-hot encoding** — avoids implying a false order. Example: `banana, kiwi, orange, apple` — no ordering |

**Answer:** When the categorical values have **no ordinal relationship** (option C).

---

### Q3 — Removing Nonlinearity (Activation Functions)

**Question:** What would happen if you removed all nonlinear activation functions from all layers of a neural network?
- A. It would become faster without impact on learning
- B. The network would collapse to a linear function ✓
- C. The model would overfit
- D. Gradient descent would stop working
- E. All of the above

**Answer: B**

**Explanation:**  
Each neuron computes `Ax + B` — a linear function. Without activation functions, stacking multiple layers of `Ax + B` still produces a linear function (a composition of linear functions is linear). The model loses the ability to represent any nonlinear relationship between features and output.

**Why activation functions enable nonlinearity:**  
Analogous to biological neurons — signals are either activated or deactivated. By selectively activating/deactivating signals at each layer, the network can represent complex, nonlinear decision boundaries.

- **Sigmoid** — maps output to (0, 1); smooth, differentiable replacement for the step function
- **ReLU** — passes positive values unchanged, zeroes out negatives; avoids the range limitation of sigmoid
- **Softmax** — normalizes outputs to a probability distribution over classes

> **This is a common ML interview question:** How is a neural network able to find nonlinear patterns? Answer: because of activation functions.

---

### Q4 — Flattening Images Before a Dense Network

**Question:** Flattening images into 1D feature vectors before a dense network mainly harms performance because:
- A. It reduces model parameters too much
- B. It destroys spatial locality ✓
- C. It prevents use of activation functions
- D. It requires fixed input size

**Answer: B**

**Explanation:** The relative position of pixels is itself a feature. If the left eye is at position (x, y), the right eye should be at a predictable relative position — that spatial relationship is information. Flattening destroys these neighborhood relationships entirely; the model sees a bag of pixel values with no positional context.

**Aside — CNNs vs. Dense layers: which has more parameters?**

In a fair comparison (same number of layers, same number of units per layer):

- **Dense layers** have more parameters — every neuron connects to every neuron in the next layer
- **CNNs** have fewer parameters — a kernel slides over the image sharing weights; no full connectivity

More parameters = more calculations = slower training + higher memory requirement + risk of overfitting. CNNs are stronger feature extractors despite having fewer parameters than equivalent dense networks.

---

### Q5 — True/False: CNNs Require Fewer Training Samples Than KNN

**Answer: False**

CNNs (and neural networks generally) require **more** data, not less. The more complex the model, the more data it needs. This is the same principle behind why transformers require vastly more data than CNNs. A sparse dataset with a complex model leads to **data sparsity** — the model can't meaningfully distinguish between class distributions.

KNN is a non-parametric model — it doesn't build any internal representation. At inference, it computes distances to all training samples. It needs the training data at runtime, which has its own costs (see Q11).

---

### Q6 — `cv2.bitwise_and` with a Binary Mask

**Question:** The function `cv2.bitwise_and` can be used to apply a binary mask to an image. Explain what it does.

**Answer:**
1. You have an original image and a binary mask (black and white)
2. `cv2.bitwise_and(image, image, mask=mask)` — wherever the mask is **white (255)**, keep the corresponding pixels from the original image; wherever the mask is **black (0)**, set pixels to black (zero out)

**`cv2.bitwise_not`** does the reverse — keeps pixels where the mask is black, zeroes out where white.

---

### Matching Column

| Term | Match |
|---|---|
| Multiple linear regression | Predicts a **continuous** target variable |
| Max pooling | Reduces **spatial dimensions** in a CNN |
| Median filter | Removes **salt-and-pepper noise** while preserving edges |
| Cross-entropy | Loss function used for **classification** |
| Data augmentation | Reduces **model variance** |

**On median filter vs. Gaussian blur:**
- **Gaussian blur** — best for Gaussian (smooth) noise; operates by weighted averaging
- **Median filter** — best for salt-and-pepper noise (isolated high/low intensity pixels); replaces each pixel with the median of its neighborhood; better at preserving edges than Gaussian blur

**On data augmentation reducing model variance:**  
Model variance refers to how much the model's fit can vary given different training data. With sparse data, the model has high variance — the decision boundary can take many shapes because there aren't enough samples to constrain it. Adding more samples (via augmentation) increases the density of the training distribution, forcing the model to fit more tightly to the actual distribution and reducing the range of shapes it can take. More density = lower variance.

---

### Q7 — Use Case for `cv2.inRange`

**Question:** Give one example of when you would use `cv2.inRange` in an image processing pipeline.

**What `cv2.inRange` does:** Takes an image and a lower/upper color bound. Pixels within the range become white (255); pixels outside the range become black (0). Produces a binary mask.

**Example answers:**
- Detecting/segmenting a colored object (e.g., isolating a yellow pedestrian sign, a green clock, a red object)
- Creating a binary mask for the "invisible cloak" effect (isolate a specific color, replace with background)
- Color-based object segmentation as a preprocessing step before further processing

The output binary mask can then be passed to `cv2.bitwise_and` to extract the region of interest from the original image.

---

### Q8 — Loss Curve Flattens After 10 Epochs

**Question:** A training loss curve decreases steeply in the first few epochs, then flattens after epoch 3–4. The curve stays flat through epoch 10. List two reasons why this may happen.

**Answer:**

1. **The model has already converged** — it has extracted all learnable information from the dataset. Additional epochs provide no new learning signal.
2. **The dataset is too small** — after a few passes through a small dataset, there's nothing more to learn. The model has seen and memorized all available samples.

**What about learning rate?**

The learning rate in this scenario is likely **just right** — not too high, not too low:
- If too high: you would see **oscillations/fluctuations** — the loss would jump up and down rather than smoothly flatten
- If too low: the initial descent would be very slow from the start, not steep
- This curve shows fast early descent (good step size) then clean flattening (convergence) — that's the sign of a well-tuned learning rate

**Recommendation:** Reduce epochs to 4–5; the model is trained after that regardless of more epochs.

---

### Q9 — Overfitting Diagram

**Question:** A training loss curve (blue) decreases through all epochs. A test loss curve (red) decreases initially, then flattens or increases while training loss continues decreasing. What is this phenomenon called?

**Answer: Overfitting**

The gap between train loss and test loss indicates the model has memorized the training data and cannot generalize to unseen data. The point where overfitting begins is where the test loss stops decreasing (e.g., around epoch 2–3 in the diagram).

**Solutions:**

| Solution | Mechanism |
|---|---|
| Early stopping | Stop training when test loss stops improving for N consecutive epochs |
| Data augmentation | Add more variation to training data to reduce memorization |
| Dropout | Randomly deactivate neurons during training |
| Regularization (L1/L2) | Penalize large weights to reduce overfitting |
| Simpler architecture | Fewer layers / fewer neurons = less capacity to memorize |

---

### Q10 — What Is One Epoch?

**Answer:** One complete pass through the entire training dataset.

**Related terms:**

- **Batch:** A subset of the training data processed in one forward + backward pass
- **Iteration / Step:** One update to the model weights (one batch processed)
- One epoch = multiple iterations if using batches

---

### Q11 — SGD vs. Batch Gradient Descent for Large Datasets

**Question:** Explain why stochastic gradient descent (SGD) is often preferred over batch gradient descent on large datasets.

**Clarification of terms:**

| Name | What it does |
|---|---|
| Batch gradient descent (GD) | Processes the entire dataset before computing one parameter update |
| Stochastic gradient descent (SGD) | Processes one sample at a time; one update per sample |
| Mini-batch gradient descent | Processes a small batch (e.g., 32 samples); one update per batch |

**Why SGD on large datasets:**

With 10 million records, batch gradient descent must process all 10M samples before making a single weight update. SGD updates parameters after every single sample — far more frequent updates, faster convergence (though noisier).

The noise in SGD convergence is an acceptable tradeoff for the speed gained from frequent updates.

**When to prefer batch GD (full):**  
Smaller datasets — where the full dataset is manageable in memory, and smooth (low-noise) convergence is preferable.

**The gradient descent update rule:**

```
W_new = W_old - α * (dL/dW)
```

In batch GD, `dL/dW` is the average gradient over all N samples. In SGD, it is the gradient from one sample. In mini-batch, it is the average over the batch.

---

### Q12 — Two Downsides of KNN

**Answer:**

1. **Slow inference (prediction time):** KNN does not build a model during training. At prediction time, it computes the distance from the query point to every training sample — O(N) per query. For large datasets, this is very slow.

2. **Memory inefficient (non-parametric):** All training data must be kept in memory at runtime. Unlike CNNs that compress dataset knowledge into learned weights, KNN requires storing every training example to function.

---

### Q13 — Role of Learning Rate in Gradient Descent

**Question:** What is the role of learning rate? What happens if it is too small or too large?

**Answer:**

Learning rate (`α`) is the step size of parameter updates — it controls how much each weight changes per update.

| Learning rate | Effect |
|---|---|
| Too high | Overshoots the minimum; loss oscillates or diverges; never converges |
| Too low | Updates are tiny; convergence is extremely slow; may plateau far from the minimum |
| Just right | Fast, stable convergence to a good minimum |

```
W_new = W_old - α * (dL/dW)
```

If `α` is large, `W` changes a lot per step → overshooting. If `α` is small, `W` barely moves per step → very slow convergence.

---

### Q14 — Forward Propagation vs. Backward Propagation

**Question:** Compare and contrast forward and backward propagation in neural networks. Include the sequence of steps, the role of each, and where the loss function and optimization come into play.

**Forward Propagation:**

1. Initialize weights randomly
2. Take input `X`
3. Compute linear combination: `z = Wx + b`
4. Apply activation function: `a = f(z)`
5. Pass `a` to the next layer; repeat through all layers
6. Produce output prediction `ŷ`
7. Compute loss: `L = loss_function(ŷ, y_true)`

**Backward Propagation:**

1. Compute gradient of loss with respect to output layer weights: `dL/dW_last`
2. Use chain rule to propagate gradients backward through each layer: `dL/dW_k = dL/da * da/dz * dz/dW_k`
3. Continue back to the first layer
4. Update all weights: `W = W - α * (dL/dW)`

**Key relationship:**
- Forward prop: produces predictions and computes loss
- Backward prop: uses the loss to compute gradients and update weights via the optimization algorithm (SGD, Adam, etc.)
- This cycle repeats for each batch/sample across all epochs until loss reaches an acceptable level

> The chain rule application through activation functions at each layer is where the math gets complex. Each layer's gradient depends on the gradient of all subsequent layers.

---

## 4. Student Q&A

### Overfitting on Assignment (Cat/Dog Classification)

**Student:** My training accuracy was ~90% but test accuracy was ~70%. Several classmates had the same issue.

**Instructor:** That gap is overfitting — the model memorized training images and doesn't generalize. With a small image dataset:
- The sample count is too low for the model to learn robust features
- Even with data augmentation, the ceiling is limited by dataset size
- **Solutions to try:** data augmentation (zoom, brightness — avoid rotation if orientation is meaningful), transfer learning (use pre-trained VGG16), early stopping, regularization (dropout, L2)
- Increasing epochs can help up to a point if you also use early stopping to prevent over-training
- If you used CNNs and still got ~70% test accuracy on a small dataset, that's an expected result — there is no direct fix beyond more data or transfer learning

### CNN Architecture — Full Visual Walkthrough

**Student:** Can you draw how a CNN works end-to-end?

**Instructor (narrated diagram):**

```
Input image (e.g., 6×6×3)
    ↓
Conv Layer 1: 5 kernels of size 3×3
    → Each kernel slides across all 3 channels
    → Produces 5 output matrices (one per kernel)
    [kernel count = number of output matrices — this is your design choice / hyperparameter]

MaxPool Layer:
    → Each of the 5 matrices is halved in spatial size
    → Still 5 matrices, just smaller

Conv Layer 2: 10 kernels of size 3×3
    → Each kernel is applied to all 5 input matrices
    → Results from all 5 applications are summed → one output matrix per kernel
    → Produces 10 output matrices
    [kernel count again = your hyperparameter choice]

MaxPool Layer:
    → Halves each of the 10 matrices

Flatten:
    → Each 3×3 matrix = 9 values
    → 10 matrices × 9 = 90 values → one vector of length 90

Dense Layer (e.g., 3 neurons):
    → Each neuron connects to all 90 inputs
    → Computes z = W·x + b, passes through activation
    → Output: 3 values

Output → loss calculated → backpropagation → weights updated
```

**Why are kernels initialized randomly?**  
You cannot know in advance what kernel values will be good feature extractors for your specific problem. Across thousands of kernels at scale, it's impossible to set by hand. Optimization (gradient descent) finds the good values for you over training iterations.

**What makes a good kernel value?**  
You either experiment (try different architectures and evaluate accuracy) or let the optimization algorithm discover them. The kernel values after training are whatever values minimized the loss.

---

## 5. Final Exam Hints (Summary)

- True/false questions tend to be **false** — same pattern as midterm
- Expect **diagrams** — overfitting curves, feature extraction layer diagrams, loss curves with fluctuations
- For overfitting diagrams: identify where overfitting starts, list two solutions
- For fluctuating loss curves: identify the cause (learning rate too high), identify the fix (reduce learning rate)
- Feature extraction: earlier layers = high-level / coarse features; middle layers = shapes, textures; later layers = fine-grained, detailed features; deeper = stronger
- CNNs: know each component by name (Conv2D, MaxPool, Flatten, Dense) and what it does
- No coding on the exam except possibly simple CV2 operations
- Final project is the coding deliverable

---

*End of Session 25 transcript.*

# Computer Vision — DPS920 — Session 17

> 1:1 markdown transcription of `Session17.pdf`.

---

## Slide 1 — Title

Computer Vision
DPS920
Session 17

---

## Slide 2 — ML in Images

---

## Slide 3 — ML in Images

(2×2 grid of values [[2,3],[4,5]] → `flatten()` → 1D array [2,3,4,5])

---

## Slide 4 — Fire Detection

(two photographs: one of a forest fire with smoke, one of a foggy forest road)

---

## Slide 5 — Code

```python
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from joblib import dump

def load_data():
    data_list = []
    labels = []

    for i, address in enumerate(glob.glob('S18/fire_dataset\\*\\*')):
        img = cv2.imread(address)
        img = cv2.resize(img, (32,32))
        img = img/255
        img = img.flatten()

        data_list.append(img)
        label = address.split("\\")[-1].split(".")[0]
        labels.append(label)

        if i % 100 == 0:
            print(f"[INFO]: {i}/1000 processed")

    data_list = np.array(data_list)

    X_train, X_test, y_train, y_test = train_test_split(data_list, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
dump(clf, 'fire_detection.z')
```

---

## Slide 6 — Code (inference)

```python
import cv2
import numpy as np
import glob
from joblib import load

clf = load('S18/fire_detector.z')

for item in glob.glob("test_images\\*"):
    img = cv2.imread(item)
    r_img = cv2.resize(img, (32,32))
    r_img = img/255
    r_img = img.flatten()
    r_img = np.array([r_img])

    label = clf.predict(r_img)[0]

    cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0,255,0), 2)
    cv2.imshow()
    cv2.waitKey(0)

cv2.destroyAllWindows()
```

---

## Slide 7 — Types of Preprocessing in Images

- Resize
- Normalization
- Flat

---

## Slide 8 — Brain

(stylized image of a human head in profile with a glowing brain, captioned "Brain")

---

## Slide 9 — Perceptron

(illustration of a biological neuron with dendrites labeled `W1X1`, `W2X2`, `W3X3`, `W4X4`, `W5X5`, `W6X6`, `W7X7`, with an "Activation Function" label in the cell body and a "Signal" along the axon)

---

## Slide 10 — Perceptron Algorithm

(diagram: INPUT VALUES x₁, x₂, x₃ → WEIGHTS w₁, w₂, w₃ → SUMMATION (Σ) → STEP FUNCTION → OUTPUT)

---

## Slide 11 — MLR

(diagram showing nodes x₀, x₁, x₂, xₙ with weights b0, b1, b2, bn feeding into a `+` node)

$$ y = b_0 + b_1 x_1 + b_2 x_2 + \ldots + b_n x_n $$

---

## Slide 12 — Perceptron Algorithm

(same perceptron diagram as slide 10, with an added bias node `b`)

$$ f(x_0 w_0 + x_1 w_1 + x_2 w_2 + x_3 w_3) $$

$$ f(\sum w_i x_i) $$

$$ f(net) $$

---

## Slide 13 — Step Function

(step function graph: u(t) = 0 for t ≤ 0, u(t) = 1 for t > 0)

$$ u(t) = \begin{cases} 1 & t > 0 \\ 0 & t > 0 \end{cases} $$

Pros: simple

Cons: no derivates

---

## Slide 14 — Framework

Data → ML algorithm (Start with random parameters) → Predict output → Evaluate output with loss → Optimization → Update parameters → (back to ML algorithm)

---

## Slide 15 — Sigmoid

(sigmoid curve from −8 to 8, crossing 0.5 at z = 0)

$$ \phi(z) = \frac{1}{1 + e^{-z}} $$

Pros: derivative at all points

Cons: small derivatives at end points

---

## Slide 16 — Multi Layer Perceptron

(diagram: Input layer → Hidden layers → Output layer)

---

## Slide 17 — 2 main steps

- Forward Pass
- Backward Pass

---

## Slide 18 — Forward Pass

---

## Slide 19 — Forward Pass

(diagram of a network: inputs x₀, x₁, 1 → Layer 0 → Layer 1; weights `w⁰_{2,0}`, `w⁰_{2,1}`, `w⁰_{2,2}` highlighted in red from the bias input)

---

## Slide 20 — Example

(network with 3 inputs i₀, i₁, i₂ → 3 hidden nodes h₀, h₁, h₂ → 1 output o₀; weights on edges: 0.351, −0.097, 0.457, 1.076, −0.165, −0.165, 1.116, 0.542, −0.331; hidden → output weights: 0.383, −0.327, −0.329)

---

## Slide 21 — Example

(inputs 0, 1, 1 placed on i₀, i₁, i₂ of the same network)

$$ h_0 = 0(0.351) + 1(1.076) + 1(1.116) = 2.192 $$

---

## Slide 22 — Example

h₀ = 0.899

$$ \frac{1}{1 + e^{-2.192}} = 0.899 $$

---

## Slide 23 — Exmaple

h₀ = 0.899

h₁ = 0.593

---

## Slide 24 — Exmaple

h₀ = 0.899

h₁ = 0.593

h₂ = 0.378

o₀ = 0.506

---

## Slide 25 — Backward Pass

---

## Slide 26 — Backpropagation

(diagram: Input Layer (x₀, x₁, 1) → Layer 0 → Hidden Layer → Layer 1 → Output Layer)

---

## Slide 27 — Simpler Backpropagation

For simplicity let's only consider summation and multiplications (no activation or bias)

(diagram: i₀, i₁ → h₀, h₁ → o₀ → Prediction; weights `w⁰_{0,0}`, `w⁰_{0,1}`, `w⁰_{1,0}`, `w⁰_{1,1}`, `w¹_{0,0}`, `w¹_{1,0}`)

---

## Slide 28 — Backpropagation

$$ Prediction = (x_0 w^0_{0,0} + x_0 w^0_{1,0}) w^1_{0,0} + (x_0 w^0_{0,1} + x_0 w^0_{1,1}) w^1_{1,0} $$

$$ Loss = \frac{(prediction - actual)^2}{2} $$

---

## Slide 29 — Goal

Final goal is to achieve to the best value for parameters

What are the parameters?

---

## Slide 30 — Backpropagation

$$ Prediction = (x_0 w^0_{0,0} + x_0 w^0_{1,0}) w^1_{0,0} + (x_0 w^0_{0,1} + x_0 w^0_{1,1}) w^1_{1,0} $$

$$ Loss = \frac{(prediction - actual)^2}{2} $$

$$ \frac{\partial loss}{\partial w^1_{1,0}} = \frac{\partial loss}{\partial Prediction} \times \frac{\partial Prediction}{\partial w^1_{1,0}} $$

---

## Slide 31 — Example

(network with inputs 2, 3 on i₀, i₁, weights 0.351, −0.097, 1.076, −0.165, 0.383, 0.327)

$$ \frac{\partial loss}{\partial w^1_{1,0}} = \frac{\partial loss}{\partial Prediction} \times \frac{\partial Prediction}{\partial w^1_{1,0}} $$

Actual y = 1

---

## Slide 32 — Example

$$ Prediction = (2 \times 0.351 + 3 \times 1.076) \, 0.383 + (2 \times -0.097 + 3 \times -0.165) \, 0.327 = 1.730 $$

---

## Slide 33 — Example

$$ Prediction = (x_0 w^0_{0,0} + x_0 w^0_{1,0}) w^1_{0,0} + (x_0 w^0_{0,1} + x_0 w^0_{1,1}) w^1_{1,0} $$

$$ \Delta = Prediction - actual = 1.730 - 1 = 0.730 $$

$$ h_1 = -0.097 \times 2 - 3 \times 0.165 = -0.689 $$

$$ \frac{\partial Error}{\partial w^1_{1,0}} = \frac{\partial Error}{\partial Prediction} \times \frac{\partial Prediction}{\partial w^1_{1,0}} \Rightarrow \Delta h_1 $$

$$ \frac{\partial Error}{\partial w^1_{1,0}} = (0.730) \times (-0.689) = \boxed{-0.502} $$

---

## Slide 34 — Example (weight update)

$$ \frac{\partial Error}{\partial w^1_{1,0}} = \frac{\partial Error}{\partial Prediction} \times \frac{\partial Prediction}{\partial w^1_{1,0}} \Rightarrow \Delta h_1 $$

$$ \frac{\partial Error}{\partial w^1_{1,0}} = (0.730) \times (-0.689) = \boxed{-0.502} $$

$$ w^+ = w^- - \alpha \frac{\partial L}{\partial w} $$

$$ w^1_{1,0\,new} = 0.327 - (-0.502) = 0.829 $$

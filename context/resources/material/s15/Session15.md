# Computer Vision — DPS920 — Session 15

> 1:1 markdown transcription of `Session15.pdf`.

---

## Slide 1 — Title

Computer Vision
DPS920
Session 15

---

## Slide 2 — Overview

- LR
- Multiple LR
- MSE, MAE
- Examples

---

## Slide 3 — Agenda

- Optimization
- Gradient Descend
- Example

---

## Slide 4 — What is Left?

1. Optimization and Loss Function
2. Logistic Regression
3. ML and Images
4. Perceptron and Neural Networks
5. Deep Neural Networks
6. Convolution Neural Networks (CNN)
7. Advanced CNNs
8. Introduction to object detection, segmentation, and image generation methods with AI
9. (Vision today, tools, future of CV)
10. Project

---

## Slide 5 — Problem

- Some datasets have millions of features -> impossible to calculate
- They are not always linear
- Some problems do not have closed form formulas

---

## Slide 6 — Problem

- Some datasets have millions of features -> impossible to calculate
- They are not always linear
- Some problems do not have closed form formals

Solution -> Optimization

---

## Slide 7 — Optimization

- Start random in space
- Take gradual steps towards your goal
- Not the best best answer but a solution close to the best

---

## Slide 8 — Example

(scatter plot of 4 points: (1,2), (2,1), (3,4), (4,3))

---

## Slide 9 — Example

(scatter plot of 4 points with a random red line)

---

## Slide 10 — What is our goal?

---

## Slide 11 — What is our goal?

Minimize loss

Gain parameter values

---

## Slide 12 — What Was Loss?

- MSE (mean squared error)

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

---

## Slide 13 — Example

(scatter plot of 4 points)

---

## Slide 14 — Example

(scatter plot of 4 points with a random red line)

---

## Slide 15 — Example

(scatter plot showing the red line shifting toward a new position — pink shows previous, red shows new)

---

## Slide 16 — Example

(scatter plot showing the line continuing to update)

---

## Slide 17 — Example

(scatter plot showing the line approaching the best fit)

---

## Slide 18 — Framework

Data → ML algorithm (Start with random parameters) → Predict output → Evaluate output with loss → Optimization → Update parameters → (back to ML algorithm)

---

## Slide 19 — Optimization Algorithms

- Gradient Descend
- Stochastic Gradient Descend
- Adam
- AdamW
- RMSProp
- Newton's Method

---

## Slide 20 — Gradient Descend

- The problem was how we get to update the line in LR

---

## Slide 21 — Mathematically How?

- We should solve this mathematically
- But how?

---

## Slide 22 — Gradient Descend

- The problem was how we get to update the line in LR
- Gradient descend does this with calculating derivatives again!

---

## Slide 23 — Gradient Descend

- The problem was how we get to update the line in LR
- Gradient descend does this with calculating derivatives again!
- It updates parameters by moving in the opposite direction of their derivatives with respect to loss!

---

## Slide 24 — Gradient Descend

- The problem was how we get to update the line in LR
- Gradient descend does this with calculating derivatives again!
- It updates parameters by moving in the opposite direction of their derivatives with respect to loss!

But what do we mean?

---

## Slide 25 — opposite direction of parameter derivatives with respect to loss!

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

(Mean-Squared Error loss curve: L vs (ŷ − y))

---

## Slide 26 — opposite direction of parameter derivatives with respect to loss!

(MSE loss curve with a red dot high on the right side)

Our first average loss with the random line

---

## Slide 27 — opposite direction of parameter derivatives with respect to loss!

(MSE loss curve with a red dot high on the right side and an arrow pointing down toward 0)

We want this loss to be as close as to 0

---

## Slide 28 — opposite direction of parameter derivatives with respect to loss!

(MSE loss curve with red dot and arrow going down)

In order to move towards that direction, we have to move in the opposite direction of slope (derivate)

---

## Slide 29 — opposite direction of parameter derivatives with respect to loss!

(MSE loss curve with red dot at the top and a tangent slope line)

---

## Slide 30 — opposite direction of parameter derivatives with respect to loss!

(MSE loss curve with red dot moved down with slope indicator)

---

## Slide 31 — opposite direction of parameter derivatives with respect to loss!

(MSE loss curve with red dot near the bottom-left with slope indicator)

$$ w^+ = w^- - \frac{\partial L}{\partial w} $$

---

## Slide 32 — opposite direction of parameter derivatives with respect to loss!

(MSE loss curve with red dot near the bottom-left with slope indicator)

$$ w^+ = w^- - \alpha \frac{\partial L}{\partial w} $$

---

## Slide 33 — Now let's see mathematically!

- opposite direction of **parameter derivatives** with respect to loss!
- Loss is calculated:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

---

## Slide 34 — Now let's see mathematically!

- opposite direction of **parameter derivatives** with respect to loss!
- Loss is calculated:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

- Given in data → Y_i
- ax+b → Ŷ_i

---

## Slide 35 — Now let's see mathematically!

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

- Given in data → Y_i
- 2x+4 → Ŷ_i

---

## Slide 36 — Now let's see mathematically!

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

Data = (1,2), (2,1), (3,4), (4,3)
y' = 2x + 4

Loss = [ (2-6)² + (1-8)² + (4-10)² + (3-12)² ] / 4 = [16 + 49 + 36 + 81] / 4 = 45.5

---

## Slide 37 — Now let's see mathematically!

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

Data = (1,2), (2,1), (3,4), (4,3)
y' = 2x + 4

Loss = [ (2-6)² + (1-8)² + (4-10)² + (3-12)² ] / 4 = [16 + 49 + 36 + 81] / 4 = 45.5

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y - (mx + c))^2 $$

---

## Slide 38 — Now let's see mathematically!

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

Data = (1,2), (2,1), (3,4), (4,3)
y' = 2x + 4

Loss = [ (2-6)² + (1-8)² + (4-10)² + (3-12)² ] / 4 = [16 + 49 + 36 + 81] / 4 = 45.5

$$ E = \frac{1}{n} \sum_{i=0}^{n} (y_i - (mx_i + c))^2 $$

---

## Slide 39 — (Gradient formulas)

$$ E = \frac{1}{n} \sum_{i=0}^{n} (y_i - (mx_i + c))^2 $$

$$ D_m = \frac{1}{n} \sum_{i=0}^{n} 2(y_i - (mx_i + c))(-x_i) $$

$$ D_m = \frac{-2}{n} \sum_{i=0}^{n} x_i (y_i - \bar{y}_i) $$

$$ m = m - \alpha \times D_m $$

$$ D_c = \frac{-2}{n} \sum_{i=0}^{n} (y_i - \bar{y}_i) $$

$$ c = c - \alpha \times D_c $$

---

## Slide 40 — (Gradient update numerical example)

$$ E = \frac{1}{n} \sum_{i=0}^{n} (y_i - (mx_i + c))^2 $$

$$ D_m = \frac{1}{n} \sum_{i=0}^{n} 2(y_i - (mx_i + c))(-x_i) $$

$$ D_m = \frac{-2}{n} \sum_{i=0}^{n} x_i (y_i - \bar{y}_i) $$

$$ m = m - \alpha \times D_m $$

new m = 2 – 0.01 * 2 = 1.98

$$ D_c = \frac{-2}{n} \sum_{i=0}^{n} (y_i - \bar{y}_i) $$

$$ c = c - \alpha \times D_c $$

new c = 4 – 0.01 * 2 = 3.98

→ y = 1.98*x + 3.98

---

## Slide 41 — (Comparison)

New line: y = 1.98*x + 3.98

Previous line: y = 2x+4

Best fitted line: y = x + 0.6

---

## Slide 42 — Framework

Data → ML algorithm (Start with random parameters) → Predict output → Evaluate output with loss → Optimization → Update parameters → (back to ML algorithm)

---

## Slide 43 — Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# DATA
data = pd.read_csv("Session14\student_scores.csv")
# print(data.describe())
# print(data.info())

X = data.drop(columns=['Scores'])
y = data['Scores']


# plt.scatter(X, y)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# MODEL
model = LinearRegression()
model.fit(X_train, y_train)

model_opt = SGDRegressor()
model_opt.fit(X_train, y_train)


# EVALUATE
print(f'y = {model.coef_[0]} * X + {model.intercept_}')
predictions = model.predict(X_test)
predictions_opt = model_opt.predict(X_test)

print(f'MAE = {mean_absolute_error(y_test, predictions)}')
print(f'MSE = {mean_squared_error(y_test, predictions)}')

print(f'MAE opt = {mean_absolute_error(y_test, predictions_opt)}')
print(f'MSE opt = {mean_squared_error(y_test, predictions_opt)}')
```

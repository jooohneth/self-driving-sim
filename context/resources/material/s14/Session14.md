# Computer Vision — DPS920 — Session 14

> 1:1 markdown transcription of `Session14.pdf`.

---

## Slide 1 — Title

Computer Vision
DPS920
Session 14

---

## Slide 2 — ML Algorithm Categorizations

- Supervised Learning (Apple)
- Unsupervised Learning
- Reinforcement Learning

---

## Slide 3 — ML Algorithm Categorizations

(Supervised Learning highlighted)

- **Supervised Learning (Apple)**
- Unsupervised Learning
- Reinforcement Learning

---

## Slide 4 — Supervised Learning Algorithms Categorization

- Classification
- Regression

---

## Slide 5 — Regression

- i) Simple
  - linear
  - nonlinear
- i) Multiple
  - linear
  - nonlinear
  - Polynomial

---

## Slide 6 — Simple Linear Regression

model the relationship between one independent variable (X) and one dependent variable (Y)

$$ y = a + bx $$

- slope
- intercept

---

## Slide 7 — Problem

Our goal:

find a line that minimizes our error

---

## Slide 8 — Solution 1

- Visualize data and plot points → not scalable
- Use Closed-Form Formulas → complex
- Use optimization algorithms

---

## Slide 9 — Formulate

Points: (1,2), (2,1), (3,4), (4,3)

Solution: map to a linear line

---

## Slide 10 — Solution

- Points: (1,2), (2,1), (3,4), (4,3)

```
a*1 + b = 2
a*2 + b = 1
a*3 + b = 4
a*4 + b = 3
```

Can't find the exact **linear** line

---

## Slide 11 — Solution

- Points: (1,2), (2,1), (3,4), (4,3)

```
a*1 + b = 2
a*2 + b = 1
a*3 + b = 4
a*4 + b = 3
```

We want to find a line to minimize the distance between the real value and the lines output

y1, y2, y3, y4
y1', y2', y3', y4'

---

## Slide 12 — Solution

- Points: (1,2), (2,1), (3,4), (4,3)

```
a*1 + b = 2
a*2 + b = 1
a*3 + b = 4
a*4 + b = 3
```

We want to find a line to minimize the distance between the real value and the lines output

y1, y2, y3, y4
y1', y2', y3', y4'

→ Minimize the distance

---

## Slide 13 — Solution

- Points: (1,2), (2,1), (3,4), (4,3)

```
a*1 + b = 2
a*2 + b = 1
a*3 + b = 4
a*4 + b = 3
```

We want to find a line to minimize the distance between the real value and the lines output

y1, y2, y3, y4
y1', y2', y3', y4'

→ min [ (y1-y1')² + (y2-y2')² + (y3-y3')² + (y4-y4')² ]

---

## Slide 14 — Example

Actual outputs: 2, 1, 4, 3

Line's outputs: -1, 0, 1, 2

We do not know the equation of the line! But we want to find it
y' = ax + b

---

## Slide 15 — Example

Actual outputs: 2, 1, 4, 3

(y1-y1')² + (y2-y2')² + (y3-y3')² + (y4-y4')²

loss/distance = (2-(a+b))² + (1-(2a+b))² + (4-(3a+b))² + (3-(4a+b))²

---

## Slide 16 — Example

loss/distance = (2-(a+b))² + (1-(2a+b))² + (4-(3a+b))² + (3-(4a+b))²

= 30*a² + 20*a*b + 56*a + 4*b² – 20*b + 30

---

## Slide 17 — Minimum Loss

- We want to move towards a point with minimum loss!
- It is not the best fit, but it is better than nothing!

---

## Slide 18 — Minimum Loss

- Mathematically speaking, when we want the minimum of something what did we do?

---

## Slide 19 — Derivative = 0

---

## Slide 20 — Equation

- 30*a² + 20*a*b + 56*a + 4*b² – 20*b + 30

df/da = 0

df/db = 0

---

## Slide 21 — Solve Equation

- 30*a² + 20*a*b + 56*a + 4*b² – 20*b + 30

df/da = 0 → 60a + 20b + 56 = 0

df/db = 0 → 20a + 8b - 20 = 0

---

## Slide 22 — Solve Equation

- 30*a² + 20*a*b + 56*a + 4*b² – 20*b + 30

df/da = 0 → 60a + 20b + 56 = 0

df/db = 0 → 20a + 8b + 20 = 0

→ a = 1
   b = 0.6

---

## Slide 23 — Final Equation

Now we have a line with minimum error

y = x + 0.6

---

## Slide 24 — Final Output

y = x + 0.6

---

## Slide 25 — Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


dataset = pd.read_csv("S12_student_scores.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

plt.scatter(X, y)
plt.title("Hours vs Marks")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.coef_)
print(regressor.intercept_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, "prediction": y_pred})

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MAE: {mean_squared_error(y_test, y_pred)}")
```

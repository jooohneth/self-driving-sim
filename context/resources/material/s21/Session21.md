# Computer Vision — DPS920 — Session 21

> 1:1 markdown transcription of `Session21.pdf`.

---

## Slide 1 — Title

Computer Vision
DPS920
Session 21

---

## Slide 2 — Overview

- Fire detection with neural networks
- NN for classification vs Regression
- Encoding overview
- Softmax
- Cross entropy loss

---

## Slide 3 — Agenda

- Convolutional Neural Networks
- Conv layer
- Padding
- Stride
- Pooling
- Code

---

## Slide 4 — What is Left?

1. Optimization and Loss Function
2. Code + Logistic Regression
3. ML and Images
4. Perceptron and Neural Networks
5. Neural Networks
6. **Convolution Neural Networks (CNN)**
7. **Advanced CNNs**
8. **Introduction to object detection, Segmentation and image generation methods with AI**

(items 1–5 are greyed out; items 6–8 are highlighted as the remaining topics)

---

## Slide 5 — Convolutional Neural Networks (CNNs)

- Images are 2D or 3D grids with local dependencies
- Traditional NNs ignore locality and translation invariance
- We need a method that captures patterns like edges, textures, shapes

- LeNet: https://www.youtube.com/watch?v=FwFduRA_L6Q
- Convolution with Optimization

(photo of Yann LeCun on the right)

---

## Slide 6 — Layers in CNNs

(horizontal bar diagram showing five layer types with icons underneath each)

- Convolutional (grid-with-kernel icon)
- Pooling (red-bordered grid icon)
- Fully Connected (small neural-net graph icon)
- Activation function (Σ / sig icon)
- Batch Normalization (stacked-layers icon)

---

## Slide 7 — Convolution Layer

(diagram: small orange square labelled "Kernel f*f" → large grey square labelled "Image n*n" → orange square labelled "output")

---

## Slide 8 — Multiple Convolutions

(diagram: stack of 4 small colored squares → large grey square → stack of 4 larger colored squares)

---

## Slide 9 — Convolution for Colored Images

(illustration from indoml.com showing a 6×6×3 input volume, a 3×3×3 filter with parameters Size f=3, #channels n_C=3, Stride s=1, Padding p=0, and the resulting single output value "2" computed as the sum of three per-channel element-wise products)

$$ n_H \times n_W \times n_C = 6 \times 6 \times 3 $$

---

## Slide 10 — Padding

(6×6 input grid padded with zeros to 8×8, convolved with a 3×3 kernel [[1,0,−1],[1,0,−1],[1,0,−1]], producing a 6×6 result grid whose first row begins with −10, −13, 1 and second row begins with −9, 3, 0)

$$ 6 \times 6 \rightarrow 8 \times 8 $$

---

## Slide 11 — Stride

(two diagrams side by side)

- **S = 1**: 4×4 input with three overlapping 2×2 windows (red, green, blue) → 3×3 output
- **S = 2**: 4×4 input with two non-overlapping 2×2 windows (red, green) → 2×2 output

---

## Slide 12 — Post Convolution Dimensions

(diagram: H₁ × W₁ × D₁ input volume → output volume H₂ × W₂ × D₂)

$$ W_2 = \frac{(W_1 - F + 2P)}{S} + 1 $$

$$ H_2 = \frac{(H_1 - F + 2P)}{S} + 1 $$

$$ D_2 = K $$

---

## Slide 13 — Layers in CNNs

(same horizontal bar diagram as Slide 6: Convolutional, Pooling, Fully Connected, Activation function, Batch Normalization)

---

## Slide 14 — MaxPooling

(4×4 input grid:
```
2 2 7 3
9 4 6 1
8 5 2 4
3 1 2 6
```
Max Pool with Filter (2×2) and Stride (2, 2) → 2×2 output:
```
9 7
8 6
```
)

---

## Slide 15 — Average Pooling

(4×4 input grid:
```
2 2 7 3
9 4 6 1
8 5 2 4
3 1 2 6
```
Average Pool with Filter (2×2) and Stride (2, 2) → 2×2 output:
```
4.25 4.25
4.25 3.5
```
)

---

## Slide 16 — Layers in CNNs

(same horizontal bar diagram as Slide 6: Convolutional, Pooling, Fully Connected, Activation function, Batch Normalization)

---

## Slide 17 — MLP

(diagram: 32×32×3 input volume → Flat → 3072-long vector → Fully Connected → 225-long vector)

---

## Slide 18 — CNNs

(LeNet-style diagram with handwritten digit "3" as input:

- input 32 × 32
- C₁ feature maps 28 × 28 (5×5 convolution)
- S₁ feature maps 14 × 14 (2×2 subsampling)
- C₂ feature maps 10 × 10 (5×5 convolution)
- S₂ feature maps 5 × 5 (2×2 subsampling)
- n₁ fully connected
- n₂ output → 0, 1, …, 8, 9

with the red dashed region labelled "feature extraction" and the blue dashed region labelled "classification")

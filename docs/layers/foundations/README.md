# Foundations: Math & Programming for AI

> **The bedrock of all AI systems.** Before diving into neural networks and transformers, master these fundamentals.

---

## Layer Navigation

| Layer | Content | Format |
|-------|---------|--------|
| Layer 0 | [Overview](#layer-0-overview) | This file |
| Layer 1 | [Concepts](./concepts.md) | Markdown |
| Layer 2 | [Deep Dive](./deep-dive.md) | Technical |
| Layer 3 | [Labs](../../labs/foundations/) | Notebooks |
| Layer 4 | [Advanced](./advanced.md) | Mastery |

---

## Layer 0: Overview

### Why Math Matters for AI

Every AI model - from simple linear regression to GPT-4 - is built on mathematical operations:

```
Input Data → Mathematical Transformations → Output
     ↓              ↓                          ↓
  Vectors     Matrix Multiplications      Predictions
```

**You don't need a PhD**, but understanding these concepts makes AI intuitive rather than magical.

---

## Core Topics

### 1. Linear Algebra (Essential)

| Topic | What It Does | AI Application |
|-------|--------------|----------------|
| **Vectors** | Represent data points | Word embeddings, features |
| **Matrices** | Transform data | Neural network layers |
| **Matrix Multiplication** | Combine transformations | Forward pass in networks |
| **Dot Product** | Measure similarity | Attention mechanisms |
| **Eigenvalues/Eigenvectors** | Find principal directions | PCA, dimensionality reduction |

### 2. Calculus (For Training)

| Topic | What It Does | AI Application |
|-------|--------------|----------------|
| **Derivatives** | Rate of change | Gradients for learning |
| **Chain Rule** | Compose derivatives | Backpropagation |
| **Partial Derivatives** | Multi-variable rates | Gradient descent |
| **Optimization** | Find minimum/maximum | Loss minimization |

### 3. Probability & Statistics

| Topic | What It Does | AI Application |
|-------|--------------|----------------|
| **Probability Distributions** | Model uncertainty | Generative models |
| **Bayes' Theorem** | Update beliefs | Bayesian inference |
| **Expectation/Variance** | Summarize distributions | Loss functions |
| **Sampling** | Generate from distributions | Diffusion models |

### 4. Programming Foundations

| Topic | Tools | AI Application |
|-------|-------|----------------|
| **Python** | Core language | Everything |
| **NumPy** | Numerical computing | Array operations |
| **PyTorch/TensorFlow** | Deep learning | Model building |
| **Jupyter** | Interactive computing | Experimentation |

---

## Quick Intuition

### What is a Vector?
```
A list of numbers representing something:

Word "king" → [0.2, -0.5, 0.8, 0.1, ...]  (embedding)
Image pixel → [255, 128, 64]               (RGB color)
User prefs → [0.9, 0.1, 0.7, 0.3]          (features)
```

### What is a Matrix?
```
A grid of numbers that transforms vectors:

[a b]   [x]   [ax + by]
[c d] × [y] = [cx + dy]

Neural network layer = matrix multiplication + activation
```

### What is Matrix Multiplication?
```
Combining transformations:

Weights × Input = Output
(2x3)    (3x1)   (2x1)

[w1 w2 w3]   [x1]   [w1*x1 + w2*x2 + w3*x3]
[w4 w5 w6] × [x2] = [w4*x1 + w5*x2 + w6*x3]
             [x3]
```

---

## Learning Path

```
Week 1-2: Vectors & Matrices
    ↓
Week 3-4: Matrix Operations & NumPy
    ↓
Week 5-6: Calculus & Gradients
    ↓
Week 7-8: PyTorch Basics
    ↓
Ready for: Neural Networks
```

---

## Notebooks (Layer 3)

| Notebook | Topic | Prerequisites |
|----------|-------|---------------|
| `01-vectors-basics.ipynb` | Vector operations | Python |
| `02-matrices-basics.ipynb` | Matrix operations | Vectors |
| `03-matrix-multiplication.ipynb` | MatMul deep-dive | Matrices |
| `04-numpy-essentials.ipynb` | NumPy for AI | Python |
| `05-gradients-intro.ipynb` | Derivatives & gradients | Calculus basics |
| `06-pytorch-tensors.ipynb` | Tensors in PyTorch | NumPy |
| `07-simple-neural-net.ipynb` | Putting it together | All above |

---

## Resources

### Interactive
- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - Visual intuition
- [Immersive Math](http://immersivemath.com/ila/) - Interactive textbook
- [Matrix Multiplication Visualizer](http://matrixmultiplication.xyz/)

### Courses
- Stanford CS229 (Machine Learning)
- fast.ai (Practical Deep Learning)
- Khan Academy (Math foundations)

### Books
- "Mathematics for Machine Learning" (Deisenroth et al.)
- "Deep Learning" (Goodfellow et al.) - Chapter 2-4

---

## Next Steps

After foundations:
→ [Neural Networks Basics](../llms/neural-networks.md)
→ [Vision AI Fundamentals](../visual-ai/README.md)
→ [Understanding Transformers](../llms/transformers.md)

---

*"The math isn't meant to scare you - it's meant to give you superpowers."*

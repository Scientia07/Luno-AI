# AI Foundations: Projects & Comparisons

> **Hands-on projects and framework comparisons for AI/ML Foundations**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Linear Regression from Scratch
**Goal**: Implement linear regression without libraries

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | Python + NumPy |
| Skills | Gradient descent, loss functions |

**Tasks**:
- [ ] Generate synthetic data
- [ ] Implement forward pass
- [ ] Calculate MSE loss
- [ ] Implement gradient descent
- [ ] Visualize learning curve

**Starter Code**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data
X = np.random.randn(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.5

# Initialize parameters
w = np.random.randn(1)
b = np.zeros(1)
lr = 0.1

# Training loop
losses = []
for epoch in range(100):
    # Forward pass
    y_pred = X @ w + b

    # Loss
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2/len(X)) * X.T @ (y_pred - y)
    db = (2/len(X)) * np.sum(y_pred - y)

    # Update
    w -= lr * dw
    b -= lr * db

print(f"Learned: w={w[0]:.3f}, b={b[0]:.3f}")
```

---

#### Project 2: Neural Network from Scratch
**Goal**: Build a 2-layer neural network

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 3-4 hours |
| Technologies | NumPy only |
| Skills | Backpropagation, activations |

**Tasks**:
- [ ] Implement sigmoid activation
- [ ] Implement forward pass
- [ ] Implement backpropagation
- [ ] Train on XOR problem
- [ ] Visualize decision boundary

---

### Intermediate Projects (L2)

#### Project 3: CNN Image Classifier
**Goal**: Build and train a CNN from scratch

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | PyTorch |
| Skills | Convolutions, pooling, training loops |

**Tasks**:
- [ ] Implement Conv2D layer
- [ ] Implement MaxPool layer
- [ ] Build CNN architecture
- [ ] Train on MNIST/CIFAR-10
- [ ] Evaluate accuracy
- [ ] Visualize filters

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

#### Project 4: Transformer from Scratch
**Goal**: Implement attention mechanism

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 8-12 hours |
| Technologies | PyTorch |
| Skills | Self-attention, positional encoding |

**Tasks**:
- [ ] Implement scaled dot-product attention
- [ ] Implement multi-head attention
- [ ] Add positional encoding
- [ ] Build transformer encoder block
- [ ] Train on simple sequence task

---

#### Project 5: Optimization Methods Comparison
**Goal**: Compare SGD, Adam, RMSprop

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | PyTorch |
| Skills | Optimizer internals, convergence |

**Tasks**:
- [ ] Implement SGD with momentum
- [ ] Implement Adam
- [ ] Compare on loss surface visualization
- [ ] Benchmark on MNIST
- [ ] Analyze learning rate sensitivity

---

### Advanced Projects (L3-L4)

#### Project 6: Autoencoder & VAE
**Goal**: Build generative models from fundamentals

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 8-12 hours |
| Technologies | PyTorch |
| Skills | Latent spaces, KL divergence |

**Tasks**:
- [ ] Implement autoencoder
- [ ] Visualize latent space
- [ ] Add variational inference (VAE)
- [ ] Implement reparameterization trick
- [ ] Generate new samples
- [ ] Interpolate in latent space

---

#### Project 7: GAN from Scratch
**Goal**: Implement adversarial training

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | PyTorch |
| Skills | Generator/discriminator, mode collapse |

**Tasks**:
- [ ] Build generator network
- [ ] Build discriminator network
- [ ] Implement adversarial loss
- [ ] Train on MNIST
- [ ] Handle mode collapse
- [ ] Implement DCGAN improvements

---

#### Project 8: Distributed Training
**Goal**: Scale training across GPUs

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | PyTorch DDP |
| Skills | Data parallelism, synchronization |

**Tasks**:
- [ ] Set up multi-GPU environment
- [ ] Implement DataParallel
- [ ] Implement DistributedDataParallel
- [ ] Compare scaling efficiency
- [ ] Profile communication overhead

---

## Framework Comparisons

### Comparison 1: Deep Learning Frameworks

**Question**: Which framework to learn?

| Framework | Ease | Speed | Ecosystem | Production | Best For |
|-----------|------|-------|-----------|------------|----------|
| **PyTorch** | ⭐⭐⭐⭐ | Fast | Large | Good | Research |
| **TensorFlow** | ⭐⭐⭐ | Fast | Largest | Best | Production |
| **JAX** | ⭐⭐ | Fastest | Growing | Good | Research |
| **Keras** | ⭐⭐⭐⭐⭐ | Fast | TF | Good | Beginners |

**Lab Exercise**: Implement same CNN in all frameworks.

```python
# PyTorch
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

# TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# JAX
@jax.jit
def forward(params, x):
    return jnp.dot(jax.nn.relu(jnp.dot(x, params['w1'])), params['w2'])
```

---

### Comparison 2: Numerical Computing Libraries

**Question**: NumPy vs alternatives?

| Library | Speed | GPU | API | Use Case |
|---------|-------|-----|-----|----------|
| **NumPy** | Baseline | No | Standard | General |
| **CuPy** | 10-100x | Yes | NumPy-like | GPU compute |
| **JAX** | Fast | Yes | NumPy-like | Autodiff + JIT |
| **PyTorch** | Fast | Yes | Different | Deep learning |

**Lab Exercise**: Benchmark matrix multiplication across libraries.

---

### Comparison 3: Activation Functions

**Question**: Which activation for your network?

| Activation | Gradient Flow | Speed | Use Case |
|------------|---------------|-------|----------|
| **ReLU** | Good (>0) | Fastest | Hidden layers |
| **LeakyReLU** | Better | Fast | Deep networks |
| **GELU** | Best | Medium | Transformers |
| **Sigmoid** | Poor | Fast | Binary output |
| **Tanh** | Better | Fast | Recurrent |
| **Swish** | Good | Medium | Mobile nets |

**Lab Exercise**: Compare training dynamics with different activations.

---

### Comparison 4: Loss Functions

**Question**: Which loss for your task?

| Loss | Task | Formula |
|------|------|---------|
| **MSE** | Regression | `(y - ŷ)²` |
| **MAE** | Robust regression | `|y - ŷ|` |
| **Cross-Entropy** | Classification | `-y·log(ŷ)` |
| **Focal Loss** | Imbalanced | `-(1-p)^γ·log(p)` |
| **Triplet Loss** | Embeddings | `max(d(a,p) - d(a,n) + m, 0)` |

**Lab Exercise**: Compare loss surfaces visually.

---

## Hands-On Labs

### Lab 1: NumPy Neural Network (4 hours)
```
Data → Forward Pass → Loss → Backward Pass → Update → Iterate
```

### Lab 2: PyTorch Basics (3 hours)
```
Tensors → Autograd → Dataset → DataLoader → Training Loop
```

### Lab 3: CNN Deep Dive (6 hours)
```
Convolution Math → Implement → Train MNIST → Visualize Filters
```

### Lab 4: Transformer Attention (8 hours)
```
Attention Math → Implement → Multi-Head → Positional → Full Block
```

### Lab 5: Training Dynamics (4 hours)
```
Learning Rates → Schedulers → Batch Size → Gradient Clipping
```

---

## Mathematical Foundations

### Pattern 1: Gradient Descent
```
θ(t+1) = θ(t) - η · ∇L(θ)
```

### Pattern 2: Chain Rule (Backprop)
```
∂L/∂w1 = ∂L/∂y · ∂y/∂h · ∂h/∂w1
```

### Pattern 3: Softmax + Cross-Entropy
```
softmax(x)_i = exp(x_i) / Σexp(x_j)
CE = -Σ y_i · log(ŷ_i)
```

### Pattern 4: Attention
```
Attention(Q,K,V) = softmax(QK^T / √d) · V
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Implementation** | 40 | Correct algorithm |
| **Understanding** | 25 | Can explain the math |
| **Experimentation** | 20 | Hyperparameter exploration |
| **Code Quality** | 10 | Clean, documented |
| **Visualization** | 5 | Insightful plots |

---

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Andrej Karpathy's Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n Stanford](http://cs231n.stanford.edu/)

---

*Part of [Luno-AI](../../../README.md) | AI Foundations Track*

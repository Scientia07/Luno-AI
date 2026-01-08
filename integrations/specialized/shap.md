# SHAP Explainability Integration

> **Explain any machine learning model's predictions**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | SHapley Additive exPlanations |
| **Why** | Model-agnostic, mathematically grounded |
| **Use Cases** | Feature importance, debugging, compliance |
| **Best For** | Understanding model decisions |

### SHAP vs Alternatives

| Method | Global | Local | Model-Agnostic |
|--------|--------|-------|----------------|
| **SHAP** | ✓ | ✓ | ✓ |
| **LIME** | ✗ | ✓ | ✓ |
| **Permutation** | ✓ | ✗ | ✓ |
| **Built-in** | Varies | Varies | ✗ |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **Model** | Any trained ML model |

---

## Quick Start (15 min)

```bash
pip install shap
```

```python
import shap
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load data
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Create explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)
```

---

## Learning Path

### L0: Basic Explanations (1-2 hours)
- [ ] Install SHAP
- [ ] Generate SHAP values
- [ ] Summary plot
- [ ] Force plot

### L1: Deep Dive (2-3 hours)
- [ ] Dependence plots
- [ ] Waterfall plots
- [ ] Bar plots
- [ ] Different explainers

### L2: Production (4-6 hours)
- [ ] Deep learning SHAP
- [ ] Text/image explanations
- [ ] Integration
- [ ] Batch processing

---

## Code Examples

### Tree Explainer (Fast)

```python
import shap
import xgboost as xgb

# For tree-based models (XGBoost, LightGBM, Random Forest)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# TreeExplainer is fast and exact
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### Single Prediction Explanation

```python
# Explain single prediction
idx = 0
shap.waterfall_plot(shap_values[idx])

# Force plot for single instance
shap.force_plot(
    explainer.expected_value,
    shap_values[idx],
    X_test[idx],
    feature_names=feature_names
)
```

### Deep Learning

```python
import shap
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("my_model.h5")

# Use DeepExplainer for neural networks
background = X_train[:100]  # Background samples
explainer = shap.DeepExplainer(model, background)

shap_values = explainer.shap_values(X_test[:10])

# Plot
shap.summary_plot(shap_values[0], X_test[:10])
```

### Kernel Explainer (Any Model)

```python
import shap
from sklearn.ensemble import RandomForestClassifier

# Works with any model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# KernelExplainer is model-agnostic but slower
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])
```

### Feature Dependence

```python
# Show how feature affects predictions
shap.dependence_plot(
    "feature_name",
    shap_values,
    X_test,
    feature_names=feature_names,
    interaction_index="auto"  # Auto-detect interaction
)
```

### Text Explanation

```python
import shap
from transformers import pipeline

# Load text classification model
classifier = pipeline("sentiment-analysis")

# Create explainer
explainer = shap.Explainer(classifier)

# Explain prediction
texts = ["This movie was amazing!", "I hated this film."]
shap_values = explainer(texts)

# Visualize
shap.plots.text(shap_values[0])
```

### Image Explanation

```python
import shap
import numpy as np
from tensorflow.keras.applications import ResNet50

# Load model
model = ResNet50(weights='imagenet')

# Create explainer
explainer = shap.GradientExplainer(model, X_train[:50])

# Explain image
shap_values = explainer.shap_values(images)

# Plot
shap.image_plot(shap_values, images)
```

### Batch Processing

```python
import shap
import pandas as pd

def explain_batch(model, X, feature_names, output_path):
    """Generate explanations for batch of samples"""
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Save feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    importance.to_csv(f"{output_path}/feature_importance.csv", index=False)

    # Save individual explanations
    explanations = []
    for i in range(len(X)):
        exp = dict(zip(feature_names, shap_values.values[i]))
        exp['prediction'] = model.predict([X[i]])[0]
        explanations.append(exp)

    pd.DataFrame(explanations).to_csv(f"{output_path}/explanations.csv", index=False)

    return importance
```

### Interactive Dashboard

```python
import shap

# Create interactive force plot
shap.initjs()

# Multiple predictions
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[:100],
    X_test[:100],
    feature_names=feature_names
)

# Save as HTML
shap.save_html("explanations.html", force_plot)
```

### Multiclass Explanation

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Train multiclass model
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train, y_train)

# Explain each class
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# Plot for each class
for i, class_name in enumerate(class_names):
    print(f"\nClass: {class_name}")
    shap.summary_plot(shap_values[i], X_test, feature_names=feature_names)
```

---

## Explainer Types

| Explainer | Best For | Speed |
|-----------|----------|-------|
| `TreeExplainer` | XGBoost, LightGBM, RF | Fast |
| `DeepExplainer` | Neural networks | Medium |
| `KernelExplainer` | Any model | Slow |
| `LinearExplainer` | Linear models | Fast |
| `GradientExplainer` | Deep learning | Medium |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow computation | Use TreeExplainer if possible |
| Memory error | Reduce background samples |
| Inconsistent values | Increase background samples |
| Visualization issues | Call `shap.initjs()` |

---

## Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [SHAP GitHub](https://github.com/slundberg/shap)
- [Original Paper](https://arxiv.org/abs/1705.07874)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/shap.html)

---

*Part of [Luno-AI](../../README.md) | Specialized Track*

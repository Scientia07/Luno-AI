# AutoGluon AutoML Integration

> **Category**: Classical ML
> **Difficulty**: Beginner
> **Setup Time**: 1-2 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
AutoGluon automates machine learning. Give it data, get a state-of-the-art model. It automatically handles feature engineering, model selection, hyperparameter tuning, and ensembling.

### Why Use It
- **Best Accuracy**: Often outperforms manual ML
- **Zero Configuration**: Works out of the box
- **Production Ready**: Models ready to deploy
- **Time Efficient**: Hours instead of weeks
- **Handles Everything**: Tabular, text, images, time series

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Tabular | Classification, regression |
| Text | NLP tasks |
| Image | Classification |
| Time Series | Forecasting |
| Multimodal | Combined modalities |
| Auto-stacking | Ensemble of ensembles |

### AutoML Comparison
| Framework | Accuracy | Speed | Ease |
|-----------|----------|-------|------|
| **AutoGluon** | Best | Medium | Easy |
| H2O AutoML | Good | Fast | Easy |
| PyCaret | Good | Fast | Easiest |
| FLAML | Good | Fastest | Easy |
| Auto-sklearn | Good | Slow | Medium |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None | NVIDIA 8GB+ (for deep learning) |
| RAM | 8 GB | 32 GB |
| Storage | 5 GB | 20 GB |

### Software Dependencies
```bash
# Full install
pip install autogluon

# Or specific modules
pip install autogluon.tabular
pip install autogluon.timeseries
pip install autogluon.text
pip install autogluon.vision
```

---

## Quick Start (10 minutes)

### 1. Install
```bash
pip install autogluon
```

### 2. Train a Model
```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Train (that's it!)
predictor = TabularPredictor(label="target").fit(train_data)

# Predict
predictions = predictor.predict(test_data)

# Evaluate
performance = predictor.evaluate(test_data)
print(performance)
```

### 3. That's It!
AutoGluon automatically:
- Detects problem type (classification/regression)
- Preprocesses features
- Trains multiple models
- Tunes hyperparameters
- Creates ensembles
- Selects best model

---

## Learning Path

### L0: Basic Training (30 min)
**Goal**: Train your first AutoML model

```python
from autogluon.tabular import TabularPredictor

# Minimal code
predictor = TabularPredictor(label="target")
predictor.fit(train_data)
predictions = predictor.predict(test_data)
```

### L1: Time & Quality Control (1 hour)
**Goal**: Balance speed vs accuracy

```python
# Quick training (minutes)
predictor = TabularPredictor(label="target").fit(
    train_data,
    time_limit=60,  # 1 minute
    presets="medium_quality",
)

# Best accuracy (hours)
predictor = TabularPredictor(label="target").fit(
    train_data,
    time_limit=3600,  # 1 hour
    presets="best_quality",
)

# Presets: 'medium_quality', 'high_quality', 'best_quality'
```

### L2: Feature Engineering & Analysis (2 hours)
**Goal**: Understand and improve models

```python
# Feature importance
importance = predictor.feature_importance(test_data)
print(importance)

# Model leaderboard
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

# Detailed model info
predictor.fit_summary()
```

### L3: Time Series Forecasting (3+ hours)
**Goal**: Forecast future values

```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    target="sales",
    prediction_length=14,  # Forecast 14 periods
).fit(train_data)

predictions = predictor.predict(test_data)
```

---

## Code Examples

### Example 1: Classification with Evaluation
```python
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

# Split data
train, test = train_test_split(df, test_size=0.2)

# Train
predictor = TabularPredictor(
    label="target",
    eval_metric="accuracy",  # or 'f1', 'roc_auc', etc.
).fit(train, time_limit=300)

# Evaluate
print(predictor.evaluate(test))
print(predictor.leaderboard(test))
```

### Example 2: Regression
```python
predictor = TabularPredictor(
    label="price",
    problem_type="regression",
    eval_metric="rmse",
).fit(train_data)

# Predict
predictions = predictor.predict(test_data)
```

### Example 3: Deploy Model
```python
# Save
predictor.save("my_model/")

# Load
predictor = TabularPredictor.load("my_model/")

# Predict on new data
new_predictions = predictor.predict(new_data)
```

### Example 4: Multimodal (Text + Tabular)
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="sentiment").fit(
    train_data,  # DataFrame with text columns + other features
    time_limit=600,
)
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| Polars | Fast data loading | [polars.md](./polars.md) |
| SHAP | Model explanations | [shap.md](../specialized/shap.md) |
| MLflow | Experiment tracking | [mlflow.md](./mlflow.md) |

### Export Options
```python
# Get best model for deployment
best_model = predictor.get_model_best()

# Export to ONNX (if supported)
# Use predictor.save() for full model
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory
```python
# Reduce models trained
predictor.fit(
    train_data,
    excluded_model_types=['NN', 'KNN'],  # Skip memory-heavy
)
```

#### Issue 2: Too Slow
```python
# Quick preset
predictor.fit(train_data, presets="medium_quality", time_limit=60)
```

---

## Resources

- [AutoGluon Docs](https://auto.gluon.ai/)
- [GitHub](https://github.com/autogluon/autogluon)
- [Tutorials](https://auto.gluon.ai/stable/tutorials/index.html)

---

*Part of [Luno-AI Integration Hub](../_index.md) | Classical ML Track*

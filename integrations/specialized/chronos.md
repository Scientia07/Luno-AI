# Chronos Integration

> **Foundation model for zero-shot time series forecasting**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Pre-trained transformer for time series |
| **Why** | Zero-shot forecasting, no training needed |
| **Creator** | Amazon (2024) |
| **Best For** | General forecasting without domain-specific training |

### Why Chronos?

| Approach | Training Needed | Accuracy | Flexibility |
|----------|-----------------|----------|-------------|
| **Chronos** | None | ⭐⭐⭐⭐ | High |
| Prophet | Per-dataset | ⭐⭐⭐ | Medium |
| ARIMA | Per-dataset | ⭐⭐⭐ | Low |
| Custom DL | Extensive | ⭐⭐⭐⭐⭐ | High |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.9+ |
| **GPU** | Optional (CPU works) |
| **RAM** | 4GB+ |

---

## Quick Start (10 min)

```bash
pip install chronos-forecasting
```

```python
import torch
from chronos import ChronosPipeline
import numpy as np
import matplotlib.pyplot as plt

# Load model
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float32
)

# Your time series data
context = torch.tensor([
    10, 12, 15, 14, 18, 20, 22, 25, 24, 28,
    30, 32, 35, 33, 38, 40, 42, 45, 44, 48
])

# Forecast
forecast = pipeline.predict(context, prediction_length=12, num_samples=20)

# Get median and quantiles
median = np.median(forecast.numpy(), axis=0)
low = np.percentile(forecast.numpy(), 10, axis=0)
high = np.percentile(forecast.numpy(), 90, axis=0)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(range(len(context)), context, label="History")
plt.plot(range(len(context), len(context) + len(median)), median, label="Forecast")
plt.fill_between(
    range(len(context), len(context) + len(median)),
    low, high, alpha=0.3, label="80% CI"
)
plt.legend()
plt.savefig("forecast.png")
```

---

## Learning Path

### L0: Basic Forecasting (1 hour)
- [ ] Install Chronos
- [ ] Forecast sample data
- [ ] Visualize predictions
- [ ] Try different horizons

### L1: Real Data (2-3 hours)
- [ ] Load CSV time series
- [ ] Handle missing values
- [ ] Multiple series forecasting
- [ ] Uncertainty quantification

### L2: Advanced (4-6 hours)
- [ ] Compare model sizes
- [ ] Ensemble methods
- [ ] Combine with other models
- [ ] Fine-tune on domain data

---

## Code Examples

### Multiple Time Series

```python
import pandas as pd

# Load data
df = pd.read_csv("sales.csv")
series_list = [
    torch.tensor(df[col].values)
    for col in df.columns if col != "date"
]

# Forecast all at once
forecasts = pipeline.predict(series_list, prediction_length=30)
```

### From Pandas DataFrame

```python
import pandas as pd

df = pd.read_csv("data.csv", parse_dates=["date"])
values = torch.tensor(df["value"].values, dtype=torch.float32)

forecast = pipeline.predict(values, prediction_length=14)
```

### With Confidence Intervals

```python
forecast = pipeline.predict(
    context,
    prediction_length=12,
    num_samples=100  # More samples = better uncertainty estimates
)

# Quantiles
q10 = np.percentile(forecast.numpy(), 10, axis=0)
q50 = np.percentile(forecast.numpy(), 50, axis=0)  # Median
q90 = np.percentile(forecast.numpy(), 90, axis=0)
```

### Model Comparison

```python
models = ["chronos-t5-tiny", "chronos-t5-small", "chronos-t5-base"]

for model_name in models:
    pipeline = ChronosPipeline.from_pretrained(f"amazon/{model_name}")
    forecast = pipeline.predict(context, prediction_length=12)
    print(f"{model_name}: MAE = {calculate_mae(forecast, actual):.2f}")
```

---

## Model Sizes

| Model | Parameters | Speed | Accuracy |
|-------|------------|-------|----------|
| chronos-t5-tiny | 8M | ⚡⚡⚡ | ⭐⭐⭐ |
| chronos-t5-mini | 20M | ⚡⚡⚡ | ⭐⭐⭐ |
| chronos-t5-small | 46M | ⚡⚡ | ⭐⭐⭐⭐ |
| chronos-t5-base | 200M | ⚡ | ⭐⭐⭐⭐⭐ |
| chronos-t5-large | 710M | ⚡ | ⭐⭐⭐⭐⭐ |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Use smaller model, reduce batch |
| NaN in forecast | Check for NaN in input |
| Poor accuracy | Try larger model, more context |
| Slow inference | Use GPU, smaller model |

---

## Resources

- [Chronos Paper](https://arxiv.org/abs/2403.07815)
- [HuggingFace Models](https://huggingface.co/amazon/chronos-t5-small)
- [GitHub](https://github.com/amazon-science/chronos-forecasting)

---

*Part of [Luno-AI](../../README.md) | Specialized Track*

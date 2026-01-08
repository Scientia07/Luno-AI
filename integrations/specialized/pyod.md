# PyOD Anomaly Detection Integration

> **Detect outliers and anomalies in your data**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Python Outlier Detection library |
| **Why** | 50+ algorithms, unified API |
| **Use Cases** | Fraud detection, quality control, intrusion detection |
| **Best For** | Tabular data anomaly detection |

### Algorithm Categories

| Category | Examples |
|----------|----------|
| **Linear** | PCA, OCSVM |
| **Proximity** | KNN, LOF |
| **Probabilistic** | GMM, COPOD |
| **Neural** | AutoEncoder, VAE |
| **Ensemble** | IForest, LODA |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **Data** | Numerical features |

---

## Quick Start (15 min)

```bash
pip install pyod
```

```python
from pyod.models.iforest import IForest
from pyod.utils.data import generate_data
import numpy as np

# Generate sample data
X_train, X_test, y_train, y_test = generate_data(
    n_train=1000,
    n_test=200,
    contamination=0.1  # 10% outliers
)

# Train Isolation Forest
clf = IForest(contamination=0.1)
clf.fit(X_train)

# Predict
y_pred = clf.predict(X_test)  # 0: inlier, 1: outlier
scores = clf.decision_function(X_test)  # Anomaly scores

# Evaluate
from pyod.utils.utility import precision_n_scores
print(f"Precision@n: {precision_n_scores(y_test, scores):.3f}")
```

---

## Learning Path

### L0: Basic Detection (1-2 hours)
- [ ] Install PyOD
- [ ] Try Isolation Forest
- [ ] Understand scores vs labels
- [ ] Evaluate results

### L1: Algorithm Selection (2-3 hours)
- [ ] Compare algorithms
- [ ] Tune contamination
- [ ] Ensemble methods
- [ ] Feature scaling

### L2: Production (4-6 hours)
- [ ] Streaming detection
- [ ] Model persistence
- [ ] Integration
- [ ] Monitoring

---

## Code Examples

### Compare Algorithms

```python
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.copod import COPOD
from sklearn.metrics import roc_auc_score

# Define models
models = {
    "Isolation Forest": IForest(contamination=0.1),
    "LOF": LOF(contamination=0.1),
    "KNN": KNN(contamination=0.1),
    "PCA": PCA(contamination=0.1),
    "COPOD": COPOD(contamination=0.1)
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train)
    scores = model.decision_function(X_test)
    auc = roc_auc_score(y_test, scores)
    results[name] = auc
    print(f"{name}: AUC = {auc:.3f}")

# Best model
best = max(results, key=results.get)
print(f"\nBest: {best} with AUC = {results[best]:.3f}")
```

### AutoEncoder

```python
from pyod.models.auto_encoder import AutoEncoder

# Deep learning anomaly detector
clf = AutoEncoder(
    hidden_neurons=[64, 32, 32, 64],
    hidden_activation='relu',
    output_activation='sigmoid',
    epochs=100,
    batch_size=32,
    contamination=0.1
)

clf.fit(X_train)
predictions = clf.predict(X_test)
```

### Ensemble

```python
from pyod.models.combination import aom, moa, average, maximization
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
import numpy as np

# Train multiple detectors
detectors = [
    IForest(contamination=0.1),
    LOF(contamination=0.1),
    KNN(contamination=0.1)
]

scores_list = []
for det in detectors:
    det.fit(X_train)
    scores_list.append(det.decision_function(X_test))

# Combine scores
scores_array = np.column_stack(scores_list)

# Different combination methods
combined_avg = average(scores_array)
combined_max = maximization(scores_array)
combined_aom = aom(scores_array, n_buckets=5)
combined_moa = moa(scores_array, n_buckets=5)

print(f"Average: AUC = {roc_auc_score(y_test, combined_avg):.3f}")
print(f"Max: AUC = {roc_auc_score(y_test, combined_max):.3f}")
```

### Real Data Example

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest

# Load data
df = pd.read_csv("transactions.csv")

# Prepare features
features = ['amount', 'time_since_last', 'distance_from_home']
X = df[features].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Detect anomalies
clf = IForest(contamination=0.05)  # Expect 5% anomalies
clf.fit(X_scaled)

# Get results
df['anomaly_score'] = clf.decision_function(X_scaled)
df['is_anomaly'] = clf.predict(X_scaled)

# Analyze anomalies
anomalies = df[df['is_anomaly'] == 1]
print(f"Found {len(anomalies)} anomalies")
print(anomalies.describe())
```

### Streaming Detection

```python
from pyod.models.iforest import IForest
import numpy as np

class StreamingAnomalyDetector:
    def __init__(self, window_size=1000, contamination=0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.buffer = []
        self.model = None

    def update(self, x):
        """Add new data point and return anomaly score"""
        self.buffer.append(x)

        # Retrain when buffer is full
        if len(self.buffer) >= self.window_size:
            X = np.array(self.buffer)
            self.model = IForest(contamination=self.contamination)
            self.model.fit(X)
            self.buffer = self.buffer[-self.window_size//2:]  # Keep recent half

        # Score if model exists
        if self.model is not None:
            return self.model.decision_function([x])[0]
        return 0.0

# Usage
detector = StreamingAnomalyDetector()

for data_point in data_stream:
    score = detector.update(data_point)
    if score > threshold:
        print(f"Anomaly detected! Score: {score}")
```

### Save and Load Model

```python
import joblib
from pyod.models.iforest import IForest

# Train
clf = IForest(contamination=0.1)
clf.fit(X_train)

# Save
joblib.dump(clf, 'anomaly_detector.pkl')

# Load
clf_loaded = joblib.load('anomaly_detector.pkl')
predictions = clf_loaded.predict(X_new)
```

---

## Algorithm Selection Guide

| Scenario | Recommended |
|----------|-------------|
| High-dimensional | IForest, PCA |
| Local anomalies | LOF, KNN |
| Large dataset | IForest, COPOD |
| Need interpretability | PCA |
| Non-linear patterns | AutoEncoder |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Too many anomalies | Lower contamination |
| Missing anomalies | Raise contamination, try different algo |
| Slow training | Use IForest or COPOD |
| Poor results | Scale features, try ensemble |

---

## Resources

- [PyOD Documentation](https://pyod.readthedocs.io/)
- [PyOD GitHub](https://github.com/yzhao062/pyod)
- [Algorithm Benchmark](https://pyod.readthedocs.io/en/latest/benchmark.html)
- [Anomaly Detection Survey](https://arxiv.org/abs/1901.03407)

---

*Part of [Luno-AI](../../README.md) | Specialized Track*

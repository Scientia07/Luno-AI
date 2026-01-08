# MLflow Integration

> **Track experiments and manage ML lifecycle**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Open-source ML lifecycle platform |
| **Why** | Experiment tracking, model registry |
| **Components** | Tracking, Projects, Models, Registry |
| **Best For** | Team collaboration, reproducibility |

### MLflow Components

| Component | Purpose |
|-----------|---------|
| **Tracking** | Log parameters, metrics, artifacts |
| **Projects** | Package code for reproducibility |
| **Models** | Standard format for models |
| **Registry** | Versioned model storage |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **Storage** | For artifacts |

---

## Quick Start (15 min)

```bash
pip install mlflow
```

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### Start UI

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Learning Path

### L0: Experiment Tracking (1-2 hours)
- [ ] Install MLflow
- [ ] Log parameters and metrics
- [ ] Use MLflow UI
- [ ] Compare runs

### L1: Model Management (2-3 hours)
- [ ] Log and load models
- [ ] Model registry
- [ ] Model versioning
- [ ] Stage transitions

### L2: Production (4-6 hours)
- [ ] Remote tracking server
- [ ] Model serving
- [ ] CI/CD integration
- [ ] Team workflows

---

## Code Examples

### Experiment Organization

```python
import mlflow

# Set experiment
mlflow.set_experiment("classification-experiments")

# Start run with name
with mlflow.start_run(run_name="random_forest_v1"):
    # Your training code
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_metric("accuracy", 0.95)
```

### Autologging

```python
import mlflow

# Auto-log for sklearn
mlflow.sklearn.autolog()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Automatically logged!

# Auto-log for other frameworks
mlflow.pytorch.autolog()
mlflow.tensorflow.autolog()
mlflow.xgboost.autolog()
```

### Log Artifacts

```python
import mlflow
import matplotlib.pyplot as plt
import json

with mlflow.start_run():
    # Log plot
    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.savefig("plot.png")
    mlflow.log_artifact("plot.png")

    # Log any file
    config = {"learning_rate": 0.01, "epochs": 100}
    with open("config.json", "w") as f:
        json.dump(config, f)
    mlflow.log_artifact("config.json")

    # Log directory
    mlflow.log_artifacts("output_dir", artifact_path="outputs")
```

### Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

# Log and register model
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model", registered_model_name="MyModel")

# Or register existing model
result = mlflow.register_model(
    "runs:/abc123/model",
    "MyModel"
)

# Manage versions
client = MlflowClient()

# Transition to staging
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Production"
)
```

### Load Models

```python
import mlflow

# Load from run
model = mlflow.sklearn.load_model("runs:/abc123/model")

# Load from registry
model = mlflow.sklearn.load_model("models:/MyModel/Production")

# Load specific version
model = mlflow.sklearn.load_model("models:/MyModel/1")

# Make predictions
predictions = model.predict(X_new)
```

### Hyperparameter Tuning

```python
import mlflow
from sklearn.model_selection import GridSearchCV

mlflow.set_experiment("hyperparameter-tuning")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

for n_est in param_grid['n_estimators']:
    for depth in param_grid['max_depth']:
        with mlflow.start_run():
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", depth)

            model = RandomForestClassifier(n_estimators=n_est, max_depth=depth)
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
```

### Remote Tracking Server

```bash
# Start server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://my-bucket/mlflow \
    --host 0.0.0.0 \
    --port 5000
```

```python
# Connect to remote server
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### Model Serving

```bash
# Serve model locally
mlflow models serve -m "models:/MyModel/Production" -p 5001

# Make prediction
curl -X POST http://localhost:5001/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [[1, 2, 3, 4]]}'
```

### Custom Model

```python
import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        probas = self.model.predict_proba(model_input)[:, 1]
        return (probas > self.threshold).astype(int)

# Save custom model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CustomModel(threshold=0.7),
        artifacts={"model": "path/to/model.pkl"}
    )
```

---

## Best Practices

| Practice | Benefit |
|----------|---------|
| Use experiments | Organize related runs |
| Log everything | Reproducibility |
| Use registry | Model versioning |
| Add tags | Easy filtering |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Artifacts not saving | Check storage permissions |
| UI not loading | Verify port, firewall |
| Model not loading | Check dependencies |
| Slow logging | Use batch logging |

---

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
- [Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/)
- [Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)

---

*Part of [Luno-AI](../../README.md) | Classical ML Track*

# XGBoost / LightGBM / CatBoost Integration

> **Category**: Classical ML
> **Difficulty**: Beginner
> **Setup Time**: 2-3 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What They Do
Gradient boosting libraries that build powerful predictive models by combining many weak decision trees. The go-to choice for tabular/structured data.

### Why Use Them
- **Best for Tabular Data**: Outperform deep learning on structured data
- **Fast Training**: Minutes vs hours for neural networks
- **Interpretable**: Feature importance built-in
- **Production Ready**: Battle-tested in industry
- **Competition Winners**: Dominate Kaggle competitions

### Key Capabilities
| Capability | XGBoost | LightGBM | CatBoost |
|------------|---------|----------|----------|
| Speed | Fast | Fastest | Medium |
| Memory | Medium | Low | High |
| Categoricals | Manual encode | Native | Best native |
| GPU Support | Yes | Yes | Yes |
| Missing Values | Native | Native | Native |
| Tuning Needed | More | More | Less |

### When to Use What
| Scenario | Recommendation |
|----------|---------------|
| Default choice | **LightGBM** (fastest, good accuracy) |
| Many categorical features | **CatBoost** (best categorical handling) |
| Need GPU training | **XGBoost** (most mature GPU) |
| Minimal tuning time | **CatBoost** (works well out of box) |
| Competitions | Try all three, ensemble |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None | NVIDIA (optional) |
| RAM | 4 GB | 16 GB |
| Storage | 500 MB | 1 GB |

### Software Dependencies
```bash
pip install xgboost lightgbm catboost

# Optional
pip install scikit-learn pandas
```

### Prior Knowledge
- [x] Python basics
- [x] Pandas DataFrames
- [ ] ML concepts (train/test split, overfitting)

---

## Quick Start (15 minutes)

### 1. Install
```bash
pip install xgboost lightgbm catboost scikit-learn pandas
```

### 2. Basic Example
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train
model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

### 3. Compare All Three
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = {
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name}: {acc:.2%}")
```

---

## Full Setup

### XGBoost
```python
import xgboost as xgb

# DMatrix format (faster)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
}

# Train with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=100,
)

# Predict
predictions = model.predict(dtest)
```

### LightGBM
```python
import lightgbm as lgb

# Dataset format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

predictions = model.predict(X_test)
```

### CatBoost
```python
from catboost import CatBoostClassifier, Pool

# With categorical features
cat_features = ['category_col1', 'category_col2']  # column names or indices

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=50,
    verbose=100,
)

model.fit(train_pool, eval_set=test_pool)
predictions = model.predict_proba(test_pool)[:, 1]
```

---

## Learning Path

### L0: Basic Training (1 hour)
**Goal**: Train your first model

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

### L1: Feature Importance & Tuning (2 hours)
**Goal**: Understand and improve models

```python
import matplotlib.pyplot as plt

# Feature importance
importance = model.feature_importances_
feature_names = X.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")

# Hyperparameter tuning with Optuna
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    }
    model = lgb.LGBMClassifier(**params, verbose=-1)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
```

### L2: Cross-Validation & Ensembling (3 hours)
**Goal**: Robust evaluation and combining models

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.2%} (+/- {scores.std()*2:.2%})")

# Ensemble all three
ensemble = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=100)),
        ('lgb', LGBMClassifier(n_estimators=100, verbose=-1)),
        ('cat', CatBoostClassifier(n_estimators=100, verbose=0)),
    ],
    voting='soft'  # Use probabilities
)
ensemble.fit(X_train, y_train)
print(f"Ensemble Accuracy: {ensemble.score(X_test, y_test):.2%}")
```

### L3: GPU Training & Production (4+ hours)
**Goal**: Scale up and deploy

```python
# XGBoost GPU
model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0,
    n_estimators=1000,
)

# LightGBM GPU
model = lgb.LGBMClassifier(
    device='gpu',
    n_estimators=1000,
)

# CatBoost GPU
model = CatBoostClassifier(
    task_type='GPU',
    devices='0',
    iterations=1000,
)

# Save model
model.save_model('model.cbm')  # CatBoost
# or
import joblib
joblib.dump(model, 'model.joblib')

# Load model
model = CatBoostClassifier()
model.load_model('model.cbm')
```

---

## Code Examples

### Example 1: Regression
```python
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = LGBMRegressor(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
```

### Example 2: Multi-class Classification
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    loss_function='MultiClass',
    classes_count=10,
    verbose=0,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Example 3: Handling Imbalanced Data
```python
# Calculate scale_pos_weight
scale = (y_train == 0).sum() / (y_train == 1).sum()

# XGBoost
model = xgb.XGBClassifier(scale_pos_weight=scale)

# LightGBM
model = lgb.LGBMClassifier(scale_pos_weight=scale)

# CatBoost
model = CatBoostClassifier(class_weights=[1, scale])
```

### Example 4: SHAP Explanations
```python
import shap

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Single prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| Polars | Fast data loading | [polars.md](./polars.md) |
| AutoGluon | AutoML wrapper | [autogluon.md](./autogluon.md) |
| MLflow | Experiment tracking | [mlflow.md](./mlflow.md) |
| SHAP | Model explanations | [shap.md](../specialized/shap.md) |
| Optuna | Hyperparameter tuning | External |

### Model Export
```python
# ONNX export (for deployment)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Overfitting
**Symptoms**: High train accuracy, low test accuracy
**Solution**:
```python
model = lgb.LGBMClassifier(
    max_depth=5,           # Limit tree depth
    num_leaves=20,         # Fewer leaves
    min_child_samples=20,  # Min samples per leaf
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=0.1,        # L2 regularization
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8,  # Column sampling
)
```

#### Issue 2: Slow Training
**Symptoms**: Training takes too long
**Solution**:
```python
# Use histogram-based (default in LightGBM)
# Reduce data size for tuning
# Use GPU
model = lgb.LGBMClassifier(device='gpu')

# Or subsample
model.fit(X_train.sample(frac=0.1), y_train.sample(frac=0.1))
```

#### Issue 3: Categorical Errors
**Symptoms**: Error with string columns
**Solution**:
```python
# CatBoost (best option)
model = CatBoostClassifier(cat_features=['col1', 'col2'])

# LightGBM
df['category'] = df['category'].astype('category')

# XGBoost (needs encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])
```

---

## Resources

### Official
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [CatBoost Docs](https://catboost.ai/docs/)

### Tutorials
- [Kaggle XGBoost](https://www.kaggle.com/learn/xgboost)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

### Community
- [Kaggle Discussions](https://www.kaggle.com/discussions)

---

*Part of [Luno-AI Integration Hub](../_index.md) | Classical ML Track*

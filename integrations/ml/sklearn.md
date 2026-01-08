# scikit-learn Integration

> **Machine learning fundamentals in Python**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Classic ML algorithms library |
| **Why** | Simple API, robust, well-documented |
| **Algorithms** | Classification, regression, clustering |
| **Best For** | Tabular data, traditional ML |

### When to Use sklearn

| Use sklearn | Use Deep Learning |
|-------------|------------------|
| Tabular data | Images, text, audio |
| Small datasets | Large datasets |
| Interpretability needed | Performance critical |
| Quick prototyping | Complex patterns |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **Data** | Numerical features |

---

## Quick Start (15 min)

```bash
pip install scikit-learn pandas
```

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

---

## Learning Path

### L0: Basics (2-3 hours)
- [ ] Install sklearn
- [ ] Train first model
- [ ] Evaluate performance
- [ ] Cross-validation

### L1: Intermediate (4-6 hours)
- [ ] Preprocessing pipelines
- [ ] Hyperparameter tuning
- [ ] Feature selection
- [ ] Model comparison

### L2: Advanced (1-2 days)
- [ ] Custom transformers
- [ ] Ensemble methods
- [ ] Model persistence
- [ ] Production pipelines

---

## Code Examples

### Full Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Define columns
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'city']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Cross-validate
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Train final model
pipeline.fit(X_train, y_train)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
```

### Multiple Models Comparison

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5)
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Best model
best = max(results, key=results.get)
print(f"\nBest: {best} with {results[best]:.3f}")
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Filter method
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Wrapper method (RFE)
model = RandomForestClassifier(n_estimators=100)
rfe = RFE(model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
rfe_features = X.columns[rfe.get_support()]
print(f"RFE features: {list(rfe_features)}")

# Embedded method (feature importance)
model.fit(X, y)
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(10))
```

### Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column])
        X['year'] = X[self.date_column].dt.year
        X['month'] = X[self.date_column].dt.month
        X['day_of_week'] = X[self.date_column].dt.dayofweek
        X = X.drop(self.date_column, axis=1)
        return X

# Use in pipeline
pipeline = Pipeline([
    ('date_features', DateFeatureExtractor('date')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
```

### Model Persistence

```python
import joblib

# Save model
joblib.dump(pipeline, 'model.pkl')

# Load model
loaded_model = joblib.load('model.pkl')
predictions = loaded_model.predict(X_new)
```

### Clustering

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

print(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}")

# Find optimal k
scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, clusters))

optimal_k = range(2, 11)[scores.index(max(scores))]
print(f"Optimal k: {optimal_k}")
```

---

## Common Algorithms

| Task | Algorithms |
|------|------------|
| Classification | LogisticRegression, RandomForest, SVM, GradientBoosting |
| Regression | LinearRegression, RandomForest, Ridge, Lasso |
| Clustering | KMeans, DBSCAN, AgglomerativeClustering |
| Dimensionality | PCA, t-SNE, UMAP |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Overfitting | Add regularization, cross-validation |
| Underfitting | More features, complex model |
| Slow training | Reduce data, use faster algorithm |
| Memory error | Use incremental learning |

---

## Resources

- [scikit-learn Docs](https://scikit-learn.org/stable/)
- [User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Examples](https://scikit-learn.org/stable/auto_examples/)
- [Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)

---

*Part of [Luno-AI](../../README.md) | Classical ML Track*

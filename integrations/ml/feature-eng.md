# Feature Engineering Integration

> **Transform raw data into predictive features**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Create meaningful features from data |
| **Why** | Better features = better models |
| **Tools** | sklearn, featuretools, pandas |
| **Best For** | Tabular data, time series |

### Impact

| Approach | Improvement |
|----------|-------------|
| Raw data only | Baseline |
| Basic features | 10-30% |
| Domain features | 30-50% |
| Automated features | 20-40% |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **Domain** | Knowledge of your data |

---

## Quick Start (20 min)

```bash
pip install scikit-learn pandas category_encoders featuretools
```

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define feature types
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'occupation']

# Create transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Transform
X_transformed = preprocessor.fit_transform(df)
```

---

## Learning Path

### L0: Basic Transformations (1-2 hours)
- [ ] Scaling and normalization
- [ ] One-hot encoding
- [ ] Missing value handling
- [ ] Pipeline creation

### L1: Advanced Features (3-4 hours)
- [ ] Feature interactions
- [ ] Polynomial features
- [ ] Binning strategies
- [ ] Target encoding

### L2: Automated & Domain (6-8 hours)
- [ ] Automated feature generation
- [ ] Time series features
- [ ] Text/embedding features
- [ ] Domain-specific features

---

## Code Examples

### Numerical Transformations

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max (0-1 range)
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

# Robust (handles outliers)
robust = RobustScaler()
X_robust = robust.fit_transform(X)

# Power transform (make normal)
power = PowerTransformer(method='yeo-johnson')
X_power = power.fit_transform(X)

# Quantile transform (uniform/normal distribution)
quantile = QuantileTransformer(output_distribution='normal')
X_quantile = quantile.fit_transform(X)
```

### Categorical Encoding

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import category_encoders as ce

# One-hot (nominal)
ohe = OneHotEncoder(sparse=False, drop='first')
X_ohe = ohe.fit_transform(df[['color']])

# Ordinal (ordered categories)
ordinal = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_ordinal = ordinal.fit_transform(df[['priority']])

# Target encoding (high cardinality)
target_encoder = ce.TargetEncoder(cols=['city'])
X_target = target_encoder.fit_transform(df['city'], df['target'])

# Binary encoding (high cardinality)
binary_encoder = ce.BinaryEncoder(cols=['category'])
X_binary = binary_encoder.fit_transform(df)

# Leave-one-out encoding
loo_encoder = ce.LeaveOneOutEncoder(cols=['category'])
X_loo = loo_encoder.fit_transform(df['category'], df['target'])
```

### Missing Values

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_knn = knn_imputer.fit_transform(X)

# Create missing indicator
from sklearn.impute import MissingIndicator
indicator = MissingIndicator()
missing_mask = indicator.fit_transform(X)

# Combined
df['value_imputed'] = imputer.fit_transform(df[['value']])
df['value_was_missing'] = df['value'].isna().astype(int)
```

### Feature Interactions

```python
from sklearn.preprocessing import PolynomialFeatures

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)

# Manual interactions
df['area'] = df['length'] * df['width']
df['density'] = df['mass'] / df['volume']
df['age_income'] = df['age'] * df['income']
df['income_per_member'] = df['income'] / (df['family_size'] + 1)
```

### Binning

```python
from sklearn.preprocessing import KBinsDiscretizer

# Equal-width binning
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_binned = discretizer.fit_transform(X)

# Quantile binning (equal frequency)
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_binned = discretizer.fit_transform(X)

# Custom bins
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                          labels=['child', 'young', 'middle', 'senior', 'elderly'])
```

### Time Series Features

```python
import pandas as pd

def create_time_features(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Basic date parts
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter

    # Cyclic encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Boolean features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    return df

def create_lag_features(df, column, lags=[1, 7, 14, 30]):
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

def create_rolling_features(df, column, windows=[7, 14, 30]):
    for window in windows:
        df[f'{column}_roll_mean_{window}'] = df[column].rolling(window).mean()
        df[f'{column}_roll_std_{window}'] = df[column].rolling(window).std()
        df[f'{column}_roll_min_{window}'] = df[column].rolling(window).min()
        df[f'{column}_roll_max_{window}'] = df[column].rolling(window).max()
    return df
```

### Automated Features (Featuretools)

```python
import featuretools as ft

# Create entity set
es = ft.EntitySet(id='customers')

# Add dataframes
es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers_df,
    index='customer_id'
)

es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions_df,
    index='transaction_id',
    time_index='date'
)

# Add relationship
es = es.add_relationship(
    'customers', 'customer_id',
    'transactions', 'customer_id'
)

# Generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2,
    agg_primitives=['sum', 'mean', 'count', 'max', 'min'],
    trans_primitives=['year', 'month', 'day']
)

print(f"Generated {len(feature_defs)} features")
```

### Text Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df['text'])

# Basic text stats
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(
    lambda x: np.mean([len(w) for w in x.split()])
)
df['uppercase_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x)
)
```

---

## Feature Pipeline Template

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

def create_feature_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Data leakage | Fit only on training data |
| High cardinality | Use target encoding |
| Many missing | Create indicator features |
| Curse of dimensionality | Feature selection |

---

## Resources

- [sklearn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Featuretools](https://www.featuretools.com/)
- [Category Encoders](https://contrib.scikit-learn.org/category_encoders/)
- [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

---

*Part of [Luno-AI](../../README.md) | Classical ML Track*

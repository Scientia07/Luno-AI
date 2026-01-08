# Time Series Forecasting Integration

> **Predict future values from historical data**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Forecasting sequential data |
| **Why** | Demand planning, anomaly detection |
| **Tools** | statsmodels, Prophet, NeuralProphet, sklearn |
| **Best For** | Sales, weather, stocks, capacity |

### Methods Overview

| Method | Best For | Complexity |
|--------|----------|------------|
| **ARIMA** | Single series | Medium |
| **Prophet** | Business data | Low |
| **XGBoost** | Multiple features | Medium |
| **LSTM** | Complex patterns | High |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **Data** | Time-indexed observations |

---

## Quick Start (20 min)

```bash
pip install statsmodels prophet scikit-learn pandas
```

```python
import pandas as pd
from prophet import Prophet

# Load data
df = pd.read_csv("sales.csv")
df = df.rename(columns={'date': 'ds', 'sales': 'y'})

# Fit model
model = Prophet()
model.fit(df)

# Make forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
fig = model.plot(forecast)
```

---

## Learning Path

### L0: Basic Forecasting (2-3 hours)
- [ ] Load and visualize time series
- [ ] Simple moving average
- [ ] Prophet basics
- [ ] Evaluate forecasts

### L1: Advanced Methods (4-6 hours)
- [ ] ARIMA/SARIMA
- [ ] Feature engineering
- [ ] ML approaches
- [ ] Seasonality handling

### L2: Production (1-2 days)
- [ ] Multiple time series
- [ ] Hierarchical forecasting
- [ ] Automated pipelines
- [ ] Uncertainty quantification

---

## Code Examples

### Data Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load and prepare
df = pd.read_csv("data.csv", parse_dates=['date'], index_col='date')

# Basic stats
print(df.describe())

# Plot time series
plt.figure(figsize=(12, 6))
plt.plot(df['value'])
plt.title('Time Series')
plt.show()

# Decompose
decomposition = seasonal_decompose(df['value'], period=12)
decomposition.plot()
plt.show()

# Check stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['value'].dropna())
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
```

### Prophet

```python
from prophet import Prophet
import pandas as pd

# Prepare data
df = df.reset_index()
df = df.rename(columns={'date': 'ds', 'value': 'y'})

# Create and fit model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

# Add holidays
model.add_country_holidays(country_name='US')

# Add regressor
model.add_regressor('temperature')

model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=90)
future['temperature'] = predict_temperature(future['ds'])  # External data
forecast = model.predict(future)

# Components
fig = model.plot_components(forecast)
```

### ARIMA/SARIMA

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# Auto ARIMA (find best parameters)
auto_model = pm.auto_arima(
    df['value'],
    seasonal=True,
    m=12,  # Seasonal period
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
print(auto_model.summary())

# Manual SARIMA
model = SARIMAX(
    df['value'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
results = model.fit()

# Forecast
forecast = results.get_forecast(steps=30)
mean = forecast.predicted_mean
ci = forecast.conf_int()

# Plot
plt.plot(df['value'], label='Actual')
plt.plot(mean.index, mean, label='Forecast')
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.3)
plt.legend()
plt.show()
```

### ML Approach (XGBoost)

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

def create_features(df, target_col, lags=[1, 7, 14, 30]):
    df = df.copy()

    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Rolling features
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window).std()

    # Date features
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear

    return df.dropna()

# Prepare data
df = create_features(df, 'value')
feature_cols = [c for c in df.columns if c != 'value']
X = df[feature_cols]
y = df['value']

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    scores.append(mape)

print(f"Average MAPE: {np.mean(scores):.2f}%")
```

### Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {'mae': mae, 'rmse': rmse, 'mape': mape}

metrics = evaluate_forecast(y_test, y_pred)
```

### Multi-Step Forecast

```python
def multi_step_forecast(model, X_last, n_steps, feature_cols):
    """Recursive multi-step forecasting"""
    predictions = []
    current_features = X_last.copy()

    for _ in range(n_steps):
        pred = model.predict(current_features.values.reshape(1, -1))[0]
        predictions.append(pred)

        # Update features for next step
        current_features = update_features(current_features, pred)

    return predictions

def update_features(features, new_value):
    """Shift lag features and add new prediction"""
    features = features.copy()
    for i in range(30, 1, -1):
        if f'lag_{i}' in features:
            features[f'lag_{i}'] = features.get(f'lag_{i-1}', new_value)
    features['lag_1'] = new_value
    return features
```

### NeuralProphet

```python
from neuralprophet import NeuralProphet

# More powerful Prophet with neural networks
model = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    n_lags=14,
    n_forecasts=7,
    learning_rate=0.1
)

# Add autoregression
model = model.add_lagged_regressor('temperature')

# Fit
metrics = model.fit(df, freq='D')

# Predict
forecast = model.predict(df)
model.plot(forecast)
```

### Backtesting

```python
def backtest(model_class, data, train_size, test_size, step_size):
    """Walk-forward validation"""
    results = []

    for i in range(0, len(data) - train_size - test_size, step_size):
        train = data.iloc[i:i+train_size]
        test = data.iloc[i+train_size:i+train_size+test_size]

        model = model_class()
        model.fit(train)
        pred = model.predict(len(test))

        error = mean_absolute_error(test['value'], pred)
        results.append({
            'train_end': train.index[-1],
            'test_end': test.index[-1],
            'mae': error
        })

    return pd.DataFrame(results)

# Run backtest
results = backtest(MyModel, df, train_size=365, test_size=30, step_size=30)
print(f"Average MAE: {results['mae'].mean():.2f}")
```

---

## Method Selection

| Data Characteristics | Recommended |
|---------------------|-------------|
| Strong trend | Prophet, ARIMA with differencing |
| Multiple seasonality | Prophet, TBATS |
| External regressors | Prophet, XGBoost |
| Short series | Moving average, ETS |
| Many series | XGBoost, LightGBM |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Non-stationary | Differencing, detrending |
| Poor forecast | Add more features, try different model |
| Overfitting | Cross-validation, regularization |
| Missing data | Interpolation, imputation |

---

## Resources

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [statsmodels Time Series](https://www.statsmodels.org/stable/tsa.html)
- [NeuralProphet](https://neuralprophet.com/)
- [Time Series Book](https://otexts.com/fpp3/)

---

*Part of [Luno-AI](../../README.md) | Classical ML Track*

# Polars Data Processing Integration

> **Lightning-fast DataFrame library**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Rust-powered DataFrame library |
| **Why** | 10-100x faster than pandas |
| **Memory** | Efficient, lazy evaluation |
| **Best For** | Large datasets, ETL, feature engineering |

### Polars vs Pandas

| Feature | Polars | Pandas |
|---------|--------|--------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Memory | Efficient | High |
| Parallel | Native | Limited |
| API | Modern | Mature |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **RAM** | Depends on data |

---

## Quick Start (15 min)

```bash
pip install polars
```

```python
import polars as pl

# Read data
df = pl.read_csv("data.csv")

# Basic operations
result = (
    df
    .filter(pl.col("age") > 25)
    .group_by("category")
    .agg([
        pl.col("value").mean().alias("avg_value"),
        pl.col("value").sum().alias("total_value"),
        pl.len().alias("count")
    ])
    .sort("avg_value", descending=True)
)

print(result)
```

---

## Learning Path

### L0: Basics (1-2 hours)
- [ ] Install Polars
- [ ] Read/write data
- [ ] Select, filter, sort
- [ ] Group by, aggregate

### L1: Intermediate (2-3 hours)
- [ ] Lazy evaluation
- [ ] Joins
- [ ] Window functions
- [ ] Complex expressions

### L2: Advanced (4-6 hours)
- [ ] Streaming large files
- [ ] Custom functions
- [ ] Interop with ML libraries
- [ ] Production pipelines

---

## Code Examples

### Basic Operations

```python
import polars as pl

df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, 30, 35, 28],
    "city": ["NYC", "LA", "NYC", "LA"],
    "salary": [50000, 60000, 75000, 55000]
})

# Select columns
df.select(["name", "age"])

# Filter
df.filter(pl.col("age") > 28)

# Add column
df.with_columns([
    (pl.col("salary") * 1.1).alias("new_salary")
])

# Sort
df.sort("salary", descending=True)

# Multiple operations
result = (
    df
    .filter(pl.col("city") == "NYC")
    .select(["name", "salary"])
    .sort("salary", descending=True)
)
```

### Lazy Evaluation

```python
# Lazy mode - builds query plan, executes on .collect()
lazy_df = pl.scan_csv("large_file.csv")

result = (
    lazy_df
    .filter(pl.col("value") > 100)
    .group_by("category")
    .agg(pl.col("value").mean())
    .sort("value", descending=True)
    .limit(10)
    .collect()  # Execute query
)

# Check query plan
print(lazy_df.explain())
```

### Group By and Aggregations

```python
# Multiple aggregations
result = df.group_by("city").agg([
    pl.col("salary").mean().alias("avg_salary"),
    pl.col("salary").median().alias("median_salary"),
    pl.col("salary").min().alias("min_salary"),
    pl.col("salary").max().alias("max_salary"),
    pl.col("salary").std().alias("std_salary"),
    pl.len().alias("count")
])

# Conditional aggregation
result = df.group_by("city").agg([
    pl.col("salary").filter(pl.col("age") > 30).mean().alias("avg_salary_senior")
])
```

### Window Functions

```python
# Ranking
df.with_columns([
    pl.col("salary").rank().over("city").alias("salary_rank"),
    pl.col("salary").mean().over("city").alias("city_avg")
])

# Running calculations
df.sort("date").with_columns([
    pl.col("value").rolling_mean(window_size=7).alias("rolling_avg"),
    pl.col("value").cumsum().alias("cumulative_sum"),
    pl.col("value").pct_change().alias("pct_change")
])
```

### Joins

```python
# Create tables
orders = pl.DataFrame({
    "order_id": [1, 2, 3],
    "customer_id": [101, 102, 101],
    "amount": [100, 200, 150]
})

customers = pl.DataFrame({
    "customer_id": [101, 102, 103],
    "name": ["Alice", "Bob", "Charlie"]
})

# Join
result = orders.join(
    customers,
    on="customer_id",
    how="left"
)

# Multiple join keys
result = df1.join(
    df2,
    on=["key1", "key2"],
    how="inner"
)
```

### Feature Engineering

```python
def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # Date features
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("day_of_week"),

        # Lag features
        pl.col("value").shift(1).alias("value_lag_1"),
        pl.col("value").shift(7).alias("value_lag_7"),

        # Rolling features
        pl.col("value").rolling_mean(window_size=7).alias("value_ma_7"),
        pl.col("value").rolling_std(window_size=7).alias("value_std_7"),

        # Ratios
        (pl.col("value") / pl.col("value").shift(1)).alias("value_ratio"),

        # Categorical encoding
        pl.col("category").cast(pl.Categorical).to_physical().alias("category_encoded")
    ])
```

### Converting to/from Pandas

```python
import pandas as pd

# Polars to Pandas
pandas_df = polars_df.to_pandas()

# Pandas to Polars
polars_df = pl.from_pandas(pandas_df)

# For ML (NumPy)
X = df.select(feature_columns).to_numpy()
y = df.select("target").to_numpy().ravel()
```

### Streaming Large Files

```python
# Process large files in chunks
lazy_df = pl.scan_csv("huge_file.csv")

# Streaming processing
result = (
    lazy_df
    .filter(pl.col("value") > 0)
    .group_by("category")
    .agg(pl.col("value").sum())
    .collect(streaming=True)  # Stream through data
)

# Write large results
(
    pl.scan_csv("input.csv")
    .filter(pl.col("status") == "active")
    .collect(streaming=True)
    .write_parquet("output.parquet")
)
```

### Complex Expressions

```python
# When/Then/Otherwise
df.with_columns([
    pl.when(pl.col("score") >= 90)
      .then(pl.lit("A"))
      .when(pl.col("score") >= 80)
      .then(pl.lit("B"))
      .when(pl.col("score") >= 70)
      .then(pl.lit("C"))
      .otherwise(pl.lit("F"))
      .alias("grade")
])

# Apply custom function
df.with_columns([
    pl.col("text").map_elements(lambda x: len(x.split())).alias("word_count")
])

# String operations
df.with_columns([
    pl.col("name").str.to_lowercase().alias("name_lower"),
    pl.col("name").str.split(" ").list.first().alias("first_name")
])
```

---

## Performance Tips

| Tip | Benefit |
|-----|---------|
| Use lazy mode | Query optimization |
| Avoid `apply` | Use native expressions |
| Use `streaming=True` | Low memory |
| Select needed columns | Less I/O |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Memory error | Use streaming, select columns |
| Slow joins | Check key types match |
| Type errors | Cast columns explicitly |
| Missing values | Use `.fill_null()` |

---

## Resources

- [Polars User Guide](https://pola-rs.github.io/polars-book/)
- [Polars API Reference](https://pola-rs.github.io/polars/py-polars/html/reference/)
- [Polars GitHub](https://github.com/pola-rs/polars)
- [Migration from Pandas](https://pola-rs.github.io/polars-book/user-guide/migration/pandas/)

---

*Part of [Luno-AI](../../README.md) | Classical ML Track*

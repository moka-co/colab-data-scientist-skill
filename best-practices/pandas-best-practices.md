# Pandas Best Practices Guide

Official Documentation: [pandas documentation (https://pandas.pydata.org/docs/)](https://pandas.pydata.org/docs/)

This reference guide outlines core best practices and patterns for agents working with **Pandas** in data analysis, cleaning, preprocessing, and feature engineering workflows.

---

## 1. Schema Contracts & Data Ingestion
- **Explicit Schema Contracts**: Always define a schema contract by explicitly selecting required columns (`usecols`) and specifying data types (`dtype`) during data loading (`pd.read_csv`, `pd.read_parquet`).
- **Post-Initialization Explicit Column Selection**: After DataFrame initialization or ingestion, explicitly select the columns to use, even if all columns are selected. This creates a clear, visible contract between the code and the user/collaborators about what is expected in the downstream data schema.
- **Downstream Consistency**: Specifying schema contracts upfront and explicitly slicing columns prevents silent type coercion, guards against unexpected upstream schema mutations, reduces memory overhead, and catches data drift early.
- **Example**:
  ```python
  schema = {
      "user_id": "int64",
      "transaction_amount": "float32",
      "category": "category",
      "status": "string",
  }
  raw_df = pd.read_csv(
      "transactions.csv",
      usecols=list(schema.keys()),
      dtype=schema,
      parse_dates=["transaction_date"]
  )

  # Explicit column contract after loading (even if using all intended columns)
  expected_features = ["user_id", "transaction_amount", "category", "status", "transaction_date"]
  df = raw_df[expected_features]
  ```

---

## 2. Descriptive Variable Names & Clean Code Conventions
- **Naming Conventions**: Use descriptive, intention-revealing variable names adhering to standard Python naming conventions (`snake_case`).
- **Avoid Ambiguity**: Avoid generic variable names like `df`, `df1`, `temp`, `data`, or single-letter names. Instead, prefer domain-clear names like `raw_transactions_df`, `cleaned_orders_df`, `monthly_summary_df`.
- **Documentation**: Document transformation workflows, assumptions, and non-obvious business logic with clear docstrings and inline comments.

---

## 3. Avoid Chained Indexing (Use `.loc` and `.iloc`)
- **Problem**: Chained indexing such as `df['col'][row]` or `df[df['col'] > 0]['other_col'] = value` triggers `SettingWithCopyWarning` and causes unpredictable behavior (modifying a temporary copy instead of the underlying dataframe).
- **Solution**: Always use `.loc[...]` (label-based) or `.iloc[...]` (position-based) in a single statement.
- **Example**:
  ```python
  # BAD: Chained indexing
  df[df["status"] == "pending"]["flag"] = 1

  # GOOD: Single-statement label indexing
  df.loc[df["status"] == "pending", "flag"] = 1
  ```

---

## 4. Re-assignment Over `inplace=True`
- **Always Opt for Re-assignment**: Always opt to re-assign operation outputs (e.g., `df = df.operation(...)` or `cleaned_df = df.operation(...)`) for significantly better readability and predictability.
- **No Performance Advantage with `inplace=True`**: In Pandas, both `inplace=True` and re-assignment perform a data copy under the hood with very few exceptions. `inplace=True` does not provide performance or memory advantages; instead, it prevents method chaining, complicates debugging, and mutates state unexpectedly.
- **Example**:
  ```python
  # BAD: inplace mutation
  df.drop(columns=["temp_id"], inplace=True)
  df.fillna(0, inplace=True)

  # GOOD: Clear re-assignment
  cleaned_df = df.drop(columns=["temp_id"])
  imputed_df = cleaned_df.fillna(0)

  # ALSO GOOD: Re-assignment to the same variable
  df = df.drop(columns=["temp_id"])
  df = df.fillna(0)
  ```

---

## 5. Use Vectorized Operations (Avoid Loops & `iterrows`)
- **Avoid `iterrows()`**: `iterrows()` serializes rows into Pandas Series objects, creating massive Python overhead. It is notoriously slow and inefficient on non-trivial datasets.
- **Prefer Vectorization**: Leverage NumPy/Pandas C/Cython-optimized vectorized methods, broadcasting, and boolean masks.
- **Hierarchy of Speed**:
  1. **Vectorized built-ins / ufuncs / expressions** (fastest, SIMD/Cython).
  2. **List comprehensions / zip** on underlying arrays or series (`[func(a, b) for a, b in zip(df['a'], df['b'])]`).
  3. **`.apply()`** with vectorized or lightweight callables.
  4. **`.itertuples()`** (if row iteration is strictly unavoidable, much faster than `iterrows()`).
  5. **`.iterrows()`** (avoid entirely).
- **Example**:
  ```python
  # BAD: Iterating over rows
  for idx, row in df.iterrows():
      df.loc[idx, "total"] = row["price"] * row["quantity"]

  # GOOD: Vectorized column operation
  df["total"] = df["price"] * df["quantity"]
  ```

---

## 6. Appropriate Data Types & Memory Optimization
- **Numeric Precision**: Downcast large numeric types when appropriate (e.g., `float64` $\rightarrow$ `float32`, `int64` $\rightarrow$ `int32`/`int16`/`int8`) to save memory.
- **Categoricals**: Convert low-cardinality string/object columns to `'category'` dtype. This significantly reduces RAM consumption and speeds up groupby operations.
- **String & Boolean Types**: Prefer the PyArrow-backed or nullable dtypes (`'string'`, `'boolean'`) over standard object columns for robust null handling.
- **Example**:
  ```python
  # Memory profiling and casting
  df["region"] = df["region"].astype("category")
  df["quantity"] = pd.to_numeric(df["quantity"], downcast="integer")
  ```

---

## 7. Robust Missing Data Handling
- **Audit Missingness**: Detect and quantify null values across features before modeling (`df.isna().sum()`, `df.isna().mean()`).
- **Explicit Handling**: Never leave missing data unaddressed or assume data is complete.
  - Impute missing values with domain-informed strategies (median, mode, forward/backward fill for time series, KNN/iterative imputation).
  - Use indicators (`df['col_was_missing'] = df['col'].isna().astype(int)`) when missingness itself provides predictive signal.
  - Drop records only with explicit justification and threshold checks (`df.dropna(subset=['critical_column'])`).

---

## 8. Aggregations with `groupby`
- **Optimized Aggregations**: Use `.groupby()` paired with native aggregation methods (`.agg()`, `.sum()`, `.mean()`, `.transform()`) rather than iterating or manually splitting data.
- **Named Aggregations**: Use named aggregation for readable, self-documenting outputs.
- **Example**:
  ```python
  aggregated_df = (
      orders_df.groupby("customer_id")
      .agg(
          total_spend=("order_amount", "sum"),
          avg_order_value=("order_amount", "mean"),
          order_count=("order_id", "count"),
      )
      .reset_index()
  )
  ```

---

## 9. Explicit Merge Parameters (`how` and `on`)
- **Explicit Parameter Specification**: Always explicitly specify the `how` and `on` (or `left_on` / `right_on`) parameters when calling `pd.merge()` or `df.merge()`, even if the defaults (such as `how="inner"`) would produce the exact same outcome.
- **Readability & Intent**: Explicit parameters make join cardinality and intent immediately clear to other engineers and review agents, preventing accidental join errors caused by silent key inference.
- **Example**:
  ```python
  # BAD: Relying on default how='inner' and inferred overlapping column keys
  merged_df = orders_df.merge(customers_df)

  # GOOD: Explicit join type and keys stated directly
  merged_df = orders_df.merge(
      customers_df,
      on="customer_id",
      how="inner",
      validate="many_to_one"
  )
  ```

---

## 10. Profiling Code & Bottleneck Identification
- **Profile Early**: Profile data pipelines and transformation code using tools like `cProfile`, `%timeit`, `%prun`, or line profilers (`line_profiler`) to identify memory/CPU bottlenecks.
- **Memory Auditing**: Check memory usage before and after major operations:
  ```python
  df.info(memory_usage="deep")
  print(f"Total Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
  ```
- **Target Bottlenecks**: Focus optimization on identified hotspots (e.g., swapping slow `.apply` calls for vectorized routines or optimizing merge keys).

---

## 11. Summary Checklist for Subagents

When drafting Pandas code, verify that:
- [ ] Columns and types are explicitly declared during loading (schema contract).
- [ ] Columns to use are explicitly selected right after DataFrame initialization/loading.
- [ ] Variables follow `snake_case` with descriptive names.
- [ ] No chained indexing is used (always single `.loc` or `.iloc`).
- [ ] Re-assignment is always used instead of `inplace=True`.
- [ ] Vectorized operations are preferred over loops and `iterrows`.
- [ ] Columns with repeated strings are converted to `category`.
- [ ] Missing values are explicitly checked and resolved.
- [ ] Groupby operations use vectorized `.agg()` with named aggregations.
- [ ] Merge calls explicitly declare `how` and `on` parameters.
- [ ] Performance and memory are verified with memory auditing / profiling.

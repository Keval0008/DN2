# Data Reconciliation Tool Backend

A comprehensive Python backend for reconciling data between original and transformed datasets using pandas. The tool provides intelligent column matching, data aggregation, and anomaly detection capabilities.

## Features

### 🔗 Column Matching
- **Exact Name Matching**: First checks for columns with identical names and ≥90% value overlap
- **Fuzzy Value Matching**: Uses Jaccard similarity to match columns based on value sets
- **Multiple Match Handling**: Identifies and allows user resolution of ambiguous matches
- **No Match Detection**: Flags columns that cannot be matched

### 📊 Data Aggregation
- **Flexible Grouping**: Groups data by matched categorical columns
- **Multiple Aggregation Functions**: Supports sum, mean, count, and other pandas aggregation functions
- **Data Validation**: Ensures numerical columns are properly formatted
- **Missing Data Handling**: Gracefully handles missing values and columns

### 🚨 Anomaly Detection
- **Threshold-Based Detection**: Flags differences exceeding user-defined thresholds
- **Absolute vs Percentage Thresholds**: Supports both fixed amount and percentage-based thresholds
- **Missing Data Anomalies**: Detects cases where data exists in one dataset but not the other
- **Comprehensive Reporting**: Provides detailed anomaly reports with filtering capabilities

## Project Structure

```
.
├── column_matcher.py          # Column matching logic
├── data_aggregator.py         # Data grouping and aggregation
├── anomaly_detector.py        # Anomaly detection and reporting
├── reconciliation_engine.py   # Main orchestrator class
├── example_usage.py          # Comprehensive usage examples
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Import the main engine:
```python
from reconciliation_engine import DataReconciliationEngine
```

## Quick Start

```python
import pandas as pd
from reconciliation_engine import DataReconciliationEngine

# Load your datasets
original_df = pd.read_csv('original_data.csv')
transformed_df = pd.read_csv('transformed_data.csv')

# Initialize the engine
engine = DataReconciliationEngine(original_df, transformed_df)

# Run full reconciliation
results = engine.run_full_reconciliation(
    categorical_columns=['region', 'product_category'],  # Columns to match and group by
    numerical_columns=['sales', 'quantity'],             # Columns to aggregate and compare
    threshold=10.0                                       # Anomaly detection threshold
)

if results['success']:
    print(f"Found {results['anomaly_results']['summary']['total_anomalies']} anomalies")
    
    # Get detailed anomaly report
    anomalies = engine.get_anomaly_report({'only_anomalies': True})
    print(anomalies)
    
    # Export results
    engine.export_results('./results')
```

## API Reference

### DataReconciliationEngine

Main class for orchestrating the reconciliation process.

#### Methods

**`__init__(original_df, transformed_df)`**
- Initialize with original and transformed datasets

**`match_columns(categorical_columns, min_name_overlap=0.9, similarity_threshold=0.5)`**
- Match categorical columns between datasets
- Returns dictionary of ColumnMatch objects

**`resolve_multiple_matches(column_resolutions)`**
- Resolve ambiguous column matches with user choices
- `column_resolutions`: Dict mapping original columns to chosen transformed columns

**`aggregate_data(numerical_columns, agg_func='sum')`**
- Group and aggregate numerical columns by matched categorical columns
- Returns aggregated datasets

**`detect_anomalies(threshold=10.0, threshold_type='absolute')`**
- Detect anomalies in aggregated data
- `threshold_type`: 'absolute' or 'percentage'

**`run_full_reconciliation(categorical_columns, numerical_columns, **kwargs)`**
- Execute complete reconciliation workflow
- Returns comprehensive results dictionary

**`get_anomaly_report(filters=None)`**
- Get filtered anomaly report
- Supports filtering by anomaly status, minimum difference, columns, etc.

**`export_results(output_dir='./reconciliation_results')`**
- Export all results to CSV files

## Advanced Usage

### Handling Multiple Matches

```python
# Run initial column matching
column_matches = engine.match_columns(['region', 'category'])

# Check for multiple matches
multiple_matches = {
    col: match for col, match in column_matches.items()
    if match.status == MatchStatus.MULTIPLE_MATCHES
}

if multiple_matches:
    # User selects the best matches (this would come from your frontend)
    resolutions = {
        'region': 'region_code',      # User chose 'region_code' for 'region'
        'category': 'product_type'    # User chose 'product_type' for 'category'
    }
    
    # Apply resolutions
    engine.resolve_multiple_matches(resolutions)
```

### Custom Filtering

```python
# Get anomalies with specific criteria
large_anomalies = engine.get_anomaly_report({
    'only_anomalies': True,
    'min_difference': 100,
    'numerical_columns': ['sales'],
    'anomaly_types': ['threshold_exceeded']
})
```

### Different Aggregation Functions

```python
# Use mean instead of sum
results = engine.run_full_reconciliation(
    categorical_columns=['region'],
    numerical_columns=['sales'],
    agg_func='mean'
)
```

### Percentage-Based Thresholds

```python
# Detect anomalies where difference is >5% of original value
results = engine.run_full_reconciliation(
    categorical_columns=['region'],
    numerical_columns=['sales'],
    threshold=5.0,
    threshold_type='percentage'
)
```

## Example Datasets

The tool includes sample dataset generators for testing:

```python
from reconciliation_engine import create_sample_datasets

original_df, transformed_df = create_sample_datasets()
```

## Column Matching Logic

1. **Exact Name Match**: If a column with the same name exists in both datasets:
   - Calculate Jaccard similarity of unique values
   - If similarity ≥ 90% (configurable), mark as single match

2. **Value-Based Matching**: If no exact name match:
   - Compare values with all categorical columns in transformed dataset
   - Use Jaccard similarity with configurable threshold (default: 0.5)
   - Return single match, multiple matches, or no match

3. **Match Resolution**: For multiple matches:
   - Present options to user with similarity scores
   - Allow user to select the best match
   - Update match status to single match

## Anomaly Detection Types

- **Threshold Exceeded**: Numerical difference exceeds specified threshold
- **Missing in Original**: Data exists in transformed but not original dataset
- **Missing in Transformed**: Data exists in original but not transformed dataset

## Error Handling

The tool provides comprehensive error handling for:
- Missing columns
- Invalid data types
- Empty datasets
- No matching columns
- Invalid thresholds

## Logging

Built-in logging provides visibility into:
- Column matching progress
- Aggregation statistics  
- Anomaly detection results
- Export operations

Set logging level:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Contributing

The modular design makes it easy to extend functionality:
- Add new similarity metrics in `column_matcher.py`
- Implement additional aggregation functions in `data_aggregator.py`
- Create custom anomaly detection rules in `anomaly_detector.py`

## License

MIT License - feel free to use and modify for your needs.

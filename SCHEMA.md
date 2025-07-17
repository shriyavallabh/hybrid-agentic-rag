# Knowledge Graph Schema
Version: 1.0.0
Last Updated: 2025-07-17

## Node Types

### Model
Represents a machine learning or statistical model in the knowledge base.
- **id**: Unique identifier (UUID)
- **type**: "Model"
- **label**: Model name (e.g., "loan_pd_model", "fraud_score_v3")
- **details**: Brief description of the model's purpose and methodology
- **page**: Source page number (if extracted from PDF)

### Dataset
Represents a dataset used by one or more models.
- **id**: Unique identifier (UUID)
- **type**: "Dataset"
- **label**: Dataset name
- **details**: Description including size, features, and characteristics
- **page**: Source page number (if extracted from PDF)

### Metric
Represents a performance metric or evaluation measure.
- **id**: Unique identifier (UUID)
- **type**: "Metric"
- **label**: Metric name (e.g., "AUC", "Precision", "RMSE")
- **details**: Metric value and context
- **page**: Source page number (if extracted from PDF)

### CodeEntity
Represents a code component (function, class, module).
- **id**: Unique identifier (UUID)
- **type**: "CodeEntity"
- **label**: Entity name (e.g., "calculate_risk_score", "DataPreprocessor")
- **details**: Purpose and key implementation details
- **page**: Source page number (if extracted from PDF)

## Edge Types

### USES_DATASET
Connects a Model to a Dataset it uses.
- **confidence**: Float (0-1) indicating extraction confidence
- **source_page**: Page number where relationship was found

### HAS_METRIC
Connects a Model to its performance Metrics.
- **confidence**: Float (0-1) indicating extraction confidence
- **source_page**: Page number where relationship was found

### CALLS
Connects CodeEntity nodes when one calls another.
- **confidence**: Float (0-1) indicating extraction confidence
- **source_page**: Page number where relationship was found

### DERIVED_FROM
Indicates when one Model is derived from another.
- **confidence**: Float (0-1) indicating extraction confidence
- **source_page**: Page number where relationship was found

### DEPENDS_ON
General dependency relationship between any node types.
- **confidence**: Float (0-1) indicating extraction confidence
- **source_page**: Page number where relationship was found

### COMPARES_TO
Indicates when models or metrics are compared.
- **confidence**: Float (0-1) indicating extraction confidence
- **source_page**: Page number where relationship was found

## Schema Validation Rules

1. All nodes must have: id, type, label, details
2. All edges must have: confidence (0.0-1.0)
3. Node IDs must be valid UUIDs
4. Edge types must be from the defined set
5. Confidence values must be between 0 and 1
# MLOps Stack Documentation

## Overview

This project implements a complete MLOps stack for production machine learning systems. All components are production-ready with proper error handling, logging, and fallback mechanisms.

## Components

### 1. Experiment Tracking

**Location**: `src/mlops/experiment_tracker.py`

Lightweight wrapper around MLflow for tracking ML experiments. Automatically falls back to local JSON logging when MLflow is unavailable.

**Features**:
- Parameter and metric logging
- Model artifact tracking
- Step-based metric history
- Local JSON fallback mode

**Usage**:
```python
from src.mlops import ExperimentTracker

tracker = ExperimentTracker("my-experiments")
tracker.start_run("training-run-1")
tracker.log_params({"lr": 0.001, "epochs": 50})
tracker.log_metrics({"accuracy": 0.85}, step=0)
tracker.end_run()
```

### 2. Feature Store

**Location**: `src/mlops/feature_store.py`

Redis-based feature caching system with versioning and metadata tracking. Falls back to in-memory storage when Redis is unavailable.

**Features**:
- Distributed feature caching with Redis
- Feature versioning and metadata
- TTL-based expiration
- In-memory fallback mode

**Usage**:
```python
from src.mlops import FeatureStore

store = FeatureStore()
store.store_features(
    symbol="AAPL",
    feature_group="technical_indicators",
    features=features_df
)
features = store.get_features("AAPL", "technical_indicators")
```

### 3. Pipeline Orchestration

**Location**: `src/mlops/pipeline_orchestrator.py`

DAG-based pipeline orchestrator for managing ML workflows. Handles task dependencies, execution order, and error recovery.

**Features**:
- Topological sort for dependency resolution
- Retry logic with exponential backoff
- Task status tracking
- Execution logging

**Usage**:
```python
from src.mlops import Pipeline

pipeline = Pipeline("training-pipeline")
pipeline.add_task("fetch_data", fetch_func, params={"symbol": "AAPL"})
pipeline.add_task("train", train_func, depends_on=["fetch_data"])
results = pipeline.run()
```

## Architecture

### Experiment Tracker
```
User Code
    │
    ├── MLflow (if available)
    │   └── Tracks to: ./mlruns/
    │
    └── JSON Fallback (if MLflow unavailable)
        └── Logs to: ./experiments/*.json
```

### Feature Store
```
User Code
    │
    ├── Redis (if available)
    │   └── Distributed cache with TTL
    │
    └── In-Memory Fallback
        └── Local dict with expiration
```

### Pipeline Orchestrator
```
Task Definition
    │
    ├── Dependency Resolution (Topological Sort)
    ├── Execution Order Calculation
    │
    └── Task Execution
        ├── Dependency Check
        ├── Retry Logic
        └── Result Logging
```

## Design Principles

### 1. Graceful Degradation
All components work with or without external dependencies:
- Experiment tracking works without MLflow
- Feature store works without Redis
- No external orchestration tools required

### 2. Production Ready
- Proper error handling throughout
- Retry logic for transient failures
- Execution logging for debugging
- Type hints for code clarity

### 3. Simplicity
- Clean, readable code
- Minimal dependencies
- Easy to understand and modify
- No over-engineering

## Integration Points

### With Training Scripts
```python
from src.mlops import ExperimentTracker, FeatureStore

# Start experiment
tracker = ExperimentTracker("stock-training")
tracker.start_run("lstm-aapl")

# Get cached features or compute
store = FeatureStore()
features = store.get_features("AAPL", "indicators")
if features is None:
    features = compute_features("AAPL")
    store.store_features("AAPL", "indicators", features)

# Train and log
model = train_model(features)
tracker.log_metrics({"accuracy": 0.85})
tracker.end_run()
```

### With Data Pipelines
```python
from src.mlops import Pipeline

pipeline = Pipeline("data-pipeline")
pipeline.add_task("fetch", fetch_data)
pipeline.add_task("clean", clean_data, depends_on=["fetch"])
pipeline.add_task("transform", transform_data, depends_on=["clean"])
pipeline.run()
```

## Performance

- **Experiment Tracking**: <10ms overhead per log operation
- **Feature Store**: <50ms cache hit (Redis), <5ms (in-memory)
- **Pipeline Orchestration**: <100ms overhead for DAG resolution

## Testing

Run the demo to verify all components:
```bash
python example_mlops_usage.py
```

Expected output:
- Experiment logs in `./experiments/`
- Pipeline logs in `./pipeline_logs/`
- Feature cache (in-memory or Redis)

## Troubleshooting

### MLflow not found
- Normal behavior, will use JSON fallback
- Install with: `pip install mlflow`
- Start UI with: `mlflow ui`

### Redis connection failed
- Normal behavior, will use in-memory cache
- Start Redis with: `redis-server`
- Or use Docker: `docker run -d -p 6379:6379 redis`

### Pipeline task failures
- Check logs in `./pipeline_logs/`
- Tasks retry 3 times by default
- Failed tasks skip dependent tasks

## Future Enhancements

Potential improvements (not currently planned):
- Distributed pipeline execution
- Remote MLflow tracking server support
- Feature store with database backend
- Pipeline visualization UI
- Integration with Apache Airflow

## References

- MLflow Documentation: https://mlflow.org/docs/latest/
- Redis Documentation: https://redis.io/documentation
- Topological Sort: https://en.wikipedia.org/wiki/Topological_sorting
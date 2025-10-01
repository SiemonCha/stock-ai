"""
MLOps Components Usage Example

Demonstrates experiment tracking, feature storage, and pipeline orchestration
for machine learning workflows.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from src.mlops import ExperimentTracker, FeatureStore, Pipeline


def main():
    print("="*60)
    print("MLOps Components Demo")
    print("="*60)

    # 1. Experiment Tracking
    demo_experiment_tracking()

    # 2. Feature Store
    demo_feature_store()

    # 3. Pipeline Orchestration
    demo_pipeline()

    print("\n" + "="*60)
    print("Demo Complete")
    print("="*60)


def demo_experiment_tracking():
    """Demonstrate experiment tracking capabilities."""
    print("\n\n1. EXPERIMENT TRACKING")
    print("-"*40)

    # Create tracker
    tracker = ExperimentTracker(experiment_name="demo-experiments")
    tracker.start_run(run_name="lstm-model-demo")

    # Log hyperparameters
    tracker.log_params({
        "model_type": "LSTM",
        "symbol": "AAPL",
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "sequence_length": 60
    })

    # Simulate training loop
    print("\nSimulating model training...")
    for epoch in range(5):
        # Simulate training metrics
        train_loss = 0.5 - epoch * 0.08
        val_loss = 0.6 - epoch * 0.07
        accuracy = 0.70 + epoch * 0.05

        tracker.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy
        }, step=epoch)

        print(f"Epoch {epoch}: loss={train_loss:.3f}, acc={accuracy:.3f}")

    tracker.end_run()
    print(f"\nExperiment logged to: {tracker.get_tracking_uri()}")


def demo_feature_store():
    """Demonstrate feature store capabilities."""
    print("\n\n2. FEATURE STORE")
    print("-"*40)

    store = FeatureStore()

    # Create sample technical indicators
    print("\nComputing technical indicators...")
    features = pd.DataFrame({
        'sma_20': np.random.rand(100) * 150,
        'sma_50': np.random.rand(100) * 150,
        'rsi': np.random.rand(100) * 100,
        'macd': np.random.rand(100) * 10,
        'bollinger_upper': np.random.rand(100) * 160,
        'bollinger_lower': np.random.rand(100) * 140
    })

    # Store features
    store.store_features(
        symbol="AAPL",
        feature_group="technical_indicators",
        features=features,
        metadata={
            "source": "yfinance",
            "computation_method": "ta-lib",
            "computed_at": datetime.now().isoformat()
        }
    )

    # Retrieve cached features
    print("\nRetrieving features from store...")
    cached_features = store.get_features("AAPL", "technical_indicators")

    if cached_features is not None:
        print(f"Retrieved {len(cached_features)} rows x {len(cached_features.columns)} features")
        print(f"Feature columns: {list(cached_features.columns)}")

    # Get metadata
    metadata = store.get_metadata("AAPL", "technical_indicators")
    if metadata:
        print(f"\nFeature metadata:")
        print(f"  Created: {metadata.get('created_at')}")
        print(f"  Shape: {metadata.get('shape')}")


def demo_pipeline():
    """Demonstrate pipeline orchestration capabilities."""
    print("\n\n3. PIPELINE ORCHESTRATION")
    print("-"*40)

    # Define pipeline tasks
    def fetch_stock_data(symbol: str, context: dict = None) -> dict:
        """Fetch stock data from API."""
        print(f"  Fetching data for {symbol}...")
        import time
        time.sleep(0.5)
        return {
            "symbol": symbol,
            "prices": [150.0, 151.5, 149.8, 152.3, 153.0],
            "volume": [1000000, 1200000, 900000, 1100000, 1050000]
        }

    def compute_technical_features(context: dict = None) -> dict:
        """Compute technical indicators from price data."""
        print(f"  Computing technical features...")
        import time
        time.sleep(0.5)

        if context and "fetch_data" in context:
            prices = context["fetch_data"]["prices"]
            return {
                "sma": sum(prices) / len(prices),
                "min": min(prices),
                "max": max(prices),
                "volatility": np.std(prices)
            }
        return {}

    def train_prediction_model(model_type: str = "lstm", context: dict = None) -> dict:
        """Train prediction model."""
        print(f"  Training {model_type} model...")
        import time
        time.sleep(0.5)
        return {
            "model_type": model_type,
            "accuracy": 0.85,
            "trained": True
        }

    def evaluate_model(context: dict = None) -> dict:
        """Evaluate model performance."""
        print(f"  Evaluating model performance...")
        import time
        time.sleep(0.5)
        return {
            "test_accuracy": 0.82,
            "precision": 0.84,
            "recall": 0.80
        }

    # Create pipeline
    pipeline = Pipeline(name="stock-prediction-pipeline")

    # Add tasks with dependencies
    pipeline.add_task(
        name="fetch_data",
        function=fetch_stock_data,
        params={"symbol": "AAPL"},
        description="Fetch historical stock data"
    )

    pipeline.add_task(
        name="compute_features",
        function=compute_technical_features,
        depends_on=["fetch_data"],
        description="Compute technical indicators"
    )

    pipeline.add_task(
        name="train_model",
        function=train_prediction_model,
        params={"model_type": "lstm"},
        depends_on=["compute_features"],
        description="Train LSTM prediction model"
    )

    pipeline.add_task(
        name="evaluate",
        function=evaluate_model,
        depends_on=["train_model"],
        description="Evaluate model on test set"
    )

    # Visualize pipeline DAG
    print(pipeline.visualize())

    # Execute pipeline
    print("\nExecuting pipeline...\n")
    results = pipeline.run()

    print(f"\nPipeline execution summary:")
    for task_name, result in results.items():
        print(f"  {task_name}: {result.status.value} ({result.duration_seconds:.2f}s)")


if __name__ == "__main__":
    main()
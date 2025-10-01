"""
Experiment Tracking Module

Simple wrapper around MLflow for tracking ML experiments.
Falls back to local JSON logging if MLflow is not installed.
"""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import mlflow
    import mlflow.tensorflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """
    Lightweight experiment tracker with MLflow integration.

    Automatically falls back to local JSON logging if MLflow is unavailable.
    Designed for simplicity and ease of use in ML training loops.
    """

    def __init__(self, experiment_name: str = "experiments", tracking_uri: str = "./mlruns"):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment group
            tracking_uri: Directory for storing experiment data
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id = None
        self.use_fallback = not MLFLOW_AVAILABLE

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        else:
            # Setup local logging directory
            self.log_dir = Path("experiments")
            self.log_dir.mkdir(exist_ok=True)
            self.current_run = None

    def start_run(self, run_name: Optional[str] = None):
        """Start a new experiment run."""
        if self.use_fallback:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_run = {
                "run_name": run_name or f"run_{timestamp}",
                "start_time": datetime.now().isoformat(),
                "params": {},
                "metrics": {},
                "artifacts": []
            }
        else:
            mlflow.start_run(run_name=run_name)
            self.run_id = mlflow.active_run().info.run_id

    def log_params(self, params: Dict[str, Any]):
        """
        Log experiment parameters.

        Args:
            params: Dictionary of hyperparameters
        """
        if self.use_fallback:
            self.current_run["params"].update(params)
        else:
            mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Training step or epoch number
        """
        if self.use_fallback:
            if key not in self.current_run["metrics"]:
                self.current_run["metrics"][key] = []
            self.current_run["metrics"][key].append({
                "value": value,
                "step": step
            })
        else:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step or epoch number
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, file_path: str):
        """
        Log a file artifact (plot, model checkpoint, etc).

        Args:
            file_path: Path to the file to log
        """
        if self.use_fallback:
            self.current_run["artifacts"].append(file_path)
        else:
            mlflow.log_artifact(file_path)

    def log_model(self, model, artifact_path: str = "model", framework: str = "tensorflow"):
        """
        Log a trained model.

        Args:
            model: Trained model object
            artifact_path: Name for the model artifact
            framework: Model framework ("tensorflow" or "pytorch")
        """
        if self.use_fallback:
            # Can't save model without MLflow, just log the path
            self.current_run["artifacts"].append(f"{artifact_path} ({framework})")
        else:
            if framework == "tensorflow":
                mlflow.tensorflow.log_model(model, artifact_path)
            elif framework == "pytorch":
                mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self):
        """End the current experiment run and save results."""
        if self.use_fallback:
            self.current_run["end_time"] = datetime.now().isoformat()

            # Save to JSON file
            filename = self.log_dir / f"{self.current_run['run_name']}.json"
            with open(filename, 'w') as f:
                json.dump(self.current_run, f, indent=2)
        else:
            mlflow.end_run()

    def get_tracking_uri(self) -> str:
        """Get the tracking URI or local directory."""
        return str(self.log_dir) if self.use_fallback else self.tracking_uri


def quick_log_experiment(
    experiment_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model=None,
    run_name: Optional[str] = None
):
    """
    Convenience function for logging a complete experiment in one call.

    Args:
        experiment_name: Name of the experiment
        params: Hyperparameters dictionary
        metrics: Final metrics dictionary
        model: Optional trained model to log
        run_name: Optional name for this run

    Returns:
        ExperimentTracker instance

    Example:
        quick_log_experiment(
            "stock-prediction",
            params={"model": "lstm", "lr": 0.001},
            metrics={"accuracy": 0.85},
            model=trained_model
        )
    """
    tracker = ExperimentTracker(experiment_name)
    tracker.start_run(run_name)
    tracker.log_params(params)
    tracker.log_metrics(metrics)
    if model is not None:
        tracker.log_model(model)
    tracker.end_run()
    return tracker


if __name__ == "__main__":
    # Simple test
    tracker = ExperimentTracker("test-experiment")
    tracker.start_run("demo-run")

    tracker.log_params({
        "model_type": "LSTM",
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32
    })

    # Simulate training
    for epoch in range(5):
        tracker.log_metrics({
            "train_loss": 0.5 - epoch * 0.05,
            "val_loss": 0.6 - epoch * 0.04,
            "accuracy": 0.7 + epoch * 0.04
        }, step=epoch)

    tracker.end_run()
    print(f"Experiment logged to: {tracker.get_tracking_uri()}")
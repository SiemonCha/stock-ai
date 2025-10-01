"""
Pipeline Orchestration Module

Lightweight DAG-based pipeline orchestrator for ML workflows.
Handles task dependencies, execution order, and error recovery.
"""

import time
import json
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """
    Pipeline task definition.

    Attributes:
        name: Unique task identifier
        function: Callable to execute
        params: Parameters to pass to function
        depends_on: List of task names that must complete first
        retry_count: Number of retry attempts on failure
        timeout_seconds: Maximum execution time
        description: Human-readable task description
    """
    name: str
    function: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout_seconds: Optional[int] = None
    description: str = ""


@dataclass
class TaskResult:
    """Result of task execution."""
    task_name: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0


class Pipeline:
    """
    Simple pipeline orchestrator with DAG support.

    Manages task dependencies and execution order using topological sort.
    Includes retry logic and execution logging.
    """

    def __init__(self, name: str = "pipeline", log_dir: str = "./pipeline_logs"):
        """
        Initialize pipeline.

        Args:
            name: Pipeline identifier
            log_dir: Directory for execution logs
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.execution_order: List[str] = []

    def add_task(
        self,
        name: str,
        function: Callable,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        description: str = ""
    ):
        """
        Add a task to the pipeline.

        Args:
            name: Unique task name
            function: Function to execute
            params: Parameters to pass to function
            depends_on: List of prerequisite task names
            description: Task description

        Example:
            pipeline.add_task(
                name="fetch_data",
                function=fetch_stock_data,
                params={"symbol": "AAPL"},
                depends_on=[],
                description="Fetch stock data from API"
            )
        """
        task = Task(
            name=name,
            function=function,
            params=params or {},
            depends_on=depends_on or [],
            description=description
        )
        self.tasks[name] = task

    def _resolve_dependencies(self) -> List[str]:
        """
        Compute task execution order using topological sort.

        Returns:
            List of task names in execution order

        Raises:
            ValueError: If circular dependencies are detected
        """
        visited = set()
        order = []

        def visit(task_name: str):
            if task_name in visited:
                return
            visited.add(task_name)

            task = self.tasks[task_name]
            for dep in task.depends_on:
                if dep not in self.tasks:
                    raise ValueError(f"Task '{task_name}' depends on unknown task '{dep}'")
                visit(dep)

            order.append(task_name)

        for task_name in self.tasks:
            visit(task_name)

        return order

    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task with retry logic."""
        result = TaskResult(
            task_name=task.name,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Check dependency status
            for dep in task.depends_on:
                if self.results[dep].status != TaskStatus.SUCCESS:
                    result.status = TaskStatus.SKIPPED
                    result.error = f"Dependency '{dep}' did not succeed"
                    return result

            # Execute with retries
            last_error = None
            for attempt in range(task.retry_count):
                try:
                    # Pass dependency results as context
                    context = {
                        dep: self.results[dep].result
                        for dep in task.depends_on
                    }

                    # Execute function
                    task_result = task.function(**task.params, context=context)
                    result.result = task_result
                    result.status = TaskStatus.SUCCESS
                    break

                except Exception as e:
                    last_error = e
                    if attempt < task.retry_count - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        result.status = TaskStatus.FAILED
                        result.error = str(e)

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)

        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    def run(self, continue_on_error: bool = False) -> Dict[str, TaskResult]:
        """
        Execute the pipeline.

        Args:
            continue_on_error: Continue executing independent tasks after failures

        Returns:
            Dictionary mapping task names to their results
        """
        start_time = datetime.now()

        # Resolve execution order
        try:
            self.execution_order = self._resolve_dependencies()
        except Exception as e:
            print(f"Failed to resolve dependencies: {e}")
            return {}

        # Execute tasks in order
        for task_name in self.execution_order:
            task = self.tasks[task_name]
            result = self._execute_task(task)
            self.results[task_name] = result

            # Stop on error if not continuing
            if result.status == TaskStatus.FAILED and not continue_on_error:
                break

        # Log execution summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        success_count = sum(1 for r in self.results.values() if r.status == TaskStatus.SUCCESS)
        failed_count = sum(1 for r in self.results.values() if r.status == TaskStatus.FAILED)
        skipped_count = sum(1 for r in self.results.values() if r.status == TaskStatus.SKIPPED)

        print(f"\nPipeline '{self.name}' completed in {duration:.2f}s")
        print(f"Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")

        self._save_log()

        return self.results

    def _save_log(self):
        """Save pipeline execution log to JSON file."""
        log_data = {
            "pipeline_name": self.name,
            "execution_time": datetime.now().isoformat(),
            "execution_order": self.execution_order,
            "results": {
                name: {
                    "status": result.status.value,
                    "duration_seconds": result.duration_seconds,
                    "error": result.error,
                    "start_time": result.start_time.isoformat() if result.start_time else None,
                    "end_time": result.end_time.isoformat() if result.end_time else None
                }
                for name, result in self.results.items()
            }
        }

        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def visualize(self) -> str:
        """
        Generate text-based visualization of the pipeline DAG.

        Returns:
            String representation of pipeline structure
        """
        lines = [f"\nPipeline: {self.name}\n", "-" * 40]

        for task_name in self.execution_order:
            task = self.tasks[task_name]
            status = self.results.get(task_name, TaskResult(task_name, TaskStatus.PENDING)).status

            status_symbol = {
                TaskStatus.SUCCESS: "[OK]",
                TaskStatus.FAILED: "[FAIL]",
                TaskStatus.SKIPPED: "[SKIP]",
                TaskStatus.PENDING: "[ ]"
            }.get(status, "[ ]")

            lines.append(f"\n{status_symbol} {task_name}")
            if task.description:
                lines.append(f"    {task.description}")
            if task.depends_on:
                lines.append(f"    Requires: {', '.join(task.depends_on)}")

        return '\n'.join(lines)


if __name__ == "__main__":
    # Simple test
    def fetch_data(symbol: str, context: Dict = None) -> Dict:
        """Example task: fetch data."""
        time.sleep(0.5)
        return {"symbol": symbol, "data": [100, 101, 102]}

    def compute_features(context: Dict = None) -> Dict:
        """Example task: compute features."""
        time.sleep(0.5)
        if context and "fetch_data" in context:
            data = context["fetch_data"]["data"]
            return {"features": [x * 2 for x in data]}
        return {"features": []}

    def train_model(model_type: str = "lstm", context: Dict = None) -> Dict:
        """Example task: train model."""
        time.sleep(0.5)
        return {"model": model_type, "accuracy": 0.85}

    # Create and run pipeline
    pipeline = Pipeline("test-pipeline")

    pipeline.add_task(
        name="fetch_data",
        function=fetch_data,
        params={"symbol": "AAPL"},
        description="Fetch stock data"
    )

    pipeline.add_task(
        name="compute_features",
        function=compute_features,
        depends_on=["fetch_data"],
        description="Compute indicators"
    )

    pipeline.add_task(
        name="train_model",
        function=train_model,
        params={"model_type": "lstm"},
        depends_on=["compute_features"],
        description="Train model"
    )

    print(pipeline.visualize())
    results = pipeline.run()
    print(f"\nResults: {[(k, v.status.value) for k, v in results.items()]}")
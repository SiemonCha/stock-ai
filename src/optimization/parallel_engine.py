"""
Advanced Parallel Processing Engine
Ultra-fast parallel processing for stock predictions and analysis
"""

import asyncio
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import time
import queue
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import dask
    from dask.distributed import Client, LocalCluster
    from dask import delayed as dask_delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

@dataclass
class ProcessingTask:
    """Task definition for parallel processing"""
    id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ProcessingResult:
    """Result from parallel processing"""
    task_id: str
    result: Any
    execution_time: float
    success: bool
    error: Optional[str] = None
    worker_id: Optional[str] = None

class ParallelExecutionEngine:
    """
    Advanced parallel execution engine with multiple backends
    """
    
    def __init__(self, 
                 max_workers: int = None,
                 backend: str = "thread",
                 chunk_size: int = 1,
                 timeout: float = 300.0):
        
        self.max_workers = max_workers or mp.cpu_count()
        self.backend = backend
        self.chunk_size = chunk_size
        self.timeout = timeout
        
        # Initialize backends
        self.executor = None
        self.ray_initialized = False
        self.dask_client = None
        
        self._setup_backend()
        
        # Performance tracking
        self.execution_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
    def _setup_backend(self):
        """Setup the parallel processing backend"""
        
        if self.backend == "ray" and RAY_AVAILABLE:
            try:
                if not ray.is_initialized():
                    ray.init(num_cpus=self.max_workers, ignore_reinit_error=True)
                self.ray_initialized = True
                print(f"✅ Ray backend initialized with {self.max_workers} workers")
            except Exception as e:
                print(f"⚠️ Ray initialization failed: {e}")
                self.backend = "thread"
                
        elif self.backend == "dask" and DASK_AVAILABLE:
            try:
                cluster = LocalCluster(n_workers=self.max_workers, 
                                     threads_per_worker=1,
                                     processes=True)
                self.dask_client = Client(cluster)
                print(f"✅ Dask backend initialized with {self.max_workers} workers")
            except Exception as e:
                print(f"⚠️ Dask initialization failed: {e}")
                self.backend = "thread"
        
        # Fallback to concurrent.futures
        if self.backend in ["thread", "process"] or not hasattr(self, 'ray_initialized'):
            executor_class = ThreadPoolExecutor if self.backend == "thread" else ProcessPoolExecutor
            self.executor = executor_class(max_workers=self.max_workers)
            print(f"✅ {self.backend.title()} pool initialized with {self.max_workers} workers")
    
    def execute_parallel(self, 
                         tasks: List[ProcessingTask],
                         progress_callback: Callable = None) -> List[ProcessingResult]:
        """Execute tasks in parallel with selected backend"""
        
        if not tasks:
            return []
        
        print(f"🚀 Executing {len(tasks)} tasks with {self.backend} backend...")
        start_time = time.time()
        
        if self.backend == "ray" and self.ray_initialized:
            results = self._execute_with_ray(tasks, progress_callback)
        elif self.backend == "dask" and self.dask_client:
            results = self._execute_with_dask(tasks, progress_callback)
        elif self.backend == "joblib" and JOBLIB_AVAILABLE:
            results = self._execute_with_joblib(tasks, progress_callback)
        else:
            results = self._execute_with_concurrent(tasks, progress_callback)
        
        # Update statistics
        total_time = time.time() - start_time
        self._update_stats(results, total_time)
        
        print(f"✅ Completed {len(tasks)} tasks in {total_time:.2f}s")
        
        return results
    
    def _execute_with_ray(self, tasks: List[ProcessingTask], 
                         progress_callback: Callable) -> List[ProcessingResult]:
        """Execute tasks using Ray"""
        
        @ray.remote
        def ray_task_wrapper(task: ProcessingTask) -> ProcessingResult:
            start_time = time.time()
            try:
                result = task.function(*task.args, **task.kwargs)
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task.id,
                    result=result,
                    execution_time=execution_time,
                    success=True,
                    worker_id=f"ray_{ray.get_runtime_context().get_worker_id()}"
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                return ProcessingResult(
                    task_id=task.id,
                    result=None,
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
        
        # Submit all tasks
        futures = [ray_task_wrapper.remote(task) for task in tasks]
        
        # Collect results with progress tracking
        results = []
        for i, future in enumerate(futures):
            try:
                result = ray.get(future, timeout=self.timeout)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(tasks))
                    
            except ray.exceptions.GetTimeoutError:
                results.append(ProcessingResult(
                    task_id=tasks[i].id,
                    result=None,
                    execution_time=self.timeout,
                    success=False,
                    error="Task timeout"
                ))
        
        return results
    
    def _execute_with_dask(self, tasks: List[ProcessingTask], 
                          progress_callback: Callable) -> List[ProcessingResult]:
        """Execute tasks using Dask"""
        
        def dask_task_wrapper(task: ProcessingTask) -> ProcessingResult:
            start_time = time.time()
            try:
                result = task.function(*task.args, **task.kwargs)
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task.id,
                    result=result,
                    execution_time=execution_time,
                    success=True,
                    worker_id="dask_worker"
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                return ProcessingResult(
                    task_id=task.id,
                    result=None,
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
        
        # Create dask delayed objects
        delayed_tasks = [dask_delayed(dask_task_wrapper)(task) for task in tasks]
        
        # Compute all tasks
        results = dask.compute(*delayed_tasks)
        
        return list(results)
    
    def _execute_with_joblib(self, tasks: List[ProcessingTask], 
                           progress_callback: Callable) -> List[ProcessingResult]:
        """Execute tasks using Joblib"""
        
        def joblib_task_wrapper(task: ProcessingTask) -> ProcessingResult:
            start_time = time.time()
            try:
                result = task.function(*task.args, **task.kwargs)
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task.id,
                    result=result,
                    execution_time=execution_time,
                    success=True,
                    worker_id="joblib_worker"
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                return ProcessingResult(
                    task_id=task.id,
                    result=None,
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
        
        # Execute with joblib parallel
        results = Parallel(n_jobs=self.max_workers, backend='threading')(
            delayed(joblib_task_wrapper)(task) for task in tasks
        )
        
        return results
    
    def _execute_with_concurrent(self, tasks: List[ProcessingTask], 
                               progress_callback: Callable) -> List[ProcessingResult]:
        """Execute tasks using concurrent.futures"""
        
        def concurrent_task_wrapper(task: ProcessingTask) -> ProcessingResult:
            start_time = time.time()
            try:
                result = task.function(*task.args, **task.kwargs)
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task.id,
                    result=result,
                    execution_time=execution_time,
                    success=True,
                    worker_id=f"{self.backend}_worker"
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                return ProcessingResult(
                    task_id=task.id,
                    result=None,
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
        
        results = []
        
        # Submit tasks
        future_to_task = {
            self.executor.submit(concurrent_task_wrapper, task): task 
            for task in tasks
        }
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_task, timeout=self.timeout):
            try:
                result = future.result()
                results.append(result)
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(tasks))
                    
            except Exception as e:
                task = future_to_task[future]
                results.append(ProcessingResult(
                    task_id=task.id,
                    result=None,
                    execution_time=0.0,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def _update_stats(self, results: List[ProcessingResult], total_time: float):
        """Update execution statistics"""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        self.execution_stats['total_tasks'] += len(results)
        self.execution_stats['successful_tasks'] += successful
        self.execution_stats['failed_tasks'] += failed
        self.execution_stats['total_execution_time'] += total_time
        
        if self.execution_stats['total_tasks'] > 0:
            self.execution_stats['average_execution_time'] = (
                self.execution_stats['total_execution_time'] / 
                self.execution_stats['total_tasks']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()
    
    def shutdown(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.ray_initialized:
            try:
                ray.shutdown()
            except:
                pass
        
        if self.dask_client:
            try:
                self.dask_client.close()
            except:
                pass

class StockPredictionParallelEngine:
    """
    Specialized parallel engine for stock predictions
    """
    
    def __init__(self, max_workers: int = None, backend: str = "thread"):
        self.engine = ParallelExecutionEngine(
            max_workers=max_workers,
            backend=backend,
            timeout=600.0  # 10 minutes timeout for predictions
        )
        
    def predict_multiple_symbols(self, 
                                symbols: List[str],
                                predictor_func: Callable,
                                predictor_kwargs: Dict = None,
                                progress_callback: Callable = None) -> Dict[str, Any]:
        """Parallel prediction for multiple symbols"""
        
        if not predictor_kwargs:
            predictor_kwargs = {}
        
        # Create tasks
        tasks = []
        for symbol in symbols:
            task = ProcessingTask(
                id=symbol,
                function=predictor_func,
                args=(symbol,),
                kwargs=predictor_kwargs.copy(),
                priority=1
            )
            tasks.append(task)
        
        # Execute in parallel
        results = self.engine.execute_parallel(tasks, progress_callback)
        
        # Format results
        prediction_results = {}
        for result in results:
            if result.success:
                prediction_results[result.task_id] = result.result
            else:
                prediction_results[result.task_id] = {
                    'error': result.error,
                    'execution_time': result.execution_time
                }
        
        return prediction_results
    
    def batch_feature_engineering(self, 
                                 symbol_data_dict: Dict[str, pd.DataFrame],
                                 feature_func: Callable,
                                 feature_kwargs: Dict = None) -> Dict[str, pd.DataFrame]:
        """Parallel feature engineering for multiple symbols"""
        
        if not feature_kwargs:
            feature_kwargs = {}
        
        # Create tasks
        tasks = []
        for symbol, data in symbol_data_dict.items():
            task = ProcessingTask(
                id=symbol,
                function=feature_func,
                args=(data,),
                kwargs=feature_kwargs.copy(),
                priority=2
            )
            tasks.append(task)
        
        # Execute in parallel
        results = self.engine.execute_parallel(tasks)
        
        # Format results
        feature_results = {}
        for result in results:
            if result.success and isinstance(result.result, pd.DataFrame):
                feature_results[result.task_id] = result.result
        
        return feature_results
    
    def parallel_backtesting(self, 
                           symbols: List[str],
                           backtest_func: Callable,
                           backtest_params: Dict) -> Dict[str, Any]:
        """Parallel backtesting for multiple symbols"""
        
        # Create tasks
        tasks = []
        for symbol in symbols:
            params = backtest_params.copy()
            params['symbol'] = symbol
            
            task = ProcessingTask(
                id=f"backtest_{symbol}",
                function=backtest_func,
                args=(),
                kwargs=params,
                priority=3,
                timeout=1800.0  # 30 minutes for backtesting
            )
            tasks.append(task)
        
        # Execute in parallel
        results = self.engine.execute_parallel(tasks)
        
        # Format results
        backtest_results = {}
        for result in results:
            symbol = result.task_id.replace('backtest_', '')
            if result.success:
                backtest_results[symbol] = result.result
            else:
                backtest_results[symbol] = {'error': result.error}
        
        return backtest_results

class AsyncStockProcessor:
    """
    Asynchronous processor for real-time stock data
    """
    
    def __init__(self, max_concurrent: int = 50):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
    async def __aenter__(self):
        # Initialize async session (e.g., aiohttp session)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def process_symbol_async(self, 
                                 symbol: str, 
                                 processor_func: Callable,
                                 **kwargs) -> Tuple[str, Any]:
        """Process single symbol asynchronously"""
        async with self.semaphore:
            try:
                # Convert sync function to async if needed
                if asyncio.iscoroutinefunction(processor_func):
                    result = await processor_func(symbol, **kwargs)
                else:
                    # Run in thread pool for CPU-bound operations
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, processor_func, symbol, **kwargs
                    )
                
                return symbol, result
                
            except Exception as e:
                return symbol, {'error': str(e)}
    
    async def process_symbols_batch(self, 
                                  symbols: List[str],
                                  processor_func: Callable,
                                  **kwargs) -> Dict[str, Any]:
        """Process multiple symbols asynchronously"""
        
        # Create async tasks
        tasks = [
            self.process_symbol_async(symbol, processor_func, **kwargs)
            for symbol in symbols
        ]
        
        # Execute with progress tracking
        results = {}
        completed_tasks = 0
        
        for task in asyncio.as_completed(tasks):
            symbol, result = await task
            results[symbol] = result
            
            completed_tasks += 1
            if completed_tasks % 10 == 0:  # Progress every 10 completions
                print(f"🔄 Processed {completed_tasks}/{len(symbols)} symbols")
        
        return results

class WorkerPoolManager:
    """
    Advanced worker pool manager with load balancing
    """
    
    def __init__(self, 
                 pool_size: int = None,
                 worker_type: str = "thread"):
        
        self.pool_size = pool_size or mp.cpu_count()
        self.worker_type = worker_type
        
        # Worker pools
        self.cpu_pool = ProcessPoolExecutor(max_workers=self.pool_size)
        self.io_pool = ThreadPoolExecutor(max_workers=self.pool_size * 2)
        
        # Task queues
        self.cpu_queue = queue.PriorityQueue()
        self.io_queue = queue.PriorityQueue()
        
        # Worker statistics
        self.worker_stats = {
            'cpu_tasks_completed': 0,
            'io_tasks_completed': 0,
            'cpu_avg_time': 0.0,
            'io_avg_time': 0.0
        }
        
    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """Submit CPU-intensive task"""
        future = self.cpu_pool.submit(func, *args, **kwargs)
        return future
    
    def submit_io_task(self, func: Callable, *args, **kwargs):
        """Submit I/O-intensive task"""
        future = self.io_pool.submit(func, *args, **kwargs)
        return future
    
    def auto_submit_task(self, func: Callable, *args, task_type: str = "auto", **kwargs):
        """Automatically choose appropriate worker pool"""
        
        if task_type == "cpu" or (task_type == "auto" and self._is_cpu_intensive(func)):
            return self.submit_cpu_task(func, *args, **kwargs)
        else:
            return self.submit_io_task(func, *args, **kwargs)
    
    def _is_cpu_intensive(self, func: Callable) -> bool:
        """Heuristic to determine if function is CPU-intensive"""
        
        cpu_keywords = ['predict', 'model', 'calculate', 'compute', 'process']
        func_name = func.__name__.lower()
        
        return any(keyword in func_name for keyword in cpu_keywords)
    
    def shutdown(self):
        """Shutdown all worker pools"""
        self.cpu_pool.shutdown(wait=True)
        self.io_pool.shutdown(wait=True)

# Example usage and benchmarking
if __name__ == "__main__":
    print("⚡ Advanced Parallel Processing Engine")
    print("=" * 50)
    
    # Test function for parallel execution
    def sample_prediction_task(symbol: str, delay: float = 0.1) -> Dict[str, Any]:
        """Sample prediction task for testing"""
        time.sleep(delay)  # Simulate processing time
        
        return {
            'symbol': symbol,
            'predicted_price': np.random.uniform(100, 200),
            'confidence': np.random.uniform(0.7, 0.95),
            'processing_time': delay
        }
    
    # Test symbols
    test_symbols = [f"STOCK{i:03d}" for i in range(20)]
    
    # Benchmark different backends
    backends = ["thread", "process"]
    if RAY_AVAILABLE:
        backends.append("ray")
    if JOBLIB_AVAILABLE:
        backends.append("joblib")
    
    print(f"🔬 Benchmarking {len(backends)} backends with {len(test_symbols)} symbols...")
    
    for backend in backends:
        print(f"\n🚀 Testing {backend} backend...")
        
        # Initialize engine
        engine = StockPredictionParallelEngine(backend=backend)
        
        # Progress callback
        def progress(completed, total):
            if completed % 5 == 0:
                print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        # Execute predictions
        start_time = time.time()
        results = engine.predict_multiple_symbols(
            symbols=test_symbols,
            predictor_func=sample_prediction_task,
            predictor_kwargs={'delay': 0.05},
            progress_callback=progress
        )
        execution_time = time.time() - start_time
        
        # Report results
        successful = sum(1 for r in results.values() if 'error' not in r)
        print(f"  ✅ Completed: {successful}/{len(test_symbols)} symbols")
        print(f"  ⏱️ Time: {execution_time:.2f}s")
        print(f"  🚄 Throughput: {len(test_symbols)/execution_time:.1f} symbols/sec")
        
        # Cleanup
        engine.engine.shutdown()
    
    print(f"\n✅ Parallel processing benchmarks complete!")
    
    # Test async processing
    print(f"\n🔄 Testing async processing...")
    
    async def test_async_processing():
        async with AsyncStockProcessor(max_concurrent=10) as processor:
            results = await processor.process_symbols_batch(
                symbols=test_symbols[:10],
                processor_func=sample_prediction_task,
                delay=0.05
            )
            return results
    
    # Run async test
    async_start = time.time()
    async_results = asyncio.run(test_async_processing())
    async_time = time.time() - async_start
    
    successful_async = sum(1 for r in async_results.values() if 'error' not in r)
    print(f"✅ Async completed: {successful_async}/10 symbols in {async_time:.2f}s")
    
    print("\n🎯 Parallel processing system ready for production!")
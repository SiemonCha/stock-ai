"""
Performance Optimization System
Ultra-fast inference and processing optimization for professional trading
"""

import time
import functools
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import hashlib
import os
from pathlib import Path

try:
    import numba
    from numba import jit, cuda, vectorize
    import cupy as cp
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import joblib
    from joblib import Memory
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

class PerformanceCache:
    """
    High-performance caching system with multiple backends
    """
    
    def __init__(self, cache_dir: str = "/tmp/stock_ai_cache", redis_url: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                print("✅ Redis cache connected")
            except:
                print("⚠️ Redis connection failed, using file cache only")
        
        # Initialize joblib memory
        if JOBLIB_AVAILABLE:
            self.memory = Memory(str(self.cache_dir), verbose=0)
        else:
            self.memory = None
        
        # In-memory cache for frequently accessed data
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature"""
        key_data = f"{func_name}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Any:
        """Get item from cache with fallback hierarchy"""
        # 1. Check memory cache first (fastest)
        if key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[key]
        
        # 2. Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    data = pickle.loads(cached_data)
                    self.memory_cache[key] = data  # Promote to memory cache
                    self.cache_hits += 1
                    return data
            except:
                pass
        
        # 3. Check file cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_cache[key] = data  # Promote to memory cache
                    self.cache_hits += 1
                    return data
            except:
                pass
        
        self.cache_misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set item in cache with multiple backends"""
        # Store in memory cache
        self.memory_cache[key] = value
        
        # Store in Redis with TTL
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, pickle.dumps(value))
            except:
                pass
        
        # Store in file cache
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except:
            pass
    
    def cache_stats(self) -> dict:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache)
        }

def performance_cache(ttl: int = 3600):
    """Decorator for caching expensive function calls"""
    def decorator(func: Callable) -> Callable:
        cache = PerformanceCache()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache.cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        wrapper.cache_stats = cache.cache_stats
        wrapper.clear_cache = lambda: cache.memory_cache.clear()
        
        return wrapper
    return decorator

def async_performance_cache(ttl: int = 3600):
    """Async version of performance cache decorator"""
    def decorator(func: Callable) -> Callable:
        cache = PerformanceCache()
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = cache.cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute async function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        wrapper.cache_stats = cache.cache_stats
        wrapper.clear_cache = lambda: cache.memory_cache.clear()
        
        return wrapper
    return decorator

class ModelInferenceOptimizer:
    """
    Optimizes model inference performance with various techniques
    """
    
    def __init__(self):
        self.optimized_models = {}
        self.batch_size = 32
        self.use_gpu = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        if NUMBA_AVAILABLE:
            try:
                cuda.detect()
                return True
            except:
                return False
        return False
    
    def optimize_model(self, model, model_name: str) -> Any:
        """Optimize model for faster inference"""
        print(f"🚀 Optimizing {model_name} for performance...")
        
        # Model-specific optimizations
        if hasattr(model, 'predict_proba'):
            # For scikit-learn models
            optimized_model = self._optimize_sklearn_model(model)
        elif hasattr(model, 'predict') and 'xgb' in str(type(model)):
            # For XGBoost
            optimized_model = self._optimize_xgboost_model(model)
        elif hasattr(model, 'predict') and 'lgb' in str(type(model)):
            # For LightGBM
            optimized_model = self._optimize_lightgbm_model(model)
        elif hasattr(model, 'predict') and 'catboost' in str(type(model)):
            # For CatBoost
            optimized_model = self._optimize_catboost_model(model)
        else:
            optimized_model = model
        
        self.optimized_models[model_name] = optimized_model
        print(f"✅ {model_name} optimization complete")
        
        return optimized_model
    
    def _optimize_sklearn_model(self, model):
        """Optimize scikit-learn models"""
        # Enable threading for tree-based models
        if hasattr(model, 'n_jobs'):
            model.n_jobs = -1
        
        return model
    
    def _optimize_xgboost_model(self, model):
        """Optimize XGBoost models"""
        # Set optimal number of threads
        model.set_param('n_jobs', -1)
        model.set_param('tree_method', 'gpu_hist' if self.use_gpu else 'hist')
        
        return model
    
    def _optimize_lightgbm_model(self, model):
        """Optimize LightGBM models"""
        # LightGBM automatically uses all cores
        return model
    
    def _optimize_catboost_model(self, model):
        """Optimize CatBoost models"""
        # CatBoost has built-in optimizations
        return model
    
    def batch_predict(self, model, data: np.ndarray) -> np.ndarray:
        """Optimized batch prediction"""
        if len(data) <= self.batch_size:
            return model.predict(data)
        
        # Process in batches
        predictions = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_pred = model.predict(batch)
            predictions.extend(batch_pred)
        
        return np.array(predictions)

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Ultra-fast rolling mean calculation using Numba"""
    n = len(data)
    result = np.empty(n)
    
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = np.mean(data[start:i + 1])
    
    return result

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """Ultra-fast rolling standard deviation using Numba"""
    n = len(data)
    result = np.empty(n)
    
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = np.std(data[start:i + 1])
    
    return result

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Ultra-fast RSI calculation"""
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gains = fast_rolling_mean(gains, period)
    avg_losses = fast_rolling_mean(losses, period)
    
    rs = avg_gains / np.where(avg_losses == 0, 1e-10, avg_losses)
    rsi = 100 - (100 / (1 + rs))
    
    return np.concatenate([np.array([50.0]), rsi])  # First value is neutral

class FeatureEngineeringOptimizer:
    """
    High-performance feature engineering with vectorized operations
    """
    
    def __init__(self):
        self.cache = PerformanceCache()
    
    @performance_cache(ttl=1800)  # 30 minutes cache
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ultra-fast technical indicator calculation"""
        prices = df['close'].values
        
        # Vectorized calculations
        df['sma_10'] = fast_rolling_mean(prices, 10)
        df['sma_20'] = fast_rolling_mean(prices, 20)
        df['sma_50'] = fast_rolling_mean(prices, 50)
        
        df['std_10'] = fast_rolling_std(prices, 10)
        df['std_20'] = fast_rolling_std(prices, 20)
        
        df['rsi'] = fast_rsi(prices)
        
        # Bollinger Bands
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        df['bb_position'] = (prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price momentum
        df['momentum_5'] = prices / np.roll(prices, 5) - 1
        df['momentum_10'] = prices / np.roll(prices, 10) - 1
        df['momentum_20'] = prices / np.roll(prices, 20) - 1
        
        return df
    
    def parallel_feature_engineering(self, symbols: List[str], 
                                   data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Parallel feature engineering for multiple symbols"""
        
        def process_symbol(symbol):
            if symbol in data_dict:
                return symbol, self.calculate_technical_indicators(data_dict[symbol])
            return symbol, None
        
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = executor.map(process_symbol, symbols)
        
        return {symbol: df for symbol, df in results if df is not None}

class ParallelPredictionEngine:
    """
    High-performance parallel prediction engine
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.model_optimizer = ModelInferenceOptimizer()
    
    def predict_multiple_symbols(self, symbols: List[str], 
                                models: Dict[str, Any],
                                data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Parallel prediction for multiple symbols"""
        
        def predict_single_symbol(symbol: str) -> Tuple[str, Dict[str, Any]]:
            if symbol not in data_dict:
                return symbol, {'error': 'No data available'}
            
            symbol_data = data_dict[symbol]
            predictions = {}
            
            # Run predictions for all models
            for model_name, model in models.items():
                try:
                    # Get features (last row for prediction)
                    features = self._extract_features(symbol_data)
                    
                    # Make prediction
                    prediction = self.model_optimizer.batch_predict(model, features)
                    predictions[model_name] = float(prediction[0]) if len(prediction) > 0 else 0.0
                    
                except Exception as e:
                    predictions[model_name] = {'error': str(e)}
            
            return symbol, predictions
        
        # Parallel execution
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(predict_single_symbol, symbols)
        
        return dict(results)
    
    def _extract_features(self, df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
        """Extract features for prediction"""
        # Get last 'lookback' rows and select numeric columns
        recent_data = df.tail(lookback)
        numeric_cols = recent_data.select_dtypes(include=[np.number]).columns
        
        # Fill any NaN values
        features = recent_data[numeric_cols].fillna(method='ffill').fillna(0)
        
        # Return as 2D array (for batch prediction)
        return features.values.reshape(1, -1)

class AsyncDataProcessor:
    """
    Asynchronous data processing for real-time performance
    """
    
    def __init__(self):
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
        
    @async_performance_cache(ttl=600)  # 10 minutes cache
    async def fetch_and_process_data(self, symbol: str) -> pd.DataFrame:
        """Async data fetching and processing"""
        async with self.semaphore:
            # Simulate async data fetching
            await asyncio.sleep(0.1)  # Replace with actual async API call
            
            # Generate sample data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            data = pd.DataFrame({
                'date': dates,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000000, 10000000, 100)
            })
            
            return data
    
    async def batch_fetch_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Async batch data fetching"""
        tasks = [self.fetch_and_process_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception):
                data_dict[symbol] = result
            else:
                print(f"❌ Error fetching {symbol}: {result}")
        
        return data_dict

class PerformanceMonitor:
    """
    Monitor and profile system performance
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing and record metrics"""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append(elapsed)
            del self.start_times[operation]
            
            return elapsed
        return None
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        stats = {}
        
        for operation, times in self.metrics.items():
            stats[operation] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times)
            }
        
        return stats
    
    def print_performance_report(self):
        """Print detailed performance report"""
        print("\n📊 PERFORMANCE OPTIMIZATION REPORT")
        print("=" * 60)
        
        stats = self.get_performance_stats()
        
        for operation, metrics in stats.items():
            print(f"\n🎯 {operation.upper()}:")
            print(f"   Executions: {metrics['count']}")
            print(f"   Avg Time: {metrics['avg_time']:.4f}s")
            print(f"   Min Time: {metrics['min_time']:.4f}s") 
            print(f"   Max Time: {metrics['max_time']:.4f}s")
            print(f"   Total Time: {metrics['total_time']:.4f}s")

def performance_profiler(func: Callable) -> Callable:
    """Decorator to profile function performance"""
    monitor = PerformanceMonitor()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        operation = func.__name__
        monitor.start_timer(operation)
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = monitor.end_timer(operation)
            if elapsed and elapsed > 1.0:  # Log slow operations
                print(f"⏱️ {operation} took {elapsed:.4f}s")
    
    wrapper.get_stats = monitor.get_performance_stats
    wrapper.print_report = monitor.print_performance_report
    
    return wrapper

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Performance Optimization System")
    print("=" * 50)
    
    # Test caching system
    @performance_cache(ttl=300)
    def expensive_calculation(n: int) -> float:
        time.sleep(0.1)  # Simulate expensive operation
        return sum(i**2 for i in range(n))
    
    # Test performance
    start = time.time()
    result1 = expensive_calculation(1000)  # Cache miss
    first_time = time.time() - start
    
    start = time.time()
    result2 = expensive_calculation(1000)  # Cache hit
    second_time = time.time() - start
    
    print(f"First call (cache miss): {first_time:.4f}s")
    print(f"Second call (cache hit): {second_time:.4f}s")
    print(f"Speedup: {first_time/second_time:.1f}x")
    print(f"Cache stats: {expensive_calculation.cache_stats()}")
    
    # Test Numba optimization
    if NUMBA_AVAILABLE:
        print(f"\n✅ Numba acceleration available")
        
        # Test fast calculations
        data = np.random.randn(10000)
        
        start = time.time()
        sma = fast_rolling_mean(data, 20)
        numba_time = time.time() - start
        
        start = time.time()
        sma_pandas = pd.Series(data).rolling(20).mean().values
        pandas_time = time.time() - start
        
        print(f"Numba rolling mean: {numba_time:.4f}s")
        print(f"Pandas rolling mean: {pandas_time:.4f}s")
        print(f"Numba speedup: {pandas_time/numba_time:.1f}x")
    else:
        print("⚠️ Numba not available - install with: pip install numba")
    
    print("\n✅ Performance optimization system ready!")
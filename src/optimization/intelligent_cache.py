"""
Intelligent Caching System for Stock Prediction
Advanced caching with smart invalidation and compression
"""

import pickle
import gzip
import json
import hashlib
import time
import threading
import functools
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

try:
    import redis
    import redis.sentinel
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    from diskcache import Cache, FanoutCache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

class SmartCache:
    """
    Intelligent multi-tier caching system optimized for stock data
    """
    
    def __init__(self, 
                 cache_dir: str = "/tmp/stock_ai_smart_cache",
                 memory_limit_mb: int = 512,
                 disk_limit_gb: int = 10,
                 redis_config: Dict = None):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache (L1 - fastest)
        self.memory_cache = {}
        self.memory_access_times = {}
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.memory_lock = threading.RLock()
        
        # Disk cache (L2 - persistent)
        if DISKCACHE_AVAILABLE:
            self.disk_cache = FanoutCache(
                str(self.cache_dir / "disk_cache"),
                size_limit=disk_limit_gb * 1024**3,
                shards=4
            )
        else:
            self.disk_cache = None
        
        # Redis cache (L3 - distributed)
        self.redis_client = None
        if REDIS_AVAILABLE and redis_config:
            self._setup_redis(redis_config)
        
        # Cache statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'redis_hits': 0,
            'redis_misses': 0,
            'evictions': 0,
            'compressions': 0
        }
        
        # Cache metadata
        self.metadata_cache = {}
        self.dependency_graph = {}
        
    def _setup_redis(self, config: Dict):
        """Setup Redis connection with failover"""
        try:
            if 'sentinel' in config:
                # Redis Sentinel for HA
                sentinel = redis.sentinel.Sentinel(config['sentinel']['hosts'])
                self.redis_client = sentinel.master_for(
                    config['sentinel']['service_name'],
                    decode_responses=False
                )
            else:
                # Direct Redis connection
                self.redis_client = redis.Redis(
                    host=config.get('host', 'localhost'),
                    port=config.get('port', 6379),
                    db=config.get('db', 0),
                    password=config.get('password'),
                    decode_responses=False
                )
            
            # Test connection
            self.redis_client.ping()
            print("✅ Redis cache connected")
            
        except Exception as e:
            print(f"⚠️ Redis setup failed: {e}")
            self.redis_client = None
    
    def _generate_key(self, namespace: str, identifier: str, params: Dict = None) -> str:
        """Generate cache key with namespace and parameters"""
        key_parts = [namespace, identifier]
        
        if params:
            # Sort parameters for consistent keys
            param_str = json.dumps(params, sort_keys=True, default=str)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            key_parts.append(param_hash)
        
        return ":".join(key_parts)
    
    def _serialize_data(self, data: Any, compress: bool = True) -> bytes:
        """Serialize data with optional compression"""
        serialized = pickle.dumps(data)
        
        if compress and len(serialized) > 1024:  # Only compress large objects
            self.stats['compressions'] += 1
            return gzip.compress(serialized)
        
        return serialized
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with automatic decompression"""
        try:
            # Try decompression first
            decompressed = gzip.decompress(data)
            return pickle.loads(decompressed)
        except:
            # If decompression fails, try direct deserialization
            return pickle.loads(data)
    
    def _estimate_memory_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            # Rough estimate for other objects
            return len(pickle.dumps(obj))
    
    def _evict_memory_cache(self):
        """Evict least recently used items from memory cache"""
        with self.memory_lock:
            if len(self.memory_cache) == 0:
                return
            
            # Sort by access time (oldest first)
            sorted_items = sorted(
                self.memory_access_times.items(),
                key=lambda x: x[1]
            )
            
            # Evict oldest 25% of items
            items_to_evict = len(sorted_items) // 4 + 1
            
            for key, _ in sorted_items[:items_to_evict]:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    del self.memory_access_times[key]
                    self.stats['evictions'] += 1
    
    def _check_memory_limit(self):
        """Check and enforce memory cache limits"""
        with self.memory_lock:
            total_size = sum(
                self._estimate_memory_size(obj) 
                for obj in self.memory_cache.values()
            )
            
            if total_size > self.memory_limit_bytes:
                self._evict_memory_cache()
    
    def get(self, namespace: str, identifier: str, params: Dict = None) -> Optional[Any]:
        """Get item from cache with intelligent fallback"""
        key = self._generate_key(namespace, identifier, params)
        
        # L1: Memory cache
        with self.memory_lock:
            if key in self.memory_cache:
                self.memory_access_times[key] = time.time()
                self.stats['memory_hits'] += 1
                return self.memory_cache[key]
            else:
                self.stats['memory_misses'] += 1
        
        # L2: Disk cache
        if self.disk_cache:
            try:
                cached_data = self.disk_cache.get(key)
                if cached_data is not None:
                    data = self._deserialize_data(cached_data)
                    
                    # Promote to memory cache
                    with self.memory_lock:
                        self.memory_cache[key] = data
                        self.memory_access_times[key] = time.time()
                    
                    self.stats['disk_hits'] += 1
                    return data
                else:
                    self.stats['disk_misses'] += 1
            except Exception as e:
                print(f"⚠️ Disk cache error: {e}")
                self.stats['disk_misses'] += 1
        
        # L3: Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    data = self._deserialize_data(cached_data)
                    
                    # Promote to memory and disk cache
                    with self.memory_lock:
                        self.memory_cache[key] = data
                        self.memory_access_times[key] = time.time()
                    
                    if self.disk_cache:
                        self.disk_cache.set(key, self._serialize_data(data))
                    
                    self.stats['redis_hits'] += 1
                    return data
                else:
                    self.stats['redis_misses'] += 1
            except Exception as e:
                print(f"⚠️ Redis cache error: {e}")
                self.stats['redis_misses'] += 1
        
        return None
    
    def set(self, namespace: str, identifier: str, data: Any, 
            ttl: int = 3600, params: Dict = None, dependencies: List[str] = None):
        """Set item in cache with TTL and dependencies"""
        key = self._generate_key(namespace, identifier, params)
        
        # Store metadata
        self.metadata_cache[key] = {
            'created_at': time.time(),
            'ttl': ttl,
            'namespace': namespace,
            'dependencies': dependencies or []
        }
        
        # Track dependencies
        if dependencies:
            for dep in dependencies:
                if dep not in self.dependency_graph:
                    self.dependency_graph[dep] = []
                self.dependency_graph[dep].append(key)
        
        # Serialize data once
        serialized_data = self._serialize_data(data)
        
        # L1: Memory cache
        with self.memory_lock:
            self.memory_cache[key] = data
            self.memory_access_times[key] = time.time()
            self._check_memory_limit()
        
        # L2: Disk cache
        if self.disk_cache:
            try:
                self.disk_cache.set(key, serialized_data, expire=ttl)
            except Exception as e:
                print(f"⚠️ Disk cache write error: {e}")
        
        # L3: Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, serialized_data)
            except Exception as e:
                print(f"⚠️ Redis cache write error: {e}")
    
    def invalidate(self, namespace: str = None, identifier: str = None, 
                   dependency: str = None):
        """Intelligent cache invalidation"""
        keys_to_invalidate = []
        
        if dependency and dependency in self.dependency_graph:
            # Invalidate all keys that depend on this
            keys_to_invalidate.extend(self.dependency_graph[dependency])
            del self.dependency_graph[dependency]
        
        if namespace and identifier:
            # Invalidate specific key
            key = self._generate_key(namespace, identifier)
            keys_to_invalidate.append(key)
        elif namespace:
            # Invalidate all keys in namespace
            keys_to_invalidate.extend([
                key for key in self.metadata_cache.keys()
                if self.metadata_cache[key]['namespace'] == namespace
            ])
        
        # Remove from all cache levels
        for key in keys_to_invalidate:
            # Memory cache
            with self.memory_lock:
                self.memory_cache.pop(key, None)
                self.memory_access_times.pop(key, None)
            
            # Disk cache
            if self.disk_cache:
                try:
                    self.disk_cache.delete(key)
                except:
                    pass
            
            # Redis cache
            if self.redis_client:
                try:
                    self.redis_client.delete(key)
                except:
                    pass
            
            # Metadata
            self.metadata_cache.pop(key, None)
    
    def cleanup_expired(self):
        """Remove expired items from cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, metadata in self.metadata_cache.items():
            if current_time - metadata['created_at'] > metadata['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.invalidate(key=key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = (
            self.stats['memory_hits'] + self.stats['memory_misses']
        )
        
        memory_hit_rate = (
            (self.stats['memory_hits'] / total_requests * 100) 
            if total_requests > 0 else 0
        )
        
        # Memory usage
        with self.memory_lock:
            memory_usage = sum(
                self._estimate_memory_size(obj) 
                for obj in self.memory_cache.values()
            )
        
        return {
            **self.stats,
            'memory_hit_rate': memory_hit_rate,
            'memory_usage_mb': memory_usage / (1024 * 1024),
            'memory_items': len(self.memory_cache),
            'total_keys': len(self.metadata_cache)
        }
    
    def print_stats(self):
        """Print cache performance statistics"""
        stats = self.get_stats()
        
        print("\n📊 SMART CACHE PERFORMANCE")
        print("=" * 40)
        print(f"Memory Hit Rate: {stats['memory_hit_rate']:.1f}%")
        print(f"Memory Usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"Memory Items: {stats['memory_items']}")
        print(f"Total Cached Keys: {stats['total_keys']}")
        print(f"Cache Evictions: {stats['evictions']}")
        print(f"Compressions: {stats['compressions']}")
        
        print("\nCache Level Performance:")
        print(f"  L1 (Memory): {stats['memory_hits']} hits, {stats['memory_misses']} misses")
        print(f"  L2 (Disk): {stats['disk_hits']} hits, {stats['disk_misses']} misses")
        print(f"  L3 (Redis): {stats['redis_hits']} hits, {stats['redis_misses']} misses")

class StockDataCache(SmartCache):
    """
    Specialized cache for stock market data with intelligent invalidation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.market_hours = {
            'open': {'hour': 9, 'minute': 30},
            'close': {'hour': 16, 'minute': 0}
        }
    
    def cache_stock_data(self, symbol: str, data: pd.DataFrame, 
                        data_type: str = "prices", timeframe: str = "daily"):
        """Cache stock data with smart expiration"""
        
        # Determine TTL based on data type and market hours
        ttl = self._get_smart_ttl(data_type, timeframe)
        
        # Dependencies for invalidation
        dependencies = [f"market_data:{symbol}", f"data_type:{data_type}"]
        
        self.set(
            namespace="stock_data",
            identifier=f"{symbol}_{data_type}_{timeframe}",
            data=data,
            ttl=ttl,
            dependencies=dependencies
        )
    
    def get_stock_data(self, symbol: str, data_type: str = "prices", 
                      timeframe: str = "daily") -> Optional[pd.DataFrame]:
        """Get cached stock data"""
        return self.get(
            namespace="stock_data",
            identifier=f"{symbol}_{data_type}_{timeframe}"
        )
    
    def cache_prediction(self, symbol: str, prediction_data: Dict, 
                        model_name: str, horizon: str):
        """Cache prediction results with model dependencies"""
        
        ttl = 300  # 5 minutes for predictions
        dependencies = [f"model:{model_name}", f"symbol:{symbol}"]
        
        self.set(
            namespace="predictions",
            identifier=f"{symbol}_{model_name}_{horizon}",
            data=prediction_data,
            ttl=ttl,
            dependencies=dependencies
        )
    
    def get_prediction(self, symbol: str, model_name: str, 
                      horizon: str) -> Optional[Dict]:
        """Get cached prediction"""
        return self.get(
            namespace="predictions",
            identifier=f"{symbol}_{model_name}_{horizon}"
        )
    
    def invalidate_symbol(self, symbol: str):
        """Invalidate all data for a specific symbol"""
        self.invalidate(dependency=f"symbol:{symbol}")
        self.invalidate(dependency=f"market_data:{symbol}")
    
    def invalidate_model_predictions(self, model_name: str):
        """Invalidate all predictions from a specific model"""
        self.invalidate(dependency=f"model:{model_name}")
    
    def _get_smart_ttl(self, data_type: str, timeframe: str) -> int:
        """Calculate smart TTL based on data type and market schedule"""
        
        base_ttls = {
            "intraday": 60,      # 1 minute
            "daily": 3600,       # 1 hour
            "weekly": 86400,     # 1 day
            "monthly": 604800    # 1 week
        }
        
        # Get base TTL
        ttl = base_ttls.get(timeframe, 3600)
        
        # Adjust based on market hours
        now = datetime.now()
        
        # During market hours, use shorter TTL for real-time data
        if self._is_market_hours(now) and data_type in ["prices", "volume"]:
            ttl = min(ttl, 300)  # Max 5 minutes during market hours
        
        return ttl
    
    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if current time is during market hours"""
        # Monday = 0, Sunday = 6
        if dt.weekday() >= 5:  # Weekend
            return False
        
        market_open = dt.replace(
            hour=self.market_hours['open']['hour'],
            minute=self.market_hours['open']['minute'],
            second=0,
            microsecond=0
        )
        
        market_close = dt.replace(
            hour=self.market_hours['close']['hour'],
            minute=self.market_hours['close']['minute'],
            second=0,
            microsecond=0
        )
        
        return market_open <= dt <= market_close

def smart_cache_decorator(
    namespace: str,
    ttl: int = 3600,
    dependencies: List[str] = None,
    cache_instance: SmartCache = None
):
    """Decorator for automatic caching with smart invalidation"""
    
    if cache_instance is None:
        cache_instance = SmartCache()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function signature
            func_params = {
                'args': str(args),
                'kwargs': sorted(kwargs.items())
            }
            
            identifier = func.__name__
            
            # Try to get from cache
            cached_result = cache_instance.get(namespace, identifier, func_params)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(
                namespace=namespace,
                identifier=identifier,
                data=result,
                ttl=ttl,
                params=func_params,
                dependencies=dependencies
            )
            
            return result
        
        # Add cache management methods
        wrapper.invalidate_cache = lambda: cache_instance.invalidate(namespace, func.__name__)
        wrapper.cache_stats = cache_instance.get_stats
        
        return wrapper
    return decorator

# Example usage and testing
if __name__ == "__main__":
    print("🧠 Intelligent Caching System")
    print("=" * 40)
    
    # Initialize cache
    cache = StockDataCache(
        memory_limit_mb=256,
        disk_limit_gb=5
    )
    
    # Test stock data caching
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # Cache data
    print("📦 Caching stock data...")
    cache.cache_stock_data("AAPL", sample_data, "prices", "daily")
    
    # Retrieve data
    print("🔍 Retrieving cached data...")
    cached_data = cache.get_stock_data("AAPL", "prices", "daily")
    
    if cached_data is not None:
        print(f"✅ Successfully retrieved {len(cached_data)} rows")
    else:
        print("❌ Failed to retrieve cached data")
    
    # Test prediction caching
    prediction_data = {
        'predicted_price': 150.25,
        'confidence': 0.87,
        'model_accuracy': 0.92
    }
    
    cache.cache_prediction("AAPL", prediction_data, "xgboost", "1_day")
    cached_prediction = cache.get_prediction("AAPL", "xgboost", "1_day")
    
    print(f"✅ Prediction cached and retrieved: {cached_prediction}")
    
    # Print performance stats
    cache.print_stats()
    
    print("\n✅ Intelligent caching system ready!")
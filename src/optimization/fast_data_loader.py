"""
Ultra-Fast Data Loading and Preprocessing System
High-performance data pipeline for stock market data
"""

import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import pickle
import gzip
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    import psycopg2
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class FastDataLoader:
    """
    Ultra-fast data loading with multiple optimization strategies
    """
    
    def __init__(self, 
                 cache_dir: str = "/tmp/stock_data_cache",
                 max_workers: int = 8,
                 use_compression: bool = True,
                 preferred_format: str = "parquet"):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.use_compression = use_compression
        self.preferred_format = preferred_format
        
        # Performance tracking
        self.load_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize optimized backends
        self.backends = self._initialize_backends()
        
        print(f"⚡ FastDataLoader initialized with {len(self.backends)} backends")
    
    def _initialize_backends(self) -> Dict[str, bool]:
        """Initialize available optimization backends"""
        backends = {
            'polars': POLARS_AVAILABLE,
            'arrow': ARROW_AVAILABLE,
            'hdf5': HDF5_AVAILABLE,
            'vaex': VAEX_AVAILABLE,
            'database': DATABASE_AVAILABLE
        }
        
        for backend, available in backends.items():
            status = "✅" if available else "❌"
            print(f"  {status} {backend.title()}")
        
        return backends
    
    def load_stock_data_optimized(self, 
                                 symbol: str,
                                 start_date: str = None,
                                 end_date: str = None,
                                 interval: str = "1d") -> pd.DataFrame:
        """Load stock data with optimal performance strategy"""
        
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        
        # Try cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            self.cache_hits += 1
            return cached_data
        
        self.cache_misses += 1
        
        # Load data using best available method
        start_time = time.time()
        
        if self.backends['polars']:
            data = self._load_with_polars(symbol, start_date, end_date, interval)
        elif self.backends['arrow']:
            data = self._load_with_arrow(symbol, start_date, end_date, interval)
        else:
            data = self._load_with_pandas(symbol, start_date, end_date, interval)
        
        load_time = time.time() - start_time
        self.load_times[cache_key] = load_time
        
        # Cache for future use
        self._save_to_cache(cache_key, data)
        
        print(f"📊 Loaded {symbol} data: {len(data)} rows in {load_time:.4f}s")
        
        return data
    
    def batch_load_symbols(self, 
                          symbols: List[str],
                          start_date: str = None,
                          end_date: str = None,
                          interval: str = "1d",
                          parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """Load multiple symbols with parallel processing"""
        
        if not parallel or len(symbols) == 1:
            # Sequential loading
            results = {}
            for symbol in symbols:
                try:
                    results[symbol] = self.load_stock_data_optimized(
                        symbol, start_date, end_date, interval
                    )
                except Exception as e:
                    print(f"❌ Failed to load {symbol}: {e}")
            return results
        
        # Parallel loading
        print(f"🚀 Loading {len(symbols)} symbols in parallel...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self.load_stock_data_optimized,
                    symbol, start_date, end_date, interval
                ): symbol
                for symbol in symbols
            }
            
            results = {}
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    print(f"❌ Failed to load {symbol}: {e}")
        
        total_time = time.time() - start_time
        print(f"✅ Loaded {len(results)} symbols in {total_time:.2f}s")
        
        return results
    
    async def async_load_symbols(self, 
                                symbols: List[str],
                                start_date: str = None,
                                end_date: str = None,
                                interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Asynchronous data loading for maximum speed"""
        
        async def load_symbol_async(symbol: str) -> Tuple[str, pd.DataFrame]:
            try:
                # Run synchronous load in thread pool
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    self.load_stock_data_optimized,
                    symbol, start_date, end_date, interval
                )
                return symbol, data
            except Exception as e:
                print(f"❌ Async load failed for {symbol}: {e}")
                return symbol, pd.DataFrame()
        
        print(f"🔄 Async loading {len(symbols)} symbols...")
        start_time = time.time()
        
        # Create async tasks
        tasks = [load_symbol_async(symbol) for symbol in symbols]
        
        # Execute concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {}
        for result in results_list:
            if isinstance(result, tuple) and len(result) == 2:
                symbol, data = result
                if not data.empty:
                    results[symbol] = data
        
        total_time = time.time() - start_time
        print(f"✅ Async loaded {len(results)} symbols in {total_time:.2f}s")
        
        return results
    
    def _load_with_polars(self, symbol: str, start_date: str, 
                         end_date: str, interval: str) -> pd.DataFrame:
        """Load data using Polars for maximum performance"""
        
        # Use yfinance to get data, then convert to Polars
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            return pd.DataFrame()
        
        # Convert to Polars DataFrame for fast processing
        pl_df = pl.from_pandas(data.reset_index())
        
        # Optimize data types
        pl_df = self._optimize_polars_dtypes(pl_df)
        
        # Convert back to pandas
        return pl_df.to_pandas().set_index('Date')
    
    def _load_with_arrow(self, symbol: str, start_date: str, 
                        end_date: str, interval: str) -> pd.DataFrame:
        """Load data using PyArrow for columnar efficiency"""
        
        # Use yfinance to get data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            return pd.DataFrame()
        
        # Convert to Arrow table for efficient processing
        table = pa.Table.from_pandas(data.reset_index())
        
        # Optimize data types
        optimized_schema = self._optimize_arrow_schema(table.schema)
        table = table.cast(optimized_schema)
        
        # Convert back to pandas
        return table.to_pandas().set_index('Date')
    
    def _load_with_pandas(self, symbol: str, start_date: str, 
                         end_date: str, interval: str) -> pd.DataFrame:
        """Fallback pandas loading with optimization"""
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            return pd.DataFrame()
        
        # Optimize pandas dtypes
        data = self._optimize_pandas_dtypes(data)
        
        return data
    
    def _optimize_polars_dtypes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimize Polars data types for memory efficiency"""
        
        optimization_map = {
            'Open': pl.Float32,
            'High': pl.Float32,
            'Low': pl.Float32,
            'Close': pl.Float32,
            'Volume': pl.UInt32,
        }
        
        for col, dtype in optimization_map.items():
            if col in df.columns:
                try:
                    df = df.with_columns(pl.col(col).cast(dtype))
                except:
                    pass  # Skip if conversion fails
        
        return df
    
    def _optimize_arrow_schema(self, schema: pa.Schema) -> pa.Schema:
        """Optimize Arrow schema for better performance"""
        
        new_fields = []
        for field in schema:
            if field.name in ['Open', 'High', 'Low', 'Close']:
                new_fields.append(pa.field(field.name, pa.float32()))
            elif field.name == 'Volume':
                new_fields.append(pa.field(field.name, pa.uint32()))
            else:
                new_fields.append(field)
        
        return pa.schema(new_fields)
    
    def _optimize_pandas_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas data types for memory efficiency"""
        
        float_cols = ['Open', 'High', 'Low', 'Close']
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(np.uint32)
        
        return df
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache with multiple format support"""
        
        # Try different cache formats in order of preference
        cache_formats = []
        
        if self.preferred_format == "parquet" and ARROW_AVAILABLE:
            cache_formats.append(("parquet", self._load_parquet_cache))
        
        if self.preferred_format == "hdf5" and HDF5_AVAILABLE:
            cache_formats.append(("h5", self._load_hdf5_cache))
        
        # Fallback formats
        cache_formats.extend([
            ("pkl.gz", self._load_pickle_cache),
            ("pkl", self._load_pickle_cache)
        ])
        
        for ext, load_func in cache_formats:
            cache_file = self.cache_dir / f"{cache_key}.{ext}"
            if cache_file.exists():
                try:
                    return load_func(cache_file)
                except Exception as e:
                    print(f"⚠️ Cache load error ({ext}): {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache with optimal format"""
        
        try:
            if self.preferred_format == "parquet" and ARROW_AVAILABLE:
                cache_file = self.cache_dir / f"{cache_key}.parquet"
                data.to_parquet(cache_file, compression='snappy', index=True)
            
            elif self.preferred_format == "hdf5" and HDF5_AVAILABLE:
                cache_file = self.cache_dir / f"{cache_key}.h5"
                data.to_hdf(cache_file, key='data', mode='w', complevel=9)
            
            else:
                # Fallback to compressed pickle
                cache_file = self.cache_dir / f"{cache_key}.pkl.gz"
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            
        except Exception as e:
            print(f"⚠️ Cache save error: {e}")
    
    def _load_parquet_cache(self, cache_file: Path) -> pd.DataFrame:
        """Load from Parquet cache"""
        return pd.read_parquet(cache_file)
    
    def _load_hdf5_cache(self, cache_file: Path) -> pd.DataFrame:
        """Load from HDF5 cache"""
        return pd.read_hdf(cache_file, key='data')
    
    def _load_pickle_cache(self, cache_file: Path) -> pd.DataFrame:
        """Load from pickle cache"""
        if cache_file.suffix == '.gz':
            with gzip.open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        cache_size = sum(
            f.stat().st_size for f in self.cache_dir.glob('*') if f.is_file()
        ) / (1024 * 1024)  # MB
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size_mb': cache_size,
            'cached_files': len(list(self.cache_dir.glob('*')))
        }
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear old cache files"""
        
        cutoff_time = time.time() - (older_than_days * 24 * 3600)
        removed_count = 0
        
        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Error removing cache file {cache_file}: {e}")
        
        print(f"🧹 Removed {removed_count} old cache files")

class FastPreprocessor:
    """
    Ultra-fast data preprocessing with vectorized operations
    """
    
    def __init__(self, use_numba: bool = True):
        self.use_numba = use_numba
        
        try:
            import numba
            self.numba_available = True
        except ImportError:
            self.numba_available = False
            self.use_numba = False
    
    def preprocess_batch(self, 
                        data_dict: Dict[str, pd.DataFrame],
                        operations: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Preprocess multiple DataFrames with vectorized operations"""
        
        if operations is None:
            operations = ['clean', 'normalize', 'feature_engineer']
        
        print(f"🔧 Preprocessing {len(data_dict)} datasets...")
        start_time = time.time()
        
        # Use parallel processing for preprocessing
        with ProcessPoolExecutor(max_workers=min(8, len(data_dict))) as executor:
            future_to_symbol = {
                executor.submit(self._preprocess_single, symbol, df, operations): symbol
                for symbol, df in data_dict.items()
            }
            
            results = {}
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    print(f"❌ Preprocessing failed for {symbol}: {e}")
        
        processing_time = time.time() - start_time
        print(f"✅ Preprocessing completed in {processing_time:.4f}s")
        
        return results
    
    def _preprocess_single(self, 
                          symbol: str, 
                          df: pd.DataFrame, 
                          operations: List[str]) -> pd.DataFrame:
        """Preprocess single DataFrame"""
        
        result_df = df.copy()
        
        for operation in operations:
            if operation == 'clean':
                result_df = self._clean_data(result_df)
            elif operation == 'normalize':
                result_df = self._normalize_data(result_df)
            elif operation == 'feature_engineer':
                result_df = self._fast_feature_engineering(result_df)
        
        return result_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast data cleaning with vectorized operations"""
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Handle missing values efficiently
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method (vectorized)
        for col in numeric_cols:
            if col != 'Volume':  # Don't remove volume outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast data normalization"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Use robust scaling (less sensitive to outliers)
        for col in numeric_cols:
            if col != 'Volume':  # Volume needs different treatment
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                df[f'{col}_normalized'] = (df[col] - median) / (mad + 1e-8)
        
        # Log-transform volume
        if 'Volume' in df.columns:
            df['Volume_log'] = np.log1p(df['Volume'])
        
        return df
    
    def _fast_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast feature engineering with vectorized operations"""
        
        if 'Close' not in df.columns:
            return df
        
        prices = df['Close'].values
        
        # Fast moving averages using convolution
        for window in [5, 10, 20]:
            df[f'SMA_{window}'] = self._fast_sma(prices, window)
        
        # Price momentum (vectorized)
        for period in [1, 5, 10]:
            df[f'momentum_{period}'] = df['Close'].pct_change(period)
        
        # Volatility (rolling standard deviation)
        for window in [10, 20]:
            df[f'volatility_{window}'] = df['Close'].rolling(window).std()
        
        # Price position within recent range
        for window in [10, 20]:
            rolling_min = df['Close'].rolling(window).min()
            rolling_max = df['Close'].rolling(window).max()
            df[f'price_position_{window}'] = (
                (df['Close'] - rolling_min) / (rolling_max - rolling_min)
            )
        
        return df
    
    def _fast_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Ultra-fast simple moving average using convolution"""
        
        if self.use_numba and self.numba_available:
            return self._numba_sma(prices, window)
        else:
            # Use pandas rolling for fallback
            return pd.Series(prices).rolling(window).mean().values
    
    def _numba_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Numba-accelerated SMA calculation"""

        if not NUMBA_AVAILABLE:
            return pd.Series(prices).rolling(window).mean().values

        @numba.jit(nopython=True)
        def _sma_kernel(prices, window):
            n = len(prices)
            result = np.empty(n)
            result[:window-1] = np.nan
            
            for i in range(window-1, n):
                result[i] = np.mean(prices[i-window+1:i+1])
            
            return result
        
        return _sma_kernel(prices, window)

class DataPipelineOptimizer:
    """
    End-to-end data pipeline optimization
    """
    
    def __init__(self):
        self.loader = FastDataLoader()
        self.preprocessor = FastPreprocessor()
        
        # Performance metrics
        self.pipeline_times = {}
    
    def run_optimized_pipeline(self, 
                              symbols: List[str],
                              start_date: str = None,
                              end_date: str = None,
                              preprocessing_ops: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Run complete optimized data pipeline"""
        
        print("🚀 Running optimized data pipeline...")
        total_start_time = time.time()
        
        # Step 1: Load data
        load_start_time = time.time()
        raw_data = self.loader.batch_load_symbols(symbols, start_date, end_date)
        load_time = time.time() - load_start_time
        
        # Step 2: Preprocess data
        preprocess_start_time = time.time()
        processed_data = self.preprocessor.preprocess_batch(raw_data, preprocessing_ops)
        preprocess_time = time.time() - preprocess_start_time
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        # Store performance metrics
        self.pipeline_times = {
            'load_time': load_time,
            'preprocess_time': preprocess_time,
            'total_time': total_time,
            'symbols_count': len(processed_data),
            'throughput': len(processed_data) / total_time
        }
        
        print(f"✅ Pipeline completed in {total_time:.2f}s")
        print(f"   Loading: {load_time:.2f}s")
        print(f"   Preprocessing: {preprocess_time:.2f}s")
        print(f"   Throughput: {self.pipeline_times['throughput']:.1f} symbols/sec")
        
        return processed_data
    
    async def run_async_pipeline(self, 
                                symbols: List[str],
                                start_date: str = None,
                                end_date: str = None,
                                preprocessing_ops: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Run asynchronous data pipeline for maximum speed"""
        
        print("🔄 Running async data pipeline...")
        total_start_time = time.time()
        
        # Step 1: Async load data
        raw_data = await self.loader.async_load_symbols(symbols, start_date, end_date)
        
        # Step 2: Preprocess data (still sync but parallel)
        processed_data = self.preprocessor.preprocess_batch(raw_data, preprocessing_ops)
        
        total_time = time.time() - total_start_time
        
        print(f"✅ Async pipeline completed in {total_time:.2f}s")
        print(f"   Throughput: {len(processed_data)/total_time:.1f} symbols/sec")
        
        return processed_data
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        cache_stats = self.loader.get_cache_stats()
        
        return {
            'pipeline_performance': self.pipeline_times,
            'cache_performance': cache_stats,
            'optimization_backends': self.loader.backends
        }

# Example usage and benchmarking
if __name__ == "__main__":
    print("⚡ Ultra-Fast Data Loading System")
    print("=" * 50)
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Initialize pipeline
    pipeline = DataPipelineOptimizer()
    
    # Test regular pipeline
    print("\n🔬 Testing regular pipeline...")
    start_time = time.time()
    results = pipeline.run_optimized_pipeline(
        symbols=test_symbols,
        start_date='2023-01-01',
        end_date='2024-01-01',
        preprocessing_ops=['clean', 'normalize', 'feature_engineer']
    )
    regular_time = time.time() - start_time
    
    print(f"Regular pipeline: {len(results)} symbols in {regular_time:.2f}s")
    
    # Test async pipeline
    print("\n🔄 Testing async pipeline...")
    start_time = time.time()
    
    async def test_async():
        return await pipeline.run_async_pipeline(
            symbols=test_symbols,
            start_date='2023-01-01', 
            end_date='2024-01-01',
            preprocessing_ops=['clean', 'normalize']
        )
    
    async_results = asyncio.run(test_async())
    async_time = time.time() - start_time
    
    print(f"Async pipeline: {len(async_results)} symbols in {async_time:.2f}s")
    print(f"Speedup: {regular_time/async_time:.1f}x faster")
    
    # Performance report
    print("\n📊 PERFORMANCE REPORT")
    print("=" * 30)
    report = pipeline.get_performance_report()
    
    for category, metrics in report.items():
        print(f"\n{category.upper()}:")
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    # Cache statistics
    cache_stats = pipeline.loader.get_cache_stats()
    print(f"\nCache hit rate: {cache_stats['hit_rate']:.1f}%")
    print(f"Cache size: {cache_stats['cache_size_mb']:.2f} MB")
    
    print(f"\n🎯 Fast data loading system ready!")
    
    # Example of individual optimized loading
    print(f"\n📈 Testing individual symbol optimization...")
    
    loader = FastDataLoader(preferred_format="parquet")
    
    # First load (cache miss)
    start_time = time.time()
    data1 = loader.load_stock_data_optimized('AAPL', '2023-01-01', '2024-01-01')
    first_load_time = time.time() - start_time
    
    # Second load (cache hit)
    start_time = time.time() 
    data2 = loader.load_stock_data_optimized('AAPL', '2023-01-01', '2024-01-01')
    second_load_time = time.time() - start_time
    
    print(f"First load: {first_load_time:.4f}s")
    print(f"Second load (cached): {second_load_time:.4f}s")
    print(f"Cache speedup: {first_load_time/second_load_time:.1f}x")
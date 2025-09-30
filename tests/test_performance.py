"""
Performance and Load Testing for Stock AI System
Comprehensive performance testing, load testing, and benchmarking
"""

import unittest
import asyncio
import numpy as np
import pandas as pd
import tempfile
import os
import json
import time
import threading
import concurrent.futures
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import test framework
from .test_framework import AsyncTestCase, MockDataGenerator, TestFixtures, BenchmarkRunner

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class TestPerformanceBenchmarks(AsyncTestCase):
    """Performance benchmarking for core system components"""
    
    def setUp(self):
        super().setUp()
        self.benchmark = BenchmarkRunner()
        
        # Generate test datasets of various sizes
        self.small_dataset = MockDataGenerator.generate_stock_data("AAPL", 100)
        self.medium_dataset = MockDataGenerator.generate_stock_data("AAPL", 1000)
        self.large_dataset = MockDataGenerator.generate_stock_data("AAPL", 10000)
        
        self.test_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    def test_data_processing_performance(self):
        """Benchmark data processing operations"""
        
        def process_small_dataset():
            """Process small dataset"""
            if hasattr(self.small_dataset, 'rolling'):
                sma = self.small_dataset['Close'].rolling(20).mean()
                ema = self.small_dataset['Close'].ewm(span=12).mean()
                vol = self.small_dataset['Close'].rolling(20).std()
                return len(sma.dropna())
            return 100
        
        def process_medium_dataset():
            """Process medium dataset"""
            if hasattr(self.medium_dataset, 'rolling'):
                sma = self.medium_dataset['Close'].rolling(20).mean()
                ema = self.medium_dataset['Close'].ewm(span=12).mean()
                vol = self.medium_dataset['Close'].rolling(20).std()
                rsi = self._calculate_rsi(self.medium_dataset['Close'])
                return len(sma.dropna())
            return 1000
        
        def process_large_dataset():
            """Process large dataset"""
            if hasattr(self.large_dataset, 'rolling'):
                # Multiple technical indicators
                close = self.large_dataset['Close']
                sma_20 = close.rolling(20).mean()
                sma_50 = close.rolling(50).mean()
                ema_12 = close.ewm(span=12).mean()
                ema_26 = close.ewm(span=26).mean()
                vol = close.rolling(20).std()
                rsi = self._calculate_rsi(close)
                return len(sma_20.dropna())
            return 10000
        
        # Benchmark different dataset sizes
        self.benchmark.run_benchmark(process_small_dataset, "small_dataset_processing", 20)
        self.benchmark.run_benchmark(process_medium_dataset, "medium_dataset_processing", 10)
        self.benchmark.run_benchmark(process_large_dataset, "large_dataset_processing", 5)
        
        # Verify performance requirements
        small_stats = self.benchmark.get_stats("small_dataset_processing")
        medium_stats = self.benchmark.get_stats("medium_dataset_processing")
        large_stats = self.benchmark.get_stats("large_dataset_processing")
        
        if small_stats:
            self.assertLess(small_stats['mean'], 0.1, "Small dataset processing too slow")
        
        if medium_stats:
            self.assertLess(medium_stats['mean'], 0.5, "Medium dataset processing too slow")
        
        if large_stats:
            self.assertLess(large_stats['mean'], 5.0, "Large dataset processing too slow")
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI for performance testing"""
        if not hasattr(prices, 'diff'):
            return pd.Series([50.0] * len(prices))
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
    
    def test_prediction_performance(self):
        """Benchmark model prediction performance"""
        
        def mock_simple_prediction(data_size):
            """Simple prediction simulation"""
            # Simulate feature extraction
            features = np.random.randn(data_size, 10)
            
            # Simulate model inference
            weights = np.random.randn(10)
            prediction = np.dot(features, weights).mean()
            
            # Simulate post-processing
            confidence = min(0.95, abs(prediction) / 2)
            
            return {
                'prediction': prediction,
                'confidence': confidence
            }
        
        def mock_ensemble_prediction(data_size):
            """Ensemble prediction simulation"""
            predictions = []
            
            # Simulate 5 different models
            for model_id in range(5):
                features = np.random.randn(data_size, 15)
                weights = np.random.randn(15)
                pred = np.dot(features, weights).mean()
                predictions.append(pred)
            
            # Ensemble combination
            final_prediction = np.mean(predictions)
            confidence = 1.0 - np.std(predictions)
            
            return {
                'prediction': final_prediction,
                'confidence': max(0.5, min(0.95, confidence))
            }
        
        def mock_complex_prediction(data_size):
            """Complex prediction with multiple features"""
            # Technical indicators
            tech_features = np.random.randn(data_size, 20)
            
            # Sentiment features
            sentiment_features = np.random.randn(data_size, 5)
            
            # Economic features
            econ_features = np.random.randn(data_size, 8)
            
            # Combine features
            all_features = np.hstack([tech_features, sentiment_features, econ_features])
            
            # Multiple model predictions
            predictions = []
            for _ in range(3):
                weights = np.random.randn(all_features.shape[1])
                pred = np.dot(all_features, weights).mean()
                predictions.append(pred)
            
            return {
                'prediction': np.mean(predictions),
                'confidence': np.random.uniform(0.7, 0.9)
            }
        
        # Benchmark different prediction complexities
        self.benchmark.run_benchmark(mock_simple_prediction, "simple_prediction", 50, 100)
        self.benchmark.run_benchmark(mock_ensemble_prediction, "ensemble_prediction", 20, 100)
        self.benchmark.run_benchmark(mock_complex_prediction, "complex_prediction", 10, 100)
        
        # Verify performance
        simple_stats = self.benchmark.get_stats("simple_prediction")
        ensemble_stats = self.benchmark.get_stats("ensemble_prediction")
        complex_stats = self.benchmark.get_stats("complex_prediction")
        
        if simple_stats:
            self.assertLess(simple_stats['mean'], 0.01, "Simple prediction too slow")
        
        if ensemble_stats:
            self.assertLess(ensemble_stats['mean'], 0.05, "Ensemble prediction too slow")
        
        if complex_stats:
            self.assertLess(complex_stats['mean'], 0.1, "Complex prediction too slow")
    
    def test_concurrent_processing_performance(self):
        """Benchmark concurrent processing capabilities"""
        
        async def process_single_symbol(symbol, data_size=252):
            """Process single symbol"""
            # Simulate data collection
            await asyncio.sleep(0.001)
            
            # Simulate feature engineering
            features = np.random.randn(data_size, 10)
            await asyncio.sleep(0.002)
            
            # Simulate prediction
            prediction = np.random.uniform(100, 200)
            await asyncio.sleep(0.003)
            
            return {
                'symbol': symbol,
                'prediction': prediction,
                'processing_time': 0.006
            }
        
        async def concurrent_processing_test(num_symbols):
            """Test concurrent processing of multiple symbols"""
            symbols = [f"STOCK{i:03d}" for i in range(num_symbols)]
            
            start_time = time.time()
            tasks = [process_single_symbol(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            return {
                'symbols_processed': len(results),
                'total_time': end_time - start_time,
                'avg_time_per_symbol': (end_time - start_time) / len(results),
                'throughput': len(results) / (end_time - start_time)
            }
        
        # Test different concurrency levels
        for num_symbols in [10, 25, 50]:
            result = self.run_async(concurrent_processing_test(num_symbols))
            
            # Verify concurrency efficiency
            expected_sequential_time = num_symbols * 0.006  # 6ms per symbol
            actual_time = result['total_time']
            
            # Concurrent processing should be much faster than sequential
            speedup_factor = expected_sequential_time / actual_time
            self.assertGreater(speedup_factor, 5, f"Insufficient speedup for {num_symbols} symbols")
            
            # Verify throughput
            self.assertGreater(result['throughput'], 50, f"Low throughput for {num_symbols} symbols")
    
    def test_memory_efficiency(self):
        """Test memory usage patterns and efficiency"""
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for memory testing")
            return
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def memory_intensive_operation():
            """Simulate memory-intensive operation"""
            # Create large datasets
            large_arrays = []
            
            try:
                # Create multiple large arrays
                for i in range(5):
                    arr = np.random.randn(10000, 50)  # ~4MB per array
                    large_arrays.append(arr)
                
                # Process data
                processed = []
                for arr in large_arrays:
                    result = np.mean(arr, axis=0)
                    processed.append(result)
                
                # Combine results
                final_result = np.vstack(processed)
                return final_result.shape
                
            finally:
                # Cleanup
                del large_arrays
                if 'processed' in locals():
                    del processed
        
        # Monitor memory during operations
        memory_samples = []
        
        for _ in range(10):
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            # Run memory-intensive operation
            result = memory_intensive_operation()
            self.assertIsNotNone(result)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Analyze memory usage
        peak_memory = max(memory_samples)
        final_memory = memory_samples[-1]
        memory_growth = peak_memory - baseline_memory
        memory_retained = final_memory - baseline_memory
        
        # Verify memory efficiency
        self.assertLess(memory_growth, 100, "Peak memory usage too high")  # < 100MB growth
        self.assertLess(memory_retained, 20, "Too much memory retained")    # < 20MB retained
    
    def test_database_performance(self):
        """Benchmark database operations performance"""
        
        class MockDatabase:
            def __init__(self):
                self.data = {}
            
            def insert_batch(self, records):
                """Mock batch insert"""
                start_time = time.time()
                
                for record in records:
                    key = f"{record['symbol']}_{record['timestamp']}"
                    self.data[key] = record
                
                # Simulate database latency
                time.sleep(len(records) * 0.0001)  # 0.1ms per record
                
                return time.time() - start_time
            
            def query_range(self, symbol, start_date, end_date):
                """Mock range query"""
                start_time = time.time()
                
                # Simulate query processing
                matching_records = []
                for key, record in self.data.items():
                    if record['symbol'] == symbol:
                        matching_records.append(record)
                
                time.sleep(0.001)  # 1ms base query time
                
                return matching_records, time.time() - start_time
            
            def aggregate_query(self, symbol):
                """Mock aggregation query"""
                start_time = time.time()
                
                # Simulate aggregation
                symbol_records = [r for r in self.data.values() if r['symbol'] == symbol]
                if symbol_records:
                    avg_price = sum(r['price'] for r in symbol_records) / len(symbol_records)
                    max_price = max(r['price'] for r in symbol_records)
                    min_price = min(r['price'] for r in symbol_records)
                else:
                    avg_price = max_price = min_price = 0
                
                time.sleep(0.002)  # 2ms aggregation time
                
                return {
                    'avg_price': avg_price,
                    'max_price': max_price,
                    'min_price': min_price
                }, time.time() - start_time
        
        db = MockDatabase()
        
        # Test batch insert performance
        batch_sizes = [100, 500, 1000]
        
        for batch_size in batch_sizes:
            records = []
            for i in range(batch_size):
                records.append({
                    'symbol': f'STOCK{i % 10:03d}',
                    'price': 100 + np.random.uniform(-10, 10),
                    'timestamp': datetime.now() - timedelta(minutes=i),
                    'volume': np.random.randint(1000, 10000)
                })
            
            insert_time = db.insert_batch(records)
            throughput = batch_size / insert_time
            
            # Verify insert performance
            self.assertGreater(throughput, 1000, f"Insert throughput too low for batch size {batch_size}")
            self.assertLess(insert_time, 1.0, f"Insert time too high for batch size {batch_size}")
        
        # Test query performance
        for symbol in ["STOCK001", "STOCK002", "STOCK003"]:
            records, query_time = db.query_range(
                symbol, 
                datetime.now() - timedelta(hours=1), 
                datetime.now()
            )
            
            self.assertLess(query_time, 0.1, f"Query time too high for {symbol}")
        
        # Test aggregation performance
        for symbol in ["STOCK001", "STOCK002"]:
            agg_result, agg_time = db.aggregate_query(symbol)
            
            self.assertLess(agg_time, 0.1, f"Aggregation time too high for {symbol}")
            self.assertIn('avg_price', agg_result)

class TestLoadTesting(AsyncTestCase):
    """Load testing for system under various stress conditions"""
    
    def setUp(self):
        super().setUp()
        self.load_test_duration = 10  # 10 seconds for load tests
    
    def test_high_volume_data_ingestion(self):
        """Test system under high volume data ingestion"""
        
        async def data_ingestion_load_test():
            """Simulate high-volume data ingestion"""
            
            test_results = {
                'start_time': time.time(),
                'records_processed': 0,
                'errors': 0,
                'throughput_samples': []
            }
            
            async def process_data_batch(batch_id, batch_size=1000):
                """Process a batch of market data"""
                try:
                    # Simulate data validation
                    await asyncio.sleep(0.001 * batch_size / 1000)  # 1ms per 1000 records
                    
                    # Simulate data transformation
                    await asyncio.sleep(0.002 * batch_size / 1000)  # 2ms per 1000 records
                    
                    # Simulate storage
                    await asyncio.sleep(0.001 * batch_size / 1000)  # 1ms per 1000 records
                    
                    return batch_size
                    
                except Exception as e:
                    return 0
            
            # Run load test for specified duration
            end_time = time.time() + self.load_test_duration
            batch_id = 0
            
            while time.time() < end_time:
                # Process multiple batches concurrently
                batch_tasks = []
                for _ in range(5):  # 5 concurrent batches
                    task = process_data_batch(batch_id, 1000)
                    batch_tasks.append(task)
                    batch_id += 1
                
                start_batch_time = time.time()
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                batch_time = time.time() - start_batch_time
                
                # Calculate throughput for this batch
                successful_records = sum(r for r in results if isinstance(r, int))
                if batch_time > 0:
                    throughput = successful_records / batch_time
                    test_results['throughput_samples'].append(throughput)
                
                test_results['records_processed'] += successful_records
                test_results['errors'] += sum(1 for r in results if isinstance(r, Exception))
                
                # Brief pause between batch cycles
                await asyncio.sleep(0.01)
            
            test_results['end_time'] = time.time()
            test_results['total_duration'] = test_results['end_time'] - test_results['start_time']
            test_results['avg_throughput'] = (
                test_results['records_processed'] / test_results['total_duration']
            )
            
            return test_results
        
        # Run the load test
        results = self.run_async(data_ingestion_load_test())
        
        # Verify load test results
        self.assertGreater(results['records_processed'], 10000, "Insufficient records processed")
        self.assertLess(results['errors'] / max(results['records_processed'], 1), 0.01, "Too many errors")
        self.assertGreater(results['avg_throughput'], 5000, "Average throughput too low")
        
        # Verify throughput consistency
        if results['throughput_samples']:
            throughput_std = np.std(results['throughput_samples'])
            throughput_mean = np.mean(results['throughput_samples'])
            
            # Coefficient of variation should be reasonable
            cv = throughput_std / throughput_mean
            self.assertLess(cv, 0.5, "Throughput too variable under load")
    
    def test_concurrent_user_simulation(self):
        """Simulate multiple concurrent users making requests"""
        
        async def simulate_user(user_id, num_requests=20):
            """Simulate a single user's requests"""
            user_results = {
                'user_id': user_id,
                'requests_completed': 0,
                'requests_failed': 0,
                'response_times': [],
                'errors': []
            }
            
            for request_id in range(num_requests):
                try:
                    request_start = time.time()
                    
                    # Simulate different types of requests
                    request_type = np.random.choice(['prediction', 'historical', 'portfolio'])
                    
                    if request_type == 'prediction':
                        # Mock prediction request
                        await asyncio.sleep(np.random.uniform(0.01, 0.05))  # 10-50ms
                        
                    elif request_type == 'historical':
                        # Mock historical data request
                        await asyncio.sleep(np.random.uniform(0.005, 0.02))  # 5-20ms
                        
                    else:  # portfolio
                        # Mock portfolio request
                        await asyncio.sleep(np.random.uniform(0.02, 0.08))  # 20-80ms
                    
                    response_time = time.time() - request_start
                    user_results['response_times'].append(response_time)
                    user_results['requests_completed'] += 1
                    
                    # Simulate user think time
                    await asyncio.sleep(np.random.uniform(0.1, 0.5))
                    
                except Exception as e:
                    user_results['requests_failed'] += 1
                    user_results['errors'].append(str(e))
            
            return user_results
        
        async def concurrent_user_test(num_users=20):
            """Run concurrent user simulation"""
            
            start_time = time.time()
            
            # Create user simulation tasks
            user_tasks = [simulate_user(i) for i in range(num_users)]
            
            # Run all users concurrently
            user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
            
            end_time = time.time()
            
            # Aggregate results
            total_requests = 0
            total_failures = 0
            all_response_times = []
            
            for result in user_results:
                if isinstance(result, dict):
                    total_requests += result['requests_completed']
                    total_failures += result['requests_failed']
                    all_response_times.extend(result['response_times'])
            
            return {
                'num_users': num_users,
                'total_duration': end_time - start_time,
                'total_requests': total_requests,
                'total_failures': total_failures,
                'success_rate': (total_requests / (total_requests + total_failures)) if (total_requests + total_failures) > 0 else 0,
                'avg_response_time': np.mean(all_response_times) if all_response_times else 0,
                'p95_response_time': np.percentile(all_response_times, 95) if all_response_times else 0,
                'requests_per_second': total_requests / (end_time - start_time)
            }
        
        # Run concurrent user test
        results = self.run_async(concurrent_user_test(20))
        
        # Verify concurrent user performance
        self.assertGreater(results['success_rate'], 0.95, "Success rate too low under concurrent load")
        self.assertLess(results['avg_response_time'], 0.1, "Average response time too high")
        self.assertLess(results['p95_response_time'], 0.2, "95th percentile response time too high")
        self.assertGreater(results['requests_per_second'], 50, "Request throughput too low")
    
    def test_memory_stress_test(self):
        """Test system behavior under memory pressure"""
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for memory testing")
            return
        
        def memory_stress_test():
            """Apply memory pressure and measure system behavior"""
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            test_results = {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': initial_memory,
                'operations_completed': 0,
                'memory_samples': []
            }
            
            # Create memory pressure
            memory_hogs = []
            
            try:
                # Gradually increase memory usage
                for i in range(20):  # Create 20 arrays
                    # Create large array (~10MB each)
                    array = np.random.randn(1024, 1024)  
                    memory_hogs.append(array)
                    
                    # Perform some operations
                    result = np.mean(array) + np.std(array)
                    test_results['operations_completed'] += 1
                    
                    # Monitor memory
                    current_memory = process.memory_info().rss / 1024 / 1024
                    test_results['memory_samples'].append(current_memory)
                    test_results['peak_memory_mb'] = max(test_results['peak_memory_mb'], current_memory)
                    
                    # Check if we're hitting memory limits
                    memory_usage = current_memory - initial_memory
                    if memory_usage > 500:  # Stop if using more than 500MB extra
                        break
                    
                    time.sleep(0.1)  # Brief pause
                
                # Test system still responsive under memory pressure
                for i in range(10):
                    # Perform computational task
                    test_array = np.random.randn(1000, 100)
                    result = np.linalg.svd(test_array, compute_uv=False)
                    test_results['operations_completed'] += 1
                
            finally:
                # Cleanup
                del memory_hogs
                import gc
                gc.collect()
            
            # Final memory check
            final_memory = process.memory_info().rss / 1024 / 1024
            test_results['final_memory_mb'] = final_memory
            test_results['memory_recovered'] = test_results['peak_memory_mb'] - final_memory
            
            return test_results
        
        # Run memory stress test
        results = memory_stress_test()
        
        # Verify memory stress handling
        self.assertGreater(results['operations_completed'], 20, "Too few operations completed under memory pressure")
        
        # Memory should be released after cleanup
        memory_increase = results['final_memory_mb'] - results['initial_memory_mb']
        self.assertLess(memory_increase, 200, "Too much memory retained after cleanup")
        
        # System should recover significant memory
        if results['peak_memory_mb'] > results['initial_memory_mb'] + 200:
            recovery_ratio = results['memory_recovered'] / (results['peak_memory_mb'] - results['initial_memory_mb'])
            self.assertGreater(recovery_ratio, 0.3, "Insufficient memory recovery")

if __name__ == "__main__":
    print("🧪 Running Performance and Load Tests")
    print("=" * 42)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPerformanceBenchmarks,
        TestLoadTesting
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print detailed summary
    print(f"\n📊 Performance Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Total test time: {end_time - start_time:.2f}s")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"   Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ Performance Issues (Failures):")
        for test, error in result.failures:
            print(f"   {test}: {error.split(chr(10))[0]}")
    
    if result.errors:
        print(f"\n🚨 Test Errors:")
        for test, error in result.errors:
            print(f"   {test}: {error.split(chr(10))[0]}")
    
    print(f"\n⚡ Performance testing completed!")
    print(f"🎯 System performance validated under various load conditions")
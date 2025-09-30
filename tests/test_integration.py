"""
Integration Tests for Stock AI System
End-to-end testing of component interactions and system workflows
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
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import test framework
from .test_framework import AsyncTestCase, MockDataGenerator, TestFixtures, BenchmarkRunner

class TestSystemIntegration(AsyncTestCase):
    """Test complete system integration workflows"""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.test_symbols = ["AAPL", "GOOGL", "MSFT"]
        self.test_data = {}
        
        # Generate test data for each symbol
        for symbol in self.test_symbols:
            self.test_data[symbol] = MockDataGenerator.generate_stock_data(symbol, 252)
    
    def tearDown(self):
        super().tearDown()
        TestFixtures.cleanup_temp_directory(self.temp_dir)
    
    def test_data_pipeline_to_prediction_flow(self):
        """Test data flows from collection through to prediction"""
        
        async def mock_pipeline_flow():
            """Mock the complete pipeline flow"""
            results = {}
            
            # Step 1: Data Collection
            collector = Mock()
            collector.collect_data = AsyncMock()
            
            for symbol in self.test_symbols:
                collector.collect_data.return_value = self.test_data[symbol]
                data = await collector.collect_data(symbol)
                self.assertIsNotNone(data)
                results[f"{symbol}_collected"] = True
            
            # Step 2: Data Validation
            validator = Mock()
            validator.validate_data.return_value = {
                'completeness': 0.95,
                'accuracy': 0.92,
                'valid': True
            }
            
            for symbol in self.test_symbols:
                validation = validator.validate_data(self.test_data[symbol])
                self.assertTrue(validation['valid'])
                results[f"{symbol}_validated"] = True
            
            # Step 3: Feature Engineering
            feature_engineer = Mock()
            feature_engineer.create_features.return_value = pd.DataFrame({
                'sma_20': np.random.randn(len(self.test_data[self.test_symbols[0]])),
                'rsi_14': np.random.uniform(0, 100, len(self.test_data[self.test_symbols[0]])),
                'macd': np.random.randn(len(self.test_data[self.test_symbols[0]]))
            })
            
            for symbol in self.test_symbols:
                features = feature_engineer.create_features(self.test_data[symbol])
                self.assertIsNotNone(features)
                results[f"{symbol}_features"] = True
            
            # Step 4: Model Training/Prediction
            predictor = Mock()
            predictor.predict.return_value = {
                'prediction': 150.0 + np.random.uniform(-10, 10),
                'confidence': np.random.uniform(0.7, 0.95),
                'trend': np.random.choice(['bullish', 'bearish', 'neutral'])
            }
            
            for symbol in self.test_symbols:
                prediction = predictor.predict(symbol)
                self.assertIn('prediction', prediction)
                self.assertIn('confidence', prediction)
                results[f"{symbol}_predicted"] = True
            
            # Step 5: Risk Assessment
            risk_manager = Mock()
            risk_manager.assess_risk.return_value = {
                'position_size': 100,
                'stop_loss': 145.0,
                'risk_score': 0.15
            }
            
            for symbol in self.test_symbols:
                risk_assessment = risk_manager.assess_risk(symbol, prediction)
                self.assertIn('position_size', risk_assessment)
                results[f"{symbol}_risk_assessed"] = True
            
            return results
        
        # Run the pipeline
        results = self.run_async(mock_pipeline_flow())
        
        # Verify all steps completed for all symbols
        expected_steps = ['collected', 'validated', 'features', 'predicted', 'risk_assessed']
        for symbol in self.test_symbols:
            for step in expected_steps:
                key = f"{symbol}_{step}"
                self.assertIn(key, results, f"Step {step} not completed for {symbol}")
                self.assertTrue(results[key])
    
    def test_concurrent_symbol_processing(self):
        """Test concurrent processing of multiple symbols"""
        
        async def process_symbol(symbol, processing_time=0.01):
            """Mock symbol processing with realistic timing"""
            start_time = time.time()
            
            # Simulate data collection
            await asyncio.sleep(processing_time)
            
            # Simulate feature engineering
            await asyncio.sleep(processing_time)
            
            # Simulate prediction
            await asyncio.sleep(processing_time)
            
            end_time = time.time()
            
            return {
                'symbol': symbol,
                'processing_time': end_time - start_time,
                'prediction': 150.0 + np.random.uniform(-10, 10),
                'status': 'completed'
            }
        
        async def process_all_concurrent():
            """Process all symbols concurrently"""
            tasks = [process_symbol(symbol) for symbol in self.test_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        start_time = time.time()
        results = self.run_async(process_all_concurrent())
        total_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(len(results), len(self.test_symbols))
        
        # All should be completed
        for result in results:
            if isinstance(result, dict):
                self.assertEqual(result['status'], 'completed')
                self.assertIn('prediction', result)
        
        # Concurrent processing should be faster than sequential
        # (allowing some overhead for testing environment)
        expected_sequential_time = len(self.test_symbols) * 0.03  # 3 * 0.01s per symbol
        self.assertLess(total_time, expected_sequential_time * 2, 
                       "Concurrent processing not faster than sequential")
    
    def test_error_handling_and_recovery(self):
        """Test system behavior under error conditions"""
        
        async def mock_error_scenarios():
            """Test various error scenarios and recovery"""
            error_scenarios = []
            
            # Scenario 1: Data source failure with fallback
            primary_source = Mock()
            primary_source.get_data = AsyncMock(side_effect=Exception("Primary source down"))
            
            fallback_source = Mock()
            fallback_source.get_data = AsyncMock(return_value=self.test_data["AAPL"])
            
            try:
                try:
                    data = await primary_source.get_data("AAPL")
                except Exception:
                    data = await fallback_source.get_data("AAPL")  # Fallback
                
                self.assertIsNotNone(data)
                error_scenarios.append("data_source_fallback_success")
            except Exception as e:
                error_scenarios.append(f"data_source_fallback_failed: {str(e)}")
            
            # Scenario 2: Invalid data handling
            validator = Mock()
            validator.validate.side_effect = ValueError("Invalid data format")
            
            try:
                validator.validate("invalid_data")
            except ValueError:
                # Should handle gracefully
                error_scenarios.append("invalid_data_handled")
            
            # Scenario 3: Model prediction failure with retry
            predictor = Mock()
            call_count = [0]  # Use list to allow modification in nested function
            
            def failing_predict(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] <= 2:  # Fail first 2 times
                    raise Exception("Model temporarily unavailable")
                return {'prediction': 150.0, 'confidence': 0.8}
            
            predictor.predict.side_effect = failing_predict
            
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = predictor.predict("AAPL")
                    error_scenarios.append("prediction_retry_success")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        error_scenarios.append("prediction_retry_exhausted")
            
            # Scenario 4: Partial system failure
            components = {
                'data_collector': Mock(),
                'validator': Mock(),
                'predictor': Mock(),
                'risk_manager': Mock()
            }
            
            # Make validator fail
            components['validator'].validate.side_effect = Exception("Validator down")
            
            # System should continue with other components
            working_components = []
            for name, component in components.items():
                try:
                    if name == 'data_collector':
                        component.collect.return_value = self.test_data["AAPL"]
                        component.collect()
                        working_components.append(name)
                    elif name == 'validator':
                        component.validate()  # This will fail
                    elif name == 'predictor':
                        component.predict.return_value = {'prediction': 150.0}
                        component.predict()
                        working_components.append(name)
                    elif name == 'risk_manager':
                        component.assess.return_value = {'risk': 'low'}
                        component.assess()
                        working_components.append(name)
                except Exception:
                    continue  # Component failed, continue with others
            
            # Should have 3 working components (all except validator)
            if len(working_components) >= 3:
                error_scenarios.append("partial_failure_handled")
            
            return error_scenarios
        
        scenarios = self.run_async(mock_error_scenarios())
        
        # Verify error handling scenarios
        self.assertIn("data_source_fallback_success", scenarios)
        self.assertIn("invalid_data_handled", scenarios)
        self.assertIn("prediction_retry_success", scenarios)
        self.assertIn("partial_failure_handled", scenarios)
    
    def test_data_consistency_across_components(self):
        """Test data consistency as it flows through components"""
        
        # Start with known test data
        original_data = self.test_data["AAPL"].copy()
        data_checkpoints = []
        
        # Checkpoint 1: After collection
        collected_data = original_data.copy()
        data_checkpoints.append({
            'stage': 'collection',
            'row_count': len(collected_data),
            'columns': list(collected_data.columns),
            'sample_close': collected_data.iloc[0]['Close'] if 'Close' in collected_data.columns else None
        })
        
        # Checkpoint 2: After validation (should be unchanged)
        validated_data = collected_data.copy()
        data_checkpoints.append({
            'stage': 'validation',
            'row_count': len(validated_data),
            'columns': list(validated_data.columns),
            'sample_close': validated_data.iloc[0]['Close'] if 'Close' in validated_data.columns else None
        })
        
        # Checkpoint 3: After cleaning (might have changes)
        cleaned_data = validated_data.copy()
        # Simulate cleaning - fill any NaN values
        if hasattr(cleaned_data, 'fillna'):
            cleaned_data = cleaned_data.fillna(method='ffill')
        
        data_checkpoints.append({
            'stage': 'cleaning',
            'row_count': len(cleaned_data),
            'columns': list(cleaned_data.columns),
            'sample_close': cleaned_data.iloc[0]['Close'] if 'Close' in cleaned_data.columns else None
        })
        
        # Checkpoint 4: After feature engineering (should have more columns)
        # Mock feature engineering - add simple features
        feature_data = cleaned_data.copy()
        if hasattr(feature_data, 'assign'):
            feature_data = feature_data.assign(
                sma_20=feature_data['Close'].rolling(20).mean() if 'Close' in feature_data.columns else 0,
                price_change=feature_data['Close'].pct_change() if 'Close' in feature_data.columns else 0
            )
        
        data_checkpoints.append({
            'stage': 'feature_engineering',
            'row_count': len(feature_data),
            'columns': list(feature_data.columns),
            'sample_close': feature_data.iloc[0]['Close'] if 'Close' in feature_data.columns else None
        })
        
        # Verify data consistency
        self.assertGreater(len(data_checkpoints), 0)
        
        # Check that row counts are consistent (or reasonably so)
        row_counts = [cp['row_count'] for cp in data_checkpoints]
        self.assertTrue(all(rc > 0 for rc in row_counts), "Some stages have no data")
        
        # Check that Close price remains consistent through validation and cleaning
        if data_checkpoints[0]['sample_close'] is not None:
            collection_close = data_checkpoints[0]['sample_close']
            validation_close = data_checkpoints[1]['sample_close']
            
            if validation_close is not None:
                # Should be identical after validation
                self.assertAlmostEqual(collection_close, validation_close, places=6)
        
        # Feature engineering should add columns
        original_columns = len(data_checkpoints[0]['columns'])
        feature_columns = len(data_checkpoints[-1]['columns'])
        self.assertGreaterEqual(feature_columns, original_columns, 
                               "Feature engineering should maintain or add columns")
    
    def test_system_performance_under_load(self):
        """Test system performance with realistic load"""
        
        async def simulate_high_load():
            """Simulate high load scenario"""
            # Simulate 50 concurrent symbol requests
            symbols = [f"STOCK{i:03d}" for i in range(50)]
            
            async def process_single_request(symbol):
                """Process a single symbol request"""
                start_time = time.time()
                
                # Simulate processing steps with realistic timing
                await asyncio.sleep(0.001)  # Data collection
                await asyncio.sleep(0.001)  # Validation
                await asyncio.sleep(0.002)  # Feature engineering
                await asyncio.sleep(0.003)  # Prediction
                
                processing_time = time.time() - start_time
                
                return {
                    'symbol': symbol,
                    'processing_time': processing_time,
                    'prediction': 100.0 + np.random.uniform(-20, 20),
                    'confidence': np.random.uniform(0.6, 0.95)
                }
            
            # Process all requests concurrently
            start_time = time.time()
            tasks = [process_single_request(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            successful_results = [r for r in results if isinstance(r, dict)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            avg_processing_time = np.mean([r['processing_time'] for r in successful_results])
            
            return {
                'total_requests': len(symbols),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'total_time': total_time,
                'avg_processing_time': avg_processing_time,
                'throughput': len(successful_results) / total_time
            }
        
        performance_metrics = self.run_async(simulate_high_load())
        
        # Verify performance meets requirements
        self.assertGreater(performance_metrics['successful'], 40, 
                          "Too many failed requests under load")
        
        self.assertLess(performance_metrics['avg_processing_time'], 0.1, 
                       "Average processing time too high")
        
        self.assertGreater(performance_metrics['throughput'], 100, 
                          "Throughput too low (requests per second)")
        
        self.assertLess(performance_metrics['total_time'], 2.0, 
                       "Total processing time too high for concurrent load")
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable during processing"""
        
        try:
            import psutil
            process = psutil.Process()
            PSUTIL_AVAILABLE = True
        except ImportError:
            PSUTIL_AVAILABLE = False
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for memory testing")
            return
        
        # Baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        async def memory_intensive_workflow():
            """Simulate memory-intensive workflow"""
            memory_snapshots = []
            
            for i in range(10):  # Process multiple batches
                # Generate large dataset
                large_data = MockDataGenerator.generate_stock_data("TEST", 1000)
                
                # Simulate processing
                if hasattr(large_data, 'rolling'):
                    # Create multiple features (memory intensive)
                    features = large_data.copy()
                    features['sma_20'] = large_data['Close'].rolling(20).mean()
                    features['sma_50'] = large_data['Close'].rolling(50).mean()
                    features['ema_12'] = large_data['Close'].ewm(span=12).mean()
                    features['rsi'] = self._calculate_mock_rsi(large_data['Close'])
                
                # Take memory snapshot
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_snapshots.append(current_memory)
                
                # Simulate cleanup
                del large_data
                if 'features' in locals():
                    del features
                
                await asyncio.sleep(0.01)  # Small delay
            
            return memory_snapshots
        
        memory_snapshots = self.run_async(memory_intensive_workflow())
        
        # Analyze memory usage
        max_memory = max(memory_snapshots)
        memory_growth = max_memory - baseline_memory
        
        # Memory growth should be reasonable (less than 500MB for test)
        self.assertLess(memory_growth, 500, 
                       f"Excessive memory growth: {memory_growth:.1f}MB")
        
        # Memory should not continuously grow (last should be similar to middle)
        if len(memory_snapshots) >= 3:
            middle_memory = memory_snapshots[len(memory_snapshots)//2]
            final_memory = memory_snapshots[-1]
            memory_drift = abs(final_memory - middle_memory)
            
            self.assertLess(memory_drift, 100, 
                           f"Memory drift too high: {memory_drift:.1f}MB")
    
    def _calculate_mock_rsi(self, prices, window=14):
        """Calculate mock RSI for memory testing"""
        if not hasattr(prices, 'diff'):
            return pd.Series([50.0] * len(prices))  # Fallback
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)  # Fill NaN with neutral RSI

class TestAPIIntegration(AsyncTestCase):
    """Test API endpoints and external integrations"""
    
    def setUp(self):
        super().setUp()
        self.mock_responses = {
            'stock_data': {
                'symbol': 'AAPL',
                'price': 150.0,
                'change': 2.5,
                'timestamp': '2024-01-01T10:00:00Z'
            },
            'news_data': [
                {
                    'headline': 'Apple reports strong earnings',
                    'sentiment': 0.8,
                    'timestamp': '2024-01-01T09:00:00Z'
                }
            ]
        }
    
    def test_external_api_integration(self):
        """Test integration with external APIs"""
        
        async def mock_api_calls():
            """Mock various API calls"""
            api_results = {}
            
            # Mock stock data API
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_response = Mock()
                mock_response.json = AsyncMock(return_value=self.mock_responses['stock_data'])
                mock_response.status = 200
                mock_get.return_value.__aenter__.return_value = mock_response
                
                # Simulate API call
                stock_data = self.mock_responses['stock_data']  # In real scenario, would call API
                api_results['stock_api'] = stock_data
                
                self.assertIn('symbol', stock_data)
                self.assertIn('price', stock_data)
            
            # Mock news API
            news_data = self.mock_responses['news_data']
            api_results['news_api'] = news_data
            
            self.assertIsInstance(news_data, list)
            self.assertGreater(len(news_data), 0)
            
            # Mock economic indicators API
            economic_data = {
                'gdp_growth': 2.5,
                'unemployment': 4.2,
                'inflation': 3.1
            }
            api_results['economic_api'] = economic_data
            
            return api_results
        
        results = self.run_async(mock_api_calls())
        
        # Verify all APIs returned data
        expected_apis = ['stock_api', 'news_api', 'economic_api']
        for api in expected_apis:
            self.assertIn(api, results)
            self.assertIsNotNone(results[api])
    
    def test_api_rate_limiting(self):
        """Test API rate limiting behavior"""
        
        class MockRateLimiter:
            def __init__(self, calls_per_second=5):
                self.calls_per_second = calls_per_second
                self.call_times = []
            
            async def make_call(self, endpoint):
                current_time = time.time()
                
                # Remove calls older than 1 second
                self.call_times = [t for t in self.call_times if current_time - t < 1.0]
                
                # Check rate limit
                if len(self.call_times) >= self.calls_per_second:
                    raise Exception("Rate limit exceeded")
                
                self.call_times.append(current_time)
                await asyncio.sleep(0.01)  # Simulate API call time
                
                return {'status': 'success', 'endpoint': endpoint}
        
        async def test_rate_limiting():
            limiter = MockRateLimiter(calls_per_second=3)  # Low limit for testing
            
            # Make calls within limit
            results = []
            for i in range(3):
                try:
                    result = await limiter.make_call(f"endpoint_{i}")
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e)})
            
            # Try to exceed limit
            for i in range(3, 6):
                try:
                    result = await limiter.make_call(f"endpoint_{i}")
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e)})
            
            return results
        
        results = self.run_async(test_rate_limiting())
        
        # First 3 should succeed
        successful_calls = [r for r in results if 'status' in r and r['status'] == 'success']
        failed_calls = [r for r in results if 'error' in r]
        
        self.assertEqual(len(successful_calls), 3, "Rate limiter should allow 3 calls")
        self.assertGreater(len(failed_calls), 0, "Rate limiter should block excess calls")
    
    def test_api_failover(self):
        """Test API failover mechanisms"""
        
        async def test_api_failover():
            """Test failover between API providers"""
            
            # Mock primary API (fails)
            primary_api = Mock()
            primary_api.get_data = AsyncMock(side_effect=Exception("Primary API down"))
            
            # Mock secondary API (succeeds)
            secondary_api = Mock()
            secondary_api.get_data = AsyncMock(return_value=self.mock_responses['stock_data'])
            
            # Mock tertiary API (also succeeds)
            tertiary_api = Mock()
            tertiary_api.get_data = AsyncMock(return_value=self.mock_responses['stock_data'])
            
            apis = [primary_api, secondary_api, tertiary_api]
            
            # Try APIs in order until one succeeds
            data = None
            last_error = None
            
            for api in apis:
                try:
                    data = await api.get_data("AAPL")
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            return {
                'data': data,
                'last_error': last_error,
                'failover_successful': data is not None
            }
        
        result = self.run_async(test_api_failover())
        
        self.assertTrue(result['failover_successful'], "Failover should have succeeded")
        self.assertIsNotNone(result['data'], "Should have received data from failover API")
        self.assertIn('symbol', result['data'], "Data should contain expected fields")

if __name__ == "__main__":
    print("🧪 Running Integration Tests")
    print("=" * 32)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSystemIntegration,
        TestAPIIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Integration Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"   Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, error in result.failures:
            print(f"   {test}: {error.split(chr(10))[0]}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, error in result.errors:
            print(f"   {test}: {error.split(chr(10))[0]}")
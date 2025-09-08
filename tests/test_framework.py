"""
Professional Testing Framework for Stock AI System
Comprehensive unit testing framework with advanced features
"""

import os
import sys
import unittest
import asyncio
import json
import tempfile
import shutil
import time
import random
import warnings
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager

# Suppress warnings during testing
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

# Configure test logging
logging.basicConfig(level=logging.WARNING)
test_logger = logging.getLogger('stock_ai_tests')

@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    warning_messages: List[str] = None
    metadata: Dict[str, Any] = None

class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    @staticmethod
    def generate_stock_data(symbol: str = "AAPL", days: int = 100) -> pd.DataFrame:
        """Generate realistic stock price data"""
        if not PANDAS_AVAILABLE:
            return None
            
        dates = pd.date_range(start='2024-01-01', periods=days)
        base_price = random.uniform(100, 200)
        
        data = []
        for i, date in enumerate(dates):
            # Add some realistic price movement
            price_change = random.uniform(-0.05, 0.05)
            base_price *= (1 + price_change)
            
            # Ensure positive prices
            base_price = max(base_price, 10)
            
            # Generate OHLCV data
            high = base_price * random.uniform(1.0, 1.03)
            low = base_price * random.uniform(0.97, 1.0)
            open_price = random.uniform(low, high)
            close_price = base_price
            volume = random.randint(1000000, 50000000)
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close_price,
                'Volume': volume,
                'Symbol': symbol
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_news_data(count: int = 50) -> List[Dict[str, Any]]:
        """Generate mock news data"""
        news_templates = [
            "Company reports strong quarterly earnings",
            "New product launch expected to boost revenue",
            "Market volatility continues amid economic uncertainty",
            "Industry analysts upgrade stock rating",
            "CEO announces strategic partnership",
            "Regulatory approval received for new initiative"
        ]
        
        news = []
        for i in range(count):
            news.append({
                'headline': random.choice(news_templates),
                'timestamp': datetime.now() - timedelta(hours=random.randint(1, 720)),
                'sentiment': random.uniform(-1, 1),
                'relevance': random.uniform(0.3, 1.0),
                'source': random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ'])
            })
        
        return news
    
    @staticmethod
    def generate_financial_metrics() -> Dict[str, float]:
        """Generate mock financial metrics"""
        return {
            'pe_ratio': random.uniform(10, 30),
            'price_to_book': random.uniform(1, 5),
            'debt_to_equity': random.uniform(0.1, 2.0),
            'roe': random.uniform(0.05, 0.25),
            'revenue_growth': random.uniform(-0.1, 0.3),
            'profit_margin': random.uniform(0.05, 0.30)
        }

class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""
    
    def setUp(self):
        """Set up test case"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up test case"""
        self.loop.close()
    
    def run_async(self, coro):
        """Run async coroutine in test"""
        return self.loop.run_until_complete(coro)

class TestRunner:
    """Advanced test runner with comprehensive reporting"""
    
    def __init__(self, test_directory: str = None):
        self.test_directory = test_directory or os.path.dirname(__file__)
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
    def discover_tests(self) -> List[str]:
        """Discover all test files"""
        test_files = []
        
        for root, dirs, files in os.walk(self.test_directory):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
        
        return test_files
    
    def run_test_file(self, test_file: str) -> List[TestResult]:
        """Run tests from a single file"""
        results = []
        
        # Import the test module
        module_name = os.path.splitext(os.path.basename(test_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            results.append(TestResult(
                test_name=f"{module_name} (import)",
                passed=False,
                execution_time=0.0,
                error_message=str(e)
            ))
            return results
        
        # Find test classes and methods
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                # Run test class
                suite = unittest.TestLoader().loadTestsFromTestCase(obj)
                runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
                
                for test in suite:
                    start_time = time.time()
                    result = runner.run(test)
                    execution_time = time.time() - start_time
                    
                    test_name = f"{obj.__name__}.{test._testMethodName}"
                    passed = result.wasSuccessful()
                    error_message = None
                    
                    if result.failures:
                        error_message = result.failures[0][1]
                    elif result.errors:
                        error_message = result.errors[0][1]
                    
                    results.append(TestResult(
                        test_name=test_name,
                        passed=passed,
                        execution_time=execution_time,
                        error_message=error_message
                    ))
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all discovered tests"""
        self.start_time = time.time()
        test_files = self.discover_tests()
        
        print(f"🧪 Running tests from {len(test_files)} files...")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_file in test_files:
            print(f"   📁 {os.path.basename(test_file)}")
            file_results = self.run_test_file(test_file)
            self.results.extend(file_results)
            
            for result in file_results:
                total_tests += 1
                if result.passed:
                    passed_tests += 1
                    print(f"      ✅ {result.test_name} ({result.execution_time:.3f}s)")
                else:
                    failed_tests += 1
                    print(f"      ❌ {result.test_name} ({result.execution_time:.3f}s)")
                    if result.error_message:
                        print(f"         Error: {result.error_message[:100]}...")
        
        self.end_time = time.time()
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'total_time': self.end_time - self.start_time,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.results:
            return "No tests run"
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        report_lines = [
            "🧪 STOCK AI TESTING REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Execution Time: {total_time:.2f}s",
            "",
            "📊 TEST SUMMARY",
            "-" * 20,
            f"Total Tests: {total_tests}",
            f"✅ Passed: {passed_tests}",
            f"❌ Failed: {failed_tests}",
            f"Success Rate: {passed_tests/total_tests*100:.1f}%",
            ""
        ]
        
        if failed_tests > 0:
            report_lines.extend([
                "🚨 FAILED TESTS",
                "-" * 15
            ])
            
            for result in self.results:
                if not result.passed:
                    report_lines.append(f"❌ {result.test_name}")
                    if result.error_message:
                        # First line of error message
                        error_line = result.error_message.split('\n')[0]
                        report_lines.append(f"   Error: {error_line[:80]}...")
            
            report_lines.append("")
        
        # Performance summary
        report_lines.extend([
            "⚡ PERFORMANCE SUMMARY",
            "-" * 22
        ])
        
        execution_times = [r.execution_time for r in self.results]
        if execution_times:
            report_lines.extend([
                f"Average Test Time: {np.mean(execution_times):.3f}s",
                f"Fastest Test: {min(execution_times):.3f}s",
                f"Slowest Test: {max(execution_times):.3f}s"
            ])
        
        # Top slowest tests
        slowest_tests = sorted(self.results, key=lambda r: r.execution_time, reverse=True)[:5]
        if slowest_tests:
            report_lines.extend([
                "",
                "🐌 SLOWEST TESTS",
                "-" * 15
            ])
            
            for result in slowest_tests:
                status = "✅" if result.passed else "❌"
                report_lines.append(f"{status} {result.test_name}: {result.execution_time:.3f}s")
        
        return "\n".join(report_lines)

class BenchmarkRunner:
    """Performance benchmarking for critical functions"""
    
    def __init__(self):
        self.benchmarks: Dict[str, List[float]] = {}
    
    @contextmanager
    def benchmark(self, name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            if name not in self.benchmarks:
                self.benchmarks[name] = []
            self.benchmarks[name].append(execution_time)
    
    def run_benchmark(self, func: Callable, name: str, iterations: int = 100, *args, **kwargs):
        """Run function multiple times and collect timing data"""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    asyncio.run(result)
            except Exception as e:
                test_logger.warning(f"Benchmark {name} iteration failed: {e}")
                continue
            
            execution_time = time.time() - start_time
            times.append(execution_time)
        
        self.benchmarks[name] = times
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistical summary of benchmark"""
        if name not in self.benchmarks or not self.benchmarks[name]:
            return {}
        
        times = self.benchmarks[name]
        return {
            'count': len(times),
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': min(times),
            'max': max(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    def generate_report(self) -> str:
        """Generate benchmark report"""
        if not self.benchmarks:
            return "No benchmarks run"
        
        report_lines = [
            "⚡ PERFORMANCE BENCHMARK REPORT",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for name in sorted(self.benchmarks.keys()):
            stats = self.get_stats(name)
            if not stats:
                continue
            
            report_lines.extend([
                f"📊 {name.upper()}",
                "-" * len(name),
                f"Iterations: {stats['count']}",
                f"Mean: {stats['mean']*1000:.2f}ms",
                f"Median: {stats['median']*1000:.2f}ms",
                f"Std Dev: {stats['std']*1000:.2f}ms",
                f"Min: {stats['min']*1000:.2f}ms",
                f"Max: {stats['max']*1000:.2f}ms",
                f"P95: {stats['p95']*1000:.2f}ms",
                f"P99: {stats['p99']*1000:.2f}ms",
                ""
            ])
        
        return "\n".join(report_lines)

# Test utilities and fixtures
class TestFixtures:
    """Common test fixtures and utilities"""
    
    @staticmethod
    def create_temp_directory() -> str:
        """Create temporary directory for tests"""
        return tempfile.mkdtemp()
    
    @staticmethod
    def cleanup_temp_directory(path: str):
        """Clean up temporary directory"""
        if os.path.exists(path):
            shutil.rmtree(path)
    
    @staticmethod
    def create_mock_api_response(status_code: int = 200, data: Dict = None) -> Mock:
        """Create mock API response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data or {}
        mock_response.text = json.dumps(data or {})
        return mock_response
    
    @staticmethod
    def create_mock_database_connection() -> Mock:
        """Create mock database connection"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = None
        return mock_connection

# Coverage analysis helper
class CoverageAnalyzer:
    """Simple code coverage analysis"""
    
    def __init__(self):
        self.covered_lines = set()
        self.total_lines = 0
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze code coverage for a file"""
        if not os.path.exists(file_path):
            return {'coverage': 0.0, 'lines': 0}
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        executable_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                executable_lines += 1
        
        return {
            'coverage': 0.0,  # Placeholder - would need actual execution tracing
            'executable_lines': executable_lines,
            'total_lines': len(lines)
        }

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Stock AI Testing Framework")
    print("=" * 40)
    
    # Test the framework itself
    class SampleTest(unittest.TestCase):
        """Sample test class"""
        
        def test_mock_data_generation(self):
            """Test mock data generation"""
            if PANDAS_AVAILABLE:
                data = MockDataGenerator.generate_stock_data("AAPL", 30)
                self.assertIsNotNone(data)
                self.assertEqual(len(data), 30)
                self.assertTrue('Close' in data.columns)
        
        def test_news_generation(self):
            """Test news data generation"""
            news = MockDataGenerator.generate_news_data(10)
            self.assertEqual(len(news), 10)
            self.assertTrue('headline' in news[0])
        
        def test_financial_metrics(self):
            """Test financial metrics generation"""
            metrics = MockDataGenerator.generate_financial_metrics()
            self.assertTrue('pe_ratio' in metrics)
            self.assertGreater(metrics['pe_ratio'], 0)
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(SampleTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Test benchmark runner
    benchmark = BenchmarkRunner()
    
    def sample_function():
        time.sleep(0.001)  # Simulate work
        return random.random()
    
    print(f"\n⚡ Running benchmarks...")
    benchmark.run_benchmark(sample_function, "sample_function", iterations=10)
    
    print(benchmark.generate_report())
    
    print(f"\n✅ Testing framework ready!")
    print(f"📋 Features:")
    print(f"   • Mock data generation")
    print(f"   • Async test support")
    print(f"   • Performance benchmarking")
    print(f"   • Comprehensive reporting")
    print(f"   • Coverage analysis")
    print(f"   • Test discovery")
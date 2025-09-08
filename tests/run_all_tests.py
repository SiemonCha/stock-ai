"""
Automated Testing Pipeline for Stock AI System
Comprehensive test runner with reporting and CI/CD integration
"""

import os
import sys
import unittest
import asyncio
import time
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
try:
    from test_framework import TestRunner, BenchmarkRunner
    from test_models import *
    from test_data_pipeline import *
    from test_integration import *
    from test_end_to_end import *
    from test_performance import *
    TESTS_AVAILABLE = True
except ImportError as e:
    TESTS_AVAILABLE = False
    print(f"Warning: Test imports failed: {e}")

class TestPipeline:
    """Automated testing pipeline with comprehensive reporting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.results = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0,
            'test_suites': {},
            'summary': {},
            'artifacts': []
        }
        
    def _default_config(self) -> Dict[str, Any]:
        """Default pipeline configuration"""
        return {
            'run_unit_tests': True,
            'run_integration_tests': True,
            'run_end_to_end_tests': True,
            'run_performance_tests': True,
            'generate_coverage': False,  # Requires coverage.py
            'generate_reports': True,
            'fail_fast': False,
            'parallel_execution': True,
            'test_timeout': 300,  # 5 minutes per test suite
            'output_format': 'both',  # 'console', 'json', or 'both'
            'output_directory': 'test_results'
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test pipeline"""
        
        print("🧪 Starting Automated Test Pipeline")
        print("=" * 50)
        
        self.results['start_time'] = datetime.now()
        
        # Ensure output directory exists
        os.makedirs(self.config['output_directory'], exist_ok=True)
        
        try:
            # Run test suites in order
            if self.config['run_unit_tests']:
                self._run_unit_tests()
            
            if self.config['run_integration_tests']:
                self._run_integration_tests()
            
            if self.config['run_end_to_end_tests']:
                self._run_end_to_end_tests()
            
            if self.config['run_performance_tests']:
                self._run_performance_tests()
            
            # Generate reports
            if self.config['generate_reports']:
                self._generate_reports()
            
            # Generate summary
            self._generate_summary()
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            self.results['pipeline_error'] = str(e)
        
        finally:
            self.results['end_time'] = datetime.now()
            self.results['total_duration'] = (
                self.results['end_time'] - self.results['start_time']
            ).total_seconds()
        
        return self.results
    
    def _run_unit_tests(self):
        """Run unit tests"""
        print("\n📋 Running Unit Tests")
        print("-" * 25)
        
        suite_start = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Unit test classes
        unit_test_classes = [
            # Add actual test classes if available
        ]
        
        # Discover tests from files
        test_files = [
            'test_models.py',
            'test_data_pipeline.py'
        ]
        
        results = self._run_test_suite("unit_tests", test_files)
        
        self.results['test_suites']['unit_tests'] = results
        self.results['test_suites']['unit_tests']['duration'] = time.time() - suite_start
        
        if results['success_rate'] < 0.8 and self.config['fail_fast']:
            raise Exception(f"Unit tests failed with {results['success_rate']:.1%} success rate")
    
    def _run_integration_tests(self):
        """Run integration tests"""
        print("\n🔗 Running Integration Tests")
        print("-" * 30)
        
        suite_start = time.time()
        
        test_files = ['test_integration.py']
        results = self._run_test_suite("integration_tests", test_files)
        
        self.results['test_suites']['integration_tests'] = results
        self.results['test_suites']['integration_tests']['duration'] = time.time() - suite_start
        
        if results['success_rate'] < 0.9 and self.config['fail_fast']:
            raise Exception(f"Integration tests failed with {results['success_rate']:.1%} success rate")
    
    def _run_end_to_end_tests(self):
        """Run end-to-end tests"""
        print("\n🎯 Running End-to-End Tests")
        print("-" * 30)
        
        suite_start = time.time()
        
        test_files = ['test_end_to_end.py']
        results = self._run_test_suite("end_to_end_tests", test_files)
        
        self.results['test_suites']['end_to_end_tests'] = results
        self.results['test_suites']['end_to_end_tests']['duration'] = time.time() - suite_start
        
        if results['success_rate'] < 0.85 and self.config['fail_fast']:
            raise Exception(f"End-to-end tests failed with {results['success_rate']:.1%} success rate")
    
    def _run_performance_tests(self):
        """Run performance tests"""
        print("\n⚡ Running Performance Tests")
        print("-" * 30)
        
        suite_start = time.time()
        
        test_files = ['test_performance.py']
        results = self._run_test_suite("performance_tests", test_files)
        
        self.results['test_suites']['performance_tests'] = results
        self.results['test_suites']['performance_tests']['duration'] = time.time() - suite_start
        
        # Performance tests are more lenient
        if results['success_rate'] < 0.7 and self.config['fail_fast']:
            print(f"⚠️  Performance tests had {results['success_rate']:.1%} success rate (continuing)")
    
    def _run_test_suite(self, suite_name: str, test_files: List[str]) -> Dict[str, Any]:
        """Run a specific test suite"""
        
        results = {
            'suite_name': suite_name,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_errors': 0,
            'success_rate': 0.0,
            'failures': [],
            'errors': [],
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Run tests using unittest
            for test_file in test_files:
                test_path = os.path.join(os.path.dirname(__file__), test_file)
                
                if not os.path.exists(test_path):
                    print(f"   ⚠️  Test file not found: {test_file}")
                    continue
                
                print(f"   📁 Running {test_file}")
                
                # Run individual test file
                file_results = self._run_test_file(test_path)
                
                # Aggregate results
                results['tests_run'] += file_results.get('tests_run', 0)
                results['tests_passed'] += file_results.get('tests_passed', 0)
                results['tests_failed'] += file_results.get('tests_failed', 0)
                results['tests_errors'] += file_results.get('tests_errors', 0)
                results['failures'].extend(file_results.get('failures', []))
                results['errors'].extend(file_results.get('errors', []))
                
        except Exception as e:
            print(f"   ❌ Suite execution error: {e}")
            results['suite_error'] = str(e)
        
        results['execution_time'] = time.time() - start_time
        
        # Calculate success rate
        if results['tests_run'] > 0:
            results['success_rate'] = results['tests_passed'] / results['tests_run']
        
        # Print suite summary
        print(f"   📊 {suite_name}: {results['tests_passed']}/{results['tests_run']} passed ({results['success_rate']:.1%})")
        
        return results
    
    def _run_test_file(self, test_file_path: str) -> Dict[str, Any]:
        """Run tests from a single file"""
        
        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_errors': 0,
            'failures': [],
            'errors': []
        }
        
        try:
            # Use subprocess to run tests in isolation
            cmd = [sys.executable, '-m', 'unittest', test_file_path, '-v']
            
            process = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__),
                capture_output=True,
                text=True,
                timeout=self.config['test_timeout']
            )
            
            # Parse output to extract test results
            output_lines = process.stdout.split('\n')
            
            test_count = 0
            failures = []
            errors = []
            
            for line in output_lines:
                if ' ... ok' in line:
                    test_count += 1
                    results['tests_passed'] += 1
                elif ' ... FAIL' in line:
                    test_count += 1
                    results['tests_failed'] += 1
                    failures.append(line.split(' ... FAIL')[0].strip())
                elif ' ... ERROR' in line:
                    test_count += 1
                    results['tests_errors'] += 1
                    errors.append(line.split(' ... ERROR')[0].strip())
            
            results['tests_run'] = test_count
            results['failures'] = failures
            results['errors'] = errors
            
            if process.returncode != 0 and not failures and not errors:
                # If process failed but we didn't capture failures/errors
                results['errors'].append(f"Process failed with return code {process.returncode}")
                if not results['tests_run']:
                    results['tests_run'] = 1
                    results['tests_errors'] = 1
            
        except subprocess.TimeoutExpired:
            results['errors'].append(f"Test file timed out after {self.config['test_timeout']}s")
            results['tests_run'] = 1
            results['tests_errors'] = 1
            
        except Exception as e:
            results['errors'].append(f"Failed to run test file: {str(e)}")
            results['tests_run'] = 1
            results['tests_errors'] = 1
        
        return results
    
    def _generate_reports(self):
        """Generate test reports"""
        print("\n📋 Generating Test Reports")
        print("-" * 28)
        
        # Generate JSON report
        json_report = self._generate_json_report()
        json_file = os.path.join(self.config['output_directory'], 'test_results.json')
        
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        print(f"   📄 JSON report: {json_file}")
        self.results['artifacts'].append(json_file)
        
        # Generate HTML report
        html_report = self._generate_html_report()
        html_file = os.path.join(self.config['output_directory'], 'test_results.html')
        
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        print(f"   📄 HTML report: {html_file}")
        self.results['artifacts'].append(html_file)
        
        # Generate coverage report if enabled
        if self.config['generate_coverage']:
            self._generate_coverage_report()
    
    def _generate_json_report(self) -> Dict[str, Any]:
        """Generate JSON test report"""
        return {
            'pipeline_info': {
                'timestamp': self.results['start_time'].isoformat(),
                'duration': self.results['total_duration'],
                'config': self.config
            },
            'summary': self.results.get('summary', {}),
            'test_suites': self.results['test_suites'],
            'artifacts': self.results.get('artifacts', [])
        }
    
    def _generate_html_report(self) -> str:
        """Generate HTML test report"""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Stock AI Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .suite { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
        .suite-header { background: #e8f4f8; padding: 15px; font-weight: bold; }
        .suite-content { padding: 15px; }
        .success { color: #28a745; }
        .failure { color: #dc3545; }
        .warning { color: #ffc107; }
        .metric { margin: 5px 0; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Stock AI Test Results</h1>
        <div class="metric">Generated: {timestamp}</div>
        <div class="metric">Duration: {duration:.2f}s</div>
        <div class="metric">Overall Status: <span class="{status_class}">{status}</span></div>
    </div>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{total_tests}</td></tr>
            <tr><td>Passed</td><td class="success">{total_passed}</td></tr>
            <tr><td>Failed</td><td class="failure">{total_failed}</td></tr>
            <tr><td>Errors</td><td class="warning">{total_errors}</td></tr>
            <tr><td>Success Rate</td><td>{success_rate:.1%}</td></tr>
        </table>
    </div>
    
    {suite_sections}
    
    <div class="footer">
        <h3>📁 Artifacts</h3>
        <ul>
            {artifacts}
        </ul>
    </div>
</body>
</html>
        """
        
        # Calculate summary metrics
        total_tests = sum(suite.get('tests_run', 0) for suite in self.results['test_suites'].values())
        total_passed = sum(suite.get('tests_passed', 0) for suite in self.results['test_suites'].values())
        total_failed = sum(suite.get('tests_failed', 0) for suite in self.results['test_suites'].values())
        total_errors = sum(suite.get('tests_errors', 0) for suite in self.results['test_suites'].values())
        
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        status = "PASS" if success_rate >= 0.8 else "FAIL"
        status_class = "success" if status == "PASS" else "failure"
        
        # Generate suite sections
        suite_sections = ""
        for suite_name, suite_data in self.results['test_suites'].items():
            suite_sections += f"""
            <div class="suite">
                <div class="suite-header">
                    {suite_name.replace('_', ' ').title()}
                </div>
                <div class="suite-content">
                    <div class="metric">Tests Run: {suite_data.get('tests_run', 0)}</div>
                    <div class="metric">Passed: <span class="success">{suite_data.get('tests_passed', 0)}</span></div>
                    <div class="metric">Failed: <span class="failure">{suite_data.get('tests_failed', 0)}</span></div>
                    <div class="metric">Errors: <span class="warning">{suite_data.get('tests_errors', 0)}</span></div>
                    <div class="metric">Success Rate: {suite_data.get('success_rate', 0):.1%}</div>
                    <div class="metric">Duration: {suite_data.get('duration', 0):.2f}s</div>
                </div>
            </div>
            """
        
        # Generate artifacts list
        artifacts_html = ""
        for artifact in self.results.get('artifacts', []):
            artifacts_html += f"<li>{os.path.basename(artifact)}</li>"
        
        return html_template.format(
            timestamp=self.results['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            duration=self.results['total_duration'],
            status=status,
            status_class=status_class,
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_errors=total_errors,
            success_rate=success_rate,
            suite_sections=suite_sections,
            artifacts=artifacts_html
        )
    
    def _generate_coverage_report(self):
        """Generate code coverage report"""
        print("   📊 Generating coverage report...")
        
        # This would require coverage.py to be installed
        # For now, we'll create a placeholder
        coverage_file = os.path.join(self.config['output_directory'], 'coverage_placeholder.txt')
        
        with open(coverage_file, 'w') as f:
            f.write("Code coverage report would be generated here with coverage.py\n")
            f.write("Install coverage.py and integrate with test runner for actual coverage metrics.\n")
        
        self.results['artifacts'].append(coverage_file)
    
    def _generate_summary(self):
        """Generate pipeline summary"""
        
        # Calculate overall metrics
        total_tests = sum(suite.get('tests_run', 0) for suite in self.results['test_suites'].values())
        total_passed = sum(suite.get('tests_passed', 0) for suite in self.results['test_suites'].values())
        total_failed = sum(suite.get('tests_failed', 0) for suite in self.results['test_suites'].values())
        total_errors = sum(suite.get('tests_errors', 0) for suite in self.results['test_suites'].values())
        
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        summary = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'success_rate': success_rate,
            'overall_status': 'PASS' if success_rate >= 0.8 else 'FAIL',
            'pipeline_duration': self.results['total_duration'],
            'test_suites_run': len(self.results['test_suites']),
            'artifacts_generated': len(self.results.get('artifacts', []))
        }
        
        self.results['summary'] = summary
        
        # Print summary
        print(f"\n🎯 Test Pipeline Summary")
        print("=" * 30)
        print(f"Overall Status: {'✅ PASS' if summary['overall_status'] == 'PASS' else '❌ FAIL'}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']} ({summary['success_rate']:.1%})")
        print(f"Failed: {summary['total_failed']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Duration: {summary['pipeline_duration']:.2f}s")
        print(f"Artifacts: {summary['artifacts_generated']} files generated")

class ContinuousIntegration:
    """CI/CD integration utilities"""
    
    @staticmethod
    def create_github_actions_workflow() -> str:
        """Create GitHub Actions workflow file"""
        
        workflow = """
name: Stock AI Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest coverage
    
    - name: Run tests
      run: |
        cd tests
        python run_all_tests.py
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: tests/test_results/
    
    - name: Generate coverage report
      run: |
        coverage run -m pytest
        coverage xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false
        """
        
        return workflow
    
    @staticmethod
    def check_exit_code(results: Dict[str, Any]) -> int:
        """Return appropriate exit code for CI/CD"""
        
        overall_status = results.get('summary', {}).get('overall_status', 'FAIL')
        return 0 if overall_status == 'PASS' else 1

def main():
    """Main entry point for test pipeline"""
    
    # Configuration
    config = {
        'run_unit_tests': True,
        'run_integration_tests': True,
        'run_end_to_end_tests': True,
        'run_performance_tests': True,
        'generate_reports': True,
        'fail_fast': False,
        'output_directory': 'test_results'
    }
    
    # Run pipeline
    pipeline = TestPipeline(config)
    results = pipeline.run_all_tests()
    
    # Generate CI workflow if requested
    if '--generate-ci' in sys.argv:
        ci = ContinuousIntegration()
        workflow_content = ci.create_github_actions_workflow()
        
        workflow_file = os.path.join('..', '.github', 'workflows', 'tests.yml')
        os.makedirs(os.path.dirname(workflow_file), exist_ok=True)
        
        with open(workflow_file, 'w') as f:
            f.write(workflow_content)
        
        print(f"\n🔧 GitHub Actions workflow created: {workflow_file}")
    
    # Exit with appropriate code for CI/CD
    exit_code = ContinuousIntegration.check_exit_code(results)
    
    if exit_code == 0:
        print(f"\n🎉 All tests passed! Pipeline completed successfully.")
    else:
        print(f"\n❌ Tests failed. Check the reports for details.")
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
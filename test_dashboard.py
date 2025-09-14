#!/usr/bin/env python3
"""
Simple test script for the Stock AI Dashboard

Just a basic test to make sure everything is working.
This is my first time writing tests, so it's pretty simple!
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if we can import our dashboard"""
    print("🧪 Testing imports...")
    try:
        from src.frontend.simple_dashboard import StockDashboard
        print("✅ Dashboard import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dependencies():
    """Test if required packages are installed"""
    print("🧪 Testing dependencies...")
    required_packages = ['dash', 'plotly', 'pandas', 'yfinance', 'numpy']
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            all_good = False
    
    return all_good

def test_dashboard_creation():
    """Test creating dashboard instance"""
    print("🧪 Testing dashboard creation...")
    try:
        from src.frontend.simple_dashboard import StockDashboard
        dashboard = StockDashboard()
        print("✅ Dashboard created successfully")
        return True
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
        return False

def test_demo_data():
    """Test demo data generation"""
    print("🧪 Testing demo data...")
    try:
        from src.frontend.simple_dashboard import StockDashboard
        dashboard = StockDashboard()
        data = dashboard.create_demo_data('AAPL')
        if not data.empty:
            print(f"✅ Demo data created: {len(data)} rows")
            return True
        else:
            print("❌ Demo data is empty")
            return False
    except Exception as e:
        print(f"❌ Demo data test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("🚀 Stock AI Dashboard - Simple Tests")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_imports,
        test_dashboard_creation,
        test_demo_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Empty line between tests
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work.")
        return 0
    else:
        print("😅 Some tests failed. Check the requirements.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
Pytest configuration and fixtures
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing"""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for return_rate in returns[1:]:
        new_price = prices[-1] * (1 + return_rate)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': [p * np.random.uniform(1.01, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 0.99) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return df

@pytest.fixture
def sample_sequences():
    """Generate sample sequence data for model testing"""
    sequence_length = 60
    n_features = 5
    n_samples = 100
    
    np.random.seed(42)
    X = np.random.normal(0, 1, (n_samples, sequence_length, n_features))
    y = np.random.normal(0, 1, n_samples)
    
    return X, y

@pytest.fixture
def mock_model_config():
    """Mock configuration for testing"""
    return {
        'sequence_length': 60,
        'n_features': 5,
        'epochs': 1,  # Minimal for testing
        'batch_size': 32,
        'learning_rate': 0.001
    }

@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_dir, exist_ok=True)
    return test_dir
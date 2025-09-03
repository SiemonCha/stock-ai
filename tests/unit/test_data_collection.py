"""
Unit tests for data collection modules
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

try:
    from data.stock_data import StockDataCollector, get_stock_data_yfinance
except ImportError:
    pytest.skip("Data collection modules not available", allow_module_level=True)

class TestStockDataCollector:
    """Test StockDataCollector class"""
    
    def test_init(self):
        """Test collector initialization"""
        collector = StockDataCollector()
        assert collector.cache == {}
    
    @patch('data.stock_data.yf.Ticker')
    def test_get_stock_data_success(self, mock_ticker, sample_stock_data):
        """Test successful data retrieval"""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector()
        result = collector.get_stock_data('AAPL', period='1y')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Close' in result.columns
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('data.stock_data.yf.Ticker')
    def test_get_stock_data_empty_response(self, mock_ticker):
        """Test handling of empty data response"""
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector()
        result = collector.get_stock_data('INVALID', period='1y')
        
        assert result is None
    
    @patch('data.stock_data.yf.Ticker')
    def test_get_stock_data_exception(self, mock_ticker):
        """Test handling of exceptions during data retrieval"""
        # Mock exception
        mock_ticker.side_effect = Exception("API Error")
        
        collector = StockDataCollector()
        result = collector.get_stock_data('AAPL', period='1y')
        
        assert result is None
    
    def test_cache_functionality(self, sample_stock_data):
        """Test data caching functionality"""
        collector = StockDataCollector()
        
        # Add data to cache
        cache_key = "AAPL_1y_1d"
        collector.cache[cache_key] = sample_stock_data
        
        # Should return cached data
        result = collector.get_stock_data('AAPL', period='1y')
        pd.testing.assert_frame_equal(result, sample_stock_data)
    
    def test_get_multiple_stocks(self, sample_stock_data):
        """Test multiple stock data retrieval"""
        collector = StockDataCollector()
        
        # Mock get_stock_data method
        def mock_get_data(symbol, period):
            if symbol in ['AAPL', 'GOOGL']:
                return sample_stock_data.copy()
            return None
        
        collector.get_stock_data = mock_get_data
        
        symbols = ['AAPL', 'GOOGL', 'INVALID']
        results = collector.get_multiple_stocks(symbols)
        
        assert len(results) == 2
        assert 'AAPL' in results
        assert 'GOOGL' in results
        assert 'INVALID' not in results
    
    def test_get_market_data(self):
        """Test market indices data retrieval"""
        collector = StockDataCollector()
        
        # Mock get_multiple_stocks method
        mock_data = {'SPY': pd.DataFrame({'Close': [100, 101, 102]})}
        collector.get_multiple_stocks = Mock(return_value=mock_data)
        
        results = collector.get_market_data()
        
        assert isinstance(results, dict)
        collector.get_multiple_stocks.assert_called_once()
    
    def test_clear_cache(self):
        """Test cache clearing functionality"""
        collector = StockDataCollector()
        collector.cache['test'] = 'data'
        
        collector.clear_cache()
        assert collector.cache == {}

class TestLegacyFunctions:
    """Test legacy compatibility functions"""
    
    @patch('data.stock_data.StockDataCollector')
    def test_get_stock_data_yfinance(self, mock_collector_class, sample_stock_data):
        """Test legacy function wrapper"""
        mock_collector = Mock()
        mock_collector.get_stock_data.return_value = sample_stock_data
        mock_collector_class.return_value = mock_collector
        
        result = get_stock_data_yfinance('AAPL', '1y')
        
        pd.testing.assert_frame_equal(result, sample_stock_data)
        mock_collector.get_stock_data.assert_called_once_with('AAPL', '1y')
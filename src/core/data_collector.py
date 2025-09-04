"""
Stock Data Collection Module
Unified interface for stock data collection from various sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class StockDataCollector:
    """Unified stock data collector"""
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol: str, period: str = "5y", 
                      interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get stock data using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache first
            if cache_key in self.cache:
                logger.info(f"Using cached data for {symbol}")
                return self.cache[cache_key]
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.error(f"No data found for symbol {symbol}")
                return None
            
            # Clean data
            df = df.dropna()
            
            # Cache the data
            self.cache[cache_key] = df
            
            logger.info(f"Downloaded {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            df = self.get_stock_data(symbol, period)
            if df is not None:
                results[symbol] = df
        
        return results
    
    def get_market_data(self, indices: List[str] = None, period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        Get market indices data
        
        Args:
            indices: List of index symbols (default: major indices)
            period: Data period
        
        Returns:
            Dictionary mapping index symbols to DataFrames
        """
        if indices is None:
            indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']  # S&P 500, Dow, Nasdaq, Russell, VIX
        
        return self.get_multiple_stocks(indices, period)
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("Data cache cleared")

# For backward compatibility
def get_stock_data_yfinance(symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """Legacy function for backward compatibility"""
    collector = StockDataCollector()
    return collector.get_stock_data(symbol, period)
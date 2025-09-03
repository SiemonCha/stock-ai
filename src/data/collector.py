import yfinance as yf
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

class StockDataCollector:
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if self.alpha_vantage_key:
            self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
    
    def get_stock_data_yfinance(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetch stock data using yfinance (free, no API key required)
        """
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        # Standardize column names
        data.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
        data = data.drop(['dividends', 'stock_splits'], axis=1)
        data.index.name = 'date'
        
        return data
    
    def get_stock_data_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock data using Alpha Vantage API
        """
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not provided")
        
        data, _ = self.ts.get_daily_adjusted(symbol=symbol, outputsize='full')
        
        # Standardize column names
        data.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
        data = data.drop(['dividend_amount', 'split_coefficient'], axis=1)
        data.index.name = 'date'
        data = data.sort_index()
        
        return data
    
    def get_sp500_symbols(self) -> List[str]:
        """
        Get list of S&P 500 stock symbols
        """
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return table['Symbol'].tolist()
        except:
            # Fallback list of popular S&P 500 stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
    
    def collect_multiple_stocks(self, symbols: List[str], use_alpha_vantage: bool = False) -> dict:
        """
        Collect data for multiple stocks
        """
        data = {}
        
        for symbol in symbols:
            try:
                if use_alpha_vantage and self.alpha_vantage_key:
                    stock_data = self.get_stock_data_alpha_vantage(symbol)
                else:
                    stock_data = self.get_stock_data_yfinance(symbol)
                
                data[symbol] = stock_data
                print(f"Successfully collected data for {symbol}")
                
            except Exception as e:
                print(f"Error collecting data for {symbol}: {str(e)}")
        
        return data
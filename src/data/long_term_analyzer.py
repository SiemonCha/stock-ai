import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import requests
from fredapi import FRED
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

class LongTermMarketAnalyzer:
    """
    Comprehensive long-term market analysis using 10+ years of data
    """
    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred = FRED(api_key=fred_api_key) if fred_api_key else None
        self.market_cycles = []
        self.seasonal_patterns = {}
        self.long_term_trends = {}
        
    def collect_comprehensive_data(self, symbol: str, years: int = 20) -> Dict:
        """
        Collect comprehensive multi-year data including market context
        """
        print(f"📊 Collecting {years} years of comprehensive data for {symbol}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        data_collection = {
            'stock_data': None,
            'market_data': {},
            'economic_data': {},
            'sector_data': {},
            'peer_data': {},
            'options_data': {},
            'insider_data': {}
        }
        
        # Collect stock data with multiple timeframes
        data_collection['stock_data'] = self._collect_multi_timeframe_data(symbol, start_date, end_date)
        
        # Collect market context data in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self._collect_market_indices, start_date, end_date): 'market_data',
                executor.submit(self._collect_economic_indicators, start_date, end_date): 'economic_data',
                executor.submit(self._collect_sector_data, symbol, start_date, end_date): 'sector_data',
                executor.submit(self._collect_peer_analysis, symbol, start_date, end_date): 'peer_data',
                executor.submit(self._collect_options_data, symbol): 'options_data',
                executor.submit(self._collect_insider_data, symbol): 'insider_data'
            }
            
            for future in futures:
                data_type = futures[future]
                try:
                    result = future.result(timeout=60)
                    data_collection[data_type] = result
                    print(f"✅ Collected {data_type}")
                except Exception as e:
                    print(f"⚠️ Error collecting {data_type}: {str(e)}")
                    data_collection[data_type] = {}
        
        return data_collection
    
    def _collect_multi_timeframe_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """
        Collect stock data across multiple timeframes
        """
        timeframes = {
            'daily': '1d',
            'weekly': '1wk', 
            'monthly': '1mo'
        }
        
        stock_data = {}
        ticker = yf.Ticker(symbol)
        
        for tf_name, tf_interval in timeframes.items():
            try:
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=tf_interval
                )
                
                if not data.empty:
                    stock_data[tf_name] = data
                    print(f"   📈 {tf_name}: {len(data)} periods")
                else:
                    print(f"   ⚠️ No {tf_name} data available")
                    
            except Exception as e:
                print(f"   ❌ Error collecting {tf_name} data: {str(e)}")
                stock_data[tf_name] = pd.DataFrame()
        
        return stock_data
    
    def _collect_market_indices(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Collect major market indices for context
        """
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'IWM': 'Russell 2000',
            'VTI': 'Total Stock Market',
            'EFA': 'International Developed',
            'EEM': 'Emerging Markets',
            'TLT': '20+ Year Treasury',
            'GLD': 'Gold',
            'VIX': 'Volatility Index',
            'DXY': 'Dollar Index'
        }
        
        market_data = {}
        
        for symbol, name in indices.items():
            try:
                if symbol == 'VIX':
                    # VIX needs special handling
                    data = yf.download('^VIX', start=start_date, end=end_date)
                elif symbol == 'DXY':
                    # Dollar Index
                    data = yf.download('DX-Y.NYB', start=start_date, end=end_date)
                else:
                    data = yf.download(symbol, start=start_date, end=end_date)
                
                if not data.empty:
                    market_data[symbol] = {
                        'name': name,
                        'data': data,
                        'returns': data['Close'].pct_change(),
                        'volatility': data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
                    }
                    
            except Exception as e:
                print(f"      ⚠️ Error collecting {symbol}: {str(e)}")
        
        return market_data
    
    def _collect_economic_indicators(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Collect economic indicators from FRED
        """
        economic_indicators = {
            'GDP': 'GDPC1',           # Real GDP
            'INFLATION': 'CPIAUCSL',  # CPI
            'UNEMPLOYMENT': 'UNRATE', # Unemployment Rate
            'FED_RATE': 'FEDFUNDS',   # Federal Funds Rate
            'YIELD_10Y': 'DGS10',     # 10-Year Treasury
            'YIELD_2Y': 'DGS2',       # 2-Year Treasury  
            'CONSUMER_SENTIMENT': 'UMCSENT',  # Consumer Sentiment
            'ISM_PMI': 'ISMMI',       # ISM Manufacturing PMI
            'HOUSING_STARTS': 'HOUST', # Housing Starts
            'RETAIL_SALES': 'RSALES', # Retail Sales
        }
        
        economic_data = {}
        
        if not self.fred:
            # Fallback: Create synthetic economic data based on market patterns
            print("      📊 Using synthetic economic indicators (no FRED API)")
            for indicator in economic_indicators.keys():
                dates = pd.date_range(start_date, end_date, freq='D')
                values = np.random.normal(0, 0.01, len(dates))  # Synthetic data
                economic_data[indicator] = pd.Series(values, index=dates)
            return economic_data
        
        for indicator_name, fred_code in economic_indicators.items():
            try:
                data = self.fred.get_series(
                    fred_code,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty:
                    # Resample to daily and forward fill
                    data_daily = data.resample('D').ffill()
                    economic_data[indicator_name] = data_daily
                    
            except Exception as e:
                print(f"      ⚠️ Error collecting {indicator_name}: {str(e)}")
                # Create fallback synthetic data
                dates = pd.date_range(start_date, end_date, freq='D')
                values = np.random.normal(0, 0.01, len(dates))
                economic_data[indicator_name] = pd.Series(values, index=dates)
        
        return economic_data
    
    def _collect_sector_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """
        Collect sector and industry data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
        except:
            sector = 'Unknown'
            industry = 'Unknown'
        
        # Major sector ETFs
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financial Services': 'XLF',
            'Consumer Cyclical': 'XLY',
            'Consumer Defensive': 'XLP',
            'Industrial': 'XLI',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB',
            'Communication Services': 'XLC'
        }
        
        sector_data = {
            'company_sector': sector,
            'company_industry': industry,
            'sector_performance': {}
        }
        
        # Get sector ETF data
        for sector_name, etf_symbol in sector_etfs.items():
            try:
                data = yf.download(etf_symbol, start=start_date, end=end_date)
                if not data.empty:
                    sector_data['sector_performance'][sector_name] = {
                        'etf': etf_symbol,
                        'data': data,
                        'returns': data['Close'].pct_change()
                    }
            except Exception as e:
                print(f"      ⚠️ Error collecting sector {sector_name}: {str(e)}")
        
        return sector_data
    
    def _collect_peer_analysis(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """
        Collect peer company data for relative analysis
        """
        try:
            # Get basic info to determine peers
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            market_cap = info.get('marketCap', 0)
        except:
            return {'peers': [], 'peer_data': {}}
        
        # Define common peers by symbol (simplified approach)
        common_peers = {
            'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META'],
            'GOOGL': ['AAPL', 'MSFT', 'AMZN', 'META'],
            'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'META'],
            'TSLA': ['GM', 'F', 'NIO', 'RIVN'],
            'AMZN': ['AAPL', 'MSFT', 'GOOGL', 'META'],
            'META': ['AAPL', 'MSFT', 'GOOGL', 'TWTR'],
            'JPM': ['BAC', 'WFC', 'C', 'GS'],
            'JNJ': ['PFE', 'MRK', 'UNH', 'ABBV']
        }
        
        peers = common_peers.get(symbol, [])
        peer_data = {}
        
        for peer in peers:
            try:
                data = yf.download(peer, start=start_date, end=end_date)
                if not data.empty:
                    peer_data[peer] = {
                        'data': data,
                        'returns': data['Close'].pct_change(),
                        'correlation': None  # Will be calculated later
                    }
            except Exception as e:
                print(f"      ⚠️ Error collecting peer {peer}: {str(e)}")
        
        return {
            'peers': peers,
            'peer_data': peer_data,
            'sector': sector,
            'industry': industry
        }
    
    def _collect_options_data(self, symbol: str) -> Dict:
        """
        Collect options data for sentiment analysis
        """
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return {'available': False}
            
            # Get options for next expiration
            next_expiry = options_dates[0]
            options_chain = ticker.option_chain(next_expiry)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Calculate put/call ratio
            call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            put_call_ratio = put_volume / (call_volume + 1e-8)
            
            # Calculate implied volatility
            call_iv = calls['impliedVolatility'].mean() if 'impliedVolatility' in calls.columns else 0
            put_iv = puts['impliedVolatility'].mean() if 'impliedVolatility' in puts.columns else 0
            
            return {
                'available': True,
                'next_expiry': next_expiry,
                'put_call_ratio': put_call_ratio,
                'call_implied_vol': call_iv,
                'put_implied_vol': put_iv,
                'call_volume': call_volume,
                'put_volume': put_volume
            }
            
        except Exception as e:
            print(f"      ⚠️ Error collecting options data: {str(e)}")
            return {'available': False}
    
    def _collect_insider_data(self, symbol: str) -> Dict:
        """
        Collect insider trading data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get institutional holders
            institutional_holders = ticker.institutional_holders
            major_holders = ticker.major_holders
            
            insider_data = {
                'available': True,
                'institutional_holders': institutional_holders if institutional_holders is not None else pd.DataFrame(),
                'major_holders': major_holders if major_holders is not None else pd.DataFrame(),
                'insider_sentiment': 'neutral'  # Simplified
            }
            
            # Calculate institutional ownership percentage
            if not institutional_holders.empty and 'Shares' in institutional_holders.columns:
                total_institutional = institutional_holders['Shares'].sum()
                ticker_info = ticker.info
                shares_outstanding = ticker_info.get('sharesOutstanding', 0)
                
                if shares_outstanding > 0:
                    institutional_pct = (total_institutional / shares_outstanding) * 100
                    insider_data['institutional_ownership_pct'] = institutional_pct
            
            return insider_data
            
        except Exception as e:
            print(f"      ⚠️ Error collecting insider data: {str(e)}")
            return {'available': False}
    
    def analyze_long_term_patterns(self, comprehensive_data: Dict) -> Dict:
        """
        Analyze long-term patterns and cycles
        """
        print("🔍 Analyzing long-term patterns and cycles...")
        
        analysis = {
            'market_cycles': self._detect_market_cycles(comprehensive_data),
            'seasonal_patterns': self._detect_seasonal_patterns(comprehensive_data),
            'long_term_trends': self._analyze_long_term_trends(comprehensive_data),
            'correlation_analysis': self._perform_correlation_analysis(comprehensive_data),
            'regime_analysis': self._analyze_market_regimes(comprehensive_data)
        }
        
        return analysis
    
    def _detect_market_cycles(self, data: Dict) -> Dict:
        """
        Detect market cycles and phases
        """
        stock_data = data['stock_data'].get('daily', pd.DataFrame())
        
        if stock_data.empty:
            return {'cycles_detected': False}
        
        prices = stock_data['Close']
        
        # Detect major cycles using rolling highs/lows
        cycle_analysis = {
            'cycles_detected': True,
            'bull_markets': [],
            'bear_markets': [],
            'current_cycle_phase': 'unknown',
            'cycle_duration_avg': 0,
            'cycle_strength': 0
        }
        
        # Simple cycle detection using 200-day moving average crossovers
        ma_200 = prices.rolling(200).mean()
        
        # Find crossovers
        above_ma = prices > ma_200
        cycle_changes = above_ma != above_ma.shift(1)
        
        change_dates = prices[cycle_changes].index
        
        bull_periods = []
        bear_periods = []
        
        for i, date in enumerate(change_dates[:-1]):
            next_date = change_dates[i + 1]
            duration = (next_date - date).days
            
            if above_ma.loc[date]:
                bull_periods.append({
                    'start': date,
                    'end': next_date,
                    'duration_days': duration,
                    'return': (prices.loc[next_date] - prices.loc[date]) / prices.loc[date]
                })
            else:
                bear_periods.append({
                    'start': date,
                    'end': next_date, 
                    'duration_days': duration,
                    'return': (prices.loc[next_date] - prices.loc[date]) / prices.loc[date]
                })
        
        cycle_analysis['bull_markets'] = bull_periods[-5:]  # Last 5 bull markets
        cycle_analysis['bear_markets'] = bear_periods[-5:]  # Last 5 bear markets
        
        # Determine current phase
        current_price = prices.iloc[-1]
        current_ma = ma_200.iloc[-1]
        cycle_analysis['current_cycle_phase'] = 'bull' if current_price > current_ma else 'bear'
        
        # Calculate average cycle duration
        all_cycles = bull_periods + bear_periods
        if all_cycles:
            cycle_analysis['cycle_duration_avg'] = np.mean([c['duration_days'] for c in all_cycles])
        
        return cycle_analysis
    
    def _detect_seasonal_patterns(self, data: Dict) -> Dict:
        """
        Detect seasonal patterns in stock performance
        """
        stock_data = data['stock_data'].get('daily', pd.DataFrame())
        
        if stock_data.empty:
            return {'seasonal_patterns_detected': False}
        
        prices = stock_data['Close']
        returns = prices.pct_change()
        
        seasonal_analysis = {
            'seasonal_patterns_detected': True,
            'monthly_performance': {},
            'quarterly_performance': {},
            'day_of_week_performance': {},
            'best_months': [],
            'worst_months': [],
            'holiday_effects': {}
        }
        
        # Monthly analysis
        monthly_returns = returns.groupby(returns.index.month).mean()
        seasonal_analysis['monthly_performance'] = monthly_returns.to_dict()
        
        # Find best and worst months
        sorted_months = monthly_returns.sort_values(ascending=False)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        seasonal_analysis['best_months'] = [
            {'month': month_names[int(month)-1], 'avg_return': ret} 
            for month, ret in sorted_months.head(3).items()
        ]
        
        seasonal_analysis['worst_months'] = [
            {'month': month_names[int(month)-1], 'avg_return': ret} 
            for month, ret in sorted_months.tail(3).items()
        ]
        
        # Quarterly analysis
        quarterly_returns = returns.groupby(returns.index.quarter).mean()
        seasonal_analysis['quarterly_performance'] = {
            f'Q{q}': ret for q, ret in quarterly_returns.items()
        }
        
        # Day of week analysis
        dow_returns = returns.groupby(returns.index.dayofweek).mean()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        seasonal_analysis['day_of_week_performance'] = {
            day_names[dow]: ret for dow, ret in dow_returns.items() if dow < 5
        }
        
        return seasonal_analysis
    
    def _analyze_long_term_trends(self, data: Dict) -> Dict:
        """
        Analyze long-term trends and secular patterns
        """
        stock_data = data['stock_data'].get('daily', pd.DataFrame())
        
        if stock_data.empty:
            return {'trend_analysis_available': False}
        
        prices = stock_data['Close']
        
        trend_analysis = {
            'trend_analysis_available': True,
            'secular_trend': 'unknown',
            'trend_strength': 0,
            'support_resistance_levels': {},
            'fibonacci_levels': {},
            'breakout_analysis': {}
        }
        
        # Calculate long-term trend using linear regression
        x = np.arange(len(prices))
        y = np.log(prices)  # Log prices for better trend analysis
        
        trend_coef = np.polyfit(x, y, 1)[0]
        
        # Determine secular trend
        if trend_coef > 0.001:  # More than 0.1% daily growth
            trend_analysis['secular_trend'] = 'strong_uptrend'
        elif trend_coef > 0.0001:
            trend_analysis['secular_trend'] = 'moderate_uptrend'
        elif trend_coef > -0.0001:
            trend_analysis['secular_trend'] = 'sideways'
        elif trend_coef > -0.001:
            trend_analysis['secular_trend'] = 'moderate_downtrend'
        else:
            trend_analysis['secular_trend'] = 'strong_downtrend'
        
        trend_analysis['trend_strength'] = abs(trend_coef) * 252 * 100  # Annualized percentage
        
        # Calculate support and resistance levels
        recent_high = prices.tail(252).max()  # 1-year high
        recent_low = prices.tail(252).min()   # 1-year low
        current_price = prices.iloc[-1]
        
        trend_analysis['support_resistance_levels'] = {
            'recent_high': recent_high,
            'recent_low': recent_low,
            'current_price': current_price,
            'resistance_1': recent_high * 0.95,
            'support_1': recent_low * 1.05,
            'distance_from_high_pct': ((recent_high - current_price) / recent_high) * 100,
            'distance_from_low_pct': ((current_price - recent_low) / recent_low) * 100
        }
        
        # Fibonacci retracement levels
        price_range = recent_high - recent_low
        trend_analysis['fibonacci_levels'] = {
            'fib_23.6': recent_high - (price_range * 0.236),
            'fib_38.2': recent_high - (price_range * 0.382),
            'fib_50.0': recent_high - (price_range * 0.500),
            'fib_61.8': recent_high - (price_range * 0.618),
            'fib_78.6': recent_high - (price_range * 0.786)
        }
        
        return trend_analysis
    
    def _perform_correlation_analysis(self, data: Dict) -> Dict:
        """
        Perform correlation analysis with market and economic factors
        """
        stock_data = data['stock_data'].get('daily', pd.DataFrame())
        market_data = data.get('market_data', {})
        economic_data = data.get('economic_data', {})
        
        if stock_data.empty:
            return {'correlation_analysis_available': False}
        
        stock_returns = stock_data['Close'].pct_change()
        
        correlation_analysis = {
            'correlation_analysis_available': True,
            'market_correlations': {},
            'economic_correlations': {},
            'sector_correlations': {},
            'correlation_stability': {}
        }
        
        # Market correlations
        for symbol, market_info in market_data.items():
            try:
                market_returns = market_info['data']['Close'].pct_change()
                
                # Align dates
                aligned_stock, aligned_market = stock_returns.align(market_returns, join='inner')
                
                if len(aligned_stock) > 50:  # Need sufficient data
                    correlation = aligned_stock.corr(aligned_market)
                    
                    if not np.isnan(correlation):
                        correlation_analysis['market_correlations'][symbol] = {
                            'correlation': correlation,
                            'name': market_info['name'],
                            'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'
                        }
                        
            except Exception as e:
                continue
        
        # Economic correlations
        for indicator, econ_series in economic_data.items():
            try:
                # Convert economic data to returns
                econ_returns = econ_series.pct_change()
                
                # Align dates
                aligned_stock, aligned_econ = stock_returns.align(econ_returns, join='inner')
                
                if len(aligned_stock) > 50:
                    correlation = aligned_stock.corr(aligned_econ)
                    
                    if not np.isnan(correlation):
                        correlation_analysis['economic_correlations'][indicator] = {
                            'correlation': correlation,
                            'strength': 'strong' if abs(correlation) > 0.3 else 'moderate' if abs(correlation) > 0.1 else 'weak'
                        }
                        
            except Exception as e:
                continue
        
        return correlation_analysis
    
    def _analyze_market_regimes(self, data: Dict) -> Dict:
        """
        Analyze different market regimes and their impact
        """
        stock_data = data['stock_data'].get('daily', pd.DataFrame())
        market_data = data.get('market_data', {})
        
        if stock_data.empty:
            return {'regime_analysis_available': False}
        
        # Get SPY data for market regime detection
        spy_data = market_data.get('SPY', {}).get('data', pd.DataFrame())
        
        regime_analysis = {
            'regime_analysis_available': True,
            'current_regime': 'unknown',
            'regime_performance': {},
            'volatility_regimes': {},
            'regime_transitions': []
        }
        
        if not spy_data.empty:
            spy_returns = spy_data['Close'].pct_change()
            stock_returns = stock_data['Close'].pct_change()
            
            # Define market regimes based on SPY performance and volatility
            spy_vol = spy_returns.rolling(20).std() * np.sqrt(252)  # Annualized volatility
            
            # Regime classification
            regimes = pd.Series(index=spy_returns.index, dtype='object')
            
            for date in spy_returns.index:
                if date in spy_vol.index:
                    vol = spy_vol[date]
                    ret_20d = spy_returns.rolling(20).mean().get(date, 0) * 252  # Annualized
                    
                    if vol > 0.3:  # High volatility
                        regimes[date] = 'crisis' if ret_20d < -0.1 else 'volatile'
                    elif ret_20d > 0.1:  # Strong positive returns
                        regimes[date] = 'bull'
                    elif ret_20d < -0.05:  # Negative returns
                        regimes[date] = 'bear'
                    else:
                        regimes[date] = 'sideways'
            
            # Analyze performance in each regime
            for regime in ['bull', 'bear', 'sideways', 'volatile', 'crisis']:
                regime_mask = regimes == regime
                if regime_mask.sum() > 0:
                    regime_returns = stock_returns[regime_mask]
                    
                    regime_analysis['regime_performance'][regime] = {
                        'avg_return': regime_returns.mean(),
                        'volatility': regime_returns.std(),
                        'sharpe_ratio': regime_returns.mean() / (regime_returns.std() + 1e-8),
                        'periods': int(regime_mask.sum()),
                        'win_rate': (regime_returns > 0).mean()
                    }
            
            # Current regime
            if len(regimes) > 0:
                regime_analysis['current_regime'] = regimes.iloc[-1]
        
        return regime_analysis
    
    def create_adaptive_sequences(self, comprehensive_data: Dict, base_sequence_length: int = 252) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create adaptive sequences based on market conditions and data availability
        """
        stock_data = comprehensive_data['stock_data'].get('daily', pd.DataFrame())
        
        if stock_data.empty:
            raise ValueError("No daily stock data available")
        
        print(f"🔧 Creating adaptive sequences with {len(stock_data)} days of data...")
        
        # Create comprehensive feature matrix
        enhanced_features = self._create_comprehensive_features(comprehensive_data)
        
        # Dynamic sequence length based on data availability and market regime
        market_volatility = stock_data['Close'].pct_change().rolling(20).std().iloc[-1]
        
        # Adjust sequence length based on volatility (higher volatility = shorter memory)
        if market_volatility > 0.03:  # High volatility
            sequence_length = base_sequence_length // 2  # 126 days
        elif market_volatility > 0.02:  # Medium volatility
            sequence_length = int(base_sequence_length * 0.75)  # 189 days
        else:  # Low volatility
            sequence_length = base_sequence_length  # 252 days (1 year)
        
        print(f"   📊 Using adaptive sequence length: {sequence_length} days")
        print(f"   🎯 Based on market volatility: {market_volatility:.1%}")
        
        # Create sequences
        X, y = self._create_sequences_from_features(enhanced_features, sequence_length)
        
        feature_names = list(enhanced_features.columns)
        
        print(f"   ✅ Created {len(X)} sequences with {len(feature_names)} features")
        
        return X, y, feature_names
    
    def _create_comprehensive_features(self, comprehensive_data: Dict) -> pd.DataFrame:
        """
        Create comprehensive feature matrix from all collected data
        """
        stock_data = comprehensive_data['stock_data'].get('daily', pd.DataFrame())
        market_data = comprehensive_data.get('market_data', {})
        economic_data = comprehensive_data.get('economic_data', {})
        
        # Start with stock price data
        features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Add market context features
        for symbol, market_info in market_data.items():
            try:
                market_prices = market_info['data']['Close']
                market_returns = market_prices.pct_change()
                
                # Align with stock data
                aligned_returns = market_returns.reindex(features.index, method='ffill')
                
                features[f'{symbol}_price'] = market_prices.reindex(features.index, method='ffill')
                features[f'{symbol}_return'] = aligned_returns
                features[f'{symbol}_vol'] = aligned_returns.rolling(20).std()
                
            except Exception as e:
                continue
        
        # Add economic features
        for indicator, econ_series in economic_data.items():
            try:
                # Align economic data with stock data
                aligned_econ = econ_series.reindex(features.index, method='ffill')
                features[f'econ_{indicator}'] = aligned_econ
                features[f'econ_{indicator}_change'] = aligned_econ.pct_change()
                
            except Exception as e:
                continue
        
        # Add derived features
        features = self._add_advanced_derived_features(features)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    def _add_advanced_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced derived features
        """
        enhanced = features.copy()
        
        # Multi-timeframe moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            enhanced[f'sma_{window}'] = features['Close'].rolling(window).mean()
            enhanced[f'price_vs_sma_{window}'] = features['Close'] / enhanced[f'sma_{window}'] - 1
        
        # Volatility features across multiple windows
        returns = features['Close'].pct_change()
        for window in [5, 10, 20, 50, 100]:
            enhanced[f'vol_{window}'] = returns.rolling(window).std()
            enhanced[f'vol_ratio_{window}'] = enhanced[f'vol_{window}'] / enhanced['vol_20']
        
        # Momentum features
        for window in [5, 10, 20, 50, 100]:
            enhanced[f'momentum_{window}'] = features['Close'].pct_change(window)
            enhanced[f'roc_{window}'] = (features['Close'] - features['Close'].shift(window)) / features['Close'].shift(window)
        
        # Volume features
        enhanced['volume_sma_20'] = features['Volume'].rolling(20).mean()
        enhanced['volume_ratio'] = features['Volume'] / enhanced['volume_sma_20']
        enhanced['volume_price_trend'] = (features['Volume'] * (features['Close'] - features['Close'].shift(1))).rolling(10).sum()
        
        # Advanced technical indicators
        enhanced['rsi'] = self._calculate_rsi(features['Close'])
        enhanced['macd'], enhanced['macd_signal'] = self._calculate_macd(features['Close'])
        enhanced['bb_upper'], enhanced['bb_lower'] = self._calculate_bollinger_bands(features['Close'])
        
        # Market structure features
        enhanced['high_low_ratio'] = features['High'] / features['Low']
        enhanced['open_close_ratio'] = features['Open'] / features['Close']
        enhanced['true_range'] = np.maximum(
            features['High'] - features['Low'],
            np.maximum(
                abs(features['High'] - features['Close'].shift(1)),
                abs(features['Low'] - features['Close'].shift(1))
            )
        )
        
        return enhanced
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def _create_sequences_from_features(self, features: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training sequences from feature matrix
        """
        # Remove non-numeric columns and handle NaN
        numeric_features = features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Target is next day's close price return
        target = numeric_features['Close'].pct_change().shift(-1)
        
        X, y = [], []
        
        for i in range(sequence_length, len(numeric_features) - 1):
            X.append(numeric_features.iloc[i-sequence_length:i].values)
            y.append(target.iloc[i])
        
        return np.array(X), np.array(y)
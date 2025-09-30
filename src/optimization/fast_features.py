"""
Ultra-Fast Feature Engineering Pipeline
Optimized feature engineering with vectorized operations and caching
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import ta  # Technical Analysis library

try:
    import numba
    from numba import jit, vectorize, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Fast vectorized technical indicators using Numba
@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Ultra-fast Simple Moving Average"""
    sma = np.empty(len(prices))
    sma[:period-1] = np.nan
    
    for i in range(period-1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    
    return sma

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Ultra-fast Exponential Moving Average"""
    alpha = 2.0 / (period + 1)
    ema = np.empty(len(prices))
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Ultra-fast RSI calculation"""
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)
    
    # Calculate average gains and losses
    avg_gains = np.empty(len(changes))
    avg_losses = np.empty(len(changes))
    
    # Initial values
    avg_gains[period-1] = np.mean(gains[:period])
    avg_losses[period-1] = np.mean(losses[:period])
    
    # Smooth with EMA
    alpha = 1.0 / period
    for i in range(period, len(changes)):
        avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i-1]
        avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i-1]
    
    # Calculate RSI
    rs = avg_gains / np.where(avg_losses == 0, 1e-10, avg_losses)
    rsi = 100 - (100 / (1 + rs))
    
    # Prepend NaN for period adjustment
    return np.concatenate([np.full(period, np.nan), rsi])

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_bollinger_bands(prices: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast Bollinger Bands"""
    sma = fast_sma(prices, period)
    
    # Calculate rolling standard deviation
    std = np.empty(len(prices))
    std[:period-1] = np.nan
    
    for i in range(period-1, len(prices)):
        std[i] = np.std(prices[i-period+1:i+1])
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast MACD calculation"""
    ema_fast = fast_ema(prices, fast_period)
    ema_slow = fast_ema(prices, slow_period)
    
    macd = ema_fast - ema_slow
    signal = fast_ema(macd, signal_period)
    histogram = macd - signal
    
    return macd, signal, histogram

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Ultra-fast Stochastic Oscillator"""
    k_percent = np.empty(len(close))
    
    for i in range(k_period-1, len(close)):
        highest_high = np.max(high[i-k_period+1:i+1])
        lowest_low = np.min(low[i-k_period+1:i+1])
        
        if highest_high == lowest_low:
            k_percent[i] = 50.0
        else:
            k_percent[i] = 100.0 * (close[i] - lowest_low) / (highest_high - lowest_low)
    
    k_percent[:k_period-1] = np.nan
    d_percent = fast_sma(k_percent, 3)  # %D is 3-period SMA of %K
    
    return k_percent, d_percent

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Ultra-fast Average True Range"""
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = fast_ema(tr, period)
    return atr

class FastFeatureEngine:
    """
    Ultra-fast feature engineering engine with vectorized operations
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.scalers = {}
        
        # Feature computation cache
        self.computation_cache = {}
        
        # Feature groups for parallel processing
        self.feature_groups = {
            'price_features': self._compute_price_features,
            'volume_features': self._compute_volume_features,
            'volatility_features': self._compute_volatility_features,
            'momentum_features': self._compute_momentum_features,
            'trend_features': self._compute_trend_features,
            'pattern_features': self._compute_pattern_features,
            'statistical_features': self._compute_statistical_features,
            'market_features': self._compute_market_features
        }
        
    def compute_all_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Compute all features efficiently with parallel processing"""
        
        # Validate input data
        if not self._validate_input(df):
            raise ValueError("Invalid input data format")
        
        # Convert to numpy arrays for faster computation
        prices = df['close'].values
        high = df['high'].values if 'high' in df.columns else prices
        low = df['low'].values if 'low' in df.columns else prices
        open_prices = df['open'].values if 'open' in df.columns else prices
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(prices))
        
        # Move to GPU if available and requested
        if self.use_gpu:
            prices = cp.array(prices)
            high = cp.array(high)
            low = cp.array(low)
            open_prices = cp.array(open_prices)
            volume = cp.array(volume)
        
        # Create result DataFrame
        result_df = df.copy()
        
        # Compute feature groups in parallel
        feature_data = {
            'prices': prices,
            'high': high,
            'low': low,
            'open': open_prices,
            'volume': volume
        }
        
        # Sequential computation for now (can be parallelized with joblib)
        for group_name, compute_func in self.feature_groups.items():
            try:
                group_features = compute_func(feature_data)
                
                # Convert back from GPU if needed
                if self.use_gpu:
                    group_features = {k: cp.asnumpy(v) if hasattr(v, 'get') else v 
                                    for k, v in group_features.items()}
                
                # Add to result DataFrame
                for feature_name, feature_values in group_features.items():
                    result_df[feature_name] = feature_values
                    
            except Exception as e:
                print(f"⚠️ Error computing {group_name}: {e}")
        
        # Add time-based features
        result_df = self._add_time_features(result_df)
        
        # Add lag features
        result_df = self._add_lag_features(result_df)
        
        # Add interaction features
        result_df = self._add_interaction_features(result_df)
        
        return result_df
    
    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input data format"""
        required_columns = ['close']
        return all(col in df.columns for col in required_columns)
    
    def _compute_price_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute price-based features"""
        prices = data['prices']
        high = data['high']
        low = data['low']
        open_prices = data['open']
        
        features = {}
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = fast_sma(prices, period)
            features[f'ema_{period}'] = fast_ema(prices, period)
            
            # Price relative to MA
            features[f'price_sma_{period}_ratio'] = prices / features[f'sma_{period}']
            features[f'price_ema_{period}_ratio'] = prices / features[f'ema_{period}']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = fast_bollinger_bands(prices, 20)
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (prices - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Price channels
        for period in [10, 20, 50]:
            features[f'highest_high_{period}'] = self._rolling_max(high, period)
            features[f'lowest_low_{period}'] = self._rolling_min(low, period)
            features[f'price_channel_position_{period}'] = (
                (prices - features[f'lowest_low_{period}']) / 
                (features[f'highest_high_{period}'] - features[f'lowest_low_{period}'])
            )
        
        # Typical price and weighted close
        features['typical_price'] = (high + low + prices) / 3
        features['weighted_close'] = (high + low + 2 * prices) / 4
        
        return features
    
    def _compute_volume_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute volume-based features"""
        volume = data['volume']
        prices = data['prices']
        
        features = {}
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            features[f'volume_sma_{period}'] = fast_sma(volume, period)
            features[f'volume_ratio_{period}'] = volume / features[f'volume_sma_{period}']
        
        # Volume-weighted price
        features['vwap'] = self._compute_vwap(prices, volume)
        
        # On-Balance Volume
        features['obv'] = self._compute_obv(prices, volume)
        
        # Volume Rate of Change
        for period in [5, 10, 20]:
            features[f'volume_roc_{period}'] = self._compute_roc(volume, period)
        
        return features
    
    def _compute_volatility_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute volatility-based features"""
        prices = data['prices']
        high = data['high']
        low = data['low']
        
        features = {}
        
        # Historical volatility
        returns = np.diff(np.log(prices))
        for period in [5, 10, 20, 50]:
            features[f'volatility_{period}'] = self._rolling_std(returns, period) * np.sqrt(252)
        
        # Average True Range
        features['atr'] = fast_atr(high, low, prices)
        features['atr_14'] = fast_atr(high, low, prices, 14)
        features['atr_21'] = fast_atr(high, low, prices, 21)
        
        # True Range
        tr = np.empty(len(prices))
        tr[0] = high[0] - low[0]
        for i in range(1, len(prices)):
            hl = high[i] - low[i]
            hc = abs(high[i] - prices[i-1])
            lc = abs(low[i] - prices[i-1])
            tr[i] = max(hl, hc, lc)
        features['true_range'] = tr
        
        # Volatility ratios
        features['atr_price_ratio'] = features['atr'] / prices
        
        return features
    
    def _compute_momentum_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute momentum-based features"""
        prices = data['prices']
        high = data['high']
        low = data['low']
        
        features = {}
        
        # RSI
        features['rsi'] = fast_rsi(prices, 14)
        features['rsi_9'] = fast_rsi(prices, 9)
        features['rsi_21'] = fast_rsi(prices, 21)
        
        # MACD
        macd, macd_signal, macd_histogram = fast_macd(prices)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_histogram
        
        # Stochastic Oscillator
        stoch_k, stoch_d = fast_stochastic(high, low, prices)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        # Rate of Change
        for period in [1, 5, 10, 20]:
            features[f'roc_{period}'] = self._compute_roc(prices, period)
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = prices - np.roll(prices, period)
        
        return features
    
    def _compute_trend_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute trend-based features"""
        prices = data['prices']
        
        features = {}
        
        # Linear regression slopes
        for period in [10, 20, 50]:
            features[f'trend_slope_{period}'] = self._rolling_linear_regression_slope(prices, period)
        
        # Moving average convergence/divergence
        sma_10 = fast_sma(prices, 10)
        sma_20 = fast_sma(prices, 20)
        sma_50 = fast_sma(prices, 50)
        
        features['sma_10_20_diff'] = sma_10 - sma_20
        features['sma_10_50_diff'] = sma_10 - sma_50
        features['sma_20_50_diff'] = sma_20 - sma_50
        
        # Trend strength
        features['trend_strength'] = abs(features['trend_slope_20'])
        
        return features
    
    def _compute_pattern_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute pattern-based features"""
        prices = data['prices']
        high = data['high']
        low = data['low']
        open_prices = data['open']
        
        features = {}
        
        # Candlestick patterns (simplified)
        body = abs(prices - open_prices)
        upper_shadow = high - np.maximum(prices, open_prices)
        lower_shadow = np.minimum(prices, open_prices) - low
        
        features['body_size'] = body
        features['upper_shadow'] = upper_shadow
        features['lower_shadow'] = lower_shadow
        features['body_upper_shadow_ratio'] = body / (upper_shadow + 1e-10)
        features['body_lower_shadow_ratio'] = body / (lower_shadow + 1e-10)
        
        # Doji pattern
        features['is_doji'] = (body < (high - low) * 0.1).astype(float)
        
        # Hammer pattern
        features['is_hammer'] = (
            (lower_shadow > 2 * body) & 
            (upper_shadow < body)
        ).astype(float)
        
        return features
    
    def _compute_statistical_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute statistical features"""
        prices = data['prices']
        
        features = {}
        
        # Rolling statistics
        for period in [10, 20, 50]:
            returns = np.diff(np.log(prices))
            
            features[f'skewness_{period}'] = self._rolling_skewness(returns, period)
            features[f'kurtosis_{period}'] = self._rolling_kurtosis(returns, period)
            features[f'mean_return_{period}'] = self._rolling_mean(returns, period)
            features[f'std_return_{period}'] = self._rolling_std(returns, period)
        
        return features
    
    def _compute_market_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute market-based features"""
        prices = data['prices']
        volume = data['volume']
        
        features = {}
        
        # Market cap proxy (price * volume)
        features['price_volume'] = prices * volume
        
        # Liquidity measures
        features['price_volume_trend'] = self._compute_roc(features['price_volume'], 20)
        
        return features
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        
        # Cyclical encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        key_features = ['close', 'volume', 'rsi', 'macd']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features"""
        
        # Price-volume interactions
        if 'close' in df.columns and 'volume' in df.columns:
            df['price_volume_interaction'] = df['close'] * np.log1p(df['volume'])
        
        # RSI-volatility interaction
        if 'rsi' in df.columns and 'volatility_20' in df.columns:
            df['rsi_volatility_interaction'] = df['rsi'] * df['volatility_20']
        
        return df
    
    # Helper methods for vectorized operations
    def _rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling maximum"""
        result = np.empty(len(data))
        result[:window-1] = np.nan
        
        for i in range(window-1, len(data)):
            result[i] = np.max(data[i-window+1:i+1])
        
        return result
    
    def _rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling minimum"""
        result = np.empty(len(data))
        result[:window-1] = np.nan
        
        for i in range(window-1, len(data)):
            result[i] = np.min(data[i-window+1:i+1])
        
        return result
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling standard deviation"""
        result = np.empty(len(data))
        result[:window-1] = np.nan
        
        for i in range(window-1, len(data)):
            result[i] = np.std(data[i-window+1:i+1])
        
        return result
    
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling mean"""
        return fast_sma(data, window)
    
    def _rolling_skewness(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling skewness"""
        result = np.empty(len(data))
        result[:window-1] = np.nan
        
        for i in range(window-1, len(data)):
            window_data = data[i-window+1:i+1]
            result[i] = stats.skew(window_data)
        
        return result
    
    def _rolling_kurtosis(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling kurtosis"""
        result = np.empty(len(data))
        result[:window-1] = np.nan
        
        for i in range(window-1, len(data)):
            window_data = data[i-window+1:i+1]
            result[i] = stats.kurtosis(window_data)
        
        return result
    
    def _rolling_linear_regression_slope(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling linear regression slope"""
        result = np.empty(len(data))
        result[:window-1] = np.nan
        
        x = np.arange(window)
        
        for i in range(window-1, len(data)):
            y = data[i-window+1:i+1]
            slope = np.polyfit(x, y, 1)[0]
            result[i] = slope
        
        return result
    
    def _compute_vwap(self, prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Compute Volume Weighted Average Price"""
        cumulative_pv = np.cumsum(prices * volume)
        cumulative_volume = np.cumsum(volume)
        
        return cumulative_pv / np.where(cumulative_volume == 0, 1, cumulative_volume)
    
    def _compute_obv(self, prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Compute On-Balance Volume"""
        price_changes = np.diff(prices)
        price_changes = np.concatenate([[0], price_changes])  # Add zero for first value
        
        obv = np.empty(len(prices))
        obv[0] = volume[0]
        
        for i in range(1, len(prices)):
            if price_changes[i] > 0:
                obv[i] = obv[i-1] + volume[i]
            elif price_changes[i] < 0:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _compute_roc(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute Rate of Change"""
        shifted_data = np.roll(data, period)
        roc = ((data - shifted_data) / shifted_data) * 100
        roc[:period] = np.nan  # Set initial values to NaN
        
        return roc

# Example usage and benchmarking
if __name__ == "__main__":
    print("⚡ Ultra-Fast Feature Engineering")
    print("=" * 40)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=2000, freq='D')
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.random.randn(2000).cumsum(),
        'high': 102 + np.random.randn(2000).cumsum(),
        'low': 98 + np.random.randn(2000).cumsum(), 
        'close': 100 + np.random.randn(2000).cumsum(),
        'volume': np.random.randint(1000000, 10000000, 2000)
    })
    sample_data.set_index('date', inplace=True)
    
    # Initialize feature engine
    engine = FastFeatureEngine(use_gpu=CUPY_AVAILABLE)
    
    # Benchmark feature computation
    print(f"📊 Computing features for {len(sample_data)} data points...")
    
    start_time = time.time()
    features_df = engine.compute_all_features(sample_data)
    computation_time = time.time() - start_time
    
    print(f"✅ Generated {features_df.shape[1]} features")
    print(f"⏱️ Computation time: {computation_time:.4f}s")
    print(f"🚄 Speed: {len(sample_data)/computation_time:.0f} rows/sec")
    
    # Show feature categories
    feature_columns = [col for col in features_df.columns if col not in sample_data.columns]
    print(f"\n📋 Generated {len(feature_columns)} new features:")
    
    feature_categories = {}
    for col in feature_columns:
        category = col.split('_')[0]
        if category not in feature_categories:
            feature_categories[category] = 0
        feature_categories[category] += 1
    
    for category, count in sorted(feature_categories.items()):
        print(f"  {category}: {count} features")
    
    # Test individual indicator speed
    if NUMBA_AVAILABLE:
        print(f"\n🔬 Benchmarking individual indicators...")
        prices = sample_data['close'].values
        
        # Benchmark fast_sma
        start_time = time.time()
        sma_fast = fast_sma(prices, 20)
        fast_time = time.time() - start_time
        
        # Benchmark pandas rolling
        start_time = time.time()
        sma_pandas = sample_data['close'].rolling(20).mean().values
        pandas_time = time.time() - start_time
        
        print(f"SMA (Numba): {fast_time:.6f}s")
        print(f"SMA (Pandas): {pandas_time:.6f}s")
        print(f"Speedup: {pandas_time/fast_time:.1f}x")
        
    else:
        print("⚠️ Numba not available - install with: pip install numba")
    
    print(f"\n🎯 Fast feature engineering system ready!")
    print(f"Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
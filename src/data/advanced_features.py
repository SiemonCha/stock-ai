import pandas as pd
import numpy as np
import yfinance as yf
import ta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced market microstructure features
        """
        data = df.copy()
        
        # Price gaps and overnight returns
        data['price_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['intraday_return'] = (data['close'] - data['open']) / data['open']
        
        # Volatility measures
        data['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        
        # Average True Range with multiple periods
        data['atr_14'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
        data['atr_30'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=30)
        
        # Volume-Price Relationship
        data['vwap'] = ta.volume.volume_weighted_average_price(data['high'], data['low'], data['close'], data['volume'])
        data['volume_profile'] = data['volume'] * data['close']
        data['price_volume_trend'] = ta.volume.volume_price_trend(data['close'], data['volume'])
        
        # Order Flow Approximation
        data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        data['selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
        
        return data
    
    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators
        """
        data = df.copy()
        
        # Trend Indicators
        data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'])
        data['cci'] = ta.trend.cci(data['high'], data['low'], data['close'])
        data['dpo'] = ta.trend.dpo(data['close'])
        data['kst'] = ta.trend.kst(data['close'])
        data['mass_index'] = ta.trend.mass_index(data['high'], data['low'])
        data['trix'] = ta.trend.trix(data['close'])
        data['vortex_pos'] = ta.trend.vortex_indicator_pos(data['high'], data['low'], data['close'])
        data['vortex_neg'] = ta.trend.vortex_indicator_neg(data['high'], data['low'], data['close'])
        
        # Momentum Indicators
        data['ao'] = ta.momentum.awesome_oscillator(data['high'], data['low'])
        data['kama'] = ta.momentum.kama(data['close'])
        data['ppo'] = ta.momentum.ppo(data['close'])
        data['pvo'] = ta.momentum.pvo(data['volume'])
        data['roc'] = ta.momentum.roc(data['close'])
        data['stoch_rsi'] = ta.momentum.stochrsi(data['close'])
        data['tsi'] = ta.momentum.tsi(data['close'])
        data['ultimate_osc'] = ta.momentum.ultimate_oscillator(data['high'], data['low'], data['close'])
        data['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'])
        
        # Volume Indicators
        data['adi'] = ta.volume.acc_dist_index(data['high'], data['low'], data['close'], data['volume'])
        data['cmf'] = ta.volume.chaikin_money_flow(data['high'], data['low'], data['close'], data['volume'])
        data['em'] = ta.volume.ease_of_movement(data['high'], data['low'], data['volume'])
        data['fi'] = ta.volume.force_index(data['close'], data['volume'])
        data['mfi'] = ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume'])
        data['nvi'] = ta.volume.negative_volume_index(data['close'], data['volume'])
        data['pvi'] = ta.volume.positive_volume_index(data['close'], data['volume'])
        data['vpt'] = ta.volume.volume_price_trend(data['close'], data['volume'])
        
        # Volatility Indicators
        data['bb_width'] = ta.volatility.bollinger_wband(data['close'])
        data['bb_pband'] = ta.volatility.bollinger_pband(data['close'])
        data['dc_width'] = ta.volatility.donchian_channel_wband(data['high'], data['low'], data['close'])
        data['dc_pband'] = ta.volatility.donchian_channel_pband(data['high'], data['low'], data['close'])
        data['kc_width'] = ta.volatility.keltner_channel_wband(data['high'], data['low'], data['close'])
        data['kc_pband'] = ta.volatility.keltner_channel_pband(data['high'], data['low'], data['close'])
        data['ui'] = ta.volatility.ulcer_index(data['close'])
        
        return data
    
    def add_fractal_and_chaos_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fractal and chaos theory indicators
        """
        data = df.copy()
        
        # Hurst Exponent approximation
        def hurst_exponent(ts, max_lag=20):
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        # Calculate Hurst for rolling windows
        hurst_values = []
        for i in range(60, len(data)):
            window_data = data['close'].iloc[i-60:i].values
            try:
                h = hurst_exponent(window_data)
                hurst_values.append(h)
            except:
                hurst_values.append(0.5)
        
        # Pad with default values
        data['hurst_exponent'] = [0.5] * 60 + hurst_values
        
        # Fractal dimension approximation
        data['fractal_dimension'] = 2 - data['hurst_exponent']
        
        return data
    
    def add_support_resistance_levels(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Identify support and resistance levels
        """
        data = df.copy()
        
        # Local maxima and minima
        data['local_max'] = data['high'].rolling(window=window, center=True).max()
        data['local_min'] = data['low'].rolling(window=window, center=True).min()
        
        data['is_resistance'] = (data['high'] == data['local_max']).astype(int)
        data['is_support'] = (data['low'] == data['local_min']).astype(int)
        
        # Distance to nearest support/resistance
        resistance_levels = data[data['is_resistance'] == 1]['high'].values
        support_levels = data[data['is_support'] == 1]['low'].values
        
        data['dist_to_resistance'] = np.nan
        data['dist_to_support'] = np.nan
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            
            if len(resistance_levels) > 0:
                nearest_resistance = resistance_levels[np.argmin(np.abs(resistance_levels - current_price))]
                data.iloc[i, data.columns.get_loc('dist_to_resistance')] = (nearest_resistance - current_price) / current_price
            
            if len(support_levels) > 0:
                nearest_support = support_levels[np.argmin(np.abs(support_levels - current_price))]
                data.iloc[i, data.columns.get_loc('dist_to_support')] = (current_price - nearest_support) / current_price
        
        return data
    
    def add_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime detection features
        """
        data = df.copy()
        
        # Volatility regimes
        returns = data['close'].pct_change()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        
        data['vol_regime'] = np.where(vol_20 > vol_60.quantile(0.75), 2,  # High vol
                                     np.where(vol_20 < vol_60.quantile(0.25), 0, 1))  # Low/Med vol
        
        # Trend strength
        data['trend_strength'] = abs(data['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        
        # Market phase detection
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        sma_200 = data['close'].rolling(200).mean()
        
        data['bull_market'] = ((sma_20 > sma_50) & (sma_50 > sma_200)).astype(int)
        data['bear_market'] = ((sma_20 < sma_50) & (sma_50 < sma_200)).astype(int)
        data['sideways_market'] = ((data['bull_market'] == 0) & (data['bear_market'] == 0)).astype(int)
        
        return data
    
    def add_momentum_and_reversal_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum and reversal signal indicators
        """
        data = df.copy()
        
        # Momentum signals
        data['price_momentum_5'] = data['close'].pct_change(5)
        data['price_momentum_10'] = data['close'].pct_change(10)
        data['price_momentum_20'] = data['close'].pct_change(20)
        
        # Volume momentum
        data['volume_momentum_5'] = data['volume'].pct_change(5)
        data['volume_momentum_10'] = data['volume'].pct_change(10)
        
        # Reversal signals
        data['doji'] = (abs(data['open'] - data['close']) / (data['high'] - data['low']) < 0.1).astype(int)
        data['hammer'] = ((data['close'] > data['open']) & 
                         ((data['close'] - data['open']) / (data['high'] - data['low']) > 0.6) &
                         ((data['open'] - data['low']) / (data['high'] - data['low']) < 0.1)).astype(int)
        
        # Divergence signals
        price_change = data['close'].pct_change(10)
        rsi_change = data['rsi'].pct_change(10) if 'rsi' in data.columns else 0
        
        data['bullish_divergence'] = ((price_change < 0) & (rsi_change > 0)).astype(int)
        data['bearish_divergence'] = ((price_change > 0) & (rsi_change < 0)).astype(int)
        
        return data
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all advanced features
        """
        print("Creating comprehensive feature set...")
        
        data = df.copy()
        
        # Apply all feature engineering methods
        data = self.add_market_microstructure_features(data)
        data = self.add_advanced_technical_indicators(data)
        data = self.add_fractal_and_chaos_features(data)
        data = self.add_support_resistance_levels(data)
        data = self.add_regime_detection_features(data)
        data = self.add_momentum_and_reversal_signals(data)
        
        # Fill NaN values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"Created {len(data.columns)} features")
        return data
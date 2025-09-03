import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from typing import Tuple, List

class StockDataPreprocessor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.feature_columns = None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset
        """
        data = df.copy()
        
        # Price-based indicators
        data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
        data['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
        data['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)
        
        # RSI
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        
        # MACD
        data['macd'] = ta.trend.macd_diff(data['close'])
        data['macd_signal'] = ta.trend.macd_signal(data['close'])
        
        # Bollinger Bands
        data['bb_upper'] = ta.volatility.bollinger_hband(data['close'])
        data['bb_lower'] = ta.volatility.bollinger_lband(data['close'])
        data['bb_middle'] = ta.volatility.bollinger_mavg(data['close'])
        
        # Volume indicators
        data['volume_sma'] = ta.volume.volume_sma(data['close'], data['volume'], window=20)
        
        # Price changes
        data['price_change'] = data['close'].pct_change()
        data['price_change_2'] = data['close'].pct_change(2)
        data['price_change_5'] = data['close'].pct_change(5)
        
        # Volatility
        data['volatility'] = data['price_change'].rolling(window=20).std()
        
        # High-Low ratio
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        
        # Volume ratio
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
        
        return data
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training
        """
        data = self.add_technical_indicators(df)
        
        # Select features for model
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_middle',
            'volume_sma', 'price_change', 'price_change_2', 'price_change_5',
            'volatility', 'hl_ratio', 'volume_ratio'
        ]
        
        # Remove rows with NaN values
        data = data[feature_columns].dropna()
        
        self.feature_columns = feature_columns
        
        return data
    
    def create_sequences(self, data: np.ndarray, target_col_index: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        target_col_index: index of the target column (close price)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, target_col_index])  # Close price
        
        return np.array(X), np.array(y)
    
    def preprocess_for_training(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for training
        """
        # Prepare features
        processed_data = self.prepare_features(df)
        
        # Split data
        split_index = int(len(processed_data) * (1 - test_size))
        train_data = processed_data[:split_index]
        test_data = processed_data[split_index:]
        
        # Scale the data
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled)
        X_test, y_test = self.create_sequences(test_scaled)
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_for_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for making predictions
        """
        processed_data = self.prepare_features(df)
        
        # Use the last sequence_length rows for prediction
        recent_data = processed_data.tail(self.sequence_length)
        
        # Scale the data
        scaled_data = self.scaler.transform(recent_data)
        
        # Reshape for prediction
        return scaled_data.reshape(1, self.sequence_length, -1)
    
    def inverse_transform_prediction(self, prediction: np.ndarray) -> float:
        """
        Convert scaled prediction back to original price scale
        """
        # Create a dummy array with the same shape as the original features
        dummy = np.zeros((1, len(self.feature_columns)))
        dummy[0, 3] = prediction  # Close price is at index 3
        
        # Inverse transform
        inverse_scaled = self.scaler.inverse_transform(dummy)
        
        return inverse_scaled[0, 3]
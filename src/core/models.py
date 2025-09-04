import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys

# Import advanced models
try:
    from ..ai.advanced_models import NextGenEnsembleModel, AdvancedTrainingSystem
    from ..ai.advanced_models import AdvancedEnsembleModel
except ImportError:
    print("Advanced models not available, using basic models only")

class UnifiedStockModels:
    """
    Unified model system containing all model architectures including next-gen models
    """
    def __init__(self, use_advanced=True):
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.use_advanced = use_advanced
        self.advanced_ensemble = None
        self.nextgen_ensemble = None
        
    def create_lstm_model(self, sequence_length: int, n_features: int, 
                         units: List[int] = [64, 32], dropout_rate: float = 0.2) -> Model:
        """Create LSTM model"""
        model = Sequential([
            LSTM(units[0], return_sequences=True, input_shape=(sequence_length, n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            LSTM(units[1], return_sequences=False),
            BatchNormalization(), 
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def create_gru_model(self, sequence_length: int, n_features: int,
                        units: List[int] = [64, 32], dropout_rate: float = 0.2) -> Model:
        """Create GRU model"""
        model = Sequential([
            GRU(units[0], return_sequences=True, input_shape=(sequence_length, n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            GRU(units[1], return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def create_transformer_model(self, sequence_length: int, n_features: int,
                               d_model: int = 64, num_heads: int = 4, 
                               num_layers: int = 2) -> Model:
        """Create Transformer model"""
        inputs = Input(shape=(sequence_length, n_features))
        
        # Project to model dimension
        x = Dense(d_model)(inputs)
        
        # Transformer blocks
        for _ in range(num_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
            x = Add()([x, attn_output])
            x = LayerNormalization()(x)
            
            # Feed forward
            ffn_output = Sequential([
                Dense(d_model * 4, activation='relu'),
                Dense(d_model)
            ])(x)
            x = Add()([x, ffn_output])
            x = LayerNormalization()(x)
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def create_cnn_lstm_model(self, sequence_length: int, n_features: int) -> Model:
        """Create CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', padding='same', 
                   input_shape=(sequence_length, n_features)),
            Conv1D(32, 5, activation='relu', padding='same'),
            MaxPooling1D(2),
            
            LSTM(64, return_sequences=True, dropout=0.2),
            LSTM(32, dropout=0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def create_ensemble_model(self, sequence_length: int, n_features: int) -> Dict:
        """Create ensemble of multiple models"""
        models = {
            'lstm': self.create_lstm_model(sequence_length, n_features),
            'gru': self.create_gru_model(sequence_length, n_features), 
            'transformer': self.create_transformer_model(sequence_length, n_features),
            'cnn_lstm': self.create_cnn_lstm_model(sequence_length, n_features)
        }
        
        return models
    
    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100,
                   model_path: str = None) -> Dict:
        """Train a specific model type"""
        
        if model_type == 'ensemble':
            return self.train_ensemble(X_train, y_train, X_val, y_val, epochs, model_path)
        
        # Create single model
        model_creators = {
            'lstm': self.create_lstm_model,
            'gru': self.create_gru_model,
            'transformer': self.create_transformer_model,
            'cnn_lstm': self.create_cnn_lstm_model
        }
        
        if model_type not in model_creators:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model_creators[model_type](X_train.shape[1], X_train.shape[2])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models[model_type] = model
        self.history[model_type] = history.history
        
        return history.history
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100,
                      model_path: str = None) -> Dict:
        """Train ensemble of models"""
        
        ensemble_models = self.create_ensemble_model(X_train.shape[1], X_train.shape[2])
        histories = {}
        
        for name, model in ensemble_models.items():
            print(f"Training {name} model...")
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
            ]
            
            if model_path:
                model_file = model_path.replace('.h5', f'_{name}.h5')
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            histories[name] = history.history
        
        self.models['ensemble'] = ensemble_models
        self.history['ensemble'] = histories
        
        return histories
    
    def predict(self, model_type: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with specified model"""
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        if model_type == 'ensemble':
            # Ensemble prediction (average)
            predictions = []
            for name, model in self.models['ensemble'].items():
                pred = model.predict(X, verbose=0)
                predictions.append(pred.flatten())
            
            return np.mean(predictions, axis=0)
        else:
            # Single model prediction
            return self.models[model_type].predict(X, verbose=0).flatten()
    
    def evaluate_model(self, model_type: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        
        predictions = self.predict(model_type, X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - predictions) ** 2)
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        if len(y_test) > 1:
            actual_direction = np.diff(y_test) > 0
            pred_direction = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = 0
        
        return {
            'mse': mse,
            'mae': mae, 
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def save_models(self, base_path: str):
        """Save all trained models"""
        os.makedirs(base_path, exist_ok=True)
        
        for model_type, model in self.models.items():
            if model_type == 'ensemble':
                for name, individual_model in model.items():
                    model_path = os.path.join(base_path, f'ensemble_{name}.h5')
                    individual_model.save(model_path)
            else:
                model_path = os.path.join(base_path, f'{model_type}.h5')
                model.save(model_path)
        
        # Save scalers if any
        if self.scalers:
            scaler_path = os.path.join(base_path, 'scalers.pkl')
            joblib.dump(self.scalers, scaler_path)
    
    def load_models(self, base_path: str):
        """Load trained models"""
        model_files = [f for f in os.listdir(base_path) if f.endswith('.h5')]
        
        for model_file in model_files:
            model_path = os.path.join(base_path, model_file)
            model_name = model_file.replace('.h5', '')
            
            if model_name.startswith('ensemble_'):
                # Ensemble model
                if 'ensemble' not in self.models:
                    self.models['ensemble'] = {}
                
                individual_name = model_name.replace('ensemble_', '')
                self.models['ensemble'][individual_name] = tf.keras.models.load_model(model_path)
            else:
                # Single model
                self.models[model_name] = tf.keras.models.load_model(model_path)
        
        # Load scalers if available
        scaler_path = os.path.join(base_path, 'scalers.pkl')
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
    
    def create_nextgen_ensemble(self, sequence_length: int, n_features: int, 
                               config: Optional[Dict] = None) -> 'NextGenEnsembleModel':
        """Create next-generation ensemble model"""
        if not self.use_advanced:
            raise ValueError("Advanced models not enabled")
        
        try:
            if config is None:
                config = {
                    'sequence_length': sequence_length,
                    'n_features': n_features,
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001
                }
            
            self.nextgen_ensemble = NextGenEnsembleModel(**config)
            return self.nextgen_ensemble.build_advanced_ensemble()
        except NameError:
            print("NextGenEnsembleModel not available, falling back to basic ensemble")
            return self.create_ensemble_model(sequence_length, n_features)
    
    def create_advanced_ensemble(self, sequence_length: int, n_features: int,
                                config: Optional[Dict] = None) -> 'AdvancedEnsembleModel':
        """Create advanced ensemble with uncertainty quantification"""
        if not self.use_advanced:
            raise ValueError("Advanced models not enabled")
        
        try:
            if config is None:
                config = {
                    'sequence_length': sequence_length,
                    'n_features': n_features
                }
            
            self.advanced_ensemble = AdvancedEnsembleModel(**config)
            return self.advanced_ensemble.build_ensemble_model()
        except NameError:
            print("AdvancedEnsembleModel not available, falling back to basic ensemble")
            return self.create_ensemble_model(sequence_length, n_features)
    
    def train_advanced_ensemble(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100,
                               config: Optional[Dict] = None) -> Dict:
        """Train advanced ensemble models with sophisticated techniques"""
        
        if model_type == 'nextgen_ensemble':
            if self.nextgen_ensemble is None:
                self.create_nextgen_ensemble(X_train.shape[1], X_train.shape[2], config)
            
            # Use advanced training system
            trainer = AdvancedTrainingSystem(self.nextgen_ensemble)
            history = trainer.train_with_advanced_techniques(
                X_train, y_train, X_val, y_val, epochs
            )
            
            self.models['nextgen_ensemble'] = self.nextgen_ensemble
            self.history['nextgen_ensemble'] = history.history if hasattr(history, 'history') else history
            
            return self.history['nextgen_ensemble']
        
        elif model_type == 'advanced_ensemble':
            if self.advanced_ensemble is None:
                self.create_advanced_ensemble(X_train.shape[1], X_train.shape[2], config)
            
            history = self.advanced_ensemble.train_ensemble(
                X_train, y_train, X_val, y_val, epochs
            )
            
            self.models['advanced_ensemble'] = self.advanced_ensemble
            self.history['advanced_ensemble'] = history
            
            return history
        
        else:
            # Fall back to standard training
            return self.train_model(model_type, X_train, y_train, X_val, y_val, epochs)
    
    def predict_with_uncertainty(self, model_type: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates"""
        
        if model_type == 'advanced_ensemble' and self.advanced_ensemble:
            return self.advanced_ensemble.predict_with_uncertainty(X)
        
        elif model_type == 'nextgen_ensemble' and self.nextgen_ensemble:
            # NextGen model prediction with uncertainty
            predictions = self.nextgen_ensemble.predict(X)
            if isinstance(predictions, list) and len(predictions) >= 2:
                return predictions[0], predictions[1]
            else:
                # Fallback to basic prediction
                return predictions, np.zeros_like(predictions)
        
        else:
            # Standard prediction without uncertainty
            predictions = self.predict(model_type, X)
            return predictions, np.zeros_like(predictions)

class DataProcessor:
    """
    Enhanced data processing system with intelligent feature engineering
    """
    def __init__(self, sequence_length: int = 60, feature_level: str = 'standard'):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.feature_level = feature_level  # 'standard', 'advanced', 'intelligent', 'all'
        self.feature_engine = None
        
        # Initialize intelligent feature engine if advanced features requested
        if feature_level in ['advanced', 'intelligent', 'all']:
            try:
                from ..data.intelligent_features import IntelligentFeatureEngine
                self.feature_engine = IntelligentFeatureEngine()
            except ImportError:
                print("Warning: Intelligent features not available, using standard features")
                self.feature_level = 'standard'
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data preparation with multiple feature levels"""
        import ta
        
        data = df.copy()
        
        # Ensure proper column names (handle case variations)
        column_mapping = {}
        for col in data.columns:
            if col.lower() in ['close', 'high', 'low', 'open', 'volume']:
                column_mapping[col] = col.capitalize()
        
        if column_mapping:
            data = data.rename(columns=column_mapping)
        
        # Standard features (always included)
        if 'Close' in data.columns:
            # Basic technical indicators
            data['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['rsi'] = ta.momentum.rsi(data['Close'], window=14)
            data['macd'] = ta.trend.macd_diff(data['Close'])
            data['macd_signal'] = ta.trend.macd_signal(data['Close'])
            data['bb_upper'] = ta.volatility.bollinger_hband(data['Close'])
            data['bb_lower'] = ta.volatility.bollinger_lband(data['Close'])
            
            # Price features
            data['price_change'] = data['Close'].pct_change()
            data['volatility'] = data['price_change'].rolling(window=20).std()
            
            if 'High' in data.columns and 'Low' in data.columns:
                data['high_low_ratio'] = data['High'] / data['Low']
            
            # Volume indicators (if volume available)
            if 'Volume' in data.columns:
                data['volume_sma'] = ta.volume.volume_sma(data['Close'], data['Volume'], window=20)
        
        # Advanced features
        if self.feature_level in ['advanced', 'intelligent', 'all'] and self.feature_engine:
            try:
                # Add market microstructure features
                data = self.feature_engine.add_market_microstructure_features(data)
                
                # Add advanced technical indicators
                data = self.feature_engine.add_advanced_technical_indicators(data)
                
                if self.feature_level in ['intelligent', 'all']:
                    # Add behavioral finance features
                    data = self.feature_engine.add_behavioral_finance_features(data)
                    
                    # Add regime detection features
                    data = self.feature_engine.add_regime_adaptive_features(data)
                    
                    # Add quantum-inspired features
                    data = self.feature_engine.add_quantum_inspired_features(data)
                
                if self.feature_level == 'all':
                    # Add fractal and chaos features (computationally intensive)
                    data = self.feature_engine.add_fractal_and_chaos_features(data)
                    
                    # Add support/resistance levels
                    data = self.feature_engine.add_support_resistance_levels(data)
                
            except Exception as e:
                print(f"Warning: Advanced feature engineering failed: {e}")
                print("Falling back to standard features...")
        
        return data
    
    def create_sequences(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        
        # Prepare features
        processed_data = self.prepare_data(data)
        
        # Select feature columns (numeric only)
        feature_columns = [col for col in processed_data.columns if processed_data[col].dtype in ['float64', 'int64']]
        self.feature_columns = feature_columns
        
        # Fill NaN and scale
        processed_data = processed_data[feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
        scaled_data = self.scaler.fit_transform(processed_data)
        
        # Create sequences
        X, y = [], []
        target_idx = feature_columns.index(target_col)
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def inverse_transform_target(self, scaled_target: np.ndarray, target_col: str = 'Close') -> np.ndarray:
        """Inverse transform target values"""
        target_idx = self.feature_columns.index(target_col)
        
        # Create dummy array for inverse transform
        dummy = np.zeros((len(scaled_target), len(self.feature_columns)))
        dummy[:, target_idx] = scaled_target
        
        # Inverse transform and return target column
        inverse_scaled = self.scaler.inverse_transform(dummy)
        return inverse_scaled[:, target_idx]
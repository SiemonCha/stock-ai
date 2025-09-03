import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Optional
import os

class StockLSTMModel:
    def __init__(self, 
                 sequence_length: int = 60, 
                 n_features: int = 22,
                 lstm_units: list = [50, 50, 50],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self) -> Sequential:
        """
        Build LSTM model architecture
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=self.lstm_units[0], 
                      return_sequences=True, 
                      input_shape=(self.sequence_length, self.n_features)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(units=self.lstm_units[i], 
                          return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, 
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        self.model = model
        return model
    
    def get_callbacks(self, model_path: str = 'models/best_model.h5') -> list:
        """
        Get training callbacks
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', 
                         patience=15, 
                         restore_best_weights=True,
                         verbose=1),
            
            ReduceLROnPlateau(monitor='val_loss', 
                             factor=0.5, 
                             patience=10, 
                             min_lr=1e-6,
                             verbose=1),
            
            ModelCheckpoint(model_path, 
                           monitor='val_loss', 
                           save_best_only=True,
                           save_weights_only=False,
                           verbose=1)
        ]
        
        return callbacks
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray, 
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              model_path: str = 'models/best_model.h5') -> dict:
        """
        Train the LSTM model
        """
        if self.model is None:
            self.build_model()
        
        callbacks = self.get_callbacks(model_path)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        predictions = self.predict(X_test)
        
        # Calculate additional metrics
        mse = np.mean((y_test - predictions.flatten()) ** 2)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(predictions.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        """
        self.model = tf.keras.models.load_model(filepath)
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary
        """
        if self.model is None:
            self.build_model()
        
        return self.model.summary()
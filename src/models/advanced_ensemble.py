import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization, 
    MultiHeadAttention, LayerNormalization, Add,
    Input, Concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Dict, List
import joblib

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class AdvancedEnsembleModel:
    def __init__(self, 
                 sequence_length: int = 60, 
                 n_features: int = 80,
                 lstm_units: List[int] = [64, 32],
                 gru_units: List[int] = [64, 32],
                 transformer_embed_dim: int = 64,
                 transformer_num_heads: int = 4,
                 transformer_ff_dim: int = 128,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.gru_units = gru_units
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_num_heads = transformer_num_heads
        self.transformer_ff_dim = transformer_ff_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.individual_models = {}
        self.history = None
        
    def create_lstm_branch(self, inputs):
        """Create LSTM branch of the ensemble"""
        x = inputs
        
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = LSTM(units, return_sequences=return_sequences, name=f'lstm_{i}')(x)
            x = BatchNormalization(name=f'lstm_bn_{i}')(x)
            x = Dropout(self.dropout_rate, name=f'lstm_dropout_{i}')(x)
        
        x = Dense(32, activation='relu', name='lstm_dense')(x)
        x = Dropout(self.dropout_rate, name='lstm_final_dropout')(x)
        lstm_output = Dense(1, name='lstm_prediction')(x)
        
        return lstm_output
    
    def create_gru_branch(self, inputs):
        """Create GRU branch of the ensemble"""
        x = inputs
        
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1
            x = GRU(units, return_sequences=return_sequences, name=f'gru_{i}')(x)
            x = BatchNormalization(name=f'gru_bn_{i}')(x)
            x = Dropout(self.dropout_rate, name=f'gru_dropout_{i}')(x)
        
        x = Dense(32, activation='relu', name='gru_dense')(x)
        x = Dropout(self.dropout_rate, name='gru_final_dropout')(x)
        gru_output = Dense(1, name='gru_prediction')(x)
        
        return gru_output
    
    def create_transformer_branch(self, inputs):
        """Create Transformer branch of the ensemble"""
        x = inputs
        
        # Project to transformer dimension
        x = Dense(self.transformer_embed_dim, name='transformer_projection')(x)
        
        # Transformer blocks
        x = TransformerBlock(
            embed_dim=self.transformer_embed_dim,
            num_heads=self.transformer_num_heads,
            ff_dim=self.transformer_ff_dim,
            rate=self.dropout_rate
        )(x)
        
        x = TransformerBlock(
            embed_dim=self.transformer_embed_dim,
            num_heads=self.transformer_num_heads,
            ff_dim=self.transformer_ff_dim,
            rate=self.dropout_rate
        )(x)
        
        # Global pooling
        x = GlobalAveragePooling1D(name='transformer_pooling')(x)
        x = Dense(32, activation='relu', name='transformer_dense')(x)
        x = Dropout(self.dropout_rate, name='transformer_dropout')(x)
        transformer_output = Dense(1, name='transformer_prediction')(x)
        
        return transformer_output
    
    def create_cnn_branch(self, inputs):
        """Create CNN branch for pattern recognition"""
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
        
        x = inputs
        
        # Multiple CNN layers with different filter sizes
        x = Conv1D(filters=64, kernel_size=3, activation='relu', name='cnn_conv1')(x)
        x = BatchNormalization(name='cnn_bn1')(x)
        x = MaxPooling1D(pool_size=2, name='cnn_pool1')(x)
        x = Dropout(self.dropout_rate, name='cnn_dropout1')(x)
        
        x = Conv1D(filters=32, kernel_size=5, activation='relu', name='cnn_conv2')(x)
        x = BatchNormalization(name='cnn_bn2')(x)
        x = MaxPooling1D(pool_size=2, name='cnn_pool2')(x)
        x = Dropout(self.dropout_rate, name='cnn_dropout2')(x)
        
        x = Flatten(name='cnn_flatten')(x)
        x = Dense(32, activation='relu', name='cnn_dense')(x)
        x = Dropout(self.dropout_rate, name='cnn_final_dropout')(x)
        cnn_output = Dense(1, name='cnn_prediction')(x)
        
        return cnn_output
    
    def build_ensemble_model(self) -> Model:
        """Build the complete ensemble model"""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features), name='main_input')
        
        # Create all branches
        lstm_pred = self.create_lstm_branch(inputs)
        gru_pred = self.create_gru_branch(inputs)
        transformer_pred = self.create_transformer_branch(inputs)
        cnn_pred = self.create_cnn_branch(inputs)
        
        # Combine predictions with learned weights
        combined = Concatenate(name='combine_predictions')([lstm_pred, gru_pred, transformer_pred, cnn_pred])
        
        # Meta-learner to combine predictions
        meta_layer = Dense(16, activation='relu', name='meta_dense1')(combined)
        meta_layer = Dropout(self.dropout_rate, name='meta_dropout1')(meta_layer)
        meta_layer = Dense(8, activation='relu', name='meta_dense2')(meta_layer)
        meta_layer = Dropout(self.dropout_rate, name='meta_dropout2')(meta_layer)
        
        # Final prediction with uncertainty estimation
        main_prediction = Dense(1, name='main_prediction')(meta_layer)
        uncertainty_estimation = Dense(1, activation='softplus', name='uncertainty')(meta_layer)
        
        # Create model
        model = Model(
            inputs=inputs, 
            outputs=[main_prediction, uncertainty_estimation],
            name='advanced_ensemble_model'
        )
        
        # Custom loss function that incorporates uncertainty
        def uncertainty_loss(y_true, y_pred):
            main_pred, uncertainty = y_pred[:, 0:1], y_pred[:, 1:2]
            
            # Negative log likelihood with uncertainty
            mse_loss = tf.square(y_true - main_pred)
            uncertainty_loss = tf.log(uncertainty + 1e-8) + mse_loss / (2 * uncertainty + 1e-8)
            
            return tf.reduce_mean(uncertainty_loss)
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss={
                'main_prediction': 'mse',
                'uncertainty': lambda y_true, y_pred: tf.constant(0.0)  # No direct loss for uncertainty
            },
            loss_weights={'main_prediction': 1.0, 'uncertainty': 0.0},
            metrics={'main_prediction': ['mae']}
        )
        
        self.model = model
        return model
    
    def train_ensemble(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_val: np.ndarray, 
                      y_val: np.ndarray,
                      epochs: int = 100,
                      batch_size: int = 32,
                      model_path: str = 'models/ensemble_best_model.h5') -> Dict:
        """Train the ensemble model"""
        
        if self.model is None:
            self.build_ensemble_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Prepare targets for multiple outputs
        y_train_dict = {
            'main_prediction': y_train,
            'uncertainty': np.zeros_like(y_train)  # Placeholder
        }
        y_val_dict = {
            'main_prediction': y_val,
            'uncertainty': np.zeros_like(y_val)  # Placeholder
        }
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates"""
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        predictions = self.model.predict(X)
        main_pred = predictions[0]
        uncertainty = predictions[1]
        
        return main_pred, uncertainty
    
    def get_confidence_intervals(self, X: np.ndarray, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get prediction confidence intervals"""
        predictions, uncertainty = self.predict_with_uncertainty(X)
        
        # Calculate confidence intervals using uncertainty estimates
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        
        lower_bound = predictions - z_score * np.sqrt(uncertainty)
        upper_bound = predictions + z_score * np.sqrt(uncertainty)
        
        return predictions, lower_bound, upper_bound
    
    def evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Comprehensive evaluation of the ensemble"""
        predictions, uncertainty = self.predict_with_uncertainty(X_test)
        
        # Basic metrics
        mse = np.mean((y_test - predictions.flatten()) ** 2)
        mae = np.mean(np.abs(y_test - predictions.flatten()))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(predictions.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        # Uncertainty calibration
        residuals = np.abs(y_test - predictions.flatten())
        uncertainty_correlation = np.corrcoef(residuals, uncertainty.flatten())[0, 1]
        
        # Confidence interval coverage
        pred, lower, upper = self.get_confidence_intervals(X_test)
        coverage = np.mean((y_test >= lower.flatten()) & (y_test <= upper.flatten()))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'uncertainty_correlation': uncertainty_correlation,
            'ci_coverage_95': coverage,
            'mean_uncertainty': np.mean(uncertainty)
        }
    
    def save_ensemble(self, filepath: str):
        """Save the complete ensemble model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'gru_units': self.gru_units,
            'transformer_embed_dim': self.transformer_embed_dim,
            'transformer_num_heads': self.transformer_num_heads,
            'transformer_ff_dim': self.transformer_ff_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }
        
        joblib.dump(config, filepath.replace('.h5', '_config.pkl'))
    
    def load_ensemble(self, filepath: str):
        """Load a trained ensemble model"""
        self.model = tf.keras.models.load_model(filepath, custom_objects={
            'TransformerBlock': TransformerBlock
        })
        
        # Load configuration
        config = joblib.load(filepath.replace('.h5', '_config.pkl'))
        for key, value in config.items():
            setattr(self, key, value)
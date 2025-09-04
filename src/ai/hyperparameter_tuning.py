import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
import joblib
from concurrent.futures import ThreadPoolExecutor
import logging

class AdaptiveHyperparameterOptimizer:
    """
    Advanced hyperparameter optimization with market-aware strategies
    """
    def __init__(self, n_trials=100, n_jobs=-1):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_history = {}
        self.best_configs = {}
        
    def market_aware_objective(self, trial, model_class, X_train, y_train, 
                              X_val, y_val, market_regime='normal'):
        """
        Market regime-aware objective function
        """
        # Suggest hyperparameters based on market regime
        if market_regime == 'volatile':
            # More regularization in volatile markets
            dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        elif market_regime == 'trending':
            # Less regularization in trending markets
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
        else:  # normal market
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True)
        
        # Model architecture parameters
        n_layers = trial.suggest_int('n_layers', 2, 6)
        layer_sizes = [
            trial.suggest_int(f'layer_size_{i}', 32, 256)
            for i in range(n_layers)
        ]
        
        # Advanced parameters
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop'])
        
        # Attention parameters
        use_attention = trial.suggest_categorical('use_attention', [True, False])
        if use_attention:
            attention_heads = trial.suggest_int('attention_heads', 2, 16)
            attention_dim = trial.suggest_int('attention_dim', 32, 128)
        else:
            attention_heads = 4
            attention_dim = 64
        
        # Build and train model
        config = {
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'l2_reg': l2_reg,
            'n_layers': n_layers,
            'layer_sizes': layer_sizes,
            'batch_size': batch_size,
            'optimizer_type': optimizer_type,
            'use_attention': use_attention,
            'attention_heads': attention_heads,
            'attention_dim': attention_dim
        }
        
        try:
            model = model_class(config)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=batch_size,
                verbose=0
            )
            
            # Multi-objective optimization
            val_loss = min(history.history['val_loss'])
            val_mae = min(history.history['val_mae'])
            
            # Combine metrics with market-specific weights
            if market_regime == 'volatile':
                # Prioritize stability in volatile markets
                score = val_loss * 0.7 + val_mae * 0.3
            else:
                # Balance accuracy and precision
                score = val_loss * 0.5 + val_mae * 0.5
            
            return score
            
        except Exception as e:
            logging.error(f"Trial failed: {e}")
            return float('inf')
    
    def optimize_ensemble_weights(self, predictions_dict: Dict[str, np.ndarray], 
                                y_true: np.ndarray) -> Dict[str, float]:
        """
        Optimize ensemble weights using Optuna
        """
        def objective(trial):
            weights = {}
            weight_sum = 0
            
            # Suggest weights for each model
            for model_name in predictions_dict.keys():
                weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                weights[model_name] = weight
                weight_sum += weight
            
            # Normalize weights
            if weight_sum == 0:
                return float('inf')
            
            for model_name in weights.keys():
                weights[model_name] /= weight_sum
            
            # Calculate ensemble prediction
            ensemble_pred = np.zeros_like(y_true)
            for model_name, predictions in predictions_dict.items():
                ensemble_pred += weights[model_name] * predictions
            
            # Return MSE
            mse = np.mean((y_true - ensemble_pred) ** 2)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Extract best weights
        best_weights = {}
        weight_sum = 0
        for model_name in predictions_dict.keys():
            weight = study.best_params[f'weight_{model_name}']
            best_weights[model_name] = weight
            weight_sum += weight
        
        # Normalize
        for model_name in best_weights.keys():
            best_weights[model_name] /= weight_sum
        
        return best_weights
    
    def dynamic_architecture_search(self, X_train, y_train, X_val, y_val):
        """
        Dynamic neural architecture search
        """
        def architecture_objective(trial):
            # Search space for architecture
            config = {
                'lstm_layers': trial.suggest_int('lstm_layers', 1, 4),
                'lstm_units': [
                    trial.suggest_int(f'lstm_unit_{i}', 32, 256)
                    for i in range(trial.suggest_int('lstm_layers', 1, 4))
                ],
                'use_gru': trial.suggest_categorical('use_gru', [True, False]),
                'use_transformer': trial.suggest_categorical('use_transformer', [True, False]),
                'use_cnn': trial.suggest_categorical('use_cnn', [True, False]),
                'attention_mechanism': trial.suggest_categorical(
                    'attention_mechanism', ['none', 'self', 'multi_head', 'multi_scale']
                ),
                'residual_connections': trial.suggest_categorical('residual_connections', [True, False]),
                'batch_normalization': trial.suggest_categorical('batch_normalization', [True, False]),
                'activation': trial.suggest_categorical(
                    'activation', ['relu', 'gelu', 'swish', 'mish']
                ),
                'dropout_schedule': trial.suggest_categorical(
                    'dropout_schedule', ['constant', 'decreasing', 'adaptive']
                )
            }
            
            # Build dynamic architecture
            model = self.build_dynamic_architecture(config, X_train.shape)
            
            # Train and evaluate
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=32,
                    verbose=0
                )
                
                return min(history.history['val_loss'])
            except:
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(architecture_objective, n_trials=50)
        
        return study.best_params
    
    def build_dynamic_architecture(self, config: Dict, input_shape: Tuple) -> tf.keras.Model:
        """
        Build model architecture based on configuration
        """
        inputs = tf.keras.Input(shape=input_shape[1:])
        x = inputs
        
        # LSTM layers
        for i in range(config['lstm_layers']):
            units = config['lstm_units'][i] if i < len(config['lstm_units']) else 64
            return_sequences = i < config['lstm_layers'] - 1 or config['use_transformer']
            
            x = tf.keras.layers.LSTM(
                units, return_sequences=return_sequences,
                dropout=0.2, recurrent_dropout=0.2
            )(x)
            
            if config['batch_normalization']:
                x = tf.keras.layers.BatchNormalization()(x)
        
        # Additional architectures
        branches = []
        
        if config['use_gru']:
            gru_branch = tf.keras.layers.GRU(64, return_sequences=False)(inputs)
            branches.append(gru_branch)
        
        if config['use_transformer']:
            transformer_branch = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=64
            )(x, x)
            transformer_branch = tf.keras.layers.GlobalAveragePooling1D()(transformer_branch)
            branches.append(transformer_branch)
        
        if config['use_cnn']:
            cnn_branch = tf.keras.layers.Conv1D(64, 3, activation='relu')(inputs)
            cnn_branch = tf.keras.layers.GlobalMaxPooling1D()(cnn_branch)
            branches.append(cnn_branch)
        
        # Combine branches
        if branches:
            if len(branches) > 1:
                x = tf.keras.layers.Concatenate()(branches + [x])
            else:
                x = tf.keras.layers.Concatenate()([x, branches[0]])
        
        # Final layers
        x = tf.keras.layers.Dense(64, activation=config['activation'])(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def multi_objective_optimization(self, model_class, X_train, y_train, X_val, y_val):
        """
        Multi-objective optimization (accuracy vs. robustness vs. speed)
        """
        def multi_objective(trial):
            config = self.suggest_hyperparameters(trial)
            
            model = model_class(config)
            
            # Measure training time
            import time
            start_time = time.time()
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=config['batch_size'],
                verbose=0
            )
            
            training_time = time.time() - start_time
            
            # Accuracy metrics
            val_loss = min(history.history['val_loss'])
            val_mae = min(history.history['val_mae'])
            
            # Robustness metric (validation stability)
            val_loss_std = np.std(history.history['val_loss'][-10:])
            
            # Multi-objective scores
            accuracy_score = val_loss
            robustness_score = val_loss_std
            speed_score = training_time / 100  # Normalize
            
            # Pareto optimization
            trial.set_user_attr('accuracy', accuracy_score)
            trial.set_user_attr('robustness', robustness_score)
            trial.set_user_attr('speed', speed_score)
            
            # Weighted combination
            return accuracy_score * 0.6 + robustness_score * 0.3 + speed_score * 0.1
        
        study = optuna.create_study(direction='minimize')
        study.optimize(multi_objective, n_trials=self.n_trials)
        
        return study
    
    def suggest_hyperparameters(self, trial):
        """
        Suggest hyperparameters for optimization
        """
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
            'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True),
            'n_layers': trial.suggest_int('n_layers', 2, 6),
            'layer_size': trial.suggest_int('layer_size', 32, 256),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
            'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish'])
        }
    
    def save_optimization_results(self, study, filepath: str):
        """
        Save optimization results for future use
        """
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'trials': [
                {
                    'params': trial.params,
                    'value': trial.value,
                    'state': str(trial.state)
                }
                for trial in study.trials
            ]
        }
        
        joblib.dump(results, filepath)
    
    def load_and_continue_optimization(self, filepath: str):
        """
        Load previous optimization results and continue
        """
        try:
            results = joblib.load(filepath)
            
            # Create new study and add previous trials
            study = optuna.create_study(direction='minimize')
            
            for trial_data in results['trials']:
                trial = optuna.trial.create_trial(
                    params=trial_data['params'],
                    distributions={},
                    value=trial_data['value']
                )
                study.add_trial(trial)
            
            return study
        except FileNotFoundError:
            return optuna.create_study(direction='minimize')
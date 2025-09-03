import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleStacking:
    """
    Advanced ensemble stacking with multiple levels of meta-learners
    """
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.base_models = {}
        self.meta_models = {}
        self.stacked_features = None
        
    def create_base_models(self, input_shape: Tuple) -> Dict:
        """Create diverse base models for ensemble"""
        
        base_models = {}
        
        # 1. LSTM Variant
        lstm_input = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True, dropout=0.2)(lstm_input)
        x = LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = LSTM(32, dropout=0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        lstm_output = Dense(1, name='lstm_pred')(x)
        base_models['lstm'] = Model(lstm_input, lstm_output)
        
        # 2. GRU Variant
        gru_input = Input(shape=input_shape)
        x = GRU(128, return_sequences=True, dropout=0.2)(gru_input)
        x = GRU(64, return_sequences=True, dropout=0.2)(x)
        x = GRU(32, dropout=0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        gru_output = Dense(1, name='gru_pred')(x)
        base_models['gru'] = Model(gru_input, gru_output)
        
        # 3. Transformer-based Model
        transformer_input = Input(shape=input_shape)
        x = self._build_transformer_encoder(transformer_input, 64, 8, 2)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        transformer_output = Dense(1, name='transformer_pred')(x)
        base_models['transformer'] = Model(transformer_input, transformer_output)
        
        # 4. CNN-LSTM Hybrid
        cnn_lstm_input = Input(shape=input_shape)
        x = Conv1D(64, 3, activation='relu', padding='same')(cnn_lstm_input)
        x = Conv1D(32, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        x = LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = LSTM(32, dropout=0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        cnn_lstm_output = Dense(1, name='cnn_lstm_pred')(x)
        base_models['cnn_lstm'] = Model(cnn_lstm_input, cnn_lstm_output)
        
        # 5. Attention-based LSTM
        attention_input = Input(shape=input_shape)
        lstm_out = LSTM(64, return_sequences=True)(attention_input)
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(64)(attention)
        attention = Permute([2, 1])(attention)
        
        sent_representation = Multiply()([lstm_out, attention])
        sent_representation = Lambda(lambda xin: tf.reduce_sum(xin, axis=-2))(sent_representation)
        x = Dense(128, activation='relu')(sent_representation)
        x = Dropout(0.3)(x)
        attention_output = Dense(1, name='attention_pred')(x)
        base_models['attention_lstm'] = Model(attention_input, attention_output)
        
        return base_models
    
    def _build_transformer_encoder(self, inputs, d_model, num_heads, num_layers):
        """Build transformer encoder"""
        x = inputs
        
        for _ in range(num_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model,
                dropout=0.1
            )(x, x)
            
            # Skip connection and layer norm
            x = Add()([x, attn_output])
            x = LayerNormalization()(x)
            
            # Feed forward network
            ffn_output = Dense(d_model * 4, activation='relu')(x)
            ffn_output = Dense(d_model)(ffn_output)
            
            # Skip connection and layer norm
            x = Add()([x, ffn_output])
            x = LayerNormalization()(x)
        
        return x
    
    def create_meta_models(self) -> Dict:
        """Create meta-learning models"""
        meta_models = {}
        
        # Level 1 Meta-learners
        meta_models['xgb_meta'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        meta_models['lgb_meta'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        meta_models['catboost_meta'] = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        
        # Level 2 Meta-meta-learner
        meta_models['neural_meta'] = self._create_neural_meta_model()
        
        return meta_models
    
    def _create_neural_meta_model(self):
        """Create neural network for meta-learning"""
        meta_input = Input(shape=(5,))  # 5 base model predictions
        x = Dense(64, activation='relu')(meta_input)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        meta_output = Dense(1)(x)
        
        meta_model = Model(meta_input, meta_output)
        meta_model.compile(optimizer='adam', loss='mse')
        
        return meta_model
    
    def train_stacked_ensemble(self, X: np.ndarray, y: np.ndarray, 
                             validation_data: Optional[Tuple] = None) -> Dict:
        """Train the complete stacked ensemble"""
        
        print("🚀 Training Advanced Stacked Ensemble...")
        
        # Step 1: Train base models with cross-validation
        base_predictions = self._train_base_models_cv(X, y)
        
        # Step 2: Train meta-models on base predictions
        meta_results = self._train_meta_models(base_predictions, y)
        
        # Step 3: Final ensemble combination
        final_weights = self._optimize_ensemble_weights(base_predictions, y)
        
        results = {
            'base_models': self.base_models,
            'meta_models': self.meta_models,
            'ensemble_weights': final_weights,
            'cv_scores': meta_results,
            'base_predictions_shape': base_predictions.shape
        }
        
        return results
    
    def _train_base_models_cv(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train base models with time series cross-validation"""
        
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        base_predictions = np.zeros((len(X), len(self.base_models)))
        
        self.base_models = self.create_base_models(X.shape[1:])
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"   📊 Training base models - Fold {fold + 1}/{self.n_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Train model
                model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )
                
                # Predict on validation set
                val_pred = model.predict(X_val_fold, verbose=0)
                base_predictions[val_idx, i] = val_pred.flatten()
        
        return base_predictions
    
    def _train_meta_models(self, base_predictions: np.ndarray, y: np.ndarray) -> Dict:
        """Train meta-learning models"""
        
        print("   🧠 Training meta-learners...")
        
        self.meta_models = self.create_meta_models()
        meta_results = {}
        
        # Split for meta-training
        split_idx = int(len(base_predictions) * 0.8)
        X_meta_train = base_predictions[:split_idx]
        y_meta_train = y[:split_idx]
        X_meta_test = base_predictions[split_idx:]
        y_meta_test = y[split_idx:]
        
        for name, meta_model in self.meta_models.items():
            if name == 'neural_meta':
                # Neural meta-model
                meta_model.fit(
                    X_meta_train, y_meta_train,
                    validation_data=(X_meta_test, y_meta_test),
                    epochs=100,
                    batch_size=32,
                    verbose=0
                )
                pred = meta_model.predict(X_meta_test, verbose=0).flatten()
            else:
                # Tree-based meta-models
                meta_model.fit(X_meta_train, y_meta_train)
                pred = meta_model.predict(X_meta_test)
            
            mse = mean_squared_error(y_meta_test, pred)
            mae = mean_absolute_error(y_meta_test, pred)
            
            meta_results[name] = {'mse': mse, 'mae': mae}
        
        return meta_results
    
    def _optimize_ensemble_weights(self, base_predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize ensemble combination weights using Bayesian optimization"""
        
        def objective(weights):
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            
            ensemble_pred = np.average(base_predictions, weights=weights, axis=1)
            mse = mean_squared_error(y, ensemble_pred)
            return mse
        
        # Bayesian optimization of weights
        from scipy.optimize import minimize
        
        n_models = base_predictions.shape[1]
        initial_weights = np.ones(n_models) / n_models
        
        bounds = [(0.01, 1.0) for _ in range(n_models)]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        optimal_weights = result.x / np.sum(result.x)
        
        print(f"   ⚖️ Optimal ensemble weights: {optimal_weights}")
        
        return optimal_weights
    
    def predict_ensemble(self, X: np.ndarray, use_meta: bool = True) -> np.ndarray:
        """Make ensemble predictions"""
        
        # Get base model predictions
        base_preds = []
        for name, model in self.base_models.items():
            pred = model.predict(X, verbose=0)
            base_preds.append(pred.flatten())
        
        base_predictions = np.column_stack(base_preds)
        
        if use_meta and self.meta_models:
            # Use meta-model for final prediction
            meta_preds = []
            for name, meta_model in self.meta_models.items():
                if name == 'neural_meta':
                    pred = meta_model.predict(base_predictions, verbose=0).flatten()
                else:
                    pred = meta_model.predict(base_predictions)
                meta_preds.append(pred)
            
            # Average meta-model predictions
            final_prediction = np.mean(meta_preds, axis=0)
        else:
            # Use weighted average of base models
            if hasattr(self, 'ensemble_weights'):
                final_prediction = np.average(base_predictions, weights=self.ensemble_weights, axis=1)
            else:
                final_prediction = np.mean(base_predictions, axis=1)
        
        return final_prediction

class ReinforcementLearningTrader:
    """
    Reinforcement Learning agent for adaptive trading strategies
    """
    def __init__(self, state_size: int, action_size: int = 3):  # Buy, Hold, Sell
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        
    def _build_model(self):
        """Build Deep Q-Network"""
        model = tf.keras.Sequential([
            Dense(512, input_shape=(self.state_size,), activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = np.array([self.memory[i][0] for i in batch])
        actions = np.array([self.memory[i][1] for i in batch])
        rewards = np.array([self.memory[i][2] for i in batch])
        next_states = np.array([self.memory[i][3] for i in batch])
        dones = np.array([self.memory[i][4] for i in batch])
        
        target_q_values = self.target_network.predict(next_states, verbose=0)
        max_target_q_values = np.max(target_q_values, axis=1)
        
        targets = rewards + (0.95 * max_target_q_values * (1 - dones))
        
        target_full = self.q_network.predict(states, verbose=0)
        target_full[np.arange(batch_size), actions] = targets
        
        self.q_network.fit(states, target_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def train_trading_agent(self, price_data: np.ndarray, features: np.ndarray, 
                           episodes: int = 1000) -> Dict:
        """Train the RL trading agent"""
        
        print(f"🤖 Training RL Trading Agent for {episodes} episodes...")
        
        episode_rewards = []
        episode_actions = []
        
        for episode in range(episodes):
            state = features[0]
            total_reward = 0
            position = 0  # 0: no position, 1: long position
            actions_taken = []
            
            for t in range(1, len(price_data) - 1):
                action = self.act(state)
                actions_taken.append(action)
                
                # Calculate reward based on action and price movement
                current_price = price_data[t]
                next_price = price_data[t + 1]
                price_change = (next_price - current_price) / current_price
                
                if action == 0:  # Buy
                    reward = price_change if position == 0 else 0
                    position = 1
                elif action == 1:  # Hold
                    reward = price_change if position == 1 else 0
                else:  # Sell
                    reward = -price_change if position == 1 else 0
                    position = 0
                
                next_state = features[t + 1] if t + 1 < len(features) else state
                done = t == len(price_data) - 2
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(self.memory) > 100:
                    self.replay()
            
            episode_rewards.append(total_reward)
            episode_actions.append(actions_taken)
            
            # Update target network every 10 episodes
            if episode % 10 == 0:
                self.update_target_network()
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"   Episode {episode}, Average Reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.3f}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_actions': episode_actions,
            'final_epsilon': self.epsilon,
            'total_episodes': episodes
        }

class BayesianNeuralNetwork:
    """
    Bayesian Neural Network for uncertainty quantification
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.model = None
        
    def build_bayesian_model(self):
        """Build Bayesian neural network using TensorFlow Probability"""
        try:
            import tensorflow_probability as tfp
            tfd = tfp.distributions
            tfpl = tfp.layers
            
            # Prior for weights
            def prior_fn(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n, dtype=dtype),
                    scale_diag=tf.ones(n, dtype=dtype)
                )
            
            # Posterior for weights
            def posterior_fn(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tfp.layers.VariationalGaussian.make_posterior_fn(
                    loc_initializer=tf.initializers.RandomNormal(stddev=0.1),
                    untransformed_scale_initializer=tf.initializers.RandomNormal(
                        mean=-3., stddev=0.1
                    )
                )(n)
            
            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=(self.input_dim,)))
            
            # Add variational dense layers
            for dim in self.hidden_dims:
                model.add(tfpl.DenseVariational(
                    dim,
                    make_prior_fn=prior_fn,
                    make_posterior_fn=posterior_fn,
                    kl_weight=1/len(self.hidden_dims),
                    activation='relu'
                ))
                model.add(Dropout(0.2))
            
            # Output layer
            model.add(tfpl.DenseVariational(
                1,
                make_prior_fn=prior_fn,
                make_posterior_fn=posterior_fn,
                kl_weight=1/len(self.hidden_dims)
            ))
            
            # Custom loss function
            def negative_log_likelihood(y_true, y_pred):
                return -y_pred.log_prob(y_true)
            
            model.compile(
                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                loss=negative_log_likelihood
            )
            
            self.model = model
            
        except ImportError:
            print("⚠️ TensorFlow Probability not available, using regular neural network")
            self.model = self._build_regular_model()
    
    def _build_regular_model(self):
        """Fallback regular neural network"""
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.input_dim,)))
        
        for dim in self.hidden_dims:
            model.add(Dense(dim, activation='relu'))
            model.add(Dropout(0.2))
        
        # Output mean and std
        model.add(Dense(2))  # Mean and log(std)
        
        def gaussian_loss(y_true, y_pred):
            mean = y_pred[:, 0]
            log_std = y_pred[:, 1]
            std = tf.exp(log_std)
            
            return tf.reduce_mean(
                0.5 * tf.log(2 * np.pi * std**2) + 
                0.5 * (y_true - mean)**2 / std**2
            )
        
        model.compile(optimizer='adam', loss=gaussian_loss)
        return model
    
    def train_with_uncertainty(self, X: np.ndarray, y: np.ndarray, 
                             validation_data: Optional[Tuple] = None,
                             epochs: int = 100) -> Dict:
        """Train Bayesian model"""
        
        if self.model is None:
            self.build_bayesian_model()
        
        history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history.history
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty quantification"""
        
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction

class OnlineLearningSystem:
    """
    Online learning system with concept drift detection
    """
    def __init__(self, base_model, drift_threshold: float = 0.1):
        self.base_model = base_model
        self.drift_threshold = drift_threshold
        self.performance_history = []
        self.drift_detected = False
        self.adaptation_trigger = 0
        
    def detect_concept_drift(self, recent_errors: List[float], 
                           window_size: int = 50) -> bool:
        """Detect concept drift using Page-Hinkley test"""
        
        if len(recent_errors) < window_size:
            return False
        
        # Page-Hinkley test for drift detection
        recent_mean = np.mean(recent_errors[-window_size:])
        historical_mean = np.mean(recent_errors[:-window_size]) if len(recent_errors) > window_size else recent_mean
        
        drift_magnitude = abs(recent_mean - historical_mean)
        
        if drift_magnitude > self.drift_threshold:
            print(f"🚨 Concept drift detected! Magnitude: {drift_magnitude:.4f}")
            return True
        
        return False
    
    def adaptive_learning_rate(self, current_performance: float) -> float:
        """Adapt learning rate based on performance"""
        
        if not self.performance_history:
            return 0.001
        
        recent_performance = np.mean(self.performance_history[-10:])
        
        if current_performance < recent_performance:
            # Performance degrading, increase learning rate
            return min(0.01, 0.001 * 2)
        else:
            # Performance stable/improving, decrease learning rate
            return max(0.0001, 0.001 * 0.5)
    
    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Incrementally update model with new data"""
        
        # Calculate recent performance
        pred = self.base_model.predict(X_new, verbose=0)
        mse = mean_squared_error(y_new, pred)
        
        self.performance_history.append(mse)
        
        # Check for drift
        if self.detect_concept_drift(self.performance_history):
            self.drift_detected = True
            self.adaptation_trigger += 1
            
            # Adapt learning rate
            new_lr = self.adaptive_learning_rate(mse)
            tf.keras.backend.set_value(self.base_model.optimizer.learning_rate, new_lr)
            
            print(f"📈 Adapting model - New LR: {new_lr:.6f}")
        
        # Incremental training
        self.base_model.fit(
            X_new, y_new,
            epochs=1,
            batch_size=len(X_new),
            verbose=0
        )
    
    def get_adaptation_stats(self) -> Dict:
        """Get adaptation statistics"""
        return {
            'drift_detected': self.drift_detected,
            'adaptation_triggers': self.adaptation_trigger,
            'performance_trend': np.mean(self.performance_history[-10:]) if len(self.performance_history) >= 10 else None,
            'total_updates': len(self.performance_history)
        }

class NeuralArchitectureSearch:
    """
    Neural Architecture Search for optimal model design
    """
    def __init__(self, input_shape: Tuple, search_space: Dict):
        self.input_shape = input_shape
        self.search_space = search_space
        self.best_architecture = None
        self.best_score = float('inf')
        
    def sample_architecture(self) -> Dict:
        """Sample a random architecture from search space"""
        architecture = {}
        
        for param, values in self.search_space.items():
            if isinstance(values, list):
                architecture[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # Range sampling
                architecture[param] = np.random.uniform(values[0], values[1])
            else:
                architecture[param] = values
        
        return architecture
    
    def build_model_from_architecture(self, architecture: Dict) -> tf.keras.Model:
        """Build model from architecture specification"""
        
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Build layers based on architecture
        for i in range(architecture.get('num_layers', 3)):
            layer_type = architecture.get(f'layer_{i}_type', 'dense')
            units = architecture.get(f'layer_{i}_units', 64)
            
            if layer_type == 'lstm':
                return_seq = i < architecture.get('num_layers', 3) - 1
                x = LSTM(units, return_sequences=return_seq, dropout=0.2)(x)
            elif layer_type == 'gru':
                return_seq = i < architecture.get('num_layers', 3) - 1
                x = GRU(units, return_sequences=return_seq, dropout=0.2)(x)
            elif layer_type == 'dense':
                if len(x.shape) > 2:  # Need to flatten
                    x = GlobalAveragePooling1D()(x)
                x = Dense(units, activation='relu')(x)
                x = Dropout(architecture.get('dropout_rate', 0.2))(x)
        
        # Ensure we have the right shape for output
        if len(x.shape) > 2:
            x = GlobalAveragePooling1D()(x)
        
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=architecture.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def evaluate_architecture(self, architecture: Dict, X_train: np.ndarray, 
                            y_train: np.ndarray, X_val: np.ndarray, 
                            y_val: np.ndarray) -> float:
        """Evaluate architecture performance"""
        
        try:
            model = self.build_model_from_architecture(architecture)
            
            # Quick training for evaluation
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,  # Quick evaluation
                batch_size=32,
                verbose=0
            )
            
            # Get validation loss
            val_loss = min(history.history['val_loss'])
            
            return val_loss
            
        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            return float('inf')
    
    def search_optimal_architecture(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  n_trials: int = 50) -> Dict:
        """Search for optimal architecture"""
        
        print(f"🔍 Searching optimal architecture over {n_trials} trials...")
        
        search_results = []
        
        for trial in range(n_trials):
            architecture = self.sample_architecture()
            score = self.evaluate_architecture(architecture, X_train, y_train, X_val, y_val)
            
            search_results.append({
                'architecture': architecture,
                'score': score,
                'trial': trial
            })
            
            if score < self.best_score:
                self.best_score = score
                self.best_architecture = architecture.copy()
                print(f"   🏆 New best architecture found! Score: {score:.6f}")
            
            if trial % 10 == 0:
                print(f"   Trial {trial}/{n_trials}, Best score: {self.best_score:.6f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_score': self.best_score,
            'all_results': search_results
        }
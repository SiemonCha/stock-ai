"""
Advanced Market Regime Detection System
Provides sophisticated market state classification and transition detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import KMeans, GaussianMixture
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except ImportError:
    print("Warning: sklearn not available, using fallback implementations")
    KMeans = GaussianMixture = BayesianGaussianMixture = None
    StandardScaler = PCA = silhouette_score = None

try:
    from scipy import stats
    from scipy.signal import find_peaks
    from scipy.stats import jarque_bera
except ImportError:
    print("Warning: scipy not available, using fallback implementations")
    stats = find_peaks = jarque_bera = None

class MarketRegimeDetector:
    """Advanced market regime detection with multiple methodologies"""
    
    def __init__(self):
        self.regime_models = {}
        self.feature_scalers = {}
        self.regime_history = []
        self.transition_matrix = None
        self.regime_features = None
        
        # Regime configurations
        self.regime_configs = {
            'bull_market': {
                'volatility_threshold': 0.15,
                'trend_threshold': 0.02,
                'volume_multiplier': 1.2
            },
            'bear_market': {
                'volatility_threshold': 0.25,
                'trend_threshold': -0.02,
                'volume_multiplier': 1.5
            },
            'sideways': {
                'volatility_threshold': 0.10,
                'trend_threshold': 0.005,
                'volume_multiplier': 0.8
            },
            'high_volatility': {
                'volatility_threshold': 0.30,
                'trend_threshold': 0.0,
                'volume_multiplier': 2.0
            },
            'crisis': {
                'volatility_threshold': 0.40,
                'trend_threshold': -0.05,
                'volume_multiplier': 3.0
            }
        }
    
    def extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive features for regime detection"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std()
        features['realized_volatility'] = np.sqrt((features['returns'] ** 2).rolling(20).mean())
        
        # Trend features
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        features['trend_20'] = (data['Close'] - features['sma_20']) / features['sma_20']
        features['trend_50'] = (data['Close'] - features['sma_50']) / features['sma_50']
        features['momentum'] = data['Close'] / data['Close'].shift(10) - 1
        
        # Volume features
        features['volume_ma'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_ma']
        features['price_volume'] = features['returns'] * features['volume_ratio']
        
        # Market microstructure
        features['spread'] = (data['High'] - data['Low']) / data['Close']
        features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
        features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']
        
        # Statistical features
        features['skewness'] = features['returns'].rolling(20).skew()
        features['kurtosis'] = features['returns'].rolling(20).kurt()
        features['autocorr'] = features['returns'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else 0
        )
        
        # VIX-like features (volatility of volatility)
        features['vol_of_vol'] = features['volatility'].rolling(10).std()
        features['vol_momentum'] = features['volatility'] / features['volatility'].shift(5) - 1
        
        # Regime probability features
        features['extreme_move'] = (np.abs(features['returns']) > 2 * features['volatility']).astype(int)
        features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        return features.dropna()
    
    def detect_regimes_hmm(self, features: pd.DataFrame, n_regimes: int = 5) -> np.ndarray:
        """Hidden Markov Model regime detection"""
        try:
            from hmmlearn.hmm import GaussianHMM
            
            # Select key features for HMM
            hmm_features = features[['returns', 'volatility', 'volume_ratio', 'trend_20']].values
            
            # Standardize features
            scaler = StandardScaler() if StandardScaler else None
            if scaler:
                hmm_features = scaler.fit_transform(hmm_features)
                self.feature_scalers['hmm'] = scaler
            
            # Fit HMM model
            model = GaussianHMM(n_components=n_regimes, covariance_type="full", random_state=42)
            model.fit(hmm_features)
            
            # Predict regimes
            regimes = model.predict(hmm_features)
            self.regime_models['hmm'] = model
            
            return regimes
            
        except ImportError:
            print("Warning: hmmlearn not available, using fallback clustering")
            return self.detect_regimes_clustering(features, n_regimes)
    
    def detect_regimes_clustering(self, features: pd.DataFrame, n_regimes: int = 5) -> np.ndarray:
        """K-means clustering regime detection"""
        if not KMeans:
            return self._fallback_regime_detection(features)
        
        # Select features for clustering
        cluster_features = features[['returns', 'volatility', 'volume_ratio', 'trend_20', 'momentum']].values
        
        # Standardize features
        scaler = StandardScaler()
        cluster_features = scaler.fit_transform(cluster_features)
        self.feature_scalers['kmeans'] = scaler
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(cluster_features, max_k=min(10, n_regimes + 3))
        
        # Fit K-means
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(cluster_features)
        
        self.regime_models['kmeans'] = kmeans
        return regimes
    
    def detect_regimes_gaussian_mixture(self, features: pd.DataFrame, n_regimes: int = 5) -> np.ndarray:
        """Gaussian Mixture Model regime detection"""
        if not GaussianMixture:
            return self._fallback_regime_detection(features)
        
        # Select features
        gmm_features = features[['returns', 'volatility', 'trend_20', 'skewness', 'kurtosis']].values
        
        # Standardize
        scaler = StandardScaler()
        gmm_features = scaler.fit_transform(gmm_features)
        self.feature_scalers['gmm'] = scaler
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, covariance_type='full')
        regimes = gmm.fit_predict(gmm_features)
        
        self.regime_models['gmm'] = gmm
        return regimes
    
    def detect_regimes_threshold(self, features: pd.DataFrame) -> np.ndarray:
        """Rule-based threshold regime detection"""
        n_samples = len(features)
        regimes = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            volatility = features.iloc[i]['volatility']
            trend = features.iloc[i]['trend_20']
            volume_ratio = features.iloc[i]['volume_ratio']
            
            # Crisis regime (highest priority)
            if (volatility > self.regime_configs['crisis']['volatility_threshold'] and
                trend < self.regime_configs['crisis']['trend_threshold'] and
                volume_ratio > self.regime_configs['crisis']['volume_multiplier']):
                regimes[i] = 4  # Crisis
            
            # High volatility regime
            elif volatility > self.regime_configs['high_volatility']['volatility_threshold']:
                regimes[i] = 3  # High volatility
            
            # Bear market regime
            elif (volatility > self.regime_configs['bear_market']['volatility_threshold'] and
                  trend < self.regime_configs['bear_market']['trend_threshold']):
                regimes[i] = 2  # Bear market
            
            # Bull market regime
            elif (volatility < self.regime_configs['bull_market']['volatility_threshold'] and
                  trend > self.regime_configs['bull_market']['trend_threshold']):
                regimes[i] = 0  # Bull market
            
            # Sideways regime (default)
            else:
                regimes[i] = 1  # Sideways
        
        return regimes
    
    def ensemble_regime_detection(self, features: pd.DataFrame, n_regimes: int = 5) -> Dict[str, np.ndarray]:
        """Ensemble approach combining multiple methods"""
        results = {}
        
        # Apply different methods
        try:
            results['hmm'] = self.detect_regimes_hmm(features, n_regimes)
        except:
            print("HMM regime detection failed, skipping")
        
        try:
            results['clustering'] = self.detect_regimes_clustering(features, n_regimes)
        except:
            print("Clustering regime detection failed, skipping")
        
        try:
            results['gmm'] = self.detect_regimes_gaussian_mixture(features, n_regimes)
        except:
            print("GMM regime detection failed, skipping")
        
        results['threshold'] = self.detect_regimes_threshold(features)
        
        # Create consensus regime
        if len(results) > 1:
            results['consensus'] = self._create_consensus_regimes(results, features)
        
        return results
    
    def _create_consensus_regimes(self, regime_results: Dict[str, np.ndarray], features: pd.DataFrame) -> np.ndarray:
        """Create consensus regime from multiple methods"""
        n_samples = len(features)
        consensus_regimes = np.zeros(n_samples, dtype=int)
        
        # Convert all regimes to common scale (0-4)
        normalized_regimes = {}
        for method, regimes in regime_results.items():
            if method != 'threshold':  # Threshold already in correct scale
                normalized_regimes[method] = self._normalize_regime_labels(regimes, 5)
            else:
                normalized_regimes[method] = regimes
        
        # Voting mechanism
        for i in range(n_samples):
            votes = [regimes[i] for regimes in normalized_regimes.values()]
            consensus_regimes[i] = max(set(votes), key=votes.count)  # Majority vote
        
        # Smooth transitions
        consensus_regimes = self._smooth_regime_transitions(consensus_regimes, window=3)
        
        return consensus_regimes
    
    def _normalize_regime_labels(self, regimes: np.ndarray, n_target_regimes: int) -> np.ndarray:
        """Normalize regime labels to target number of regimes"""
        unique_regimes = np.unique(regimes)
        n_current = len(unique_regimes)
        
        if n_current == n_target_regimes:
            return regimes
        
        # Map current regimes to target regimes
        normalized = np.zeros_like(regimes)
        for i, regime in enumerate(unique_regimes):
            target_regime = int(i * (n_target_regimes - 1) / max(1, n_current - 1))
            normalized[regimes == regime] = target_regime
        
        return normalized
    
    def _smooth_regime_transitions(self, regimes: np.ndarray, window: int = 3) -> np.ndarray:
        """Smooth regime transitions to reduce noise"""
        smoothed = regimes.copy()
        n_samples = len(regimes)
        
        for i in range(window, n_samples - window):
            window_regimes = regimes[i-window:i+window+1]
            mode_regime = max(set(window_regimes), key=list(window_regimes).count)
            
            # Only change if there's strong evidence
            if list(window_regimes).count(mode_regime) >= window:
                smoothed[i] = mode_regime
        
        return smoothed
    
    def calculate_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime transition probability matrix"""
        n_regimes = len(np.unique(regimes))
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            current_regime = int(regimes[i])
            next_regime = int(regimes[i + 1])
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), 
                                    where=row_sums!=0)
        
        self.transition_matrix = transition_matrix
        return transition_matrix
    
    def predict_regime_probability(self, features: pd.DataFrame, horizon: int = 5) -> Dict[int, float]:
        """Predict future regime probabilities"""
        if self.transition_matrix is None:
            return {}
        
        # Get current regime probabilities
        current_features = features.tail(1)
        current_regime_probs = self._get_current_regime_probabilities(current_features)
        
        # Forward propagation
        future_probs = current_regime_probs.copy()
        for _ in range(horizon):
            future_probs = np.dot(future_probs, self.transition_matrix)
        
        return {i: prob for i, prob in enumerate(future_probs)}
    
    def _get_current_regime_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """Get current regime probabilities from ensemble models"""
        n_regimes = 5
        probs = np.zeros(n_regimes)
        
        # Simple approach: use threshold method
        regime = self.detect_regimes_threshold(features)[0]
        probs[regime] = 1.0
        
        return probs
    
    def _find_optimal_clusters(self, features: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using silhouette score"""
        if not silhouette_score:
            return 5  # Default
        
        best_score = -1
        best_k = 5
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def _fallback_regime_detection(self, features: pd.DataFrame) -> np.ndarray:
        """Fallback regime detection when advanced libraries unavailable"""
        return self.detect_regimes_threshold(features)
    
    def get_regime_characteristics(self, regimes: np.ndarray, features: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Analyze characteristics of each regime"""
        characteristics = {}
        
        for regime_id in np.unique(regimes):
            mask = regimes == regime_id
            regime_features = features[mask]
            
            characteristics[int(regime_id)] = {
                'avg_return': regime_features['returns'].mean(),
                'volatility': regime_features['volatility'].mean(),
                'avg_volume_ratio': regime_features['volume_ratio'].mean(),
                'trend_strength': regime_features['trend_20'].mean(),
                'duration_days': np.sum(mask),
                'frequency': np.mean(mask),
                'max_drawdown': self._calculate_regime_drawdown(regime_features['returns']),
                'sharpe_ratio': self._calculate_regime_sharpe(regime_features['returns'])
            }
        
        return characteristics
    
    def _calculate_regime_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for regime"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_regime_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio for regime"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    
    def get_regime_labels(self) -> Dict[int, str]:
        """Get human-readable regime labels"""
        return {
            0: "Bull Market",
            1: "Sideways/Consolidation", 
            2: "Bear Market",
            3: "High Volatility",
            4: "Crisis/Panic"
        }

class RegimeAwareStrategy:
    """Strategy that adapts based on detected market regimes"""
    
    def __init__(self, regime_detector: MarketRegimeDetector):
        self.regime_detector = regime_detector
        self.strategy_params = {
            0: {'position_size': 1.0, 'stop_loss': 0.15, 'take_profit': 0.25},  # Bull
            1: {'position_size': 0.5, 'stop_loss': 0.10, 'take_profit': 0.15},  # Sideways
            2: {'position_size': 0.3, 'stop_loss': 0.08, 'take_profit': 0.12},  # Bear
            3: {'position_size': 0.2, 'stop_loss': 0.05, 'take_profit': 0.08},  # High Vol
            4: {'position_size': 0.0, 'stop_loss': 0.03, 'take_profit': 0.05},  # Crisis
        }
    
    def get_strategy_parameters(self, current_regime: int) -> Dict[str, float]:
        """Get strategy parameters for current regime"""
        return self.strategy_params.get(current_regime, self.strategy_params[1])
    
    def should_trade(self, current_regime: int, regime_confidence: float = 0.7) -> bool:
        """Determine if trading should occur in current regime"""
        if current_regime == 4:  # Crisis - no trading
            return False
        
        return regime_confidence > 0.6  # Trade only with sufficient confidence
    
    def adjust_position_size(self, base_size: float, current_regime: int) -> float:
        """Adjust position size based on regime"""
        multiplier = self.strategy_params[current_regime]['position_size']
        return base_size * multiplier

def create_regime_detection_pipeline(data: pd.DataFrame) -> Dict[str, Any]:
    """Complete regime detection pipeline"""
    detector = MarketRegimeDetector()
    
    # Extract features
    features = detector.extract_regime_features(data)
    
    # Detect regimes using ensemble approach
    regime_results = detector.ensemble_regime_detection(features)
    
    # Use consensus if available, otherwise use threshold
    final_regimes = regime_results.get('consensus', regime_results['threshold'])
    
    # Calculate transition matrix
    transition_matrix = detector.calculate_transition_matrix(final_regimes)
    
    # Analyze regime characteristics
    characteristics = detector.get_regime_characteristics(final_regimes, features)
    
    # Create strategy
    strategy = RegimeAwareStrategy(detector)
    
    return {
        'detector': detector,
        'features': features,
        'regimes': final_regimes,
        'regime_results': regime_results,
        'transition_matrix': transition_matrix,
        'characteristics': characteristics,
        'strategy': strategy,
        'regime_labels': detector.get_regime_labels()
    }
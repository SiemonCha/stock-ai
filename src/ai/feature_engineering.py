import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
import talib
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx

class IntelligentFeatureEngine:
    """
    Advanced feature engineering with market intelligence
    """
    def __init__(self):
        self.feature_importance_history = {}
        self.market_graph = nx.Graph()
        
    def create_cross_asset_features(self, symbols: List[str], period='2y') -> pd.DataFrame:
        """
        Create features based on cross-asset relationships
        """
        # Download data for multiple assets
        data = {}
        for symbol in symbols:
            try:
                ticker_data = yf.download(symbol, period=period)
                data[symbol] = ticker_data['Close']
            except:
                continue
        
        df = pd.DataFrame(data)
        
        # Cross-asset correlation features
        correlation_matrix = df.corr()
        
        features = pd.DataFrame(index=df.index)
        
        for i, symbol1 in enumerate(symbols):
            if symbol1 not in df.columns:
                continue
                
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                if symbol2 not in df.columns:
                    continue
                    
                # Rolling correlation
                rolling_corr = df[symbol1].rolling(20).corr(df[symbol2])
                features[f'{symbol1}_{symbol2}_corr'] = rolling_corr
                
                # Cointegration-like feature
                spread = df[symbol1] - df[symbol2]
                features[f'{symbol1}_{symbol2}_spread'] = spread
                features[f'{symbol1}_{symbol2}_spread_zscore'] = (
                    spread - spread.rolling(60).mean()
                ) / spread.rolling(60).std()
        
        return features.fillna(0)
    
    def create_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced microstructure features
        """
        features = pd.DataFrame(index=df.index)
        
        # Order flow imbalance proxy
        features['ofi_proxy'] = (
            (df['Close'] - df['Low']) - (df['High'] - df['Close'])
        ) / (df['High'] - df['Low'])
        
        # Kyle's Lambda (price impact)
        returns = df['Close'].pct_change()
        volume_impact = returns.rolling(20).std() / np.log(df['Volume']).rolling(20).mean()
        features['kyles_lambda'] = volume_impact
        
        # Amihud Illiquidity
        features['amihud_illiquidity'] = abs(returns) / (df['Volume'] * df['Close'])
        
        # Roll's Spread Estimator
        return_autocorr = returns.rolling(20).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0
        )
        features['roll_spread'] = 2 * np.sqrt(-return_autocorr.clip(upper=0))
        
        # High-Low Spread
        features['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Realized Volatility Components
        features['rv_daily'] = returns.rolling(20).std()
        features['rv_intraday'] = (
            (np.log(df['High']) - np.log(df['Close'])) * 
            (np.log(df['High']) - np.log(df['Open'])) +
            (np.log(df['Low']) - np.log(df['Close'])) * 
            (np.log(df['Low']) - np.log(df['Open']))
        )
        
        return features.fillna(0)
    
    def create_behavioral_finance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features based on behavioral finance theory
        """
        features = pd.DataFrame(index=df.index)
        returns = df['Close'].pct_change()
        
        # Overreaction indicators
        features['overreaction_5d'] = returns.rolling(5).sum()
        features['overreaction_20d'] = returns.rolling(20).sum()
        
        # Momentum and reversal
        features['momentum_12_1'] = (
            df['Close'] / df['Close'].shift(252) - 1  # 12-month momentum
        ) - (df['Close'] / df['Close'].shift(21) - 1)  # minus 1-month
        
        # Herding behavior proxy
        features['herding_proxy'] = abs(
            returns - returns.rolling(20).mean()
        ) / returns.rolling(20).std()
        
        # Anchoring bias indicator
        features['anchoring_52w_high'] = df['Close'] / df['High'].rolling(252).max()
        features['anchoring_52w_low'] = df['Close'] / df['Low'].rolling(252).min()
        
        # Loss aversion proxy
        negative_returns = returns.where(returns < 0, 0)
        positive_returns = returns.where(returns > 0, 0)
        
        features['loss_aversion'] = (
            abs(negative_returns.rolling(60).sum()) / 
            positive_returns.rolling(60).sum().clip(lower=1e-6)
        )
        
        # Disposition effect
        unrealized_gains = (df['Close'] - df['Close'].rolling(60).min()) / df['Close'].rolling(60).min()
        unrealized_losses = (df['Close'].rolling(60).max() - df['Close']) / df['Close'].rolling(60).max()
        
        features['disposition_effect'] = unrealized_gains / (unrealized_gains + unrealized_losses)
        
        return features.fillna(0)
    
    def create_alternative_data_features(self, symbol: str) -> pd.DataFrame:
        """
        Alternative data features (social sentiment, news, etc.)
        """
        # This would typically integrate with external APIs
        # For now, we'll create synthetic alternative data features
        
        # Simulate news sentiment
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        features = pd.DataFrame(index=dates)
        
        # Simulated news sentiment (normally would come from news APIs)
        np.random.seed(42)
        features['news_sentiment'] = np.random.normal(0, 1, len(dates))
        features['news_volume'] = np.random.poisson(5, len(dates))
        
        # Social media sentiment proxy
        features['social_sentiment'] = np.random.normal(0, 1, len(dates))
        features['social_volume'] = np.random.poisson(10, len(dates))
        
        # Google Trends proxy
        features['search_interest'] = np.random.uniform(0, 100, len(dates))
        
        # Insider trading proxy
        features['insider_activity'] = np.random.binomial(1, 0.05, len(dates))
        
        return features
    
    def create_regime_adaptive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features that adapt to different market regimes
        """
        features = pd.DataFrame(index=df.index)
        returns = df['Close'].pct_change()
        
        # Volatility regimes
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        
        # Regime indicators
        features['vol_regime_low'] = (vol_20 < vol_60.quantile(0.33)).astype(int)
        features['vol_regime_med'] = (
            (vol_20 >= vol_60.quantile(0.33)) & 
            (vol_20 <= vol_60.quantile(0.67))
        ).astype(int)
        features['vol_regime_high'] = (vol_20 > vol_60.quantile(0.67)).astype(int)
        
        # Regime-specific indicators
        for regime in ['low', 'med', 'high']:
            regime_mask = features[f'vol_regime_{regime}']
            
            # RSI in different regimes
            rsi = talib.RSI(df['Close'].values)
            features[f'rsi_{regime}_regime'] = rsi * regime_mask
            
            # Moving averages in different regimes
            sma_20 = talib.SMA(df['Close'].values, timeperiod=20)
            features[f'sma_ratio_{regime}_regime'] = (
                (df['Close'] / sma_20) * regime_mask
            )
        
        # Trend strength adaptation
        trend_strength = abs(
            df['Close'].rolling(20).apply(
                lambda x: stats.linregress(range(len(x)), x)[0]
            )
        )
        
        features['adaptive_momentum'] = (
            returns.rolling(10).mean() * trend_strength
        )
        
        return features.fillna(0)
    
    def create_ensemble_meta_features(self, predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Meta-features based on ensemble model predictions
        """
        features = pd.DataFrame()
        
        # Prediction disagreement
        pred_array = np.array(list(predictions_dict.values())).T
        features['pred_std'] = np.std(pred_array, axis=1)
        features['pred_range'] = np.max(pred_array, axis=1) - np.min(pred_array, axis=1)
        
        # Prediction trend
        for window in [5, 10, 20]:
            features[f'pred_trend_{window}'] = pd.Series(
                np.mean(pred_array, axis=1)
            ).rolling(window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0
            )
        
        # Model confidence indicators
        features['high_confidence'] = (features['pred_std'] < features['pred_std'].quantile(0.2)).astype(int)
        features['low_confidence'] = (features['pred_std'] > features['pred_std'].quantile(0.8)).astype(int)
        
        return features.fillna(0)
    
    def create_quantum_inspired_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantum-inspired features for market analysis
        """
        features = pd.DataFrame(index=df.index)
        returns = df['Close'].pct_change()
        
        # Quantum superposition proxy
        # Represents market state uncertainty
        price_momentum = returns.rolling(10).mean()
        price_volatility = returns.rolling(10).std()
        
        features['quantum_superposition'] = np.sqrt(
            price_momentum**2 + price_volatility**2
        )
        
        # Entanglement proxy (correlation with market factors)
        market_proxy = df['Volume'].pct_change()  # Use volume as market proxy
        
        entanglement = returns.rolling(20).corr(market_proxy)
        features['quantum_entanglement'] = entanglement
        
        # Quantum tunneling effect (breakout probability)
        price_range = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
        current_position = (df['Close'] - df['Low'].rolling(20).min()) / price_range
        
        features['tunneling_probability'] = np.exp(-current_position**2)
        
        # Wave function collapse proxy
        features['wave_collapse'] = (
            abs(returns) > returns.rolling(60).std() * 2
        ).astype(int)
        
        return features.fillna(0)
    
    def intelligent_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                    top_k: int = 50) -> List[str]:
        """
        Intelligent feature selection using multiple criteria
        """
        from sklearn.feature_selection import mutual_info_regression, f_regression
        from sklearn.ensemble import RandomForestRegressor
        
        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.95)
        ]
        
        X_reduced = X.drop(columns=high_corr_features)
        
        # Multiple selection methods
        # 1. Mutual Information
        mi_scores = mutual_info_regression(X_reduced.fillna(0), y.fillna(0))
        
        # 2. F-statistic
        f_scores, _ = f_regression(X_reduced.fillna(0), y.fillna(0))
        
        # 3. Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_reduced.fillna(0), y.fillna(0))
        rf_importance = rf.feature_importances_
        
        # Combine scores
        feature_scores = pd.DataFrame({
            'feature': X_reduced.columns,
            'mi_score': mi_scores,
            'f_score': f_scores,
            'rf_score': rf_importance
        })
        
        # Normalize scores
        for col in ['mi_score', 'f_score', 'rf_score']:
            feature_scores[f'{col}_norm'] = (
                feature_scores[col] / feature_scores[col].max()
            )
        
        # Combined score
        feature_scores['combined_score'] = (
            feature_scores['mi_score_norm'] * 0.4 +
            feature_scores['f_score_norm'] * 0.3 +
            feature_scores['rf_score_norm'] * 0.3
        )
        
        # Select top features
        top_features = feature_scores.nlargest(top_k, 'combined_score')['feature'].tolist()
        
        return top_features
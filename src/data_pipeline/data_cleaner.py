"""
Advanced Data Cleaning and Anomaly Detection System
Professional-grade data cleaning with machine learning-based anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

from scipy import stats, signal
from scipy.interpolation import interp1d
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

try:
    import pyod
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.cblof import CBLOF
    from pyod.models.hbos import HBOS
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

try:
    import tslearn
    from tslearn.clustering import TimeSeriesKMeans
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies"""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    STRUCTURAL_BREAK = "structural_break"

class CleaningAction(Enum):
    """Data cleaning actions"""
    REMOVE = "remove"
    INTERPOLATE = "interpolate"
    REPLACE = "replace"
    FLAG = "flag"
    CORRECT = "correct"

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    index: Union[int, List[int]]
    anomaly_type: AnomalyType
    severity: float  # 0-1 score
    description: str
    suggested_action: CleaningAction
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CleaningReport:
    """Data cleaning report"""
    symbol: str
    original_rows: int
    cleaned_rows: int
    anomalies_detected: int
    cleaning_actions: List[Tuple[CleaningAction, int]]  # (action, count)
    quality_improvement: float
    timestamp: datetime
    anomalies: List[AnomalyDetection] = field(default_factory=list)

class AdvancedDataCleaner:
    """
    Advanced data cleaning system with multiple anomaly detection methods
    """
    
    def __init__(self):
        self.anomaly_detectors = self._initialize_detectors()
        self.cleaning_rules = self._initialize_cleaning_rules()
        self.scalers = {}
        
        # Cleaning statistics
        self.cleaning_history = {}
        
        logger.info("AdvancedDataCleaner initialized with multiple detection methods")
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize anomaly detection methods"""
        
        detectors = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'elliptic_envelope': EllipticEnvelope(contamination=0.1, random_state=42),
            'one_class_svm': OneClassSVM(gamma='scale', nu=0.1),
        }
        
        if PYOD_AVAILABLE:
            detectors.update({
                'knn': KNN(contamination=0.1),
                'lof': LOF(contamination=0.1),
                'cblof': CBLOF(contamination=0.1, random_state=42),
                'hbos': HBOS(contamination=0.1)
            })
        
        return detectors
    
    def _initialize_cleaning_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data cleaning rules"""
        
        return {
            'price_data': {
                'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'positive_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'price_relationships': [
                    ('High', 'Low', 'greater_equal'),
                    ('High', 'Open', 'greater_equal'),
                    ('High', 'Close', 'greater_equal'),
                    ('Low', 'Open', 'less_equal'),
                    ('Low', 'Close', 'less_equal')
                ],
                'outlier_thresholds': {
                    'price_change_pct': 0.50,  # 50% daily change
                    'volume_spike': 20.0,  # 20x normal volume
                    'gap_threshold': 0.20,  # 20% gap
                    'zero_volume_threshold': 0.05  # 5% of days can have zero volume
                },
                'interpolation_methods': {
                    'price_columns': 'linear',
                    'volume': 'nearest'
                }
            },
            'fundamental_data': {
                'required_columns': ['symbol'],
                'positive_columns': ['marketCap', 'totalRevenue', 'totalCash'],
                'ratio_bounds': {
                    'pe_ratio': (-1000, 1000),
                    'debt_to_equity': (0, 50),
                    'current_ratio': (0, 100)
                }
            }
        }
    
    def clean_data(self, data: pd.DataFrame, symbol: str, data_type: str = 'price_data') -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Comprehensive data cleaning pipeline
        """
        
        logger.info(f"Starting data cleaning for {symbol}: {len(data)} rows")
        
        original_rows = len(data)
        cleaned_data = data.copy()
        anomalies = []
        cleaning_actions = []
        
        # Initialize report
        report = CleaningReport(
            symbol=symbol,
            original_rows=original_rows,
            cleaned_rows=0,  # Will be updated
            anomalies_detected=0,
            cleaning_actions=[],
            quality_improvement=0.0,
            timestamp=datetime.now()
        )
        
        if cleaned_data.empty:
            report.cleaned_rows = 0
            return cleaned_data, report
        
        # Step 1: Basic data validation and correction
        cleaned_data, basic_anomalies = self._basic_data_validation(cleaned_data, data_type)
        anomalies.extend(basic_anomalies)
        
        # Step 2: Handle missing values
        cleaned_data, missing_actions = self._handle_missing_values(cleaned_data, data_type)
        cleaning_actions.extend(missing_actions)
        
        # Step 3: Detect and handle outliers
        cleaned_data, outlier_anomalies, outlier_actions = self._detect_and_handle_outliers(
            cleaned_data, data_type
        )
        anomalies.extend(outlier_anomalies)
        cleaning_actions.extend(outlier_actions)
        
        # Step 4: Time series specific cleaning
        if isinstance(cleaned_data.index, pd.DatetimeIndex):
            cleaned_data, ts_anomalies, ts_actions = self._time_series_cleaning(
                cleaned_data, data_type
            )
            anomalies.extend(ts_anomalies)
            cleaning_actions.extend(ts_actions)
        
        # Step 5: Advanced anomaly detection
        advanced_anomalies = self._advanced_anomaly_detection(cleaned_data, data_type)
        anomalies.extend(advanced_anomalies)
        
        # Step 6: Business logic validation
        business_anomalies, business_actions = self._business_logic_validation(
            cleaned_data, data_type
        )
        anomalies.extend(business_anomalies)
        cleaning_actions.extend(business_actions)
        
        # Step 7: Final quality assessment
        quality_improvement = self._calculate_quality_improvement(data, cleaned_data)
        
        # Update report
        report.cleaned_rows = len(cleaned_data)
        report.anomalies_detected = len(anomalies)
        report.cleaning_actions = [(action, count) for action, count in 
                                 pd.Series([action for action, _ in cleaning_actions]).value_counts().items()]
        report.quality_improvement = quality_improvement
        report.anomalies = anomalies
        
        # Store cleaning history
        self.cleaning_history[symbol] = report
        
        logger.info(f"Data cleaning complete for {symbol}: {len(anomalies)} anomalies detected, "
                   f"quality improved by {quality_improvement:.1%}")
        
        return cleaned_data, report
    
    def _basic_data_validation(self, data: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[AnomalyDetection]]:
        """Basic data validation and correction"""
        
        anomalies = []
        cleaned_data = data.copy()
        rules = self.cleaning_rules.get(data_type, {})
        
        if data_type == 'price_data':
            # Check price relationships
            price_relationships = rules.get('price_relationships', [])
            
            for col1, col2, relationship in price_relationships:
                if col1 in cleaned_data.columns and col2 in cleaned_data.columns:
                    if relationship == 'greater_equal':
                        violations = cleaned_data[col1] < cleaned_data[col2]
                    elif relationship == 'less_equal':
                        violations = cleaned_data[col1] > cleaned_data[col2]
                    else:
                        continue
                    
                    if violations.any():
                        violation_indices = cleaned_data[violations].index.tolist()
                        
                        anomalies.append(AnomalyDetection(
                            index=violation_indices,
                            anomaly_type=AnomalyType.POINT_ANOMALY,
                            severity=0.8,
                            description=f"Price relationship violation: {col1} vs {col2}",
                            suggested_action=CleaningAction.CORRECT,
                            metadata={'columns': [col1, col2], 'relationship': relationship}
                        ))
                        
                        # Correct violations (use average of surrounding valid values)
                        for idx in violation_indices:
                            try:
                                if relationship == 'greater_equal':
                                    # High should be >= Low, Open, Close
                                    if col1 == 'High':
                                        cleaned_data.loc[idx, col1] = max(
                                            cleaned_data.loc[idx, col2],
                                            cleaned_data.loc[idx, 'Open'] if 'Open' in cleaned_data.columns else cleaned_data.loc[idx, col2],
                                            cleaned_data.loc[idx, 'Close'] if 'Close' in cleaned_data.columns else cleaned_data.loc[idx, col2]
                                        )
                                elif relationship == 'less_equal':
                                    # Low should be <= High, Open, Close
                                    if col1 == 'Low':
                                        cleaned_data.loc[idx, col1] = min(
                                            cleaned_data.loc[idx, col2],
                                            cleaned_data.loc[idx, 'Open'] if 'Open' in cleaned_data.columns else cleaned_data.loc[idx, col2],
                                            cleaned_data.loc[idx, 'Close'] if 'Close' in cleaned_data.columns else cleaned_data.loc[idx, col2]
                                        )
                            except Exception as e:
                                logger.warning(f"Error correcting price relationship at {idx}: {e}")
        
        # Check for negative values in positive columns
        positive_columns = rules.get('positive_columns', [])
        
        for col in positive_columns:
            if col in cleaned_data.columns:
                negative_values = cleaned_data[col] < 0
                
                if negative_values.any():
                    negative_indices = cleaned_data[negative_values].index.tolist()
                    
                    anomalies.append(AnomalyDetection(
                        index=negative_indices,
                        anomaly_type=AnomalyType.POINT_ANOMALY,
                        severity=0.9,
                        description=f"Negative values in {col}",
                        suggested_action=CleaningAction.INTERPOLATE,
                        metadata={'column': col}
                    ))
                    
                    # Replace negative values with NaN for interpolation
                    cleaned_data.loc[negative_values, col] = np.nan
        
        return cleaned_data, anomalies
    
    def _handle_missing_values(self, data: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[Tuple[CleaningAction, int]]]:
        """Handle missing values with appropriate interpolation"""
        
        cleaned_data = data.copy()
        actions = []
        rules = self.cleaning_rules.get(data_type, {})
        
        for column in cleaned_data.columns:
            missing_count = cleaned_data[column].isna().sum()
            
            if missing_count == 0:
                continue
            
            missing_pct = missing_count / len(cleaned_data)
            
            if missing_pct > 0.5:
                # Too many missing values - remove column
                cleaned_data = cleaned_data.drop(columns=[column])
                actions.append((CleaningAction.REMOVE, missing_count))
                logger.warning(f"Removed column {column} due to {missing_pct:.1%} missing values")
                
            elif missing_pct > 0.2:
                # Many missing values - flag for review
                actions.append((CleaningAction.FLAG, missing_count))
                
            else:
                # Interpolate missing values
                if data_type == 'price_data' and column in ['Open', 'High', 'Low', 'Close']:
                    # Linear interpolation for prices
                    cleaned_data[column] = cleaned_data[column].interpolate(method='linear')
                    
                elif data_type == 'price_data' and column == 'Volume':
                    # Forward fill for volume (more conservative)
                    cleaned_data[column] = cleaned_data[column].fillna(method='ffill').fillna(method='bfill')
                    
                else:
                    # Default interpolation
                    cleaned_data[column] = cleaned_data[column].interpolate(method='linear')
                
                actions.append((CleaningAction.INTERPOLATE, missing_count))
        
        return cleaned_data, actions
    
    def _detect_and_handle_outliers(self, data: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[AnomalyDetection], List[Tuple[CleaningAction, int]]]:
        """Detect and handle outliers using multiple methods"""
        
        cleaned_data = data.copy()
        anomalies = []
        actions = []
        
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            column_data = cleaned_data[column].dropna()
            
            if len(column_data) < 10:  # Need enough data for outlier detection
                continue
            
            # Method 1: Statistical outliers (IQR)
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR  # More aggressive than typical 1.5
            upper_bound = Q3 + 3 * IQR
            
            statistical_outliers = (cleaned_data[column] < lower_bound) | (cleaned_data[column] > upper_bound)
            
            # Method 2: Z-score outliers
            z_scores = np.abs(stats.zscore(column_data))
            zscore_threshold = 4.0  # More conservative than typical 3.0
            zscore_outliers_mask = z_scores > zscore_threshold
            zscore_outliers = pd.Series(False, index=cleaned_data.index)
            zscore_outliers.loc[column_data.index] = zscore_outliers_mask
            
            # Method 3: Modified Z-score (robust to outliers)
            median = column_data.median()
            mad = np.median(np.abs(column_data - median))
            modified_z_scores = 0.6745 * (column_data - median) / mad
            modified_zscore_outliers_mask = np.abs(modified_z_scores) > 3.5
            modified_zscore_outliers = pd.Series(False, index=cleaned_data.index)
            modified_zscore_outliers.loc[column_data.index] = modified_zscore_outliers_mask
            
            # Combine outlier detection methods
            combined_outliers = statistical_outliers | zscore_outliers | modified_zscore_outliers
            
            if combined_outliers.any():
                outlier_indices = cleaned_data[combined_outliers].index.tolist()
                
                # Calculate severity based on how extreme the outliers are
                outlier_values = cleaned_data.loc[combined_outliers, column]
                severity = min(1.0, np.mean(np.abs(stats.zscore(column_data))[zscore_outliers_mask]) / 10.0)
                
                anomalies.append(AnomalyDetection(
                    index=outlier_indices,
                    anomaly_type=AnomalyType.POINT_ANOMALY,
                    severity=severity,
                    description=f"Statistical outliers in {column}: {len(outlier_indices)} values",
                    suggested_action=CleaningAction.REPLACE,
                    metadata={
                        'column': column,
                        'method': 'combined_statistical',
                        'bounds': [lower_bound, upper_bound],
                        'outlier_values': outlier_values.tolist()
                    }
                ))
                
                # Handle outliers based on data type and severity
                if data_type == 'price_data' and column in ['Open', 'High', 'Low', 'Close']:
                    # For prices, cap outliers at reasonable bounds
                    cleaned_data.loc[combined_outliers, column] = np.clip(
                        cleaned_data.loc[combined_outliers, column],
                        lower_bound,
                        upper_bound
                    )
                    actions.append((CleaningAction.REPLACE, len(outlier_indices)))
                    
                elif column == 'Volume':
                    # For volume, replace extreme outliers with median
                    very_extreme = cleaned_data[column] > upper_bound * 2  # Very extreme volume
                    if very_extreme.any():
                        cleaned_data.loc[very_extreme, column] = column_data.median()
                        actions.append((CleaningAction.REPLACE, very_extreme.sum()))
                
                else:
                    # Generic handling - interpolate
                    cleaned_data.loc[combined_outliers, column] = np.nan
                    cleaned_data[column] = cleaned_data[column].interpolate(method='linear')
                    actions.append((CleaningAction.INTERPOLATE, len(outlier_indices)))
        
        return cleaned_data, anomalies, actions
    
    def _time_series_cleaning(self, data: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[AnomalyDetection], List[Tuple[CleaningAction, int]]]:
        """Time series specific cleaning"""
        
        cleaned_data = data.copy()
        anomalies = []
        actions = []
        
        if not isinstance(cleaned_data.index, pd.DatetimeIndex):
            return cleaned_data, anomalies, actions
        
        # Check for time series gaps
        time_diffs = cleaned_data.index.to_series().diff()
        median_gap = time_diffs.median()
        
        # Large gaps detection
        large_gaps = time_diffs > median_gap * 5
        
        if large_gaps.any():
            gap_indices = cleaned_data[large_gaps].index.tolist()
            
            anomalies.append(AnomalyDetection(
                index=gap_indices,
                anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                severity=0.6,
                description=f"Large time gaps detected: {large_gaps.sum()} instances",
                suggested_action=CleaningAction.FLAG,
                metadata={'median_gap': median_gap, 'large_gaps': large_gaps.sum()}
            ))
        
        # Detect structural breaks in price series
        if data_type == 'price_data' and 'Close' in cleaned_data.columns:
            structural_breaks = self._detect_structural_breaks(cleaned_data['Close'])
            
            if structural_breaks:
                anomalies.append(AnomalyDetection(
                    index=structural_breaks,
                    anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                    severity=0.7,
                    description=f"Structural breaks detected: {len(structural_breaks)} points",
                    suggested_action=CleaningAction.FLAG,
                    metadata={'break_points': structural_breaks}
                ))
        
        # Seasonality anomalies (if enough data)
        if len(cleaned_data) > 252:  # At least one year of daily data
            seasonal_anomalies = self._detect_seasonal_anomalies(cleaned_data, data_type)
            anomalies.extend(seasonal_anomalies)
        
        return cleaned_data, anomalies, actions
    
    def _advanced_anomaly_detection(self, data: pd.DataFrame, data_type: str) -> List[AnomalyDetection]:
        """Advanced ML-based anomaly detection"""
        
        anomalies = []
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_data) < 50:  # Need enough data
            return anomalies
        
        # Prepare data for ML models
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Run multiple anomaly detection algorithms
        detection_results = {}
        
        for name, detector in self.anomaly_detectors.items():
            try:
                if hasattr(detector, 'fit_predict'):
                    # Sklearn-style detectors
                    outlier_labels = detector.fit_predict(scaled_data)
                    detection_results[name] = outlier_labels
                elif hasattr(detector, 'fit') and hasattr(detector, 'decision_function'):
                    # PyOD-style detectors
                    detector.fit(scaled_data)
                    outlier_scores = detector.decision_function(scaled_data)
                    # Convert scores to labels (top 10% as outliers)
                    threshold = np.percentile(outlier_scores, 90)
                    outlier_labels = (outlier_scores > threshold).astype(int)
                    outlier_labels = np.where(outlier_labels == 1, -1, 1)  # Convert to sklearn convention
                    detection_results[name] = outlier_labels
                
            except Exception as e:
                logger.warning(f"Anomaly detector {name} failed: {e}")
                continue
        
        if not detection_results:
            return anomalies
        
        # Ensemble approach - combine results
        detection_matrix = np.array(list(detection_results.values()))
        outlier_votes = np.sum(detection_matrix == -1, axis=0)
        
        # Require consensus from at least half of the detectors
        consensus_threshold = max(1, len(detection_results) // 2)
        consensus_outliers = outlier_votes >= consensus_threshold
        
        if consensus_outliers.any():
            outlier_indices = numeric_data.iloc[consensus_outliers].index.tolist()
            
            # Calculate severity based on consensus strength
            max_votes = len(detection_results)
            severity = np.mean(outlier_votes[consensus_outliers]) / max_votes
            
            anomalies.append(AnomalyDetection(
                index=outlier_indices,
                anomaly_type=AnomalyType.POINT_ANOMALY,
                severity=severity,
                description=f"ML consensus anomalies: {len(outlier_indices)} points",
                suggested_action=CleaningAction.FLAG,
                metadata={
                    'detectors_used': list(detection_results.keys()),
                    'consensus_threshold': consensus_threshold,
                    'votes_distribution': outlier_votes[consensus_outliers].tolist()
                }
            ))
        
        return anomalies
    
    def _business_logic_validation(self, data: pd.DataFrame, data_type: str) -> Tuple[List[AnomalyDetection], List[Tuple[CleaningAction, int]]]:
        """Business logic validation"""
        
        anomalies = []
        actions = []
        rules = self.cleaning_rules.get(data_type, {})
        
        if data_type == 'price_data':
            # Check for impossible price movements
            if 'Close' in data.columns:
                price_changes = data['Close'].pct_change().abs()
                extreme_changes = price_changes > rules.get('outlier_thresholds', {}).get('price_change_pct', 0.5)
                
                if extreme_changes.any():
                    extreme_indices = data[extreme_changes].index.tolist()
                    
                    anomalies.append(AnomalyDetection(
                        index=extreme_indices,
                        anomaly_type=AnomalyType.CONTEXTUAL_ANOMALY,
                        severity=0.8,
                        description=f"Extreme price movements: {extreme_changes.sum()} instances",
                        suggested_action=CleaningAction.FLAG,
                        metadata={
                            'max_change': price_changes.max(),
                            'threshold': rules.get('outlier_thresholds', {}).get('price_change_pct', 0.5)
                        }
                    ))
            
            # Check volume consistency
            if 'Volume' in data.columns:
                zero_volume = data['Volume'] == 0
                zero_volume_pct = zero_volume.mean()
                
                if zero_volume_pct > rules.get('outlier_thresholds', {}).get('zero_volume_threshold', 0.05):
                    anomalies.append(AnomalyDetection(
                        index=data[zero_volume].index.tolist(),
                        anomaly_type=AnomalyType.CONTEXTUAL_ANOMALY,
                        severity=0.6,
                        description=f"Excessive zero volume days: {zero_volume_pct:.1%}",
                        suggested_action=CleaningAction.FLAG,
                        metadata={'zero_volume_pct': zero_volume_pct}
                    ))
        
        return anomalies, actions
    
    def _detect_structural_breaks(self, series: pd.Series) -> List[int]:
        """Detect structural breaks in time series"""
        
        if len(series) < 100:
            return []
        
        breaks = []
        
        try:
            # Use rolling statistics to detect breaks
            window_size = min(30, len(series) // 10)
            rolling_mean = series.rolling(window_size).mean()
            rolling_std = series.rolling(window_size).std()
            
            # Detect significant changes in mean and volatility
            mean_changes = rolling_mean.diff().abs()
            std_changes = rolling_std.diff().abs()
            
            # Thresholds based on historical volatility
            mean_threshold = mean_changes.quantile(0.95)
            std_threshold = std_changes.quantile(0.95)
            
            # Find breaks
            significant_changes = (mean_changes > mean_threshold) | (std_changes > std_threshold)
            
            if significant_changes.any():
                breaks = series[significant_changes].index.tolist()
        
        except Exception as e:
            logger.warning(f"Structural break detection failed: {e}")
        
        return breaks
    
    def _detect_seasonal_anomalies(self, data: pd.DataFrame, data_type: str) -> List[AnomalyDetection]:
        """Detect seasonal anomalies"""
        
        anomalies = []
        
        if data_type != 'price_data' or 'Close' not in data.columns:
            return anomalies
        
        try:
            # Simple seasonal decomposition
            prices = data['Close'].dropna()
            
            if len(prices) < 252:  # Need at least a year
                return anomalies
            
            # Detect day-of-week effects
            prices_with_dow = prices.to_frame()
            prices_with_dow['day_of_week'] = prices_with_dow.index.dayofweek
            
            # Check for unusual day-of-week patterns
            dow_returns = prices.pct_change().groupby(prices.index.dayofweek).mean()
            
            # If any day has consistently extreme returns
            if (dow_returns.abs() > 0.01).any():  # 1% average daily return is unusual
                extreme_days = dow_returns[dow_returns.abs() > 0.01].index.tolist()
                
                anomalies.append(AnomalyDetection(
                    index=extreme_days,
                    anomaly_type=AnomalyType.CONTEXTUAL_ANOMALY,
                    severity=0.5,
                    description=f"Unusual day-of-week patterns detected",
                    suggested_action=CleaningAction.FLAG,
                    metadata={'extreme_days': extreme_days, 'avg_returns': dow_returns.to_dict()}
                ))
        
        except Exception as e:
            logger.warning(f"Seasonal anomaly detection failed: {e}")
        
        return anomalies
    
    def _calculate_quality_improvement(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> float:
        """Calculate quality improvement score"""
        
        if original.empty or cleaned.empty:
            return 0.0
        
        try:
            # Calculate various quality metrics
            original_missing_pct = original.isna().sum().sum() / (original.shape[0] * original.shape[1])
            cleaned_missing_pct = cleaned.isna().sum().sum() / (cleaned.shape[0] * cleaned.shape[1])
            
            missing_improvement = max(0, original_missing_pct - cleaned_missing_pct)
            
            # Calculate consistency improvement (for numeric columns)
            numeric_cols = original.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return missing_improvement
            
            original_cv = original[numeric_cols].std().mean() / original[numeric_cols].mean().abs().mean()
            cleaned_cv = cleaned[numeric_cols].std().mean() / cleaned[numeric_cols].mean().abs().mean()
            
            consistency_improvement = max(0, (original_cv - cleaned_cv) / original_cv) if original_cv > 0 else 0
            
            # Weighted average of improvements
            quality_improvement = 0.6 * missing_improvement + 0.4 * consistency_improvement
            
            return min(1.0, quality_improvement)  # Cap at 100%
            
        except Exception as e:
            logger.warning(f"Quality improvement calculation failed: {e}")
            return 0.0
    
    def generate_cleaning_summary(self, report: CleaningReport) -> str:
        """Generate human-readable cleaning summary"""
        
        summary_lines = [
            f"🧹 DATA CLEANING REPORT",
            f"=" * 50,
            f"Symbol: {report.symbol}",
            f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"📊 OVERVIEW",
            f"-" * 20,
            f"Original Rows: {report.original_rows:,}",
            f"Cleaned Rows: {report.cleaned_rows:,}",
            f"Data Retention: {(report.cleaned_rows/report.original_rows)*100:.1f}%",
            f"Quality Improvement: {report.quality_improvement*100:.1f}%",
            f"",
            f"🚨 ANOMALIES DETECTED",
            f"-" * 25,
            f"Total Anomalies: {report.anomalies_detected}"
        ]
        
        # Group anomalies by type
        anomaly_by_type = {}
        for anomaly in report.anomalies:
            if anomaly.anomaly_type not in anomaly_by_type:
                anomaly_by_type[anomaly.anomaly_type] = []
            anomaly_by_type[anomaly.anomaly_type].append(anomaly)
        
        for anomaly_type, anomaly_list in anomaly_by_type.items():
            summary_lines.append(f"  {anomaly_type.value}: {len(anomaly_list)}")
        
        # Cleaning actions summary
        if report.cleaning_actions:
            summary_lines.extend([
                f"",
                f"🔧 CLEANING ACTIONS",
                f"-" * 20
            ])
            
            for action, count in report.cleaning_actions:
                summary_lines.append(f"  {action.value}: {count} instances")
        
        # Top anomalies
        if report.anomalies:
            summary_lines.extend([
                f"",
                f"🔍 TOP ANOMALIES",
                f"-" * 15
            ])
            
            # Sort by severity
            top_anomalies = sorted(report.anomalies, key=lambda x: x.severity, reverse=True)[:5]
            
            for i, anomaly in enumerate(top_anomalies):
                severity_pct = anomaly.severity * 100
                summary_lines.append(f"{i+1}. {anomaly.description} (Severity: {severity_pct:.1f}%)")
        
        return "\n".join(summary_lines)

# Example usage and testing
if __name__ == "__main__":
    print("🧹 Advanced Data Cleaning System")
    print("=" * 50)
    
    # Create test data with various quality issues
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')  # One year of data
    
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.random.randn(252).cumsum(),
        'High': 105 + np.random.randn(252).cumsum(),
        'Low': 95 + np.random.randn(252).cumsum(),
        'Close': 100 + np.random.randn(252).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, 252)
    }).set_index('Date')
    
    # Introduce various data quality issues
    # Price relationship violations
    test_data.loc[test_data.index[10], 'High'] = test_data.loc[test_data.index[10], 'Low'] - 5
    
    # Missing values
    test_data.loc[test_data.index[20:25], 'Volume'] = np.nan
    test_data.loc[test_data.index[30:32], 'Close'] = np.nan
    
    # Outliers
    test_data.loc[test_data.index[50], 'Close'] = test_data.loc[test_data.index[50], 'Close'] * 5  # 500% spike
    test_data.loc[test_data.index[60], 'Volume'] = test_data.loc[test_data.index[60], 'Volume'] * 50  # Volume spike
    
    # Negative values
    test_data.loc[test_data.index[70], 'Close'] = -10
    
    # Zero volume
    test_data.loc[test_data.index[80:90], 'Volume'] = 0
    
    print(f"📊 Created test data with quality issues: {len(test_data)} rows")
    print(f"   Missing values: {test_data.isna().sum().sum()}")
    print(f"   Negative prices: {(test_data[['Open', 'High', 'Low', 'Close']] < 0).sum().sum()}")
    print(f"   Zero volume days: {(test_data['Volume'] == 0).sum()}")
    
    # Test data cleaning
    cleaner = AdvancedDataCleaner()
    
    print(f"\n🔧 Starting comprehensive data cleaning...")
    cleaned_data, report = cleaner.clean_data(test_data, 'AAPL', 'price_data')
    
    # Print results
    print(f"\n📈 Cleaning Results:")
    print(f"   Original rows: {report.original_rows}")
    print(f"   Cleaned rows: {report.cleaned_rows}")
    print(f"   Anomalies detected: {report.anomalies_detected}")
    print(f"   Quality improvement: {report.quality_improvement:.1%}")
    
    # Print detailed summary
    summary = cleaner.generate_cleaning_summary(report)
    print(f"\n{summary}")
    
    # Verify cleaning results
    print(f"\n✅ Verification:")
    print(f"   Missing values after cleaning: {cleaned_data.isna().sum().sum()}")
    print(f"   Negative prices after cleaning: {(cleaned_data[['Open', 'High', 'Low', 'Close']] < 0).sum().sum()}")
    print(f"   Price relationship violations: {(cleaned_data['High'] < cleaned_data['Low']).sum()}")
    
    print(f"\n🎯 Advanced data cleaning system ready!")
    print(f"📋 Features:")
    print(f"   • Multi-method anomaly detection")
    print(f"   • Business rule validation")
    print(f"   • Time series analysis")
    print(f"   • Statistical outlier detection")
    print(f"   • Quality scoring")
    print(f"   • Comprehensive reporting")